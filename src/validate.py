"""Pomiar dokładności kNN: parametry globalne (LOO) i strojenie per osoba (zagnieżdżona).

Użycie:
    python src/validate.py                          # loo: params/global_ga.json
    python src/validate.py loo --params params/global.json
    python src/validate.py nested                   # walidacja zagnieżdżona (≈ 15 min)
    python src/validate.py nested --restart         # od nowa (domyślnie wznawia)
    python src/validate.py nested --limit 10        # tylko 10 osób (do testów)

Miary: accuracy (trafiona dokładna ocena), acc ±1 (pomyłka o 1 też liczy się jako
trafienie, tylko do raportu) i MAE (średni błąd bezwzględny).

=== Tryb "loo" ===
Parametry (k, sposób łączenia, skala, cechy) są wczytywane z pliku JSON i NIE są tu
stroione. Liczona jest walidacja leave-one-out osobno dla każdej osoby (evaluate.py):
każdy film przewidywany z pozostałych 89. Dla porównania baseline (najczęstsza
i średnia ocena osoby).
Uwaga: parametry dobrano na tych samych danych (LOO na wszystkich osobach), więc wynik
jest lekko optymistyczny.
Wynik: tabela w konsoli i results/validation_global.csv.

=== Tryb "nested" (uczciwa ocena strojenia per osoba) ===
Dla każdej osoby jej oceny są dzielone losowo na FOLDS części (po ok. 18). Dla każdej części:
1. strojenie per osoba (tune2.tune_user: pętla po k, w środku GA) TYLKO na pozostałych ocenach,
2. przewidywanie odłożonej części: sąsiedzi wyłącznie spośród ocen treningowych.
Każda ocena jest więc przewidziana dokładnie raz, modelem, który jej nie widział.
Na tych samych podziałach przewidywane są też: baseline (najczęstsza ocena w części
treningowej) i parametry globalne (params/global_ga.json).

Próg trafień jest ten sam co w strojeniu per osoba (tune2.MARGIN_HITS), liczony na ok. 72
ocenach treningowych. Parametry globalne dobrano na wszystkich ocenach (także odłożonych), więc ich wynik jest
minimalnie zawyżony (ok. 0.2-0.5 pp), co lekko faworyzuje globalne w porównaniu.

Osoby liczą się równolegle; wynik każdej osoby jest zapisywany od razu
(results/nested_partial.json), więc przerwane obliczenia można wznowić.
Wynik: tabela w konsoli i results/validation_nested.csv.
"""
import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import tune2
from data import ROOT, load_all
from evaluate import evaluate, knn_predictor, most_frequent, user_mean
from fast_eval import holdout_predict, user_tensor
from knn import user_data
from preprocess import prepare
from sim_matrix import combined, similarity_matrices, standardize

PARAMS_FILE = ROOT / "params" / "global_ga.json"
RESULTS = ROOT / "results"

FOLDS = 5   # walidacja zagnieżdżona: liczba podziałów ocen osoby


def load_params(path):
    params = json.loads(path.read_text(encoding="utf-8"))
    for key in ("k", "method", "scale", "features"):
        assert key in params, f"brak klucza {key!r} w {path}"
    return params


def similarity(params):
    """Macierz podobieństw 200x200 dla cech i skali z parametrów."""
    _, prep = prepare()
    names, S = similarity_matrices(prep)
    unknown = set(params["features"]) - set(names)
    assert not unknown, f"nieznane cechy w parametrach: {unknown}"
    w = np.array([n in params["features"] for n in names], dtype=float)
    if params["scale"] == "std":
        return combined(standardize(S), w, rescale=True)
    return combined(S, w)


def run_loo(params_path=PARAMS_FILE):
    params = load_params(params_path)
    _, train, _ = load_all()
    users = user_data(train)
    M = similarity(params)
    k, method = params["k"], params["method"]

    print(f"Parametry z {params_path.name}: k = {k}, {method}, {params['scale']}, "
          f"cechy ({len(params['features'])}): {', '.join(params['features'])}")
    if "loo_accuracy" in params:
        print(f"Accuracy LOO zapisana przy strojeniu: {params['loo_accuracy'] * 100:.2f}%")
    print(f"Osób: {len(users)}, walidacja leave-one-out\n")

    predictors = {
        "kNN (parametry globalne)": knn_predictor(M, k, method),
        "baseline: najczęstsza ocena": most_frequent,
        "baseline: średnia ocen": user_mean,
    }
    rows, per_user = [], {}
    print(f"  {'predyktor':<30}{'accuracy':>10}{'acc ±1':>9}{'MAE':>8}")
    for name, predictor in predictors.items():
        r = evaluate(predictor, users)
        per_user[name] = np.array([r["per_user"][u] for u in users])
        rows.append({"predictor": name, "accuracy": round(r["accuracy"], 5),
                     "accuracy_1": round(r["accuracy_1"], 5), "mae": round(r["mae"], 5)})
        print(f"  {name:<30}{r['accuracy'] * 100:>9.2f}%{r['accuracy_1'] * 100:>8.2f}%"
              f"{r['mae']:>8.3f}", flush=True)

    # kNN vs baseline: różnica sparowana po osobach
    knn_name, base_name = list(predictors)[:2]
    d = per_user[knn_name] - per_user[base_name]
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"\nkNN minus baseline (najczęstsza ocena), sparowane po osobach: "
          f"{d.mean() * 100:+.2f} pp (błąd std {se * 100:.2f} pp); "
          f"lepiej: {(d > 0).sum()}, gorzej: {(d < 0).sum()}, remis: {(d == 0).sum()} osób")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "validation_global.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nZapisano {out.relative_to(ROOT)}")
    return rows


# --- Walidacja zagnieżdżona ---

def most_frequent_rating(ratings):
    """Najczęstsza ocena (przy remisie wyższa), jak evaluate.most_frequent."""
    counts = np.bincount(ratings, minlength=6)
    return int(5 - counts[::-1].argmax())


def nested_user(user, global_features, global_k):
    """Walidacja zagnieżdżona jednej osoby (w procesie równoległym, dane z tune2._ctx).

    Zwraca przewidywania każdej oceny przez 3 strategie oraz, dla każdego podziału,
    co wybrało strojenie (GA czy globalne) i zysk GA nad globalnymi na treningu.
    """
    ctx = tune2._ctx
    rated_idx, ratings = ctx.users[user]
    n = len(ratings)
    S_user = user_tensor(ctx.S, rated_idx)
    global_mask = ctx.mask(global_features)

    perm = np.random.default_rng([tune2.SEED, int(user)]).permutation(n)
    preds = {s: np.empty(n, dtype=int) for s in ("baseline", "global", "user")}
    folds = []
    for f, test_pos in enumerate(np.array_split(perm, FOLDS)):
        train_pos = np.setdiff1d(np.arange(n), test_pos)

        # 1. strojenie tylko na ocenach treningowych (odłożone są dla GA niewidoczne)
        S_train = np.ascontiguousarray(S_user[:, train_pos][:, :, train_pos])
        r = tune2.tune_user(S_train, ratings[train_pos], global_mask, global_k, ctx.names,
                            tune2.KS, tune2.POPULATION, tune2.GENERATIONS,
                            seed=[tune2.SEED, int(user), f])

        # 2. przewidywanie odłożonych, sąsiedzi tylko spośród treningowych
        preds["user"][test_pos] = holdout_predict(S_user, ratings, train_pos, test_pos,
                                                  ctx.mask(r["features"]), r["k"], tune2.METHOD)
        preds["global"][test_pos] = holdout_predict(S_user, ratings, train_pos, test_pos,
                                                    global_mask, global_k, tune2.METHOD)
        preds["baseline"][test_pos] = most_frequent_rating(ratings[train_pos])
        folds.append({"source": r["source"], "k": r["k"], "features": r["features"],
                      "gain": r["ga_best"]["hits"] - r["global_hits"]})

    return user, {"ratings": ratings.tolist(),
                  "preds": {s: p.tolist() for s, p in preds.items()}, "folds": folds}


def scores(per_user_preds, per_user_ratings):
    """Accuracy, acc ±1 (średnie po osobach), MAE (po wszystkich ocenach), accuracy osób."""
    acc = np.array([(p == r).mean() for p, r in zip(per_user_preds, per_user_ratings)])
    acc1 = np.mean([(np.abs(p - r) <= 1).mean() for p, r in zip(per_user_preds, per_user_ratings)])
    mae = np.concatenate([np.abs(p - r) for p, r in zip(per_user_preds, per_user_ratings)]).mean()
    return acc, acc1, mae


def run_nested(workers=4, restart=False, limit=None):
    ctx = tune2.Context()
    glob = load_params(PARAMS_FILE)
    config = {"folds": FOLDS, "seed": tune2.SEED, "method": tune2.METHOD, "ks": tune2.KS,
              "penalty": tune2.PENALTY_USER, "margin_hits": tune2.MARGIN_HITS,
              "population": tune2.POPULATION,
              "generations": tune2.GENERATIONS, "features": ctx.names,
              "global": {"k": glob["k"], "features": glob["features"]}}

    # Wznawianie: osoby już policzone przy tych samych ustawieniach
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "nested_partial.json"
    saved = {"config": config, "users": {}}
    if path.exists() and not restart:
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("config") == config:
            saved = old
        else:
            print("Ustawienia zmieniły się od poprzedniego uruchomienia: liczę od nowa")
    todo = [u for u in ctx.users if str(u) not in saved["users"]]
    if limit is not None:
        todo = todo[:limit]

    print(f"Walidacja zagnieżdżona: {FOLDS} podziałów na osobę, w każdym strojenie per osoba "
          f"(k {tune2.KS}, kara {tune2.PENALTY_USER}, próg {tune2.MARGIN_HITS} trafień) "
          f"na ocenach treningowych")
    print(f"Ustawienia globalne: k = {glob['k']}, cechy: {', '.join(glob['features'])}")
    print(f"Już policzone: {len(saved['users'])} osób, do policzenia: {len(todo)} "
          f"({workers} procesów równolegle)\n", flush=True)

    start = time.perf_counter()
    if todo:
        with ProcessPoolExecutor(workers, initializer=tune2._init) as pool:
            futures = [pool.submit(nested_user, u, glob["features"], glob["k"]) for u in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                user, r = fut.result()
                saved["users"][str(user)] = r
                tmp = path.with_suffix(".tmp")           # zapis po każdej osobie (wznawianie)
                tmp.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
                tmp.replace(path)

                hits = {s: sum(p == t for p, t in zip(r["preds"][s], r["ratings"]))
                        for s in r["preds"]}
                elapsed = time.perf_counter() - start
                n_ga = sum(fold["source"] == "ga" for fold in r["folds"])
                print(f"[{i}/{len(todo)}] osoba {user}: trafienia na odłożonych: "
                      f"baseline {hits['baseline']}, globalne {hits['global']}, "
                      f"per osoba {hits['user']} (z {len(r['ratings'])}); "
                      f"GA wybrany w {n_ga}/{FOLDS} podziałach"
                      f"   [zostało ≈ {elapsed / i * (len(todo) - i) / 60:.0f} min]", flush=True)

    results = list(saved["users"].values())
    ratings = [np.array(r["ratings"]) for r in results]
    strategies = {
        "baseline: najczęstsza ocena": "baseline",
        "parametry globalne": "global",
        f"per osoba (próg {tune2.MARGIN_HITS} trafień)": "user",
    }

    print(f"\nWyniki na odłożonych ocenach ({len(results)} osób, {FOLDS} podziałów):")
    print(f"  {'strategia':<32}{'accuracy':>10}{'acc ±1':>9}{'MAE':>8}"
          f"{'vs globalne':>13}{'błąd std':>10}")
    acc_global = scores([np.array(r["preds"]["global"]) for r in results], ratings)[0]
    rows = []
    for name, key in strategies.items():
        acc, acc1, mae = scores([np.array(r["preds"][key]) for r in results], ratings)
        d = acc - acc_global
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        rows.append({"strategy": name, "accuracy": round(acc.mean(), 5),
                     "accuracy_1": round(acc1, 5), "mae": round(mae, 5),
                     "diff_vs_global": round(d.mean(), 5), "se": round(se, 5)})
        print(f"  {name:<32}{acc.mean() * 100:>9.2f}%{acc1 * 100:>8.2f}%{mae:>8.3f}"
              f"{d.mean() * 100:>+11.2f}pp{se * 100:>8.2f}pp")

    folds = [fold for r in results for fold in r["folds"]]
    gains = np.array([fold["gain"] for fold in folds])
    n_ga = sum(fold["source"] == "ga" for fold in folds)
    print(f"\nGA wybrany w {n_ga}/{len(folds)} podziałach (osoba × część, "
          f"{n_ga / len(folds) * 100:.0f}%); zysk GA nad globalnymi na treningu: "
          f"średnio {gains.mean():+.1f} trafień, max {gains.max():+d}")

    out = RESULTS / "validation_nested.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nZapisano {out.relative_to(ROOT)} i {path.relative_to(ROOT)}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", choices=["loo", "nested"], default="loo")
    parser.add_argument("--params", type=str, default=str(PARAMS_FILE),
                        help="loo: plik z parametrami (domyślnie params/global_ga.json)")
    parser.add_argument("--workers", type=int, default=4, help="nested: procesy równoległe")
    parser.add_argument("--restart", action="store_true", help="nested: licz od nowa")
    parser.add_argument("--limit", type=int, help="nested: tylko tyle osób (do testów)")
    args = parser.parse_args()
    if args.mode == "loo":
        path = Path(args.params)
        if not path.is_absolute():          # ścieżka względna: względem folderu Zadanie1
            path = ROOT / path
        run_loo(path.resolve())
    else:
        run_nested(args.workers, args.restart, args.limit)
