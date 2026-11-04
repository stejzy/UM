"""Strojenie kNN: ustawienia globalne i per osoba (pętla po k, w środku algorytm genetyczny).

Użycie:
    python src/tune2.py global                       # pełne strojenie -> params/global_ga.json
    python src/tune2.py global --restart             # od nowa (domyślnie wznawia)
    python src/tune2.py global --ks 10 30            # tylko wybrane k (do testów)
    python src/tune2.py global --generations 5       # krótszy GA (do testów)
    python src/tune2.py users                        # per osoba -> params/users.json
    python src/tune2.py users --restart              # od nowa (domyślnie wznawia)
    python src/tune2.py users --limit 10             # tylko 10 osób (do testów)
"""
import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from data import ROOT, load_all
from evaluate import evaluate, most_frequent
from fast_eval import loo_accuracy_all_k, predict_all_k, user_tensor
from ga import run_ga
from knn import user_data
from preprocess import prepare
from sim_matrix import combined, similarity_matrices, standardize

PARAMS = ROOT / "params"
RESULTS = ROOT / "results"
GA_LOGS = RESULTS / "ga_logs"   # folder z logami GA, osobny plik dla każdego k

METHOD = "wvote"
SCALE = "raw"
PENALTY = 0.0001   # bardzo mała kara: przy (prawie) równym wyniku wygrywa mniej cech
SEED = 0

KS = [3, 5, 7, 9, 11, 15, 20, 30, 45]   # wartości k w pętli (dla każdego osobny GA)
POPULATION = 30                          # liczba masek w pokoleniu GA
GENERATIONS = 40                         # maksymalna liczba pokoleń GA
# zestaw startowy wybrany ręcznie w etapie 3, jedna z masek w populacji początkowej GA
START_FEATURES = ["year", "runtime", "vote_count", "genres", "keywords", "directors", "cast"]

# Strojenie per osoba
PENALTY_USER = 0.003   # kara za cechę (≈ 1/4 trafienia na 90): mniej cech, mniej przeuczenia
MARGIN_HITS = 5        # zestaw z GA tylko, gdy ma ≥ 5 trafień więcej niż globalny


class Context:
    """Wszystko, czego potrzeba do oceny: osoby, nazwy cech, macierze cech (raw i std)."""

    def __init__(self):
        _, train, _ = load_all()
        self.users = user_data(train)
        _, prep = prepare()
        self.names, self.S = similarity_matrices(prep)
        self.Z = standardize(self.S)

    def mask(self, features):
        return np.array([n in features for n in self.names], dtype=float)


def global_scores(ctx, features, scale, method, ks):
    """Średnia accuracy leave-one-out (po osobach) dla każdego k z ks."""
    w = ctx.mask(features)
    M = combined(ctx.Z, w, rescale=True) if scale == "std" else combined(ctx.S, w)
    accs = np.zeros(len(ks))
    for rated_idx, ratings in ctx.users.values():
        sim = M[np.ix_(rated_idx, rated_idx)]          # kopia, więc można zmieniać przekątną
        np.fill_diagonal(sim, -np.inf)                 # film nie jest swoim sąsiadem
        preds = predict_all_k(sim, ratings, np.minimum(ks, len(ratings) - 1), method)
        accs += (preds == ratings[None, :]).mean(axis=1)
    return accs / len(ctx.users)


_ctx = None


def _init():
    """Każdy proces równoległy wczytuje dane i macierze cech raz."""
    global _ctx
    _ctx = Context()


def ga_for_k(k, population, generations):
    """GA dla jednego k: najlepszy wspólny zestaw cech. Zwraca wiersz z wynikiem."""
    ctx, start = _ctx, time.perf_counter()
    n_features = len(ctx.names)

    def to_features(mask):
        return [n for n, m in zip(ctx.names, mask) if m]

    def fitness(mask):
        acc = global_scores(ctx, to_features(mask), SCALE, METHOD, ks=[k])[0]
        return acc - PENALTY * mask.sum()

    rng = np.random.default_rng([SEED, k])
    initial = [np.ones(n_features), ctx.mask(START_FEATURES)]   # start: wszystkie, startowe 7
    mask, fit, evaluated = run_ga(fitness, n_features, rng, initial=initial,
                                  population=population, generations=generations,
                                  log=f"k = {k:<2}", names=ctx.names,
                                  log_file=GA_LOGS / f"k_{k:02d}.csv")
    features = to_features(mask)
    return {"k": k, "fitness": round(fit, 5),
            "accuracy": round(fit + PENALTY * len(features), 5),
            "n_features": len(features), "features": features,
            "evaluated_masks": evaluated, "minutes": round((time.perf_counter() - start) / 60, 1)}


def tune_global(ks=KS, population=POPULATION, generations=GENERATIONS, workers=4,
                restart=False):
    ctx = Context()
    baseline = evaluate(most_frequent, ctx.users)["accuracy"]
    config = {"method": METHOD, "scale": SCALE, "penalty": PENALTY, "seed": SEED,
              "population": population, "generations": generations, "features": ctx.names}

    # Wznawianie: wyniki już policzonych k (przy tych samych ustawieniach)
    RESULTS.mkdir(exist_ok=True)
    partial = RESULTS / "global_ga_partial.json"
    done = {}
    if partial.exists() and not restart:
        old = json.loads(partial.read_text(encoding="utf-8"))
        if old.get("config") == config:
            done = {int(k): row for k, row in old["by_k"].items()}
    todo = [k for k in ks if k not in done]

    print(f"Baseline (najczęstsza ocena osoby): {baseline * 100:.2f}%")
    print(f"Ustawienia: {METHOD}, {SCALE}; GA: populacja {population}, maks. {generations} "
          f"pokoleń; k: {list(ks)}")
    if done:
        print(f"Wznawianie: już policzone k = {sorted(k for k in done if k in ks)}")
    print(f"Do policzenia: k = {todo}, {workers} procesów równolegle "
          f"(jedno k to zwykle kilka minut)\n", flush=True)

    if todo:
        with ProcessPoolExecutor(workers, initializer=_init) as pool:
            futures = [pool.submit(ga_for_k, k, population, generations) for k in todo]
            for fut in as_completed(futures):
                row = fut.result()
                done[row["k"]] = row
                partial.write_text(json.dumps({"config": config, "by_k": done},
                                              ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"k = {row['k']:<3} accuracy {row['accuracy'] * 100:.2f}%  "
                      f"({row['n_features']} cech, {row['evaluated_masks']} masek, "
                      f"{row['minutes']} min): {', '.join(row['features'])}", flush=True)

    rows = [done[k] for k in sorted(ks)]
    print(f"\n{'k':>4}{'accuracy':>11}{'cech':>6}  cechy")
    for r in rows:
        print(f"{r['k']:>4}{r['accuracy'] * 100:>10.2f}%{r['n_features']:>6}  {', '.join(r['features'])}")

    with open(RESULTS / "global_ga.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**r, "features": ", ".join(r["features"])} for r in rows)

    best = max(rows, key=lambda r: (r["fitness"], r["k"]))     # remis -> większe k
    params = {
        "k": best["k"],
        "method": METHOD,
        "scale": SCALE,
        "features": best["features"],
        "loo_accuracy": round(best["accuracy"], 4),
        "baseline_accuracy": round(float(baseline), 4),
        "ga": {"population": population, "generations": generations,
               "penalty": PENALTY, "seed": SEED, "ks": list(ks)},
        "note": "Pętla po k, dla każdego k GA wybiera jeden zestaw cech dla wszystkich osób. "
                "LOO na wszystkich osobach, k i cechy dobrane na tych samych danych "
                "(lekko optymistyczne).",
    }
    PARAMS.mkdir(exist_ok=True)
    (PARAMS / "global_ga.json").write_text(json.dumps(params, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"\nNajlepsze: k = {best['k']}, accuracy {best['accuracy'] * 100:.2f}% "
          f"(baseline {baseline * 100:.2f}%), cechy: {', '.join(best['features'])}")
    print("Zapisano params/global_ga.json i results/global_ga.csv")
    return params


def tune_user(S_user, ratings, global_mask, global_k, names, ks, population, generations,
              seed):
    """Dobór k i cech dla jednej osoby. Zwraca słownik do zapisania w users.json.

    S_user, ratings: macierze cech i oceny osoby (z user_tensor); seed: lista liczb dla rng.
    """
    n, n_features = len(ratings), len(names)

    def accuracy(mask, k):
        return float(loo_accuracy_all_k(S_user, ratings, mask, [k], METHOD)[0])

    def to_features(mask):
        return [nm for nm, m in zip(names, mask) if m]

    by_k, best = {}, None
    for k in ks:
        rng = np.random.default_rng([*seed, k])
        mask, fit, _ = run_ga(lambda m: accuracy(m, k) - PENALTY_USER * m.sum(), n_features,
                              rng, initial=[global_mask, np.ones(n_features)],
                              population=population, generations=generations)
        acc = accuracy(mask, k)
        by_k[str(k)] = {"accuracy": round(acc, 4), "fitness": round(fit, 4),
                        "features": to_features(mask)}
        if best is None or fit >= best["fitness"]:      # remis -> większe k
            best = {"k": k, "mask": mask, "fitness": fit, "accuracy": acc}

    # Porównanie w trafieniach (liczby całkowite), nie w accuracy: bez błędów zaokrągleń
    global_acc = accuracy(global_mask, global_k)
    global_hits, ga_hits = round(global_acc * n), round(best["accuracy"] * n)
    use_ga = ga_hits - global_hits >= MARGIN_HITS
    k, mask = (best["k"], best["mask"]) if use_ga else (global_k, global_mask)
    return {
        "k": int(k),
        "features": to_features(mask),
        "source": "ga" if use_ga else "global",
        "accuracy": round(accuracy(mask, k), 4),
        "hits": ga_hits if use_ga else global_hits,
        "global_accuracy": round(global_acc, 4),
        "global_hits": global_hits,
        "n_ratings": n,
        "ga_best": {"k": int(best["k"]), "features": to_features(best["mask"]),
                    "accuracy": round(best["accuracy"], 4), "hits": ga_hits,
                    "fitness": round(best["fitness"], 4)},
        "by_k": by_k,
    }


def _tune_user_worker(user, global_features, global_k, ks, population, generations):
    """Strojenie jednej osoby w procesie równoległym (dane z _ctx)."""
    ctx = _ctx
    rated_idx, ratings = ctx.users[user]
    result = tune_user(user_tensor(ctx.S, rated_idx), ratings, ctx.mask(global_features),
                       global_k, ctx.names, ks, population, generations,
                       seed=[SEED, int(user)])
    return user, result


def tune_users(ks=KS, population=POPULATION, generations=GENERATIONS, workers=4,
               restart=False, limit=None):
    ctx = Context()
    glob = json.loads((PARAMS / "global_ga.json").read_text(encoding="utf-8"))
    config = {"method": METHOD, "scale": SCALE, "ks": list(ks), "penalty": PENALTY_USER,
              "margin_hits": MARGIN_HITS, "population": population, "generations": generations,
              "seed": SEED, "features": ctx.names,
              "global": {"k": glob["k"], "features": glob["features"]}}

    # Wznawianie: osoby już policzone przy tych samych ustawieniach
    path = PARAMS / "users.json"
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

    print(f"Ustawienia globalne: k = {glob['k']}, cechy: {', '.join(glob['features'])}")
    print(f"GA per osoba: k {list(ks)}, kara {PENALTY_USER} za cechę, próg {MARGIN_HITS} "
          f"trafień ponad globalne; populacja {population}, maks. {generations} pokoleń")
    print(f"Już policzone: {len(saved['users'])} osób, do policzenia: {len(todo)} "
          f"({workers} procesów równolegle)\n", flush=True)

    start = time.perf_counter()
    if todo:
        with ProcessPoolExecutor(workers, initializer=_init) as pool:
            futures = [pool.submit(_tune_user_worker, u, glob["features"], glob["k"], ks,
                                   population, generations) for u in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                user, r = fut.result()
                saved["users"][str(user)] = r
                tmp = path.with_suffix(".tmp")           # zapis po każdej osobie (wznawianie)
                tmp.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
                tmp.replace(path)

                elapsed = time.perf_counter() - start
                print(f"[{i}/{len(todo)}] osoba {user}: {r['source']:<6} k = {r['k']:<3} "
                      f"trafienia {r['hits']}/{r['n_ratings']} (globalne {r['global_hits']}, "
                      f"najlepsze z GA {r['ga_best']['hits']}), cechy: {', '.join(r['features'])}"
                      f"   [zostało ≈ {elapsed / i * (len(todo) - i) / 60:.0f} min]", flush=True)

    users = saved["users"].values()
    n_ga = sum(r["source"] == "ga" for r in users)
    acc = np.mean([r["accuracy"] for r in users])
    glob_acc = np.mean([r["global_accuracy"] for r in users])
    print(f"\nGotowe: {len(users)} osób w params/users.json; zestaw z GA: {n_ga}, "
          f"globalny: {len(users) - n_ga}")
    print(f"Accuracy LOO (na danych strojenia, zawyżona): {acc * 100:.2f}% "
          f"wobec globalnych {glob_acc * 100:.2f}%")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["global", "users"])
    parser.add_argument("--ks", type=int, nargs="+", default=KS, help="sprawdzane wartości k")
    parser.add_argument("--population", type=int, default=POPULATION)
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    parser.add_argument("--workers", type=int, default=4, help="liczba procesów równoległych")
    parser.add_argument("--restart", action="store_true", help="licz od nowa, bez wznawiania")
    parser.add_argument("--limit", type=int, help="users: tylko tyle osób (do testów)")
    args = parser.parse_args()
    if args.step == "global":
        tune_global(args.ks, args.population, args.generations, args.workers, args.restart)
    elif args.step == "users":
        tune_users(args.ks, args.population, args.generations, args.workers, args.restart,
                   args.limit)
