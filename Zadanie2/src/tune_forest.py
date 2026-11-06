"""Etap 4 dla lasu losowego: dobór parametrów lasu (przeszukanie siatki walidacją krzyżową).

Jedna iteracja = jedna para (max_features, min_samples_leaf); w niej uczymy lasy
ze 100 drzew i odczytujemy wyniki dla wszystkich rozmiarów z SIZES naraz
(las z k drzew = pierwsze k drzew lasu ze 100). Podziały filmów są te same co
dla drzewa i baseline'ów (validate.make_folds), więc wyniki są porównywalne.

Wyniki (folder results/):
    validation_forest.csv          - wszystkie konfiguracje, od najlepszej (zapisywany po każdej iteracji)
    validation_summary_forest.csv  - zestawienie: baseline'y, kNN z Zadania 1, drzewo, las
    best_forest.json               - wybrane parametry (czyta je Etap 5)
    forest_size_curve.png          - trafność walidacyjna względem liczby drzew

Użycie:
    uv run python src/tune_forest.py               # liczy siatkę; wznawia przerwane obliczenia
    uv run python src/tune_forest.py --restart     # liczy wszystko od nowa
    uv run python src/tune_forest.py --workers 2   # liczba procesów (domyślnie: rdzenie - 1)
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

import validate as V
from data import ROOT
from tune_tree import BEST_FILE as BEST_TREE_FILE
from tune_tree import KNN_ZAD1, fmt_time, progress_printer

RESULTS = ROOT / "results"
GRID_FILE = RESULTS / "validation_forest.csv"
SUMMARY_FILE = RESULTS / "validation_summary_forest.csv"
BEST_FILE = RESULTS / "best_forest.json"
CURVE_FILE = RESULTS / "forest_size_curve.png"

# siatka parametrów
MAX_FEATURES = ["sqrt", 30, 60]                # "sqrt" = 13 ze 179 kolumn
MIN_SAMPLES_LEAF = [1, 3, 5, 10, 20]
SIZES = [10, 25, 50, 100]                      # liczba drzew; uczymy max(SIZES)
N_FOLDS = V.N_FOLDS
SEED = 0           # podział filmów (i losowanie w lesie), na którym dobieramy parametry
CHECK_SEED = 1     # inny podział: czy wynik zwycięzcy nie był szczęśliwym przypadkiem

COLUMNS = ["max_features", "min_samples_leaf", "min_samples_split", "n_trees",
           "train_accuracy", "val_accuracy", "val_accuracy_1", "val_mae", "oob_accuracy",
           "gap", "vs_baseline", "vs_tree", "mean_depth", "mean_leaves"]


# ---------------------------------------------------------------- pomocnicze

def forest_params(max_features, min_samples_leaf):
    """Parametry RandomForest bez n_trees i seed; drzewa bez limitu głębokości,
    min_samples_split zawsze = 2 x min_samples_leaf (jak przy drzewie)."""
    return {"max_features": max_features, "max_depth": None,
            "min_samples_split": 2 * min_samples_leaf, "min_samples_leaf": min_samples_leaf,
            "min_gain": 0.0}


def mf_label(max_features):
    return str(max_features)


def parse_mf(label):
    return "sqrt" if str(label) == "sqrt" else int(label)


def load_done():
    """Wyniki z poprzedniego (przerwanego) uruchomienia: wiersze pełnych iteracji."""
    if not GRID_FILE.exists():
        return []
    df = pd.read_csv(GRID_FILE, dtype={"max_features": str}, keep_default_na=False)
    rows = []
    for _, group in df.groupby(["max_features", "min_samples_leaf"]):
        if sorted(group.n_trees) == sorted(SIZES):
            rows += group.to_dict("records")
    return rows


def save_grid(rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sort_values(["val_accuracy", "n_trees"], ascending=[False, True])
    df.to_csv(GRID_FILE, index=False, float_format="%.5f")
    return df


def best_row(rows):
    """Najwyższa trafność walidacyjna; przy remisie mniej drzew, potem mniejsze drzewa."""
    return max(rows, key=lambda r: (round(r["val_accuracy"], 9), -r["n_trees"], -r["mean_leaves"]))


def load_best_tree():
    """Wynik najlepszego drzewa z Etapu 4 (None, jeśli tune_tree.py nie był uruchomiony)."""
    if not BEST_TREE_FILE.exists():
        return None
    return json.loads(BEST_TREE_FILE.read_text(encoding="utf-8"))


def from_row(r):
    return {"accuracy": r["val_accuracy"], "accuracy_1": r["val_accuracy_1"], "mae": r["val_mae"]}


def forest_desc(p):
    return (f"n_trees={p['n_trees']}, max_features={p['max_features']}, "
            f"min_samples_leaf={p['min_samples_leaf']}")


# ---------------------------------------------------------------- wykres

def plot_size_curve(rows, best, baseline, tree_acc):
    """Trafność walidacyjna względem liczby drzew; jedna linia na max_features
    (z najlepszym dla niego min_samples_leaf)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, axis, surface = "#0b0b0b", "#5f5e58", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]   # sloty 1-3 palety (sprawdzone walidatorem)
    markers = ["o", "s", "^"]
    xs = list(range(len(SIZES)))

    fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for color, marker, mf in zip(colors, markers, MAX_FEATURES):
        mine = [r for r in rows if str(r["max_features"]) == mf_label(mf)]
        if not mine:
            continue
        msl = best_row(mine)["min_samples_leaf"]
        curve = sorted((r for r in mine if r["min_samples_leaf"] == msl), key=lambda r: r["n_trees"])
        val = [100 * r["val_accuracy"] for r in curve]
        name = "sqrt (13 kolumn)" if mf == "sqrt" else f"{mf} kolumn"
        ax.plot(xs[:len(val)], val, color=color, lw=2, marker=marker, ms=7,
                mec=surface, mew=1.5, label=f"max_features={name}, min_samples_leaf={msl}")
        ax.text(xs[len(val) - 1] + 0.08, val[-1], f"{name}", color=ink, va="center", fontsize=9)

    ax.axhline(100 * baseline, color=muted, lw=1.2, ls="--",
               label=f"baseline: najczęstsza ocena osoby ({100 * baseline:.1f}%)")
    if tree_acc is not None:
        ax.axhline(100 * tree_acc, color=ink, lw=1.2, ls=":",
                   label=f"najlepsze drzewo ({100 * tree_acc:.1f}%)")
    bi = SIZES.index(int(best["n_trees"]))
    ax.annotate(f"{100 * best['val_accuracy']:.1f}%", (bi, 100 * best["val_accuracy"]),
                textcoords="offset points", xytext=(0, 10), ha="center", color=ink, fontsize=9)

    ax.set_xticks(xs, [str(s) for s in SIZES])
    ax.set_xlim(-0.3, len(SIZES) - 1 + 0.9)
    ax.set_xlabel("liczba drzew w lesie", color=muted)
    ax.set_ylabel("trafność walidacyjna [%]", color=muted)
    ax.set_title(f"Las losowy: trafność walidacyjna a liczba drzew\n"
                 f"{N_FOLDS}-krotna walidacja, {len(V.get_users())} osób, seed {SEED}",
                 color=ink, fontsize=11, loc="left")
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(axis)
    ax.tick_params(colors=muted, length=0)
    ax.margins(y=0.12)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2,
              fontsize=8.5, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(CURVE_FILE, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- główny przebieg

def main():
    parser = argparse.ArgumentParser(description="Dobór parametrów lasu losowego (Etap 4).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--restart", action="store_true", help="usuń poprzednie wyniki siatki")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    if args.restart and GRID_FILE.exists():
        GRID_FILE.unlink()

    users = V.get_users()
    print("Baseline'y (te same podziały filmów co las):")
    base = {name: V.baseline_scores(rule, seed=SEED) for name, rule in V.BASELINES.items()}
    for name, s in base.items():
        print(f"  {name:<26} accuracy {s['accuracy']:.1%}  ±1 {s['accuracy_1']:.1%}  "
              f"MAE {s['mae']:.3f}")
    baseline_acc = base["najczęstsza ocena osoby"]["accuracy"]
    best_tree = load_best_tree()
    tree_acc = best_tree["validation"]["accuracy"] if best_tree else None
    if best_tree:
        print(f"Najlepsze drzewo ({BEST_TREE_FILE.relative_to(ROOT)}): val {tree_acc:.1%}")
    else:
        print("Brak results/best_tree.json: pomijam porównanie z drzewem")

    pairs = [(mf, msl) for mf in MAX_FEATURES for msl in MIN_SAMPLES_LEAF]
    rows = load_done()
    done_pairs = {(str(r["max_features"]), int(r["min_samples_leaf"])) for r in rows}
    for r in rows:
        r["vs_baseline"] = r["val_accuracy"] - baseline_acc
        r["vs_tree"] = r["val_accuracy"] - tree_acc if tree_acc is not None else float("nan")

    print(f"\nSiatka: {len(MAX_FEATURES)} x max_features, {len(MIN_SAMPLES_LEAF)} x "
          f"min_samples_leaf = {len(pairs)} iteracji ({len(pairs) * len(SIZES)} konfiguracji "
          f"z {len(SIZES)} rozmiarami lasu)")
    print(f"{len(users)} osób x {N_FOLDS} części, las do {max(SIZES)} drzew, seed = {SEED}, "
          f"procesy: {args.workers}")
    if done_pairs:
        print(f"Wznowienie: {len(done_pairs)} iteracji już policzonych w "
              f"{GRID_FILE.relative_to(ROOT)} (od nowa: --restart)")

    executor = (ProcessPoolExecutor(max_workers=args.workers, initializer=V.init_worker)
                if args.workers > 1 else None)
    t_start = time.perf_counter()
    times = []
    try:
        for it, (mf, msl) in enumerate(pairs, 1):
            params = forest_params(mf, msl)
            header = (f"\n[{it:>2}/{len(pairs)}] max_features={mf}, min_samples_leaf={msl}, "
                      f"min_samples_split={params['min_samples_split']}")
            if (mf_label(mf), msl) in done_pairs:
                print(header + "  -> już policzone, pomijam")
                continue
            print(header, flush=True)

            t0 = time.perf_counter()
            results = V.evaluate_forest(params, SIZES, N_FOLDS, SEED, executor,
                                        progress_printer(t0))
            for r in results:
                r.update(max_features=mf_label(mf), min_samples_leaf=msl,
                         min_samples_split=params["min_samples_split"],
                         gap=r["train_accuracy"] - r["val_accuracy"],
                         vs_baseline=r["val_accuracy"] - baseline_acc,
                         vs_tree=r["val_accuracy"] - tree_acc if tree_acc is not None
                         else float("nan"))
                print(f"        n_trees={r['n_trees']:<4} train {r['train_accuracy']:6.1%}  "
                      f"val {r['val_accuracy']:6.1%}  OOB {r['oob_accuracy']:6.1%}  "
                      f"gap {100 * r['gap']:5.1f}")
            rows += results
            save_grid(rows)

            times.append(time.perf_counter() - t0)
            b = best_row(rows)
            left = sum((mf_label(m), s) not in done_pairs for m, s in pairs[it:])
            print(f"        najlepsza do tej pory: max_features={b['max_features']}, "
                  f"min_samples_leaf={b['min_samples_leaf']}, n_trees={b['n_trees']} "
                  f"-> val {b['val_accuracy']:.1%}")
            print(f"        czas: {fmt_time(times[-1])} | łącznie "
                  f"{fmt_time(time.perf_counter() - t_start)} | do końca całości "
                  f"~{fmt_time(sum(times) / len(times) * left)}")

        df = save_grid(rows)
        best = best_row(rows)
        best_mf, best_msl, best_n = (parse_mf(best["max_features"]), int(best["min_samples_leaf"]),
                                     int(best["n_trees"]))
        best_params = {"n_trees": best_n, **forest_params(best_mf, best_msl)}

        print(f"\nSprawdzenie zwycięzcy na innym podziale filmów (seed = {CHECK_SEED}):", flush=True)
        t0 = time.perf_counter()
        check = V.evaluate_forest(forest_params(best_mf, best_msl), [best_n], N_FOLDS,
                                  CHECK_SEED, executor, progress_printer(t0))[0]
    finally:
        if executor is not None:
            executor.shutdown()
    check_base = V.baseline_scores(V.most_frequent, seed=CHECK_SEED)

    # --- zestawienie
    summary = [
        {"predictor": "baseline: najczęstsza ocena osoby", "seed": SEED,
         **base["najczęstsza ocena osoby"], "oob_accuracy": "", "uwagi": ""},
        {"predictor": "baseline: średnia ocen osoby", "seed": SEED,
         **base["średnia ocen osoby"], "oob_accuracy": "", "uwagi": ""},
        {"predictor": "kNN z Zadania 1", "seed": "", **KNN_ZAD1, "oob_accuracy": "",
         "uwagi": "leave-one-out (uczenie na 89 filmach), porównanie orientacyjne"},
    ]
    if best_tree:
        v = best_tree["validation"]
        summary.append({"predictor": f"drzewo: najlepsze ({best_tree['params']})", "seed": v["seed"],
                        "accuracy": v["accuracy"], "accuracy_1": v["accuracy_1"], "mae": v["mae"],
                        "oob_accuracy": "", "uwagi": "z results/best_tree.json"})
    summary += [
        {"predictor": f"las: najlepszy ({forest_desc(best_params)})", "seed": SEED,
         **from_row(best), "oob_accuracy": best["oob_accuracy"],
         "uwagi": f"wybrany z {len(rows)} konfiguracji"},
        {"predictor": "las: najlepszy, inny podział filmów", "seed": CHECK_SEED,
         **from_row(check), "oob_accuracy": check["oob_accuracy"],
         "uwagi": "sprawdzenie, czy wynik nie był przypadkiem"},
        {"predictor": "baseline: najczęstsza ocena osoby, inny podział", "seed": CHECK_SEED,
         **check_base, "oob_accuracy": "", "uwagi": ""},
    ]
    pd.DataFrame(summary).to_csv(SUMMARY_FILE, index=False, float_format="%.5f")

    BEST_FILE.write_text(json.dumps({
        "params": best_params,
        "validation": {"seed": SEED, "n_folds": N_FOLDS, **from_row(best),
                       "train_accuracy": best["train_accuracy"],
                       "oob_accuracy": best["oob_accuracy"]},
        "check": {"seed": CHECK_SEED, **from_row(check), "oob_accuracy": check["oob_accuracy"]},
        "baseline": {"seed": SEED, **base["najczęstsza ocena osoby"]},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_size_curve(rows, best, baseline_acc, tree_acc)

    print("\n10 najlepszych konfiguracji:")
    print(df.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nZestawienie (accuracy / ±1 / MAE / OOB):")
    for s in summary:
        seed = f"seed {s['seed']}" if s["seed"] != "" else "Zad. 1"
        oob = f"  OOB {s['oob_accuracy']:.1%}" if s["oob_accuracy"] != "" else ""
        print(f"  {s['predictor']:<72} {seed:<7} {s['accuracy']:6.1%}  "
              f"{s['accuracy_1']:6.1%}  {s['mae']:.3f}{oob}")
    print(f"\nWybrane parametry: {best_params}")
    print(f"Zapisano: {GRID_FILE.relative_to(ROOT)}, {SUMMARY_FILE.relative_to(ROOT)}, "
          f"{BEST_FILE.relative_to(ROOT)}, {CURVE_FILE.relative_to(ROOT)}")
    print(f"Czas całości: {fmt_time(time.perf_counter() - t_start)}")


if __name__ == "__main__":
    main()
