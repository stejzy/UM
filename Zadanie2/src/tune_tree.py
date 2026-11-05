"""Etap 4: dobór parametrów drzewa decyzyjnego (przeszukanie siatki walidacją krzyżową).

Jedna iteracja = jedna para (min_samples_leaf, min_gain); w niej uczymy drzewa bez
limitu głębokości i odczytujemy wyniki dla wszystkich głębokości z DEPTHS naraz.

Wyniki (folder results/):
    validation_tree.csv     - wszystkie konfiguracje, od najlepszej (zapisywany po każdej iteracji)
    validation_summary.csv  - zestawienie: baseline'y, kNN z Zadania 1, drzewa
    best_tree.json          - wybrane parametry (czyta je Etap 5)
    tree_depth_curve.png    - trafność treningowa i walidacyjna względem głębokości

Użycie:
    uv run python src/tune_tree.py               # liczy siatkę; wznawia przerwane obliczenia
    uv run python src/tune_tree.py --restart     # liczy wszystko od nowa
    uv run python src/tune_tree.py --workers 2   # liczba procesów (domyślnie: rdzenie - 1)
"""
import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

import validate as V
from data import ROOT

RESULTS = ROOT / "results"
GRID_FILE = RESULTS / "validation_tree.csv"
SUMMARY_FILE = RESULTS / "validation_summary.csv"
BEST_FILE = RESULTS / "best_tree.json"
CURVE_FILE = RESULTS / "tree_depth_curve.png"

# siatka parametrów
MIN_SAMPLES_LEAF = [1, 3, 5, 8, 12, 16, 24, 32]
MIN_GAIN = [0.0, 0.01, 0.03]
DEPTHS = [1, 2, 3, 4, 5, 6, 8, None]          # None = bez limitu
N_FOLDS = V.N_FOLDS
SEED = 0           # podział filmów, na którym dobieramy parametry
CHECK_SEED = 1     # inny podział: czy wynik zwycięzcy nie był szczęśliwym przypadkiem

DEFAULT = {"max_depth": 5, "min_samples_leaf": 5, "min_gain": 0.0}   # parametry z Etapu 2

# wynik kNN z Zadania 1 (Zadanie1/results/validation_global.csv, metoda leave-one-out)
KNN_ZAD1 = {"accuracy": 0.41766, "accuracy_1": 0.71207, "mae": 1.04413}

COLUMNS = ["max_depth", "min_samples_leaf", "min_samples_split", "min_gain",
           "train_accuracy", "val_accuracy", "val_accuracy_1", "val_mae",
           "gap", "vs_baseline", "mean_depth", "mean_leaves"]


# ---------------------------------------------------------------- pomocnicze

def fmt_time(seconds):
    seconds = int(round(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def depth_label(d):
    return "None" if d is None else str(d)


def tree_params(min_samples_leaf, min_gain):
    """Parametry DecisionTree bez max_depth; min_samples_split zawsze = 2 x min_samples_leaf."""
    return {"min_samples_split": 2 * min_samples_leaf, "min_samples_leaf": min_samples_leaf,
            "min_gain": min_gain}


def progress_printer(t0, prefix="        "):
    """Wypisuje postęp co ok. 10% osób, z czasem i szacunkiem do końca iteracji."""
    def show(done, total):
        step = max(1, math.ceil(total / 10))
        if done % step and done != total:
            return
        elapsed = time.perf_counter() - t0
        line = f"{prefix}osoby {done:>3}/{total} ({done / total:4.0%})  {fmt_time(elapsed)}"
        if done < total:
            line += f"  | do końca iteracji ~{fmt_time(elapsed / done * (total - done))}"
        print(line, flush=True)
    return show


def load_done():
    """Wyniki z poprzedniego (przerwanego) uruchomienia: wiersze pełnych iteracji."""
    if not GRID_FILE.exists():
        return []
    # keep_default_na=False: inaczej pandas wczyta napis "None" jako NaN
    df = pd.read_csv(GRID_FILE, dtype={"max_depth": str}, keep_default_na=False)
    rows = []
    for (msl, gain), group in df.groupby(["min_samples_leaf", "min_gain"]):
        if sorted(group.max_depth) == sorted(map(depth_label, DEPTHS)):
            rows += group.to_dict("records")
    return rows


def save_grid(rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sort_values(["val_accuracy", "mean_leaves"], ascending=[False, True])
    df.to_csv(GRID_FILE, index=False, float_format="%.5f")
    return df


def best_row(rows):
    """Najwyższa trafność walidacyjna; przy remisie mniejsze drzewo."""
    return max(rows, key=lambda r: (round(r["val_accuracy"], 9), -r["mean_leaves"]))


def to_int_depth(label):
    return None if str(label) == "None" else int(label)


# ---------------------------------------------------------------- wykres

def plot_depth_curve(rows, best, baseline):
    """Trafność treningowa i walidacyjna względem głębokości dla zwycięskiej pary parametrów."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    msl, gain = best["min_samples_leaf"], best["min_gain"]
    curve = [r for r in rows if r["min_samples_leaf"] == msl and r["min_gain"] == gain]
    curve.sort(key=lambda r: math.inf if to_int_depth(r["max_depth"]) is None
               else to_int_depth(r["max_depth"]))
    labels = ["bez limitu" if r["max_depth"] == "None" else r["max_depth"] for r in curve]
    xs = range(len(curve))
    train = [100 * r["train_accuracy"] for r in curve]
    val = [100 * r["val_accuracy"] for r in curve]

    ink, muted, grid, axis = "#0b0b0b", "#5f5e58", "#e1e0d9", "#c3c2b7"
    blue, orange = "#2a78d6", "#eb6834"      # sloty 1-2 palety (sprawdzone walidatorem)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(xs, train, color=blue, lw=2, marker="o", ms=6, label="trening (72 filmy)")
    ax.plot(xs, val, color=orange, lw=2, marker="o", ms=6, label="walidacja (18 filmów)")
    ax.axhline(100 * baseline, color=muted, lw=1.2, ls="--",
               label="baseline: najczęstsza ocena osoby")
    best_i = labels.index("bez limitu" if best["max_depth"] == "None" else best["max_depth"])
    ax.annotate(f"{val[best_i]:.1f}%", (best_i, val[best_i]), textcoords="offset points",
                xytext=(0, -16), ha="center", color=ink, fontsize=9)
    ax.text(len(curve) - 1, train[-1], "  trening", color=ink, va="center", fontsize=9)
    ax.text(len(curve) - 1, val[-1], "  walidacja", color=ink, va="center", fontsize=9)

    ax.set_xticks(list(xs), labels)
    ax.set_xlim(-0.3, len(curve) - 0.3 + 0.9)
    ax.set_xlabel("max_depth", color=muted)
    ax.set_ylabel("trafność [%]", color=muted)
    ax.set_title(f"Drzewo decyzyjne: przeuczenie wraz z głębokością\n"
                 f"min_samples_leaf={msl}, min_gain={gain:g}, {N_FOLDS}-krotna walidacja, "
                 f"{len(V.get_users())} osób",
                 color=ink, fontsize=11, loc="left")
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(axis)
    ax.tick_params(colors=muted, length=0)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(CURVE_FILE, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- główny przebieg

def main():
    parser = argparse.ArgumentParser(description="Dobór parametrów drzewa (Etap 4).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--restart", action="store_true", help="usuń poprzednie wyniki siatki")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    if args.restart and GRID_FILE.exists():
        GRID_FILE.unlink()

    users = V.get_users()
    print("Baseline'y (te same podziały filmów co drzewo):")
    base = {name: V.baseline_scores(rule, seed=SEED) for name, rule in V.BASELINES.items()}
    for name, s in base.items():
        print(f"  {name:<26} accuracy {s['accuracy']:.1%}  ±1 {s['accuracy_1']:.1%}  "
              f"MAE {s['mae']:.3f}")
    baseline_acc = base["najczęstsza ocena osoby"]["accuracy"]

    pairs = [(msl, gain) for msl in MIN_SAMPLES_LEAF for gain in MIN_GAIN]
    rows = load_done()
    done_pairs = {(int(r["min_samples_leaf"]), round(float(r["min_gain"]), 6)) for r in rows}
    for r in rows:
        r["vs_baseline"] = r["val_accuracy"] - baseline_acc

    print(f"\nSiatka: {len(MIN_SAMPLES_LEAF)} x min_samples_leaf, {len(MIN_GAIN)} x min_gain = "
          f"{len(pairs)} iteracji ({len(pairs) * len(DEPTHS)} konfiguracji "
          f"z {len(DEPTHS)} głębokościami)")
    print(f"{len(users)} osób x {N_FOLDS} części, seed = {SEED}, procesy: {args.workers}")
    if done_pairs:
        print(f"Wznowienie: {len(done_pairs)} iteracji już policzonych w "
              f"{GRID_FILE.relative_to(ROOT)} (od nowa: --restart)")

    executor = (ProcessPoolExecutor(max_workers=args.workers, initializer=V.init_worker)
                if args.workers > 1 else None)
    t_start = time.perf_counter()
    times = []
    try:
        for it, (msl, gain) in enumerate(pairs, 1):
            params = tree_params(msl, gain)
            header = (f"\n[{it:>2}/{len(pairs)}] min_samples_leaf={msl}, "
                      f"min_samples_split={params['min_samples_split']}, min_gain={gain:g}")
            if (msl, round(gain, 6)) in done_pairs:
                print(header + "  -> już policzone, pomijam")
                continue
            print(header, flush=True)

            t0 = time.perf_counter()
            results = V.evaluate_tree(params, DEPTHS, N_FOLDS, SEED, executor,
                                      progress_printer(t0))
            for r in results:
                r.update(max_depth=depth_label(r["max_depth"]), min_samples_leaf=msl,
                         min_samples_split=params["min_samples_split"], min_gain=gain,
                         gap=r["train_accuracy"] - r["val_accuracy"],
                         vs_baseline=r["val_accuracy"] - baseline_acc)
                print(f"        depth={r['max_depth']:<5} train {r['train_accuracy']:6.1%}  "
                      f"val {r['val_accuracy']:6.1%}  gap {100 * r['gap']:5.1f}  "
                      f"liście {r['mean_leaves']:5.1f}")
            rows += results
            save_grid(rows)

            times.append(time.perf_counter() - t0)
            b = best_row(rows)
            left = sum((m, round(g, 6)) not in done_pairs for m, g in pairs[it:])
            print(f"        najlepsza do tej pory: depth={b['max_depth']}, "
                  f"min_samples_leaf={b['min_samples_leaf']}, min_gain={b['min_gain']:g} "
                  f"-> val {b['val_accuracy']:.1%}")
            print(f"        czas: {fmt_time(times[-1])} | łącznie "
                  f"{fmt_time(time.perf_counter() - t_start)} | do końca całości "
                  f"~{fmt_time(sum(times) / len(times) * left)}")

        df = save_grid(rows)
        best = best_row(rows)
        best_depth = to_int_depth(best["max_depth"])
        best_params = {"max_depth": best_depth, **tree_params(int(best["min_samples_leaf"]),
                                                              float(best["min_gain"]))}

        print(f"\nSprawdzenie zwycięzcy na innym podziale filmów (seed = {CHECK_SEED}):", flush=True)
        t0 = time.perf_counter()
        check = V.evaluate_tree(tree_params(int(best["min_samples_leaf"]), float(best["min_gain"])),
                                [best_depth], N_FOLDS, CHECK_SEED, executor,
                                progress_printer(t0))[0]
    finally:
        if executor is not None:
            executor.shutdown()
    check_base = V.baseline_scores(V.most_frequent, seed=CHECK_SEED)

    # --- zestawienie
    default = next(r for r in rows if r["max_depth"] == depth_label(DEFAULT["max_depth"])
                   and r["min_samples_leaf"] == DEFAULT["min_samples_leaf"]
                   and r["min_gain"] == DEFAULT["min_gain"])

    def tree_desc(p):
        return (f"max_depth={depth_label(p['max_depth'])}, min_samples_leaf={p['min_samples_leaf']}, "
                f"min_gain={p['min_gain']:g}")

    def from_row(r):
        return {"accuracy": r["val_accuracy"], "accuracy_1": r["val_accuracy_1"], "mae": r["val_mae"]}

    summary = [
        {"predictor": "baseline: najczęstsza ocena osoby", "seed": SEED,
         **base["najczęstsza ocena osoby"], "uwagi": ""},
        {"predictor": "baseline: średnia ocen osoby", "seed": SEED,
         **base["średnia ocen osoby"], "uwagi": ""},
        {"predictor": "kNN z Zadania 1", "seed": "", **KNN_ZAD1,
         "uwagi": "leave-one-out (uczenie na 89 filmach), porównanie orientacyjne"},
        {"predictor": f"drzewo: parametry z Etapu 2 ({tree_desc(DEFAULT)})", "seed": SEED,
         **from_row(default), "uwagi": ""},
        {"predictor": f"drzewo: najlepsze ({tree_desc(best_params)})", "seed": SEED,
         **from_row(best), "uwagi": f"wybrane z {len(rows)} konfiguracji"},
        {"predictor": "drzewo: najlepsze, inny podział filmów", "seed": CHECK_SEED,
         **from_row(check), "uwagi": "sprawdzenie, czy wynik nie był przypadkiem"},
        {"predictor": "baseline: najczęstsza ocena osoby, inny podział", "seed": CHECK_SEED,
         **check_base, "uwagi": ""},
    ]
    pd.DataFrame(summary).to_csv(SUMMARY_FILE, index=False, float_format="%.5f")

    BEST_FILE.write_text(json.dumps({
        "params": best_params,
        "validation": {"seed": SEED, "n_folds": N_FOLDS, **from_row(best),
                       "train_accuracy": best["train_accuracy"]},
        "check": {"seed": CHECK_SEED, **from_row(check)},
        "baseline": {"seed": SEED, **base["najczęstsza ocena osoby"]},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_depth_curve(rows, best, baseline_acc)

    print("\n10 najlepszych konfiguracji:")
    print(df.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nZestawienie (accuracy / ±1 / MAE):")
    for s in summary:
        seed = f"seed {s['seed']}" if s["seed"] != "" else "Zad. 1"
        print(f"  {s['predictor']:<72} {seed:<7} {s['accuracy']:6.1%}  "
              f"{s['accuracy_1']:6.1%}  {s['mae']:.3f}")
    print(f"\nWybrane parametry: {best_params}")
    print(f"Zapisano: {GRID_FILE.relative_to(ROOT)}, {SUMMARY_FILE.relative_to(ROOT)}, "
          f"{BEST_FILE.relative_to(ROOT)}, {CURVE_FILE.relative_to(ROOT)}")
    print(f"Czas całości: {fmt_time(time.perf_counter() - t_start)}")


if __name__ == "__main__":
    main()
