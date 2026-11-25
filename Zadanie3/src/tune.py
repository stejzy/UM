import argparse
import itertools
import json
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import RESULTS, load_all
from predict import predict, to_rating
from validate import evaluate, evaluate_baseline, make_cv, scores, summarize

GRID = {
    "measure": ["pearson", "cosine", "msd"],
    "lam": [0, 10, 25, 50],
    "min_common": [1, 20, 25],
    "k": [5, 10, 20, 40, 80],
}
AGGREGATION_GRID = (
    [{"aggregation": "mean", "centered": c, "a": None} for c in (True, False)]
    + [{"aggregation": "vote", "centered": c, "a": a}
       for c in (True, False) for a in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)]
)
FALLBACKS = ("mode", "mean")

QUICK_GRID = {"measure": ["pearson", "msd"], "lam": [0, 25], "min_common": [1], "k": [10, 20]}
QUICK_AGGREGATION_GRID = [{"aggregation": "mean", "centered": True, "a": None},
                          {"aggregation": "vote", "centered": True, "a": 1.0}]

SEED, STABILITY_SEED, TOP_N = 0, 1, 10
PARAM_COLS = ["measure", "lam", "min_common", "k", "aggregation", "centered", "a", "fallback"]


def run_grid(folds, grid, aggregation_grid):
    combos = list(itertools.product(*grid.values()))
    n_total = len(combos) * len(aggregation_grid) * len(FALLBACKS)
    print(f"siatka: {n_total} ustawień × {len(folds)} foldy")
    rows, t0, done = [], time.perf_counter(), 0

    for combo in combos:
        sim_params = dict(zip(grid.keys(), combo))
        sims = [f.similarity(sim_params["measure"], sim_params["lam"], sim_params["min_common"])
                for f in folds]
        for agg in aggregation_grid:
            per_fold = {fb: [] for fb in FALLBACKS}
            for fold, S in zip(folds, sims):
                pred, used_fb = predict(fold.R, S, fold.rows, fold.cols, k=sim_params["k"],
                                        aggregation=agg["aggregation"], centered=agg["centered"],
                                        a=agg["a"] or 0.0, fallback="mode")
                variants = {"mode": pred, "mean": np.where(used_fb, fold.mu[fold.rows], pred)}
                for fb, p in variants.items():
                    s = scores(to_rating(p), fold.truth)
                    s["fallback_rate"] = float(used_fb.mean())
                    per_fold[fb].append(s)
            for fb in FALLBACKS:
                rows.append({**sim_params, **agg, "fallback": fb, **summarize(per_fold[fb])})
            done += len(FALLBACKS)
        elapsed = time.perf_counter() - t0
        print(f"  {done}/{n_total} ({elapsed:.0f} s, zostało ~{elapsed / done * (n_total - done):.0f} s)",
              end="\r", flush=True)
    print()
    df = pd.DataFrame(rows)
    return df.sort_values(["accuracy", "accuracy_1"], ascending=False, ignore_index=True)


def params_of(row):
    p = {c: row[c] for c in PARAM_COLS}
    p["lam"], p["min_common"], p["k"] = float(p["lam"]), int(p["min_common"]), int(p["k"])
    p["centered"] = bool(p["centered"])
    p["a"] = None if pd.isna(p["a"]) else float(p["a"])
    return p


def stability(train, task, results, top_n):
    folds = make_cv(train, task, seed=STABILITY_SEED)
    out = []
    for rank, row in results.head(top_n).iterrows():
        p = params_of(row)
        res = evaluate(folds, **{**p, "a": p["a"] or 0.0})
        out.append({"rank": rank + 1, **p, "accuracy_seed0": row["accuracy"],
                    "accuracy_seed1": res["accuracy"], "accuracy_1_seed1": res["accuracy_1"]})
    return pd.DataFrame(out)


def best_per(results, by, **fixed):
    df = results
    for key, val in fixed.items():
        df = df[df[key] == val]
    return df.groupby(by)["accuracy"].max()


COLORS = {"pearson": "#2a78d6", "cosine": "#eb6834", "msd": "#1baf7a"}
MARKERS = {"pearson": "o", "cosine": "s", "msd": "^"}
INK, MUTED, GRID_COLOR = "#2b2b2a", "#6b6b66", "#e4e3df"


def style(ax, title, xlabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=MUTED)


def plot_k(results, baseline_mode, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, agg in zip(axes, ("mean", "vote")):
        for measure in COLORS:
            s = best_per(results, "k", aggregation=agg, measure=measure)
            if s.empty:
                continue
            ax.plot(s.index, s.values, color=COLORS[measure], marker=MARKERS[measure],
                    linewidth=2, markersize=7, markeredgecolor="white", label=measure)
        ax.axhline(baseline_mode, color=MUTED, linestyle="--", linewidth=1)
        ax.annotate("baseline: moda osoby", (ax.get_xlim()[1], baseline_mode), color=MUTED,
                    fontsize=8, ha="right", va="bottom")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(results.k.unique()), labels=sorted(results.k.unique()))
        style(ax, f"agregacja: {agg}", "k (liczba sąsiadów)")
    axes[0].set_ylabel("accuracy (najlepsza przy danym k)", color=MUTED)
    axes[1].legend(title="miara", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_a(results, baseline_mode, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for measure in COLORS:
        s = best_per(results, "a", aggregation="vote", measure=measure)
        if s.empty:
            continue
        ax.plot(s.index, s.values, color=COLORS[measure], marker=MARKERS[measure],
                linewidth=2, markersize=7, markeredgecolor="white", label=measure)
    ax.axhline(baseline_mode, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate("baseline: moda osoby", (ax.get_xlim()[1], baseline_mode), color=MUTED,
                fontsize=8, ha="right", va="bottom")
    style(ax, "głosowanie: waga przyzwyczajeń osoby", "a (0 = tylko sąsiedzi)")
    ax.set_ylabel("accuracy (najlepsza przy danym a)", color=MUTED)
    ax.legend(title="miara", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dobór hiperparametrów: przeszukiwanie siatki + walidacja krzyżowa.")
    parser.add_argument("--quick", action="store_true", help="mała siatka do szybkiego testu")
    args = parser.parse_args()
    grid, agg_grid = (QUICK_GRID, QUICK_AGGREGATION_GRID) if args.quick else (GRID, AGGREGATION_GRID)
    suffix = "_quick" if args.quick else ""
    RESULTS.mkdir(exist_ok=True)

    train, task = load_all()
    folds = make_cv(train, task, seed=SEED)
    base = {rule: evaluate_baseline(folds, rule) for rule in ("mode", "mean")}
    print(f"baseline moda: {base['mode']['accuracy']:.3f} | "
          f"baseline średnia: {base['mean']['accuracy']:.3f}")

    results = run_grid(folds, grid, agg_grid)
    results.to_csv(RESULTS / f"tuning{suffix}.csv", index=False)

    cols = PARAM_COLS + ["accuracy", "accuracy_std", "accuracy_1", "mae", "fallback_rate"]
    print(f"\nTOP {TOP_N} (seed {SEED}):")
    print(results[cols].head(TOP_N).to_string(float_format=lambda x: f"{x:.3f}"))

    stab = stability(train, task, results, min(TOP_N, len(results)))
    stab.to_csv(RESULTS / f"tuning_stability{suffix}.csv", index=False)
    print(f"\nTe same ustawienia na innym podziale na foldy (seed {STABILITY_SEED}):")
    print(stab[["rank", "accuracy_seed0", "accuracy_seed1", "accuracy_1_seed1"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best = params_of(results.iloc[0])
    summary = {"params": best,
               "cv": {m: float(results.iloc[0][m]) for m in
                      ("accuracy", "accuracy_std", "accuracy_1", "mae", "fallback_rate")},
               "baselines": {r: base[r]["accuracy"] for r in base},
               "n_folds": len(folds), "seed": SEED}
    with open(RESULTS / f"best_params{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_k(results, base["mode"]["accuracy"], RESULTS / f"tuning_k{suffix}.png")
    if (results.aggregation == "vote").any():
        plot_a(results, base["mode"]["accuracy"], RESULTS / f"tuning_a{suffix}.png")

    print(f"\nnajlepsze ustawienie: {best}")
    print(f"accuracy CV: {summary['cv']['accuracy']:.3f} ± {summary['cv']['accuracy_std']:.3f}")
    print(f"zapisano: results/tuning{suffix}.csv, best_params{suffix}.json, wykresy")
