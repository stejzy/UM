import json
import time

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import RESULTS, load_all
from validate import evaluate, evaluate_baseline, make_cv

SEED, STABILITY_SEED = 0, 1

# Dwie osobne ścieżki zgodne z wykładem: albo dobór n (lam=0), albo duże ustalone n + dobór lam
N_SWEEP_ONEHOT = [1, 2, 4, 6, 8, 10, 15, 20, 30]
N_SWEEP_SCALAR = [1, 2, 4, 6, 8, 10, 15, 20]

FIXED_N_ONEHOT = 150
LAM_SWEEP_ONEHOT = [1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 8.0, 12.0]

FIXED_N_SCALAR = 50
LAM_SWEEP_SCALAR = [1.0, 4.0, 8.0, 12.0, 16.0, 20.0]

COLORS = {"onehot": "#2a78d6", "scalar": "#eb6834"}
MARKERS = {"onehot": "o", "scalar": "s"}
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


def plot_sweep(df, x_col, title, xlabel, baseline_mode, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode in ("onehot", "scalar"):
        sub = df[df["mode"] == mode].sort_values(x_col)
        if sub.empty:
            continue
        label = "CF one-hot (klasy 0-5)" if mode == "onehot" else "CF skalarne (zaokrąglane)"
        ax.plot(sub[x_col], sub["accuracy"], color=COLORS[mode], marker=MARKERS[mode],
                linewidth=2, markersize=6, markeredgecolor="white", label=label)
    ax.axhline(baseline_mode, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate("baseline: moda osoby", (ax.get_xlim()[1], baseline_mode), color=MUTED,
                fontsize=8, ha="right", va="bottom")
    style(ax, title, xlabel)
    ax.set_ylabel("accuracy (5-fold CV)", color=MUTED)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def md_table(df, cols):
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df[cols].iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(df, stab, base, best, path):
    cols = ["mode", "n", "lam", "norm", "update", "accuracy", "accuracy_1", "mae", "rmse"]
    s_n = df[df["experiment"] == "path_a_n"]
    s_lam = df[df["experiment"] == "path_b_lam"]
    s_var = df[df["experiment"] == "variants"]

    parts = [
        "# Wyniki eksperymentów — Zadanie 4 (Collaborative Filtering)",
        "",
        "5-krotna walidacja krzyżowa ze stratyfikacją po osobach (`seed=0`, 18 zakrytych / 72 treningowe oceny na osobę).",
        "Algorytm: metoda najszybszego spadku z wykładu (`X`, `P`, `p0`).",
        "Zgodnie z wykładem **nie łączono** jednoczesnego doboru `n` i regularyzacji `lambda` — przetestowano dwie osobne ścieżki:",
        "- **Ścieżka A**: dobór wymiaru przestrzeni cech `n` bez regularyzacji (`lambda = 0`).",
        "- **Ścieżka B**: ustalone wystarczająco duże `n` i dobór samej regularyzacji `lambda` (dla `X` i `P`).",
        "",
        f"**Baseline'y:** średnia filmu = `{base['movie_mean']['accuracy']:.4f}`, "
        f"średnia osoby = `{base['user_mean']['accuracy']:.4f}`, "
        f"moda osoby = `{base['mode']['accuracy']:.4f}`.",
        "",
        f"**Najlepsza konfiguracja:** `mode={best['params']['mode']}`, `n={best['params']['n']}`, "
        f"`lam={best['params']['lam']}`, `norm={best['params']['norm']}`, `update={best['params']['update']}` "
        f"→ Accuracy = **`{best['cv']['accuracy']:.4f}`** (`± {best['cv']['accuracy_std']:.4f}`), "
        f"Accuracy ±1 = `{best['cv']['accuracy_1']:.4f}`, MAE = `{best['cv']['mae']:.4f}`.",
        "",
        "## 1. Ścieżka A — dobór wymiaru cech `n` bez regularyzacji (`lam = 0`)",
        "",
        md_table(s_n, cols),
        "",
        "## 2. Ścieżka B — ustalone duże `n` + dobór regularyzacji `lam`",
        "",
        md_table(s_lam, cols),
        "",
        "## 3. Porównanie aktualizacji (`joint` vs `alternating`) i normalizacji (`none`, `movie`, `user`)",
        "",
        md_table(s_var, cols),
        "",
        "## 4. Stabilność na innym podziale foldów (`seed = 1`)",
        "",
        md_table(stab, ["rank", "mode", "n", "lam", "norm", "update", "accuracy_seed0", "accuracy_seed1", "mae_seed1"]),
        "",
    ]
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    RESULTS.mkdir(exist_ok=True)
    train, task = load_all()
    folds = make_cv(train, task, seed=SEED)

    base = {r: evaluate_baseline(folds, r) for r in ("movie_mean", "user_mean", "mode")}
    print(f"baseline średnia filmu: {base['movie_mean']['accuracy']:.3f} | "
          f"średnia osoby: {base['user_mean']['accuracy']:.3f} | "
          f"moda osoby: {base['mode']['accuracy']:.3f}")

    configs = []
    for n in N_SWEEP_ONEHOT:
        configs.append(("path_a_n", dict(mode="onehot", n=n, lam=0.0, norm="none", update="joint")))
    for n in N_SWEEP_SCALAR:
        configs.append(("path_a_n", dict(mode="scalar", n=n, lam=0.0, norm="movie", update="joint")))
    for lam in LAM_SWEEP_ONEHOT:
        configs.append(("path_b_lam", dict(mode="onehot", n=FIXED_N_ONEHOT, lam=lam, norm="none", update="joint")))
    for lam in LAM_SWEEP_SCALAR:
        configs.append(("path_b_lam", dict(mode="scalar", n=FIXED_N_SCALAR, lam=lam, norm="movie", update="joint")))

    rows = []
    t0 = time.perf_counter()
    for i, (exp, cfg) in enumerate(configs, 1):
        res = evaluate(folds, **cfg)
        rows.append({"experiment": exp, **cfg, **res})
        print(f"[{i:2d}/{len(configs)}] {exp:<10} {cfg['mode']:<6} n={cfg['n']:<3} lam={cfg['lam']:<4} "
              f"-> acc={res['accuracy']:.4f} ±1={res['accuracy_1']:.4f} MAE={res['mae']:.4f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    df_init = pd.DataFrame(rows)
    best_oh = df_init[df_init["mode"] == "onehot"].sort_values("accuracy", ascending=False).iloc[0]
    best_sc = df_init[df_init["mode"] == "scalar"].sort_values("rmse", ascending=True).iloc[0]

    variant_configs = []
    for norm in ("none", "movie", "user"):
        for update in ("joint", "alternating"):
            variant_configs.append(dict(mode="onehot", n=int(best_oh["n"]), lam=float(best_oh["lam"]),
                                        norm=norm, update=update))
    for norm in ("none", "movie", "user"):
        for update in ("joint", "alternating"):
            variant_configs.append(dict(mode="scalar", n=int(best_sc["n"]), lam=float(best_sc["lam"]),
                                        norm=norm, update=update))

    print("\nPorównanie wariantów normalizacji i aktualizacji (joint vs alternating):")
    for cfg in variant_configs:
        res = evaluate(folds, **cfg)
        rows.append({"experiment": "variants", **cfg, **res})
        print(f"  {cfg['mode']:<6} n={cfg['n']:<3} lam={cfg['lam']:<4} norm={cfg['norm']:<5} "
              f"upd={cfg['update']:<11} -> acc={res['accuracy']:.4f} MAE={res['mae']:.4f} RMSE={res['rmse']:.4f}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "tuning.csv", index=False)

    ranked = (df.drop_duplicates(subset=["mode", "n", "lam", "norm", "update"])
                .sort_values(["accuracy", "accuracy_1"], ascending=False, ignore_index=True))

    folds_s1 = make_cv(train, task, seed=STABILITY_SEED)
    stab_rows = []
    for rank, row in ranked.head(5).iterrows():
        cfg = dict(mode=row["mode"], n=int(row["n"]), lam=float(row["lam"]),
                   norm=row["norm"], update=row["update"])
        res1 = evaluate(folds_s1, **cfg)
        stab_rows.append({
            "rank": rank + 1, **cfg,
            "accuracy_seed0": float(row["accuracy"]),
            "accuracy_seed1": res1["accuracy"],
            "accuracy_1_seed1": res1["accuracy_1"],
            "mae_seed1": res1["mae"],
        })
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(RESULTS / "tuning_stability.csv", index=False)

    top = ranked.iloc[0]
    best = {
        "params": {
            "mode": str(top["mode"]),
            "n": int(top["n"]),
            "lam": float(top["lam"]),
            "norm": str(top["norm"]),
            "update": str(top["update"]),
        },
        "cv": {m: float(top[m]) for m in ("accuracy", "accuracy_std", "accuracy_1", "mae", "rmse")},
        "baselines": {k: base[k]["accuracy"] for k in base},
        "n_folds": len(folds),
        "seed": SEED,
    }
    with open(RESULTS / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    plot_sweep(df[df["experiment"] == "path_a_n"], "n",
               "Ścieżka A: dobór wymiaru cech n (bez regularyzacji, λ = 0)",
               "n (liczba cech filmu)", base["mode"]["accuracy"], RESULTS / "tuning_n.png")
    plot_sweep(df[df["experiment"] == "path_b_lam"], "lam",
               "Ścieżka B: dobór regularyzacji λ (ustalone duże n)",
               "λ (parametr regularyzacji X i P)", base["mode"]["accuracy"], RESULTS / "tuning_lambda.png")

    write_report(df, stab, base, best, RESULTS / "experiment_results.md")
    print(f"\nNajlepsze ustawienie: {best['params']}")
    print(f"Accuracy CV: {best['cv']['accuracy']:.4f} ± {best['cv']['accuracy_std']:.4f}")
    print("Zapisano: results/tuning.csv, tuning_stability.csv, best_params.json, experiment_results.md, wykresy")


if __name__ == "__main__":
    main()
