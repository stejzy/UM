"""Kontrola po Etapie 5: rozkład ocen w train.csv, w submission_tree.csv i w submission.csv z Zadania 1.

Sprawdza, czy przewidywania wyglądają rozsądnie: jakie oceny drzewo przewiduje,
jak bardzo pokrywa się z kNN z Zadania 1 i z baseline'em "najczęstsza ocena osoby".

Użycie:
    uv run python src/compare_distributions.py
    uv run python src/compare_distributions.py --zad1 ../Zadanie1/submission.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import RATING_COLS, ROOT, load_all
from predict import SUBMISSION_TREE
from validate import most_frequent

ZAD1_SUBMISSION = ROOT.parent / "Zadanie1" / "submission.csv"
OUT = ROOT / "results" / "distribution_comparison.csv"


def load_submission(path):
    return pd.read_csv(path, sep=";", header=None, names=RATING_COLS)


def distribution(ratings):
    ratings = np.asarray(ratings, dtype=int)
    return {**{str(r): float((ratings == r).mean()) for r in range(6)},
            "średnia": float(ratings.mean()), "n": len(ratings)}


def per_user(sub):
    """Ile różnych ocen dostaje osoba w swoich 30 przewidywaniach."""
    return sub.groupby("user").rating.nunique()


def main():
    parser = argparse.ArgumentParser(description="Porównanie rozkładów ocen (kontrola Etapu 5).")
    parser.add_argument("--zad1", type=Path, default=ZAD1_SUBMISSION,
                        help="submission.csv z Zadania 1 (pomijany, jeśli go nie ma)")
    args = parser.parse_args()

    movies, train, task = load_all()
    if not SUBMISSION_TREE.exists():
        raise SystemExit("Brak submission_tree.csv. Uruchom najpierw: uv run python src/predict.py")
    tree = load_submission(SUBMISSION_TREE)
    assert (tree.id.to_numpy() == task.id.to_numpy()).all(), "inne wiersze niż w task.csv"

    # co zwróciłby baseline: najczęstsza ocena osoby w train.csv (przy remisie wyższa)
    mode = train.groupby("user").rating.apply(lambda r: most_frequent(r.to_numpy()))
    baseline = task.user.map(mode).to_numpy()

    sources = {"train.csv (prawdziwe oceny)": train.rating,
               "submission_tree.csv (drzewo)": tree.rating,
               "baseline: najczęstsza ocena osoby": baseline}
    zad1 = None
    if args.zad1.exists():
        zad1 = load_submission(args.zad1)
        assert (zad1.id.to_numpy() == task.id.to_numpy()).all(), "Zadanie 1: inne wiersze"
        sources["submission.csv z Zadania 1 (kNN)"] = zad1.rating
    else:
        print(f"Uwaga: brak {args.zad1}, pomijam porównanie z Zadaniem 1\n")

    rows = [{"źródło": name, **distribution(r)} for name, r in sources.items()]
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, float_format="%.4f")

    print("Rozkład ocen (udział każdej oceny):")
    print(f"  {'źródło':<36}" + "".join(f"{r:>7}" for r in range(6)) + "  średnia       n")
    for r in rows:
        print(f"  {r['źródło']:<36}" + "".join(f"{r[str(k)]:7.1%}" for k in range(6))
              + f"  {r['średnia']:7.2f}  {r['n']:>6}")

    print("\nZgodność przewidywań (ten sam wiersz task.csv, ta sama ocena):")
    pairs = [("drzewo", tree.rating.to_numpy(), "baseline", baseline)]
    if zad1 is not None:
        pairs += [("drzewo", tree.rating.to_numpy(), "kNN z Zadania 1", zad1.rating.to_numpy()),
                  ("kNN z Zadania 1", zad1.rating.to_numpy(), "baseline", baseline)]
    for a, pa, b, pb in pairs:
        print(f"  {a:<16} vs {b:<16} {np.mean(pa == pb):6.1%}   "
              f"(różnica o więcej niż 1: {np.mean(np.abs(pa - pb) > 1):5.1%})")

    print("\nRóżnorodność przewidywań na osobę (30 filmów):")
    subs = {"drzewo": tree} | ({"kNN z Zadania 1": zad1} if zad1 is not None else {})
    for name, sub in subs.items():
        n = per_user(sub)
        print(f"  {name:<16} ta sama ocena dla wszystkich 30 filmów: {np.mean(n == 1):5.1%} osób, "
              f"średnio {n.mean():.2f} różnych ocen na osobę")

    print(f"\nZapisano {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
