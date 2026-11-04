"""Predykcja ocen z task.csv i zapis submission.csv (etap 10).

Użycie:
    python src/predict.py                           # params/global_ga.json -> submission.csv
    python src/predict.py --params params/global.json
"""
import argparse
from pathlib import Path

import numpy as np

from data import ROOT, load_all
from knn import predict, user_data
from validate import PARAMS_FILE, load_params, similarity

SUBMISSION = ROOT / "submission.csv"


def run(params_path=PARAMS_FILE, out=SUBMISSION):
    params = load_params(params_path)
    _, train, task = load_all()
    users = user_data(train)
    M = similarity(params)
    k, method = params["k"], params["method"]
    print(f"Parametry z {params_path.name}: k = {k}, {method}, {params['scale']}, "
          f"cechy: {', '.join(params['features'])}")

    preds = np.array([predict(M, row.movie_id - 1, *users[row.user], k, method)
                      for row in task.itertuples()])

    # Kontrola: każda ocena to liczba całkowita 0-5, tyle samo wierszy co w task.csv
    assert len(preds) == len(task) and preds.dtype.kind == "i"
    assert preds.min() >= 0 and preds.max() <= 5

    # Zapis w formacie task.csv: id;user;movie_id;ocena, bez nagłówka, końce linii "\n"
    with open(out, "w", newline="\n", encoding="utf-8") as f:
        for row, p in zip(task.itertuples(), preds):
            f.write(f"{row.id};{row.user};{row.movie_id};{p}\n")

    dist = {r: f"{(preds == r).mean() * 100:.0f}%" for r in range(6)}
    true = {r: f"{(train.rating == r).mean() * 100:.0f}%" for r in range(6)}
    print(f"Przewidziano {len(preds)} ocen dla {task.user.nunique()} osób")
    print(f"Rozkład przewidywań: {dist}")
    print(f"Rozkład w train.csv: {true}")
    print(f"Zapisano {out.relative_to(ROOT)}")
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=str, default=str(PARAMS_FILE),
                        help="plik z parametrami (domyślnie params/global_ga.json)")
    args = parser.parse_args()
    path = Path(args.params)
    if not path.is_absolute():          # ścieżka względna: względem folderu Zadanie1
        path = ROOT / path
    run(path.resolve())
