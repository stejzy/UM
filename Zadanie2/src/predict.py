"""Etap 5: przewidywanie ocen z task.csv -> submission_tree.csv albo submission_forest.csv.

Dla każdej osoby uczymy model na wszystkich jej 90 ocenach z train.csv
(parametry wybrane w Etapie 4) i przewidujemy jej 30 filmów z task.csv.
  * drzewo: parametry z results/best_tree.json   -> submission_tree.csv
  * las:    parametry z results/best_forest.json -> submission_forest.csv
            (seed = 0, jak domyślnie w RandomForest)

Użycie:
    uv run python src/predict.py                  # drzewo
    uv run python src/predict.py --model forest   # las losowy
"""
import argparse
import json
import time

import numpy as np

from data import ROOT, load_all
from forest import RandomForest
from matrix import load_matrix, user_data
from tree import DecisionTree

BEST_TREE = ROOT / "results" / "best_tree.json"
BEST_FOREST = ROOT / "results" / "best_forest.json"
SUBMISSION_TREE = ROOT / "submission_tree.csv"
SUBMISSION_FOREST = ROOT / "submission_forest.csv"

# model: (klasa, plik z parametrami, plik wynikowy, skrypt, który tworzy plik z parametrami)
MODELS = {
    "tree": (DecisionTree, BEST_TREE, SUBMISSION_TREE, "tune_tree.py"),
    "forest": (RandomForest, BEST_FOREST, SUBMISSION_FOREST, "tune_forest.py"),
}


def load_params(path=BEST_TREE, tuner="tune_tree.py"):
    if not path.exists():
        raise SystemExit(f"Brak {path.relative_to(ROOT)}. Uruchom najpierw: "
                         f"uv run python src/{tuner}")
    return json.loads(path.read_text(encoding="utf-8"))["params"]


def predict_all(make_model, train, task, X):
    """Ocena dla każdego wiersza task.csv (w kolejności pliku).

    make_model - funkcja bez argumentów zwracająca nowy, niewyuczony model
                 (z metodami fit i predict); osobny model dla każdej osoby
    """
    preds = {}
    for user in task.user.unique():
        X_train, y_train, X_task, ids = user_data(X, train, task, user)
        model = make_model().fit(X_train, y_train)
        preds.update(zip(ids.to_numpy(), model.predict(X_task)))
    return np.array([preds[i] for i in task.id], dtype=int)


def save_submission(task, preds, out):
    """Zapis w formacie task.csv: id;user;movie_id;ocena, bez nagłówka, końce linii "\\n"."""
    assert len(preds) == len(task), "nie każdy wiersz task.csv ma przewidywanie"
    assert preds.min() >= 0 and preds.max() <= 5, "ocena spoza zakresu 0-5"
    with open(out, "w", newline="\n", encoding="utf-8") as f:
        for row, p in zip(task.itertuples(), preds):
            f.write(f"{row.id};{row.user};{row.movie_id};{p}\n")


def check_submission(path, task):
    """Plik ma te same wiersze co task.csv (id, osoba, film, kolejność), a w miejscu NULL ocenę 0-5."""
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(task), f"{len(lines)} wierszy zamiast {len(task)}"
    for line, row in zip(lines, task.itertuples()):
        i, user, movie, rating = line.split(";")
        assert (int(i), int(user), int(movie)) == (row.id, row.user, row.movie_id), line
        assert rating in {"0", "1", "2", "3", "4", "5"}, line


def main():
    parser = argparse.ArgumentParser(description="Przewidywanie ocen z task.csv (Etap 5).")
    parser.add_argument("--model", choices=list(MODELS), default="tree",
                        help="tree = drzewo decyzyjne, forest = las losowy")
    args = parser.parse_args()
    model_class, params_file, out, tuner = MODELS[args.model]

    movies, train, task = load_all()
    X = load_matrix()
    params = load_params(params_file, tuner)
    print(f"Parametry z {params_file.relative_to(ROOT)}: {params}")

    t0 = time.perf_counter()
    preds = predict_all(lambda: model_class(**params), train, task, X)
    save_submission(task, preds, out)
    check_submission(out, task)

    dist = {r: f"{(preds == r).mean():.1%}" for r in range(6)}
    print(f"Przewidziano {len(preds)} ocen dla {task.user.nunique()} osób "
          f"(czas {time.perf_counter() - t0:.0f} s)")
    print(f"Rozkład przewidywań: {dist}")
    print(f"Zapisano {out.relative_to(ROOT)} (sprawdzony: zgodny z task.csv)")


if __name__ == "__main__":
    main()
