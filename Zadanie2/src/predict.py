"""Etap 5: przewidywanie ocen z task.csv i zapis submission_tree.csv.

Dla każdej osoby uczymy drzewo na wszystkich jej 90 ocenach z train.csv
(parametry z results/best_tree.json, wybrane w Etapie 4) i przewidujemy jej
30 filmów z task.csv.

Użycie:
    uv run python src/predict.py
"""
import json

import numpy as np

from data import ROOT, load_all
from matrix import load_matrix, user_data
from tree import DecisionTree

BEST_TREE = ROOT / "results" / "best_tree.json"
SUBMISSION_TREE = ROOT / "submission_tree.csv"


def load_params(path=BEST_TREE):
    if not path.exists():
        raise SystemExit(f"Brak {path.relative_to(ROOT)}. Uruchom najpierw: "
                         f"uv run python src/tune_tree.py")
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
    movies, train, task = load_all()
    X = load_matrix()
    params = load_params()
    print(f"Parametry drzewa z {BEST_TREE.relative_to(ROOT)}: {params}")

    preds = predict_all(lambda: DecisionTree(**params), train, task, X)
    save_submission(task, preds, SUBMISSION_TREE)
    check_submission(SUBMISSION_TREE, task)

    dist = {r: f"{(preds == r).mean():.1%}" for r in range(6)}
    print(f"Przewidziano {len(preds)} ocen dla {task.user.nunique()} osób")
    print(f"Rozkład przewidywań: {dist}")
    print(f"Zapisano {SUBMISSION_TREE.relative_to(ROOT)} (sprawdzony: zgodny z task.csv)")


if __name__ == "__main__":
    main()
