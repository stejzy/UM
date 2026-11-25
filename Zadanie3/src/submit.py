import json

import numpy as np

from data import RESULTS, ROOT, load_all
from matrix import build_matrix, make_index, positions
from predict import predict, to_rating
from similarity import adjust, common_counts, similarity

PARAMS_FILE = RESULTS / "best_params.json"
SUBMISSION = ROOT / "submission.csv"


def load_params(path=PARAMS_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def predict_task(train, task, p):
    users, movies = make_index(train, task)
    R = build_matrix(train, users, movies)
    S = adjust(similarity(R, p["measure"]), common_counts(R), p["lam"], p["min_common"])
    rows, cols = positions(task, users, movies)
    pred, used_fb = predict(R, S, rows, cols, k=p["k"], aggregation=p["aggregation"],
                            centered=p["centered"], a=p["a"] or 0.0, fallback=p["fallback"])
    return to_rating(pred), used_fb


def save_submission(task, preds, out=SUBMISSION):
    assert len(preds) == len(task), "nie każdy wiersz task.csv ma predykcję"
    assert preds.min() >= 0 and preds.max() <= 5, "ocena spoza zakresu 0-5"
    with open(out, "w", newline="\n", encoding="utf-8") as f:
        for row, p in zip(task.itertuples(), preds):
            f.write(f"{row.id};{row.user};{row.movie_id};{p}\n")


def check_submission(path, task):
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(task), f"{len(lines)} wierszy zamiast {len(task)}"
    for line, row in zip(lines, task.itertuples()):
        i, user, movie, rating = line.split(";")
        assert (int(i), int(user), int(movie)) == (row.id, row.user, row.movie_id), line
        assert rating in {"0", "1", "2", "3", "4", "5"}, line


if __name__ == "__main__":
    train, task = load_all()
    best = load_params()
    p = best["params"]
    print(f"parametry z {PARAMS_FILE.relative_to(ROOT)}: {p}")
    print(f"accuracy w CV: {best['cv']['accuracy']:.3f} ± {best['cv']['accuracy_std']:.3f} "
          f"(baseline moda: {best['baselines']['mode']:.3f})")

    preds, used_fb = predict_task(train, task, p)
    save_submission(task, preds)
    check_submission(SUBMISSION, task)

    def dist(values):
        return " ".join(f"{c}: {np.mean(values == c):5.1%}" for c in range(6))

    print(f"\nzapisano {SUBMISSION.name}: {len(preds)} ocen dla {task.user.nunique()} osób, "
          f"fallback {used_fb.sum()} ({used_fb.mean():.1%})")
    print(f"rozkład predykcji:  {dist(preds)}")
    print(f"rozkład train.csv:  {dist(train.rating.to_numpy())}")
    print("kontrola formatu: OK (te same wiersze co task.csv, oceny 0-5, brak NULL)")
