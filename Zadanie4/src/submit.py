import json
import zipfile

import numpy as np

from cf import predict_matrix
from data import RESULTS, ROOT, load_all
from matrix import make_index, positions

PARAMS_FILE = RESULTS / "best_params.json"
SUBMISSION = ROOT / "submission.csv"
CODE_ZIP = ROOT / "code.zip"


def load_params(path=PARAMS_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def predict_task(train, task, p):
    users, movies = make_index(train, task)
    pred_mat, _ = predict_matrix(
        train, users, movies,
        mode=p["mode"], n=p["n"], lam=p["lam"], norm=p["norm"], update=p["update"],
    )
    m_idx, r_idx = positions(task, users, movies)
    return pred_mat[m_idx, r_idx]


def save_submission(task, preds, out=SUBMISSION):
    assert len(preds) == len(task)
    assert preds.min() >= 0 and preds.max() <= 5
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


def build_code_zip(out=CODE_ZIP):
    files = sorted((ROOT / "src").glob("*.py")) + [PARAMS_FILE, ROOT / "pyproject.toml"]
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    train, task = load_all()
    best = load_params()
    p = best["params"]
    print(f"parametry z {PARAMS_FILE.relative_to(ROOT)}: {p}")
    print(f"accuracy w CV: {best['cv']['accuracy']:.4f} ± {best['cv']['accuracy_std']:.4f} "
          f"(baseline moda: {best['baselines']['mode']:.4f})")

    preds = predict_task(train, task, p)
    save_submission(task, preds)
    check_submission(SUBMISSION, task)
    build_code_zip()

    def dist(values):
        return " ".join(f"{c}: {np.mean(values == c):5.1%}" for c in range(6))

    print(f"\nzapisano {SUBMISSION.name}: {len(preds)} ocen dla {task.user.nunique()} osób")
    print(f"rozkład predykcji: {dist(preds)}")
    print(f"rozkład train.csv: {dist(train.rating.to_numpy())}")
    print(f"zapisano {CODE_ZIP.name}")
    print("kontrola formatu: OK")
