from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "movie"
RESULTS = ROOT / "results"

RATING_COLS = ["id", "user", "movie_id", "rating"]


def load_train():
    return pd.read_csv(DATA / "train.csv", sep=";", header=None, names=RATING_COLS)


def load_task():
    return pd.read_csv(DATA / "task.csv", sep=";", header=None, names=RATING_COLS)


def validate(train, task):
    assert train.id.is_unique
    assert train.rating.notna().all()
    assert train.rating.between(0, 5).all() and (train.rating % 1 == 0).all()
    assert not train.duplicated(["user", "movie_id"]).any()

    assert task.id.is_unique
    assert task.rating.isna().all()
    assert not task.duplicated(["user", "movie_id"]).any()
    assert set(task.user) <= set(train.user)
    assert set(task.movie_id) <= set(train.movie_id)
    assert train.merge(task, on=["user", "movie_id"]).empty
    assert set(train.id).isdisjoint(task.id)


def load_all():
    train, task = load_train(), load_task()
    validate(train, task)
    train["rating"] = train.rating.astype(int)
    return train, task


if __name__ == "__main__":
    train, task = load_all()
    print(f"train: {len(train)} | task: {len(task)}")
    print(f"osoby: {train.user.nunique()} | filmy: {train.movie_id.nunique()}")
    print("rozkład ocen:", train.rating.value_counts().sort_index().to_dict())
