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
    assert set(task.user) <= set(train.user), "osoba w task.csv bez ocen w train.csv"
    assert set(task.movie_id) <= set(train.movie_id), "film w task.csv bez ocen w train.csv"

    overlap = train.merge(task, on=["user", "movie_id"])
    assert overlap.empty, "ta sama para (osoba, film) w train i task"
    assert set(train.id).isdisjoint(task.id)


def load_all():
    train, task = load_train(), load_task()
    validate(train, task)
    train["rating"] = train.rating.astype(int)
    return train, task


if __name__ == "__main__":
    train, task = load_all()
    per_user_train = train.groupby("user").size()
    per_user_task = task.groupby("user").size()
    per_movie_train = train.groupby("movie_id").size()

    print("Walidacja: OK")
    print(f"train: {len(train)} | task: {len(task)}")
    print(f"osoby: {train.user.nunique()} | filmy: {train.movie_id.nunique()}")
    print(f"ocen na osobę w train: {per_user_train.min()}-{per_user_train.max()}")
    print(f"filmów na osobę w task: {per_user_task.min()}-{per_user_task.max()}")
    print(f"ocen na film w train: {per_movie_train.min()}-{per_movie_train.max()} "
          f"(średnio {per_movie_train.mean():.1f})")
    print("rozkład ocen:", train.rating.value_counts().sort_index().to_dict())
