"""Wczytywanie i walidacja danych z folderu movie/."""
import io
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # = folder Zadanie2
DATA = ROOT / "movie"
CACHE = ROOT / "cache" / "tmdb"

RATING_COLS = ["id", "user", "movie_id", "rating"]


def load_movies():
    """movie.csv: movie_id (1-200, ID w zadaniu), tmdb_id (ID w TMDB), title."""
    # movie.csv ma końce linii "\r" (stary format Maca), więc normalizujemy je ręcznie
    txt = (DATA / "movie.csv").read_text(encoding="utf-8")
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    return pd.read_csv(io.StringIO(txt), sep=";", header=None,
                       names=["movie_id", "tmdb_id", "title"])


def load_train():
    """train.csv: znane oceny (0-5)."""
    return pd.read_csv(DATA / "train.csv", sep=";", header=None, names=RATING_COLS)


def load_task():
    """task.csv: oceny do przewidzenia; "NULL" wczytuje się jako NaN."""
    return pd.read_csv(DATA / "task.csv", sep=";", header=None, names=RATING_COLS)


def validate(movies, train, task):
    """Sprawdza założenia, na których opiera się reszta kodu."""
    # filmy
    assert movies.movie_id.is_unique and movies.tmdb_id.is_unique
    assert sorted(movies.movie_id) == list(range(1, len(movies) + 1))

    # oceny treningowe
    assert train.id.is_unique
    assert train.rating.notna().all()
    assert train.rating.between(0, 5).all() and (train.rating % 1 == 0).all()
    assert not train.duplicated(["user", "movie_id"]).any()

    # zadanie
    assert task.id.is_unique
    assert task.rating.isna().all()
    assert not task.duplicated(["user", "movie_id"]).any()
    assert set(task.user) <= set(train.user), "osoba w task.csv bez ocen w train.csv"

    # spójność między plikami
    known = set(movies.movie_id)
    assert set(train.movie_id) <= known and set(task.movie_id) <= known
    overlap = train.merge(task, on=["user", "movie_id"])
    assert overlap.empty, "ta sama para (osoba, film) w train i task"
    assert set(train.id).isdisjoint(task.id)


def load_all():
    """Zwraca (movies, train, task) po walidacji."""
    movies, train, task = load_movies(), load_train(), load_task()
    validate(movies, train, task)
    train["rating"] = train.rating.astype(int)
    return movies, train, task


if __name__ == "__main__":
    movies, train, task = load_all()
    per_user_train = train.groupby("user").size()
    per_user_task = task.groupby("user").size()

    print("Walidacja: OK")
    print(f"filmy: {len(movies)} | train: {len(train)} | task: {len(task)}")
    print(f"osoby: {train.user.nunique()}")
    print(f"ocen na osobę w train: {per_user_train.min()}-{per_user_train.max()}")
    print(f"filmów na osobę w task: {per_user_task.min()}-{per_user_task.max()}")
    print("rozkład ocen:", train.rating.value_counts().sort_index().to_dict())
