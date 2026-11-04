"""Etap 1: zamiana cech filmów z cache/features.json na macierz liczbową.

Wiersz = film, kolumna = liczba. Dzięki temu drzewo w każdym węźle zadaje
jedno i to samo pytanie: "czy kolumna <= próg?".

Użycie:
    uv run python src/matrix.py               # buduje cache/matrix.csv
    uv run python src/matrix.py --user 1642   # pokazuje dane jednej osoby
"""
import argparse
from collections import Counter

import pandas as pd

from data import ROOT, load_all
from features import load_features

MATRIX_FILE = ROOT / "cache" / "matrix.csv"
USERS_DIR = ROOT / "cache" / "users"

MIN_COUNT = 5     # kolumna 0/1 musi mieć co najmniej tyle jedynek i tyle zer
MISSING = -1      # wartość dla brakujących cech liczbowych (np. budżetu)

# cechy liczbowe: zostają bez zmian (drzewo patrzy tylko na kolejność wartości,
# więc normalizacja i logarytm niczego by nie zmieniły)
NUMERIC = ["year", "runtime", "vote_count", "vote_average", "budget"]

# cechy listowe w features.json -> przedrostek nazwy kolumny 0/1
LISTS = {
    "genres": "genre",
    "keywords": "keyword",
    "directors": "director",
    "cast_wide": "cast",
    "writers": "writer",
    "composers": "composer",
    "cinematographers": "cinematographer",
    "countries": "country",
    "companies": "company",
}


# ---------------------------------------------------------------- budowa macierzy

def values(film, name):
    """Zbiór wartości cechy filmu; osoby i wytwórnie po nazwie.

    Zbiór, bo ta sama osoba może wystąpić w obsadzie dwa razy (dwie role).
    """
    if name == "language":
        return {film["language"]}
    return {x["name"] if isinstance(x, dict) else x for x in film[name]}


def binary_columns(features, name, prefix):
    """Kolumny 0/1 "czy film ma wartość v" dla częstych wartości cechy `name`.

    Zostają tylko wartości, które ma co najmniej MIN_COUNT filmów i których
    nie ma co najmniej MIN_COUNT filmów; rzadsze dałyby podziały typu
    "ten jeden film / cała reszta", na których drzewo tylko się przeucza.
    """
    n = len(features)
    sets = {mid: values(f, name) for mid, f in features.items()}
    counts = Counter(v for s in sets.values() for v in s)
    kept = sorted((v for v, c in counts.items() if MIN_COUNT <= c <= n - MIN_COUNT),
                  key=lambda v: (-counts[v], v))       # najczęstsze pierwsze
    return pd.DataFrame({f"{prefix}={v}": [int(v in sets[mid]) for mid in features]
                         for v in kept}, index=list(features))


def build_matrix(features=None):
    """Tabela 200 filmów x cechy (indeks: movie_id). Nie wie nic o osobach i ocenach."""
    features = features or load_features()
    features = {mid: features[mid] for mid in sorted(features)}

    numeric = pd.DataFrame({c: [MISSING if f[c] is None else f[c] for f in features.values()]
                            for c in NUMERIC}, index=list(features))
    parts = [numeric, binary_columns(features, "language", "lang")]
    parts += [binary_columns(features, name, prefix) for name, prefix in LISTS.items()]

    X = pd.concat(parts, axis=1)
    X.index.name = "movie_id"
    return X


def check_matrix(X, features):
    """Sprawdza założenia, na których opiera się drzewo."""
    assert list(X.index) == sorted(features), "nie każdy film ma wiersz"
    assert not X.isna().any().any(), "puste wartości w macierzy"
    assert X.columns.is_unique, "powtórzone nazwy kolumn"
    binary = X.drop(columns=NUMERIC)
    assert binary.isin([0, 1]).all().all(), "kolumna 0/1 ma inne wartości"
    assert binary.sum().between(MIN_COUNT, len(X) - MIN_COUNT).all(), "za rzadka kolumna"


def save_matrix(X):
    X.to_csv(MATRIX_FILE, encoding="utf-8")


def load_matrix():
    """Wczytuje cache/matrix.csv zbudowany przez build_matrix()."""
    if not MATRIX_FILE.exists():
        raise SystemExit("Brak cache/matrix.csv. Uruchom najpierw: uv run python src/matrix.py")
    return pd.read_csv(MATRIX_FILE, index_col="movie_id", encoding="utf-8")


def group(column):
    """Grupa kolumny: 'liczbowe' albo przedrostek przed '=' (genre, keyword, ...)."""
    return "liczbowe" if column in NUMERIC else column.split("=", 1)[0]


def describe(row):
    """Film tak, jak widzi go drzewo: cechy liczbowe i kolumny z jedynką."""
    nums = ", ".join(f"{c}={row[c]:.10g}" for c in NUMERIC)   # .10g: bez zapisu 1.5e+08
    ones = [c for c in row.index if c not in NUMERIC and row[c] == 1]
    return f"{nums}\n  jedynki ({len(ones)}): " + ", ".join(ones)


def summary(X, features):
    print(f"Macierz: {X.shape[0]} filmów x {X.shape[1]} kolumn "
          f"(MIN_COUNT = {MIN_COUNT}) -> {MATRIX_FILE.relative_to(ROOT)}")
    print("Kolumny w grupach:")
    for name, count in Counter(map(group, X.columns)).items():
        print(f"  {name:<16} {count:>4}")
    first = X.index[0]
    print(f"\nPrzykład, film {first} '{features[first]['title']}':")
    print("  " + describe(X.loc[first]))


# ---------------------------------------------------------------- dane jednej osoby

def user_data(X, train, task, user):
    """Wycina z macierzy dane jednej osoby.

    Zwraca (indeks wszędzie: movie_id):
        X_train - wiersze filmów, które osoba oceniła w train.csv (90 x kolumny)
        y_train - jej oceny tych filmów (0-5), czyli to, czego drzewo ma się nauczyć
        X_task  - wiersze filmów do przewidzenia z task.csv (30 x kolumny)
        ids     - numery tych wierszy w task.csv, żeby wpisać przewidywania na miejsce
    """
    tr = train[train.user == user]
    tk = task[task.user == user]
    if tr.empty:
        raise ValueError(f"osoba {user} nie ma ocen w train.csv")

    X_train = X.loc[tr.movie_id]
    y_train = pd.Series(tr.rating.to_numpy(), index=X_train.index, name="rating")
    X_task = X.loc[tk.movie_id]
    ids = pd.Series(tk.id.to_numpy(), index=X_task.index, name="id")
    return X_train, y_train, X_task, ids


def show_user(user):
    """Wypisuje, co user_data() pobiera dla jednej osoby, i zapisuje to do CSV."""
    movies, train, task = load_all()
    X = load_matrix()
    try:
        X_train, y_train, X_task, ids = user_data(X, train, task, user)
    except ValueError as e:
        raise SystemExit(f"Błąd: {e}. Przykładowe osoby: {sorted(train.user.unique())[:5]}")
    titles = movies.set_index("movie_id").title.rename("title")

    print(f"Osoba {user}")
    print(f"  X_train: {X_train.shape[0]} filmów x {X_train.shape[1]} kolumn")
    print(f"  y_train: {len(y_train)} ocen, rozkład: "
          f"{y_train.value_counts().sort_index().to_dict()}")
    print(f"  X_task:  {X_task.shape[0]} filmów x {X_task.shape[1]} kolumn")
    print(f"  ids:     {len(ids)} numerów wierszy z task.csv ({ids.min()}-{ids.max()})")
    # kolumna z samymi 0 albo samymi 1 w filmach tej osoby nie da żadnego podziału
    useful = int((X_train.nunique() > 1).sum())
    print(f"  kolumny, które mogą podzielić jej filmy: {useful} / {X_train.shape[1]}")

    print("\nFilmy treningowe (ocena | movie_id | tytuł), od najwyżej ocenionych:")
    for mid, r in y_train.sort_values(ascending=False, kind="stable").items():
        print(f"  {r} | {mid:>3} | {titles[mid]}")
    print("\nFilmy do przewidzenia (id w task.csv | movie_id | tytuł):")
    for mid, i in ids.items():
        print(f"  {i} | {mid:>3} | {titles[mid]}")

    first = X_train.index[0]
    print(f"\nTak drzewo widzi film '{titles[first]}' (ocena {y_train[first]}):")
    print("  " + describe(X_train.loc[first]))

    USERS_DIR.mkdir(parents=True, exist_ok=True)
    train_file = USERS_DIR / f"user_{user}_train.csv"
    task_file = USERS_DIR / f"user_{user}_task.csv"
    pd.concat([titles.loc[X_train.index], y_train, X_train], axis=1).to_csv(
        train_file, encoding="utf-8")
    pd.concat([ids, titles.loc[X_task.index], X_task], axis=1).to_csv(
        task_file, encoding="utf-8")
    print(f"\nZapisano do wglądu: {train_file.relative_to(ROOT)}, {task_file.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Macierz cech filmów (Etap 1).")
    parser.add_argument("--user", type=int,
                        help="zamiast budować macierz, pokaż dane jednej osoby (np. 1642)")
    args = parser.parse_args()

    if args.user is None:
        features = load_features()
        X = build_matrix(features)
        check_matrix(X, features)
        save_matrix(X)
        summary(X, features)
    else:
        show_user(args.user)
