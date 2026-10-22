"""Przygotowanie cech do liczenia podobieństw (w pamięci, bez zapisu do pliku).

Wynik prepare() to słownik {nazwa cechy: wartości dla 200 filmów}, gdzie
pozycja i odpowiada filmowi MOVIE_IDS[i] (czyli movie_id = i + 1).
- cechy liczbowe: tablica numpy z wartościami w [0, 1], brak danych = NaN
- cechy kategoryczne: lista wartości (None = brak)
- cechy zbiorowe: lista zbiorów Pythona
- cechy tekstowe: macierz wektorów TF-IDF (każdy wiersz o długości 1)
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from features import load_features

# Cechy liczbowe: nazwa -> funkcja wyciągająca surową wartość (przed normalizacją)
NUMERIC = {
    "year": lambda f: f["year"],
    "runtime": lambda f: f["runtime"],
    "vote_count": lambda f: np.log1p(f["vote_count"]),   # logarytm wyrównuje rozpiętość
    "vote_average": lambda f: f["vote_average"],
    "budget": lambda f: np.log1p(f["budget"]) if f["budget"] else None,
}

# Cechy kategoryczne: nazwa -> funkcja zwracająca jedną wartość
CATEGORIES = {
    "language": lambda f: f["language"],
    # "collection" usunięta: tylko 3 serie mają w danych po 2 filmy, więc podobieństwo
    # jest niezerowe dla 3 par z 19 900 (prawie stałe zero, nic nie wnosi)
}

# Cechy zbiorowe: nazwa -> funkcja zwracająca zbiór
SETS = {
    "genres": lambda f: set(f["genres"]),
    "keywords": lambda f: set(f["keywords"]),
    "directors": lambda f: {p["id"] for p in f["directors"]},
    "cast": lambda f: {p["id"] for p in f["cast"]},
    "countries": lambda f: set(f["countries"]),
    "companies": lambda f: {c["id"] for c in f["companies"]},
    "writers": lambda f: {p["id"] for p in f["writers"]},
    "composers": lambda f: {p["id"] for p in f["composers"]},
    "cinematographers": lambda f: {p["id"] for p in f["cinematographers"]},
    "cast_wide": lambda f: {p["id"] for p in f["cast_wide"]},
}

# Cechy tekstowe: nazwa -> funkcja zwracająca tekst
TEXTS = {
    "overview": lambda f: f["overview"],
}

FEATURE_TYPES = {**{n: "numeric" for n in NUMERIC},
                 **{n: "category" for n in CATEGORIES},
                 **{n: "set" for n in SETS},
                 **{n: "text" for n in TEXTS}}


def minmax(x):
    """Skaluje wartości do [0, 1]; NaN zostaje NaN."""
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def tfidf(texts):
    """Wektory TF-IDF (bez słów typu „the”, „and”), znormalizowane do długości 1."""
    vec = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True)
    return vec.fit_transform(texts).toarray()


def prepare(features=None):
    """Zwraca (MOVIE_IDS, {nazwa cechy: przygotowane wartości})."""
    features = features or load_features()
    movie_ids = sorted(features)
    prep = {}
    for name, get in NUMERIC.items():
        raw = np.array([np.nan if get(features[m]) is None else get(features[m])
                        for m in movie_ids], dtype=float)
        prep[name] = minmax(raw)
    for name, get in CATEGORIES.items():
        prep[name] = [get(features[m]) for m in movie_ids]
    for name, get in SETS.items():
        prep[name] = [get(features[m]) for m in movie_ids]
    for name, get in TEXTS.items():
        prep[name] = tfidf([get(features[m]) for m in movie_ids])
    return movie_ids, prep


if __name__ == "__main__":
    features = load_features()
    movie_ids, prep = prepare(features)
    assert movie_ids == list(range(1, len(movie_ids) + 1))
    assert set(prep) == set(FEATURE_TYPES)
    print(f"Cech: {len(FEATURE_TYPES)} "
          f"({', '.join(f'{t}: {list(FEATURE_TYPES.values()).count(t)}' for t in ['numeric', 'category', 'set', 'text'])})\n")

    print("Cechy liczbowe po normalizacji (min / kwartyle / max, braki):")
    for name in NUMERIC:
        v = prep[name]
        q = np.nanpercentile(v, [0, 25, 50, 75, 100])
        print(f"  {name:<17} " + " / ".join(f"{x:.2f}" for x in q)
              + f", braki: {int(np.isnan(v).sum())}")

    print("\nCechy kategoryczne (różnych wartości, braki):")
    for name in CATEGORIES:
        vals = prep[name]
        print(f"  {name:<17} {len(set(vals) - {None})}, braki: {vals.count(None)}")

    print("\nCechy zbiorowe (rozmiar zbioru: min-max, puste):")
    for name in SETS:
        sizes = [len(s) for s in prep[name]]
        print(f"  {name:<17} {min(sizes)}-{max(sizes)}, puste: {sizes.count(0)}")

    print("\nCechy tekstowe:")
    for name in TEXTS:
        X = prep[name]
        print(f"  {name:<17} słownik {X.shape[1]} słów, puste wektory: {int((X.sum(1) == 0).sum())}")
