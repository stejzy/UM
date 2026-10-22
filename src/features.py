"""Wyciąga wybrane cechy filmów z cache/tmdb/ do jednego pliku cache/features.json.

Zapisywane są wartości surowe (np. rok 1957); normalizacja odbywa się w kodzie.
Klucze w pliku to wewnętrzne movie_id (1-200), takie jak w train.csv i task.csv.
"""
import json
from collections import Counter

from data import ROOT, load_movies
from fetch_tmdb import fetch_all

FEATURES_FILE = ROOT / "cache" / "features.json"
N_CAST = 5         # ilu głównych aktorów brać (wg kolejności w napisach)
N_CAST_WIDE = 15   # szersza obsada
COMPOSER_JOBS = {"Original Music Composer", "Music"}
NOT_WRITERS = {"Story Artist", "Storyboard", "Storyboard Artist"}   # rysownicy, nie autorzy


def person(p):
    return {"id": p["id"], "name": p["name"]}


def unique_people(people):
    """Osoby bez powtórzeń (ta sama osoba może mieć kilka funkcji w ekipie)."""
    seen = {}
    for p in people:
        seen.setdefault(p["id"], person(p))
    return list(seen.values())


def extract(movie_id, d):
    """Cechy jednego filmu z surowych danych TMDB."""
    cast = sorted(d["credits"]["cast"], key=lambda p: p["order"])
    crew = d["credits"]["crew"]
    return {
        "tmdb_id": d["id"],
        "title": d["title"],
        "genres": [g["name"] for g in d["genres"]],
        "keywords": [k["name"] for k in d["keywords"]["keywords"]],
        "directors": unique_people(p for p in crew if p["job"] == "Director"),
        "cast": [person(p) for p in cast[:N_CAST]],
        "year": int(d["release_date"][:4]),
        "runtime": d["runtime"] or None,        # 0 w TMDB oznacza brak danych
        "vote_count": d["vote_count"],
        # cechy dodane w kroku 9.1 (kandydaci dla algorytmu genetycznego)
        "vote_average": d["vote_average"],
        "budget": d["budget"] or None,          # 0 w TMDB oznacza brak danych
        "language": d["original_language"],
        "countries": [c["iso_3166_1"] for c in d["production_countries"]],
        "companies": [{"id": c["id"], "name": c["name"]} for c in d["production_companies"]],
        "writers": unique_people(p for p in crew
                                 if p["department"] == "Writing" and p["job"] not in NOT_WRITERS),
        "composers": unique_people(p for p in crew if p["job"] in COMPOSER_JOBS),
        "cinematographers": unique_people(p for p in crew if p["job"] == "Director of Photography"),
        "cast_wide": [person(p) for p in cast[:N_CAST_WIDE]],
        "overview": d["overview"],
    }


def build():
    """Tworzy cache/features.json z cache/tmdb/ i zwraca {movie_id: cechy}."""
    tmdb = fetch_all()
    features = {mid: extract(mid, d) for mid, d in tmdb.items()}
    FEATURES_FILE.write_text(json.dumps({str(k): v for k, v in features.items()},
                                        ensure_ascii=False, indent=1), encoding="utf-8")
    return features


def load_features():
    """Wczytuje cache/features.json; zwraca {movie_id (int): cechy}."""
    raw = json.loads(FEATURES_FILE.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def summary(features):
    """Wypisuje, jak wyglądają cechy na 200 filmach (pomoc przy ich ocenie)."""
    n = len(features)

    def set_stats(name, key=lambda x: x):
        counts = Counter(key(x) for f in features.values() for x in f[name])
        sizes = [len(f[name]) for f in features.values()]
        shared = sum(1 for c in counts.values() if c > 1)
        print(f"{name}: {len(counts)} różnych wartości, {min(sizes)}-{max(sizes)} na film "
              f"(średnio {sum(sizes) / n:.1f}); w ≥2 filmach: {shared}")
        top = ", ".join(f"{v} ({c})" for v, c in counts.most_common(8))
        print(f"  najczęstsze: {top}")

    def num_stats(name):
        vals = [f[name] for f in features.values() if f[name] is not None]
        print(f"{name}: {min(vals)}-{max(vals)}, braki: {n - len(vals)}")

    def cat_stats(name):
        counts = Counter(f[name] for f in features.values())
        top = ", ".join(f"{v} ({c})" for v, c in counts.most_common(8))
        print(f"{name}: {len(counts)} różnych wartości; najczęstsze: {top}")

    set_stats("genres")
    set_stats("keywords")
    set_stats("directors", key=lambda p: p["name"])
    set_stats("cast", key=lambda p: p["name"])
    num_stats("year")
    num_stats("runtime")
    num_stats("vote_count")
    print("\n--- cechy dodane w kroku 9.1 ---")
    num_stats("vote_average")
    num_stats("budget")
    cat_stats("language")
    set_stats("countries")
    set_stats("companies", key=lambda c: c["name"])
    set_stats("writers", key=lambda p: p["name"])
    set_stats("composers", key=lambda p: p["name"])
    set_stats("cinematographers", key=lambda p: p["name"])
    set_stats("cast_wide", key=lambda p: p["name"])
    words = [len(f["overview"].split()) for f in features.values()]
    print(f"overview: {min(words)}-{max(words)} słów, puste: {words.count(0)}")


if __name__ == "__main__":
    features = build()
    assert set(features) == set(load_movies().movie_id)
    assert load_features() == features
    print(f"Zapisano {len(features)} filmów do {FEATURES_FILE.relative_to(ROOT)}\n")
    summary(features)
