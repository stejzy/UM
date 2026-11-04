"""Pobiera dane filmów z TMDB do cache/tmdb/{tmdb_id}.json (jednorazowo)."""
import json
import os
import time

import requests

from data import CACHE, load_movies

API_URL = "https://api.themoviedb.org/3/movie/{}"


def fetch(tmdb_id, session=None, retries=3):
    """Zwraca dane filmu z cache albo pobiera je z API i zapisuje do cache."""
    path = CACHE / f"{tmdb_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        raise SystemExit("Brak zmiennej TMDB_API_KEY. W PowerShellu: $env:TMDB_API_KEY = \"twój_klucz\"")

    session = session or requests.Session()
    params = {"api_key": api_key,
              "append_to_response": "credits,keywords"}
    for attempt in range(retries):
        r = session.get(API_URL.format(tmdb_id), params=params, timeout=20)
        if r.status_code == 429:   # za dużo zapytań, odczekaj i spróbuj ponownie
            time.sleep(int(r.headers.get("Retry-After", 2)))
            continue
        r.raise_for_status()
        data = r.json()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(0.05)           # nie zasypuj API zapytaniami
        return data
    raise RuntimeError(f"Nie udało się pobrać filmu {tmdb_id}")


def fetch_all():
    """Pobiera wszystkie filmy z movie.csv; zwraca {movie_id: dane z TMDB}."""
    CACHE.mkdir(parents=True, exist_ok=True)
    movies = load_movies()
    session = requests.Session()
    result = {}
    for row in movies.itertuples():
        result[row.movie_id] = fetch(row.tmdb_id, session)
    return result


def report(movies, tmdb):
    """Wypisuje kontrolę zgodności tytułów i braki danych."""
    mismatched = [(m.movie_id, m.title, tmdb[m.movie_id].get("title"),
                   tmdb[m.movie_id].get("original_title"))
                  for m in movies.itertuples()
                  if m.title not in (tmdb[m.movie_id].get("title"),
                                     tmdb[m.movie_id].get("original_title"))]
    print(f"Tytuły różne od TMDB (do ręcznego sprawdzenia): {len(mismatched)}")
    for mid, title, t, ot in mismatched:
        print(f"  {mid:>3}: {title!r} -> TMDB: {t!r} / {ot!r}")

    checks = {
        "release_date": lambda d: not d.get("release_date"),
        "runtime = 0": lambda d: not d.get("runtime"),
        "budget = 0": lambda d: not d.get("budget"),
        "revenue = 0": lambda d: not d.get("revenue"),
        "genres": lambda d: not d.get("genres"),
        "overview": lambda d: not d.get("overview"),
        "keywords": lambda d: not d.get("keywords", {}).get("keywords"),
        "cast": lambda d: not d.get("credits", {}).get("cast"),
        "director": lambda d: not any(c.get("job") == "Director"
                                      for c in d.get("credits", {}).get("crew", [])),
        "production_countries": lambda d: not d.get("production_countries"),
        "belongs_to_collection": lambda d: not d.get("belongs_to_collection"),
    }
    print("Braki danych (liczba filmów):")
    for name, is_missing in checks.items():
        ids = [mid for mid, d in tmdb.items() if is_missing(d)]
        shown = ", ".join(map(str, ids[:15])) + (" …" if len(ids) > 15 else "")
        print(f"  {name:<22} {len(ids):>3}" + (f"   movie_id: {shown}" if ids else ""))


if __name__ == "__main__":
    tmdb = fetch_all()
    print(f"Filmów w cache: {len(list(CACHE.glob('*.json')))} / {len(tmdb)}")
    report(load_movies(), tmdb)
