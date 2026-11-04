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

    # import w funkcji, bo features.py importuje ten moduł (import cykliczny)
    from features import COMPOSER_JOBS, NOT_WRITERS

    def crew_has(d, cond):
        return any(cond(p) for p in d.get("credits", {}).get("crew", []))

    # jeden check na każdą cechę z features.extract(), w tej samej kolejności;
    # klucz = nazwa cechy, w nawiasie pole TMDB, gdy jest inne
    checks = {
        "genres": lambda d: not d.get("genres"),
        "keywords": lambda d: not d.get("keywords", {}).get("keywords"),
        "directors": lambda d: not crew_has(d, lambda p: p.get("job") == "Director"),
        "cast / cast_wide": lambda d: not d.get("credits", {}).get("cast"),
        "year (release_date)": lambda d: not d.get("release_date"),
        "runtime = 0": lambda d: not d.get("runtime"),
        "vote_count = 0": lambda d: not d.get("vote_count"),
        "vote_average = 0": lambda d: not d.get("vote_average"),
        "budget = 0": lambda d: not d.get("budget"),
        "language (original_language)": lambda d: not d.get("original_language"),
        "countries (production_countries)": lambda d: not d.get("production_countries"),
        "companies (production_companies)": lambda d: not d.get("production_companies"),
        "writers": lambda d: not crew_has(d, lambda p: p.get("department") == "Writing"
                                          and p.get("job") not in NOT_WRITERS),
        "composers": lambda d: not crew_has(d, lambda p: p.get("job") in COMPOSER_JOBS),
        "cinematographers": lambda d: not crew_has(
            d, lambda p: p.get("job") == "Director of Photography"),
        "overview": lambda d: not d.get("overview"),
    }
    width = max(map(len, checks))
    print("Braki danych w cechach z features.py (liczba filmów):")
    for name, is_missing in checks.items():
        ids = [mid for mid, d in tmdb.items() if is_missing(d)]
        shown = ", ".join(map(str, ids[:15])) + (" …" if len(ids) > 15 else "")
        print(f"  {name:<{width}} {len(ids):>3}" + (f"   movie_id: {shown}" if ids else ""))


if __name__ == "__main__":
    tmdb = fetch_all()
    print(f"Filmów w cache: {len(list(CACHE.glob('*.json')))} / {len(tmdb)}")
    report(load_movies(), tmdb)
