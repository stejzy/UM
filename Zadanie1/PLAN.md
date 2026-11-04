# Rekomendacje filmów metodą kNN: plan krok po kroku

Odznaczaj zrobione punkty, zmieniając `- [ ]` na `- [x]`.

## Postęp ogólny

- [x] Etap 0. Przygotowanie
- [x] Etap 1. Wczytanie danych
- [x] Etap 2. Pobranie danych z TMDB
- [x] Etap 3. Wybór cech
- [x] Etap 4. Przygotowanie (preprocessing) cech
- [x] Etap 5. Funkcje podobieństwa dla typów
- [x] Etap 6. Łączna miara podobieństwa filmów
- [x] Etap 7. Predykcja kNN
- [x] Etap 8. Sposób oceny: walidacja
- [ ] Etap 9. Strojenie: cechy (algorytm genetyczny) i k
- [ ] Etap 10. Generowanie `submission.csv`
- [ ] Etap 11. Przygotowanie `code.zip`

**Do oddania:** `submission.csv` (plik `task.csv` z `NULL` zastąpionymi ocenami 0–5) oraz `code.zip` (kod i zasoby).

---

## Etap 0. Przygotowanie

### Krok 0.1: Środowisko

- [x] Folder `Zadanie1` otwarty w PyCharmie jako projekt
- [x] Wirtualne środowisko `.venv` na Pythonie 3.13
- [x] Foldery `src/` i `cache/tmdb/`
- [x] Folder `src` oznaczony jako *Sources Root* (prawym → Mark Directory as → Sources Root)
- [x] Plik `requirements.txt` (`pandas`, `numpy`, `requests`, `scikit-learn`, wersje przypięte)
- [x] Biblioteki zainstalowane w `.venv`
- [x] Test środowiska: biblioteki się importują, dane wczytują się poprawnie (tymczasowy `check_env.py`, już usunięty)

Docelowa struktura projektu:

```
Zadanie1/
  movie/            ← dane od prowadzącego (nie zmieniaj)
  cache/tmdb/       ← pobrane dane z TMDB (po 1 pliku JSON na film)
  src/
    data.py         ← wczytywanie i walidacja danych
    fetch_tmdb.py   ← pobieranie danych z API
    features.py     ← wyciąganie cech do cache/features.json
    preprocess.py   ← przygotowanie cech (log, normalizacja, zbiory)
    similarity.py   ← funkcje podobieństwa dla typów
    sim_matrix.py   ← macierze podobieństw i łączna miara z wagami
    knn.py          ← predykcja kNN
    evaluate.py     ← walidacja, miary, baseline
    fast_eval.py    ← szybka ocena LOO dla wszystkich k naraz (fitness)
    ga.py           ← algorytm genetyczny: wybór cech per osoba
    tune.py         ← strojenie, walidacja zagnieżdżona, zapis params/
    predict.py      ← generowanie submission.csv
  params/           ← dostrojone parametry (global.json, users.json)
  results/          ← wyniki strojenia do sprawozdania
  requirements.txt
  README.md
```

Ścieżki do plików licz względem położenia pliku z kodem. PyCharm uruchamia skrypty z katalogiem roboczym `src/`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # = folder Zadanie1
DATA = ROOT / "movie"
CACHE = ROOT / "cache" / "tmdb"
```

### Krok 0.2: Klucz API do TMDB

- [x] Konto na themoviedb.org i wygenerowany **API Key** (krótki, 32 znaki; nie *Read Access Token*)
- [x] Klucz zapisany w zmiennej środowiskowej Windows `TMDB_API_KEY` (zmienna użytkownika)
- [x] Klucz działa: zapytanie testowe do TMDB zwraca film 389 = *12 Angry Men*
- [x] PyCharm widzi zmienną `TMDB_API_KEY`

Klucza **nie wpisuj w kodzie**, bo kod trafi do `code.zip`. W kodzie: `os.environ["TMDB_API_KEY"]`.

---

## Etap 1. Wczytanie danych

Pliki nie mają nagłówków, a separatorem jest `;`. **Uwaga:** `movie.csv` ma końce linii `\r`.

| Plik | Kolumny | Zawartość |
|---|---|---|
| `movie.csv` | `movie_id; tmdb_id; title` | 200 filmów; `movie_id` 1–200 to ID w zadaniu, `tmdb_id` to ID w TMDB |
| `train.csv` | `id; user; movie_id; rating` | 32 220 ocen; 358 osób × 90 ocen; oceny 0–5 |
| `task.csv` | `id; user; movie_id; rating` | 10 740 wierszy; 358 osób × 30 filmów; `rating = NULL` do przewidzenia |

- [x] Wczytywanie danych w module `src/data.py`: `load_movies()`, `load_train()`, `load_task()`, `load_all()`
- [x] Walidacja `validate()`: unikalne ID, `movie_id` 1–200, oceny całkowite 0–5, w `task` same `NULL`, brak wspólnych par (osoba, film) między `train` i `task`, wszystkie osoby z `task` mają oceny w `train`
- [x] `python src/data.py` wypisuje: `Walidacja: OK`, 200 filmów, 358 osób, 90 ocen w `train` i 30 filmów w `task` na osobę

W kolejnych modułach: `from data import load_all, CACHE` i `movies, train, task = load_all()`.

```python
import io, pandas as pd

txt = (DATA / "movie.csv").read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
movies = pd.read_csv(io.StringIO(txt), sep=";", header=None,
                     names=["movie_id", "tmdb_id", "title"])
train = pd.read_csv(DATA / "train.csv", sep=";", header=None,
                    names=["id", "user", "movie_id", "rating"])
task  = pd.read_csv(DATA / "task.csv",  sep=";", header=None,
                    names=["id", "user", "movie_id", "rating"])
```

---

## Etap 2. Pobranie danych z TMDB

- [x] Skrypt `src/fetch_tmdb.py`: jedno zapytanie na film, `GET https://api.themoviedb.org/3/movie/{tmdb_id}?append_to_response=credits,keywords`
- [x] Każda odpowiedź zapisana jako `cache/tmdb/{tmdb_id}.json`; jeśli plik istnieje, nie pytaj API ponownie
- [x] W `cache/tmdb/` jest 200 plików
- [x] Zanotowane braki danych (poniżej)

W kolejnych modułach: `from fetch_tmdb import fetch_all` i `tmdb = fetch_all()` daje `{movie_id: dane z TMDB}`. Czyta tylko z cache, bez klucza API.

**Wyniki kontroli (`python src/fetch_tmdb.py`):**

- Tytuły: 8 filmów ma w TMDB inny tytuł niż w `movie.csv`, ale to te same filmy (tłumaczenie lub inny zapis): 55 *GoodFellas*, 78 *Nausicaä*, 90 *Princess Mononoke*, 93 *Life of Brian*, 138 *Return of the Jedi*, 141 *Like Stars on Earth*, 196 *My Neighbor Totoro*, 200 *Three Colors: Red*.
- Kompletne dla wszystkich 200 filmów: data premiery, czas trwania, gatunki, opis, słowa kluczowe, obsada, reżyser, kraje produkcji.
- `budget = 0` (brak danych): 6 filmów (movie_id 59, 84, 107, 118, 187, 200), więc w etapie 4 zamienić na `NaN`.
- `revenue = 0`: 5 filmów (17, 46, 59, 118, 168).
- `belongs_to_collection`: tylko 54 filmy należą do jakiejś kolekcji, a wspólną kolekcję mają tylko 3 pary (Dark Knight, Indiana Jones, Terminator). **Ta cecha prawie nic nie da, można ją pominąć.**

Cache trafi do `code.zip`, więc prowadzący odtworzy wynik bez klucza API.

```python
import os, json, time, requests

def fetch(tmdb_id):
    path = CACHE / f"{tmdb_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    r = requests.get(f"https://api.themoviedb.org/3/movie/{tmdb_id}",
                     params={"api_key": os.environ["TMDB_API_KEY"],
                             "append_to_response": "credits,keywords"})
    r.raise_for_status()
    data = r.json()
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.05)   # nie zasypuj API zapytaniami
    return data
```

---

## Etap 3. Wybór cech

**Pogrubione** = wybrane na start. Resztę można dodawać i sprawdzać walidacją, czy pomaga.

| Cecha | Pole w TMDB | Typ | Dlaczego może mieć znaczenie |
|---|---|---|---|
| **Gatunki** | `genres[].name` | zbiór | Najsilniejszy sygnał gustu |
| **Rok premiery** | `release_date[:4]` | liczba | Stare kino a współczesne blockbustery |
| **Reżyser** | `credits.crew` z `job == "Director"` | zbiór | Fani Kubricka, Hitchcocka, Nolana |
| **Główni aktorzy** | `credits.cast` (pierwsze 5 osób wg `order`) | zbiór | Ulubieni aktorzy |
| **Słowa kluczowe** | `keywords.keywords[].name` | zbiór | Tematyka: wojna, mafia, podróże w czasie… |
| **Popularność** | `vote_count` (log) | liczba | Mainstream a nisza |
| **Czas trwania** | `runtime` | liczba | Ktoś nie znosi 3-godzinnych filmów |
| Średnia ocen TMDB | `vote_average` | liczba | „Oczywiste arcydzieła” a reszta |
| Język oryginału | `original_language` | kategoria | Kino anglojęzyczne a japońskie lub europejskie |
| Kraje produkcji | `production_countries[].iso_3166_1` | zbiór | Podobnie jak język |
| Budżet | `budget` (log, 0 = brak danych) | liczba | Wielkie produkcje a kino autorskie |
| Seria / kolekcja | `belongs_to_collection.id` | kategoria | Fani całej serii |
| Scenarzysta, kompozytor | `credits.crew` (`Screenplay`, `Writer`, `Original Music Composer`) | zbiór | Słabszy sygnał, opcjonalny |
| Opis fabuły | `overview` | tekst | Podobieństwo treści, opcjonalne |

- [x] Wybrany startowy zestaw cech
- [x] Spisane uzasadnienie wyboru (poniżej; przenieść do README)
- [x] Cechy wyciągnięte skryptem `src/features.py` do jednego pliku `cache/features.json`

**Startowy zestaw cech** (wartości surowe w `cache/features.json`, klucz = wewnętrzne `movie_id`):

| Cecha | Klucz | Typ | Źródło w TMDB | Na 200 filmach |
|---|---|---|---|---|
| Gatunki | `genres` | zbiór nazw | `genres[].name` | 17 gatunków, 1–4 na film; najczęściej Drama (126), Crime (52), Thriller (49) |
| Słowa kluczowe | `keywords` | zbiór nazw | `keywords.keywords[].name` | 2370 różnych, ok. 21 na film; 709 występuje w ≥2 filmach |
| Reżyser | `directors` | zbiór osób (`id`, `name`) | `credits.crew`, `job == "Director"` | 130 reżyserów, 42 z ≥2 filmami (Kubrick 8, Hitchcock 7) |
| 5 głównych aktorów | `cast` | zbiór osób (`id`, `name`) | `credits.cast`, 5 najmniejszych `order` | 799 aktorów, 134 w ≥2 filmach (De Niro 8, Stewart 7) |
| Popularność | `vote_count` | liczba | `vote_count` | 708–40 481 (skala logarytmiczna w etapie 4) |
| Rok premiery | `year` | liczba | `release_date[:4]` | 1921–2012 |
| Czas trwania | `runtime` | liczba | `runtime` | 68–238 min, bez braków |

Decyzje:
- Aktorzy wybierani po **`order`** (kolejność w napisach), nie po `popularity`. Popularność osoby w TMDB to bieżący trend wyświetleń, a nie ważność roli. Wyjątek po stronie TMDB: w *12 Angry Men* kolejność odpowiada numerom przysięgłych, więc Henry Fonda nie trafia do top 5.
- Popularność filmu jako **`vote_count`**, nie `popularity`. `popularity` to bieżący trend i zmienia się z dnia na dzień.
- Osoby porównywane po `id` (jednoznaczne), gatunki i słowa kluczowe po nazwie.
- Pominięte na start: średnia ocen, język, kraje, budżet, kolekcja (tylko 3 wspólne pary), opis fabuły. Można je dodać w kroku 9.2.

W kolejnych modułach: `from features import load_features` i `feat = load_features()` daje `{movie_id: cechy}`. Po zmianie zestawu cech uruchom ponownie `python src/features.py`.

---

## Etap 4. Przygotowanie (preprocessing) cech

Wynik: jedna tabela, 200 wierszy (filmy) × kolumny z cechami w „czystej” postaci.

- [x] **4.1** Cechy liczbowe → liczby: `year`, `runtime`, `vote_count` jako `log(1 + vote_count)`
- [x] **4.2** Cechy zbiorowe → zbiory Pythona: `genres`, `keywords` (nazwy), `directors`, `cast` (ID osób)
- [x] **4.3** Brakujące dane: kod zamienia brak na `NaN`; w startowym zestawie braków nie ma, żaden zbiór nie jest pusty
- [x] **4.4** Normalizacja cech liczbowych do [0, 1] metodą min-max po 200 filmach: `(x - min) / (max - min)`

Kod: `src/preprocess.py`. Wynik jest liczony w pamięci, bez zapisu do pliku (trwa ułamek sekundy, a sposób przygotowania może się jeszcze zmienić):

```python
from preprocess import prepare, FEATURE_TYPES
movie_ids, prep = prepare()
prep["year"]     # tablica 200 liczb z [0, 1]; pozycja i = film movie_id i + 1
prep["genres"]   # lista 200 zbiorów
FEATURE_TYPES    # {"year": "numeric", ..., "genres": "set", ...}
```

Nową cechę dodajesz jednym wpisem w `NUMERIC` lub `SETS` w `preprocess.py` (oraz w `features.py`, jeśli jeszcze jej nie wyciąga).

Wartości po normalizacji (`python src/preprocess.py`), min / kwartyle / max:

| Cecha | Wartości | Uwagi |
|---|---|---|
| `year` | 0.00 / 0.45 / 0.70 / 0.87 / 1.00 | 1921 → 0, 2012 → 1 |
| `runtime` | 0.00 / 0.22 / 0.32 / 0.42 / 1.00 | Połowa filmów w 0.22–0.42, bo skalę rozciąga film 238-minutowy. **Do sprawdzenia w etapie 9:** logarytm lub obcięcie skrajnych wartości |
| `vote_count` | 0.00 / 0.34 / 0.58 / 0.74 / 1.00 | Po logarytmie rozkład jest równomierny |

Logarytm wyrównuje ogromne rozpiętości (budżet, liczba głosów). Normalizacja sprawia, że żadna cecha nie dominuje tylko dlatego, że ma większe liczby.

---

## Etap 5. Funkcje podobieństwa dla typów

Każda funkcja zwraca wartość z **[0, 1]**: 1 = identyczne, 0 = zupełnie różne.

- [x] `sim_numeric`: `1 - |a - b|` (na znormalizowanych wartościach); brak danych → 0.5
- [x] `sim_category`: 1 gdy równe, inaczej 0 (gotowa na później, w startowym zestawie nie ma cech kategorycznych)
- [x] `sim_set`: współczynnik Jaccarda `|A ∩ B| / |A ∪ B|`; pusty zbiór → 0
- [ ] (opcjonalnie) `sim_text`: TF-IDF + podobieństwo cosinusowe (`scikit-learn`)
- [x] Rejestr cech `FEATURE_TYPES = {nazwa: typ}` (w `preprocess.py`) i słownik `SIM_FN = {typ: funkcja}` (w `similarity.py`)
- [x] Testy (`python src/similarity.py`): wartości w [0, 1], symetria, film identyczny sam ze sobą = 1

Kod: `src/similarity.py`. `feature_sims(prep, FEATURE_TYPES, i, j)` zwraca podobieństwa dwóch filmów dla każdej cechy.

Przykładowe podobieństwa:

| Para filmów | year | runtime | vote_count | genres | keywords | directors | cast |
|---|---|---|---|---|---|---|---|
| The Godfather / Pulp Fiction | 0.76 | 0.88 | 0.93 | 0.25 | 0.03 | 0 | 0 |
| The Godfather / The Matrix | 0.70 | 0.77 | 0.95 | 0 | 0 | 0 | 0 |
| The Matrix / Blade Runner | 0.81 | 0.89 | 0.85 | 0.25 | 0.07 | 0 | 0 |
| GoodFellas / The Departed | 0.82 | 0.96 | 0.97 | 0.67 | 0.11 | 1 | 0 |
| 12 Angry Men / The Matrix | 0.54 | 0.77 | 0.75 | 0 | 0 | 0 | 0 |

**Obserwacja na etap 6:** wszystkie podobieństwa mieszczą się w [0, 1], ale mają różne typowe poziomy. Liczbowe zwykle wynoszą 0.5–1, a zbiorowe zwykle 0–0.1 (słowa kluczowe, aktorzy). W zwykłej sumie cechy liczbowe będą więc dominować w „tle”, a zbiory zadziałają jak bonus za trafienie. Do sprawdzenia: wagi albo dodatkowe wyrównanie skali podobieństw każdej cechy (np. przez jej średnią lub ranking).

```python
import math

def sim_numeric(a, b):
    if a is None or b is None or math.isnan(a) or math.isnan(b):
        return 0.5
    return 1.0 - abs(a - b)

def sim_category(a, b):
    if a is None or b is None:
        return 0.0
    return 1.0 if a == b else 0.0

def sim_set(A, B):
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)
```

---

## Etap 6. Łączna miara podobieństwa filmów

$$
\text{sim}(f_1, f_2) = \frac{\sum_i w_i \cdot \text{sim}_i(f_1, f_2)}{\sum_i w_i}
$$

`w_i` to waga cechy *i*. Na początek wszystkie = 1, a waga 0 oznacza wyłączenie cechy.

- [x] Macierze podobieństw 200×200 dla każdej cechy, policzone z góry: `S` o kształcie `(7, 200, 200)`, liczone w 0.1 s
- [x] Funkcja łącząca z wagami `combined(S, w)`
- [x] Dwa warianty skali do porównania w etapie 9: `raw` i `std` (standaryzacja każdej cechy)

Kod: `src/sim_matrix.py`.

```python
from sim_matrix import build, similarity_matrices, standardize, combined
M = build("raw")    # macierz 200x200 w [0, 1], wagi = 1; M[i, j] = podobieństwo movie_id i+1 i j+1
M = build("std")    # wariant ze standaryzacją cech
# przy strojeniu wag (etap 9): S liczone raz, potem combined(S, w) dla różnych w
```

**Typowe poziomy podobieństw** (`python src/sim_matrix.py`, poza przekątną):

| Cecha | Średnia | Odchylenie | % par z podobieństwem 0 |
|---|---|---|---|
| year | 0.70 | 0.22 | 0% |
| runtime | 0.82 | 0.15 | 0% |
| vote_count | 0.72 | 0.20 | 0% |
| genres | 0.20 | 0.23 | 44% |
| keywords | 0.009 | 0.018 | 70% |
| directors | 0.008 | 0.09 | 99% |
| cast | 0.002 | 0.017 | 99% |

**Udział cech w zróżnicowaniu łącznej miary** (wagi = 1). O tym, kto jest sąsiadem, decyduje rozrzut podobieństw cechy, a nie jej średni poziom:

| Wariant | year | runtime | vote_count | genres | keywords | directors | cast |
|---|---|---|---|---|---|---|---|
| `raw` | 24% | 16% | 22% | 25% | **2%** | 10% | **2%** |
| `std` | 14% | 14% | 14% | 14% | 14% | 14% | 14% |

**Wnioski:**
- W `raw` słowa kluczowe i aktorzy prawie nie wpływają na wynik (po 2%), choć zostały wybrane jako ważne cechy.
- `std` wyrównuje wpływ, ale pojedynczy wspólny aktor staje się bardzo silnym sygnałem. Przykład: dla *Matrixa* w top 5 trafia *Mystic River* (wspólny Laurence Fishburne).
- Oba warianty dają sensownych sąsiadów: *GoodFellas* → *Casino*, *Raging Bull*, *Taxi Driver*; *Totoro* → *Nausicaä*, *Princess Mononoke*.
- **Który lepszy, rozstrzygnie walidacja (etap 8–9).** Wagi w etapie 9.4 mogą też poprawić `raw`, np. zwiększając wagę słów kluczowych.

---

## Etap 7. Predykcja kNN

Dla osoby *u* i filmu *m*: weź 90 ocenionych filmów → wybierz *k* najbardziej podobnych do *m* → połącz ich oceny.

| Sposób | Opis |
|---|---|
| Głosowanie | najczęstsza ocena wśród sąsiadów (remisy trzeba rozstrzygać) |
| Głosowanie ważone | każdy sąsiad głosuje z siłą = podobieństwo |
| Średnia | średnia ocen → zaokrąglenie |
| **Średnia ważona** | $\hat r = \frac{\sum s_j r_j}{\sum s_j}$ → zaokrąglenie (**na start**) |

- [x] Funkcja `predict` z czterema sposobami łączenia: `vote`, `wvote`, `mean`, `wmean`
- [x] Wynik zawsze jest **liczbą całkowitą 0–5** (zaokrąglenie „szkolne”: 2.5 → 3)
- [x] Remis w zwykłym głosowaniu rozstrzyga suma podobieństw (wygrywa ocena bliższych sąsiadów)

Kod: `src/knn.py`.

```python
from knn import user_data, predict, neighbours, METHODS
users = user_data(train)                 # {user: (pozycje ocenionych filmów, oceny)}
rated_idx, ratings = users[1642]
predict(M, movie_id - 1, rated_idx, ratings, k=5, method="wmean")   # -> int 0-5
```

Przykład (`python src/knn.py`, osoba 1642, `raw`, k = 5): dla *Taxi Driver* najbliższe ocenione filmy to *GoodFellas*, *Casino* i *Raging Bull*, wszystkie ocenione na 1. Przewidywanie: 1 (głosowanie) lub 2 (średnia).

Rozkład przewidywań na całym `task.csv` (`raw`, k = 5) w porównaniu z rozkładem prawdziwych ocen w `train.csv`:

| | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| `train.csv` (prawdziwe) | 5% | 10% | 19% | 26% | 21% | 19% |
| `vote` / `wvote` | 5% | 8% | 19% | 28% | 21% | 18% |
| `mean` / `wmean` | 1% | 4% | 20% | 44% | 27% | 4% |

**Obserwacje:**
- Średnia „ściąga” przewidywania do środka skali: prawie nie ma ocen 0 i 5. Głosowanie zachowuje skrajne oceny.
- W wariancie `raw` wersje ważone dają prawie to samo co zwykłe, bo podobieństwa sąsiadów są zbliżone (np. 0.45–0.51).
- Który sposób daje więcej trafień, rozstrzygnie walidacja (etap 8).

---

## Etap 8. Sposób oceny: walidacja

- [x] **8.1** Miary: **dokładność (accuracy)** jako główna, MAE pomocniczo
- [x] **8.2** Baseline: „najczęstsza ocena osoby” oraz „zaokrąglona średnia ocen osoby”
- [x] **8.3** Walidacja dla każdej osoby: leave-one-out (domyślnie) lub *n*-krotna (`folds=n`)
- [ ] **kNN lepszy od baseline**: na razie **nie** (patrz niżej), do osiągnięcia w etapie 9

Kod: `src/evaluate.py`.

```python
from evaluate import evaluate, knn_predictor, most_frequent, user_mean
r = evaluate(knn_predictor(M, k=5, method="wvote"), users)              # leave-one-out
r = evaluate(knn_predictor(M, k=5, method="wvote"), users, folds=5)     # 5-krotna
r["accuracy"], r["mae"], r["per_user"]   # średnia, MAE, {user: accuracy}
```

**Rozmiar zbioru walidacyjnego** ustawia parametr `folds`:
- `folds=None` (domyślnie): leave-one-out, w zbiorze walidacyjnym zawsze **1 film**, 90 rund na osobę,
- `folds=5`: 5 grup po **18 filmów**, 5 rund; `folds=10`: 10 grup po 9 filmów.

Filmy z zbioru walidacyjnego nigdy nie są sąsiadami. Jedna ocena całego zbioru (358 osób) trwa ok. 0.6 s.

**Wyniki** (`python src/evaluate.py`, leave-one-out, 358 osób):

| Metoda | Accuracy | MAE |
|---|---|---|
| Baseline: najczęstsza ocena osoby | **41.1%** | 1.045 |
| Baseline: średnia ocen osoby | 32.6% | **0.969** |
| kNN `raw`, k = 5, `vote` / `wvote` | 36.6% | 1.145 |
| kNN `raw`, k = 5, `mean` / `wmean` | 32.5% | 1.019 |
| kNN `std`, k = 5, `vote` | 36.5% | 1.149 |
| kNN `std`, k = 5, `wmean` | 32.3% | 1.023 |

kNN `wvote` dla większych *k* (wagi = 1):

| | k = 5 | 10 | 15 | 20 | 30 | 45 |
|---|---|---|---|---|---|---|
| `raw` | 36.6% | 38.6% | 39.4% | 40.1% | 40.5% | 41.1% |
| `std` | 36.3% | 38.6% | 39.6% | 40.2% | 40.7% | 41.1% |

**Wnioski:**
- **kNN ze wszystkimi cechami i równymi wagami nie bije jeszcze baseline.** Im większe *k*, tym bliżej najczęstszej oceny osoby (przy *k* = 89 głosowanie to po prostu najczęstsza ocena). Obecna miara podobieństwa niesie bardzo mało informacji o tym, jak osoba oceni film.
- Głosowanie wygrywa ze średnią pod względem accuracy, a średnia wygrywa pod względem MAE.
- `raw` i `std` wypadają praktycznie tak samo.
- LOO i 5-krotna walidacja dają prawie identyczne wyniki (36.57% wobec 36.62%), więc pomiar jest stabilny.
- **Zadanie dla etapu 9:** wybór cech i wagi per osoba (9.2–9.4) muszą znaleźć cechy, które faktycznie coś mówią o guście danej osoby. Wyniki trzeba porównywać z baseline 41.1%.

---

## Etap 9. Strojenie: cechy (algorytm genetyczny) i k

### Co już wiemy (prototyp, przed właściwym strojeniem)

Zanim wybrałem metodę strojenia, sprawdziłem prototypem, co da się osiągnąć. Wszystkie wyniki to accuracy na 358 osobach.

**Diagnoza danych:**
- Średnie oceny filmów prawie się nie różnią (odchylenie 0.19), za to średnie ocen osób różnią się mocno (0.68). Większość informacji siedzi w tym, **jak dana osoba w ogóle ocenia**, a nie w tym, jaki to film.
- Korelacja oceny z rokiem, popularnością i czasem trwania w obrębie osoby wynosi średnio ok. 0.03, czyli prawie zero. U części osób sygnał jednak jest: wyraźną zależność od popularności (|r| > 0.3) ma 7% osób, a przypadek dałby ok. 0.5%.
- Oceny różnych osób są ze sobą prawie nieskorelowane (średnio 0.01).
- **Wniosek:** sygnał w danych jest słaby i występuje tylko u części osób. Realny cel to kilka punktów procentowych ponad baseline 41%, a nie 60%.

**Uczciwa walidacja zagnieżdżona** (strojenie na 72 ocenach, sprawdzenie na odłożonych 18; szczegóły w kroku 9.5):

| Strategia | Accuracy |
|---|---|
| Baseline: najczęstsza ocena osoby | **40.9%** |
| Wszystkie cechy, wspólne k = 30 | 40.5% |
| Wszystkie cechy, k per osoba | 40.1% |
| Najlepszy podzbiór cech per osoba (pełne przeszukanie 127 podzbiorów), wspólne k | 40.0–40.5% |
| Najlepszy podzbiór cech per osoba + k per osoba | 38.1–38.4% |

**Globalnie** (jeden zestaw dla wszystkich, LOO): najlepszy z 255 podzbiorów 8 cech daje 41.6% przy k = 89, czyli praktycznie baseline.

**Co z tego wynika:**
1. **Przeuczenie to główny wróg.** Przy 90 ocenach na osobę dobór cech „pod osobę” znajduje przypadkowe wzorce. Wynik na danych strojenia rośnie, a na nowych spada. Im więcej opcji do wyboru (więcej cech, k per osoba), tym gorzej.
2. **Algorytm genetyczny nie znajdzie lepszego podzbioru niż pełne przeszukanie tych samych cech.** Prototyp sprawdził wszystkie 127 podzbiorów per osoba, a to górna granica tego, co GA mógłby znaleźć na 7 cechach. Wynik: gorzej niż baseline.
3. **GA ma sens przy większej liczbie cech** (np. 15 cech to 32 767 podzbiorów; pełne przeszukanie dla 358 osób jest wtedy zbyt wolne). Szansa na poprawę istnieje tylko wtedy, gdy nowe cechy niosą sygnał, a GA jest przy tym **mocno chroniony przed przeuczeniem** (kara za liczbę cech, start od ustawień globalnych, uczciwa walidacja zagnieżdżona).
4. **k: wspólne dla wszystkich, nie per osoba.** W walidacji zagnieżdżonej k per osoba było zawsze gorsze od wspólnego. Wspólne k i sposób łączenia dobieramy raz na wszystkich osobach.

### Plan strojenia (kolejność kroków)

```
9.1 więcej cech-kandydatów      → features.py, preprocess.py
9.2 szybka ocena (fitness)      → src/fast_eval.py
9.3 ustawienia globalne         → k, sposób łączenia, globalny zestaw cech
9.4 GA per osoba                → src/ga.py
9.5 uczciwa walidacja           → czy GA bije baseline i ustawienia globalne?
9.6 strojenie końcowe + zapis   → params/*.json
```

### 9.1 Więcej cech-kandydatów

GA ma z czego wybierać: dodane zostały **wszystkie cechy z tabeli z etapu 3** oraz trzy dodatkowe (wytwórnie, operator kamery, szersza obsada). Razem jest **18 cech**.

| Cecha | Klucz | Typ | Źródło w TMDB | Uwagi |
|---|---|---|---|---|
| Rok premiery | `year` | liczba | `release_date[:4]` | startowa |
| Czas trwania | `runtime` | liczba | `runtime` | startowa |
| Popularność | `vote_count` | liczba | `log(1 + vote_count)` | startowa |
| Średnia ocen TMDB | `vote_average` | liczba | `vote_average` | zakres 7.4–8.7 |
| Budżet | `budget` | liczba | `log(1 + budget)` | 6 braków → podobieństwo 0.5 |
| Język oryginału | `language` | kategoria | `original_language` | 177 z 200 to `en` |
| Seria / kolekcja | `collection` | kategoria | `belongs_to_collection.name` | 146 braków, tylko 3 wspólne pary |
| Gatunki | `genres` | zbiór | `genres[].name` | startowa |
| Słowa kluczowe | `keywords` | zbiór | `keywords.keywords[].name` | startowa |
| Reżyser | `directors` | zbiór osób | `crew`, `job == "Director"` | startowa |
| 5 głównych aktorów | `cast` | zbiór osób | `cast`, 5 najmniejszych `order` | startowa |
| Kraje produkcji | `countries` | zbiór | `production_countries[].iso_3166_1` | 167 filmów z US |
| Wytwórnie | `companies` | zbiór | `production_companies[].id` | dodatkowa |
| Scenarzyści | `writers` | zbiór osób | `crew`, `department == "Writing"` (bez rysowników) | |
| Kompozytor | `composers` | zbiór osób | `crew`, `job` = `Original Music Composer` lub `Music` | 13 braków |
| Operator kamery | `cinematographers` | zbiór osób | `crew`, `job == "Director of Photography"` | dodatkowa, 5 braków |
| Szersza obsada | `cast_wide` | zbiór osób | `cast`, 15 najmniejszych `order` | dodatkowa |
| Opis fabuły | `overview` | tekst | `overview` → TF-IDF (789 słów), cosinus | |

- [x] Nowe cechy w `features.py` (i ponowne `python src/features.py`)
- [x] Nowe wpisy w `preprocess.py`: `NUMERIC`, `SETS` oraz nowe słowniki `CATEGORIES` (typ `"category"`) i `TEXTS` (typ `"text"`)
- [x] Typ `"text"`: `sim_text` w `similarity.py` (cosinus wektorów TF-IDF)
- [x] Testy `python src/similarity.py`: brak danych nie jest traktowany jako „identyczne”
- [x] `python src/sim_matrix.py`: 18 macierzy 200×200 liczy się w **0.3 s**, więc żadna cecha nie jest kosztowna. GA ich nie przelicza, tylko łączy.

**Obserwacje** (wagi = 1, wariant `raw`): największy wpływ na łączną miarę mają `language` i `countries` (po 17%), bo prawie wszystkie filmy są z USA lub po angielsku. Wtedy liczy się tylko, czy film jest „nieamerykański”. `collection` i `cast_wide` prawie nic nie zmieniają. To argument za tym, żeby GA wybierał cechy, zamiast używać wszystkich.

### 9.2 Szybka ocena: funkcja fitness

GA wywoła ocenę setki tysięcy razy (358 osób × populacja × pokolenia), więc obecne `evaluate` (pętla w Pythonie) jest za wolne. Potrzebna jest wersja „macierzowa”.

**Pomysł:** dla jednej osoby wycinamy z `S` tylko jej filmy, czyli tablicę `(liczba_cech, 90, 90)`. Dla danego zestawu cech:
1. łączna macierz 90×90 = średnia wybranych cech,
2. na przekątną wpisujemy −∞ (film nie jest swoim sąsiadem, to jest leave-one-out),
3. **jedno sortowanie** każdego wiersza daje kolejność sąsiadów,
4. skumulowane głosy (`np.cumsum`) dają przewidywania **dla wszystkich k naraz**.

Jedno wywołanie trwa ułamek milisekundy, a prototyp potwierdził, że to działa.

- [x] `src/fast_eval.py`: `loo_accuracy_all_k(S_user, ratings, mask, ks) -> accuracy dla każdego k`
- [x] Test zgodności: dla kilku osób wynik równy `evaluate` z etapu 8 (280 porównań: 10 osób, losowe zestawy cech, 4 metody, 7 wartości k, identyczne co do bitu)
- [x] `holdout_predict(...)`: przewidywanie odłożonych filmów (do walidacji zagnieżdżonej i końcowej predykcji), zgodne z `knn.predict`

```python
from fast_eval import user_tensor, loo_accuracy_all_k, holdout_predict
S_user = user_tensor(S, rated_idx)                        # (18 cech, 90, 90), raz na osobę
loo_accuracy_all_k(S_user, ratings, mask, [30])           # fitness dla jednego k
loo_accuracy_all_k(S_user, ratings, mask, [5, 10, 30])    # accuracy dla kilku k naraz
holdout_predict(S_user, ratings, train_pos, test_pos, mask, k=30)
```

**Szybkość** (`python src/fast_eval.py`):
- jedno wywołanie fitness dla jednego k: **≈ 0.37 ms** (ok. 3× szybciej niż `evaluate.py`), dla 7 wartości k naraz ok. 10× szybciej,
- szacowany czas GA (populacja 30, 40 pokoleń, 358 osób): **≈ 2.6 min**; walidacja zagnieżdżona (GA 5×): **≈ 13 min**.
- Uwaga techniczna: tablica cech osoby musi być ciągła w pamięci (`np.ascontiguousarray`), inaczej każde wywołanie kopiuje 1 MB danych i trwa 3× dłużej.
- Ograniczenie: liczy wariant `raw`. Jeśli w kroku 9.3 wygra `std`, trzeba będzie dodać jego obsługę.

### 9.3 Ustawienia globalne (punkt odniesienia dla GA)

Jedna konfiguracja dla wszystkich osób. 358 osób × 90 ocen to dużo danych, więc ryzyko przeuczenia jest małe.

- [x] Sposób łączenia: `vote` / `wvote` / `mean` / `wmean`
- [x] Wariant skali: `raw` / `std`
- [x] Globalny zestaw cech: wybór zachłanny w przód (cecha dodawana, gdy poprawia średnią accuracy o ≥ 0.1 pp)
- [x] Wspólne `k ∈ {1, 3, 5, 7, 10, 15, 20, 30, 45, 60, 89}` (k = 89 oznacza wszystkie pozostałe filmy osoby)
- [x] Zapis: `params/global.json`, wyniki do sprawozdania: `results/global_methods.csv`, `results/global_selection.csv`

Kod: `python src/tune.py global` (ok. 20 s).

**1. Sposób łączenia i skala** (najlepsze k dla każdej kombinacji, LOO, 358 osób; baseline 41.15%):

| Cechy | Skala | `vote` | `wvote` | `mean` | `wmean` |
|---|---|---|---|---|---|
| startowe 7 | raw | 41.28% (k=60) | 41.47% (k=89) | 32.72% (k=1) | 32.72% (k=1) |
| startowe 7 | std | 41.24% (k=45) | 41.48% (k=89) | 32.99% (k=1) | 32.99% (k=1) |
| wszystkie 18 | raw | 41.34% (k=60) | **41.64% (k=89)** | 33.04% (k=1) | 33.04% (k=1) |
| wszystkie 18 | std | 41.42% (k=60) | 41.40% (k=89) | 33.02% (k=1) | 33.02% (k=1) |

→ **`wvote`, skala `raw`**. Średnia (`mean`, `wmean`) zdecydowanie odpada przy mierzeniu accuracy.

**2. Globalny zestaw cech** (wybór zachłanny): `budget` (41.56%), potem `language` (41.69%). Kolejna cecha (`vote_average`) dawała < 0.1 pp.

**3. Wspólne k** dla `budget + language`:

| k | 1 | 3 | 5 | 10 | 15 | 20 | 30 | 45 | 60 | 89 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 31.5% | 33.6% | 36.2% | 38.4% | 39.5% | 40.0% | 40.9% | 41.4% | 41.4% | **41.7%** |

**Wynik (`params/global.json`):** k = 89, `wvote`, `raw`, cechy `budget`, `language`, accuracy LOO **41.69%** wobec baseline 41.15%.

**Wnioski i konsekwencje dla 9.4:**
- **Globalnie nie ma sygnału w cechach filmów.** Najlepsze jest k = 89, czyli głosowanie wszystkich filmów osoby ważone podobieństwem. To prawie baseline, a poprawa o 0.5 pp mieści się w granicy szumu (wybrana na tych samych danych). Wybór `budget` i `language` też wygląda na przypadek, a nie na realny gust.
- Accuracy rośnie monotonicznie z k: im mniej kNN „wierzy” sąsiadom, tym lepiej. Jeden wspólny zestaw cech nie pasuje do wszystkich.
- **Ważne dla GA:** przy k = 89 wybór cech prawie nic nie zmienia, bo i tak głosują wszystkie filmy, a cechy zmieniają tylko wagi głosów. Żeby GA miał na co wpływać, **osoby z zestawem z GA muszą używać mniejszego k** (np. 20–45). Osoby, dla których GA nie znajdzie nic wyraźnie lepszego (reguła `margin`), dostają ustawienia globalne (k = 89). To k dla GA (`k_ga`) trzeba ustalić w 9.4 i sprawdzić w walidacji zagnieżdżonej (9.5).

### 9.4 Algorytm genetyczny: k i cechy per osoba

**Proces** (`python src/tune.py ga`):

```
Dla każdej osoby (pomiń, jeśli jest już w params/users.json):
    Dla k w [5, 10, 20, 30, 45, 60]:
        GA → najlepszy zestaw cech dla tego k
        fitness_k = accuracy LOO − 0.003 × liczba cech
    najlepsze (k, cechy) = max fitness_k   (remis → większe k)

    jeśli najlepszy fitness ≥ accuracy ustawień globalnych + 3/90:
        zapisz {k, cechy, source: "ga"}
    w przeciwnym razie:
        zapisz ustawienia globalne (k = 89, budget + language), source: "global"
    zapisz params/users.json od razu (wznawianie)
```

- Agregat: `wvote` (w kroku 9.3 wyraźnie lepszy od `wmean`: 41.6% wobec 33.0%).
- Accuracy LOO = to samo co `cross_val_predict` + `(preds == ratings).mean()`, liczone przez `fast_eval` (szybciej, identyczny wynik).
- k = 89 nie wchodzi do pętli z GA, bo przy nim cechy prawie nic nie zmieniają. Ta opcja jest pokryta przez ustawienia globalne.

| Element GA | Ustawienie |
|---|---|
| Chromosom | maska 0/1 długości 18 (1 = cecha używana) |
| Fitness | accuracy LOO − 0.003 × liczba cech |
| Populacja startowa | 30: zestaw globalny, wszystkie cechy, każda cecha osobno, reszta losowa |
| Selekcja | turniejowa (3) |
| Krzyżowanie | jednorodne, p = 0.9 |
| Mutacja | każdy gen z p = 1/18 |
| Elitaryzm | 2 najlepszych |
| Stop | 40 pokoleń lub 10 bez poprawy |
| Ziarno | `[0, user, k]`, więc wynik jest powtarzalny |
| Pamięć podręczna | każda maska oceniana raz (te same maski się powtarzają) |

- [x] `src/ga.py`: `run_ga(fitness, n_features, rng, initial) -> (maska, fitness, liczba ocenionych masek)`; test na problemie testowym (`python src/ga.py`): znajduje idealny wzorzec, oceniając 494 z 262 144 masek
- [x] `src/tune.py ga`: pętla po osobach i po k, zapis do `params/users.json` po każdej osobie
- [x] Wznawianie: osoby już policzone są pomijane; `--restart` liczy od nowa; zmiana ustawień GA wymusza liczenie od nowa (ustawienia zapisane w `config`)
- [x] Uruchomione dla wszystkich 358 osób: ok. 0.6–0.8 s na osobę, razem ok. 4 min (log: `results/ga_log.txt`)

Struktura `params/users.json`:

```json
{
  "config": {"method": "wvote", "ga_ks": [5, 10, 20, 30, 45, 60], "penalty": 0.003, "margin": 0.0333, "...": "..."},
  "users": {
    "19": {"k": 5, "features": ["budget", "language", "directors", "countries", "composers", "overview"],
           "source": "ga", "accuracy": 0.5889, "global_accuracy": 0.5111,
           "ga_best": {"...": "..."}, "by_k": {"5": 0.5709, "10": 0.5367, "...": "..."}},
    "5":  {"k": 89, "features": ["budget", "language"], "source": "global", "...": "..."}
  }
}
```

**Wyniki** (LOO na tych samych 90 ocenach, na których GA wybierał):

| | |
|---|---|
| Osoby z zestawem z GA / globalnym | **269** / 89 |
| Średnia accuracy wybranych ustawień | **51.4%** (globalne 41.7%) |
| k u osób z GA | 5: 96, 10: 95, 20: 48, 30: 16, 45: 6, 60: 8 |
| Liczba cech u osób z GA | najczęściej 3–5 |
| Przewaga GA nad globalnym | mediana 10 trafień na 90, maksymalnie 47 |
| Najczęstsze cechy | keywords 94, budget 81, runtime 80, vote_average 79, overview 77, …, genres 42, collection 7 |

**⚠️ Te liczby są zawyżone i nie wolno ich traktować jako wyniku.** Objawy przeuczenia:
- GA dla każdej osoby sprawdza setki zestawów × 6 wartości k na tych samych 90 ocenach. Najlepszy z tylu kandydatów zawsze wypadnie dobrze, nawet przypadkiem. Dlatego aż 75% osób przekroczyło `margin` = 3 trafienia.
- Wygrywają małe k (5 i 10), czyli najbardziej „nerwowe” ustawienia, które najłatwiej dopasować do przypadku.
- Wybierane cechy są rozrzucone prawie równomiernie. Nie widać, że jakaś cecha naprawdę opisuje gust; gatunki są nawet rzadziej niż kompozytor.
- Prototyp podobnej procedury dał w uczciwej walidacji **38%**, czyli mniej niż baseline.

**Rozstrzygnie to krok 9.5.** Walidacja zagnieżdżona pokaże prawdziwą skuteczność. Tam warto też sprawdzić surowsze ustawienia: większy `margin` (np. 6–10 trafień), większą karę, mniej wartości k (np. tylko 20, 30, 45).

### 9.5 Uczciwa ocena: walidacja zagnieżdżona

**Dlaczego to konieczne:** fitness najlepszego chromosomu jest zawyżony, bo GA wybrał go spośród tysięcy kandydatów na tych samych 90 ocenach. Żeby wiedzieć, ile GA naprawdę daje, trzeba go sprawdzić na ocenach, których **w ogóle nie widział**.

**Jak:** dla każdej osoby 5 razy:
1. odłóż 18 ocen (zbiór testowy),
2. na pozostałych 72 uruchom **całe strojenie** (GA + reguła `margin`),
3. przewiduj odłożone 18 z wybranym zestawem cech, sąsiedzi spośród 72.

Wynik porównaj w jednej tabeli (to też materiał do sprawozdania):

| Strategia | Accuracy (zagnieżdżona) |
|---|---|
| Baseline: najczęstsza ocena | |
| Globalne ustawienia (9.3) | |
| GA per osoba, bez `margin` | |
| GA per osoba, `margin` = 0.033 | |
| GA per osoba, inne λ / `margin` | |

- [ ] Walidacja zagnieżdżona w `src/tune.py`
- [ ] **Decyzja:** do `submission.csv` idzie strategia z najlepszym wynikiem w tej tabeli, nawet jeśli to nie GA

### 9.6 Strojenie końcowe i zapis parametrów

Wybraną strategię uruchom jeszcze raz, tym razem na **wszystkich 90** ocenach każdej osoby, i zapisz wynik.

**Gdzie zapisywać:** nowy folder `params/` w głównym katalogu projektu.

```
Zadanie1/
  params/
    global.json        ← ustawienia wspólne
    users.json         ← zestaw cech dla każdej osoby
  results/
    tuning.csv         ← wyniki walidacji zagnieżdżonej (do sprawozdania)
```

`params/global.json`:
```json
{
  "k": 30,
  "method": "wvote",
  "scale": "raw",
  "features": ["vote_count", "genres", "keywords", "directors"],
  "ga": {"population": 30, "generations": 40, "lambda": 0.003, "margin": 0.033, "seed": 0}
}
```

`params/users.json`:
```json
{
  "1642": {"features": ["genres", "directors", "vote_average"], "source": "ga", "fitness": 0.456},
  "1643": {"features": ["vote_count", "genres", "keywords", "directors"], "source": "global", "fitness": 0.411}
}
```

- `source` mówi, czy osoba dostała zestaw z GA, czy globalny (reguła `margin`).
- `predict.py` (etap 10) czyta oba pliki i dla każdej osoby używa jej zestawu cech oraz wspólnego k.
- Oba pliki trafiają do `code.zip`. Dzięki stałemu ziarnu można je też odtworzyć skryptem `python src/tune.py`.

- [ ] `src/tune.py` zapisuje `params/global.json`, `params/users.json`, `results/tuning.csv`
- [ ] Podsumowanie: ile osób dostało zestaw z GA, które cechy wybierane są najczęściej

---

## Etap 10. Generowanie `submission.csv`

- [ ] **10.1** Dla każdej osoby: najlepsze *k*, wagi i sposób łączenia; jako sąsiedzi **wszystkie 90** ocenionych filmów; predykcja 30 filmów z `task.csv`
- [ ] **10.2** Zapis w formacie `task.csv`:
  ```python
  task["rating"] = predictions            # int 0–5, ta sama kolejność wierszy
  task.to_csv(ROOT / "submission.csv", sep=";", header=False, index=False)
  ```
- [ ] **10.3** Kontrola przed wysłaniem:
  - [ ] 10 740 wierszy, ta sama kolejność i te same ID co w `task.csv`
  - [ ] żadnego `NULL`, tylko liczby całkowite 0–5 (nie `4.0`)
  - [ ] bez nagłówka, separator `;`
  - [ ] rozkład przewidywanych ocen ma sens (np. nie same trójki)

---

## Etap 11. Przygotowanie `code.zip`

- [ ] `src/`: cały kod
- [ ] `cache/tmdb/`: pobrane JSON-y
- [ ] `requirements.txt`
- [ ] `README.md`: jak uruchomić, wersja Pythona (3.13), klucz API potrzebny tylko do ponownego pobrania danych
- [ ] **Bez** klucza API, `.venv/`, `.idea/`, `__pycache__/`
- [ ] Stałe ziarno losowości (jeśli gdzieś jest losowość); ponowne uruchomienie daje ten sam `submission.csv`

---

## Lista kontrolna najczęstszych błędów

- [ ] `movie.csv` wczytany poprawnie (końce linii `\r`)
- [ ] Nie mylę `movie_id` (1–200) z `tmdb_id`
- [ ] Cechy liczbowe znormalizowane (żadna nie dominuje)
- [ ] Budżet lub czas trwania = 0 traktowany jako brak danych
- [ ] W walidacji ukryty film nie jest własnym sąsiadem
- [ ] Jest porównanie z baseline
- [ ] Przewidywania jako int, nie float
- [ ] Brak klucza API w `code.zip`

---

## Proponowana kolejność pracy

1. Etapy 0–2: dane i pobranie z TMDB (jednorazowo).
2. Etapy 4–7 z **3 cechami** (gatunki, rok, reżyser), wspólne *k*, sprawdzenie, że kNN bije baseline.
3. Dokładanie cech jedna po drugiej i sprawdzanie, czy pomagają (krok 9.2).
4. Dobór *k* i wag per osoba (kroki 9.3–9.5).
5. `submission.csv` i `code.zip` (etapy 10–11).
