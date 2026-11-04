"""Funkcje podobieństwa dla typów cech: 1 = identyczne, 0 = zupełnie różne."""
import math


def sim_numeric(a, b):
    """Liczby znormalizowane do [0, 1]: 1 - |a - b|. Brak danych -> 0.5 (neutralnie)."""
    if a is None or b is None or math.isnan(a) or math.isnan(b):
        return 0.5
    return 1.0 - abs(a - b)


def sim_category(a, b):
    """Pojedyncza wartość (np. język): 1 gdy równe, inaczej 0."""
    if a is None or b is None:
        return 0.0
    return 1.0 if a == b else 0.0


def sim_set(A, B):
    """Zbiory: współczynnik Jaccarda |A ∩ B| / |A ∪ B|. Pusty zbiór -> 0."""
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def sim_text(a, b):
    """Teksty jako wektory TF-IDF o długości 1: podobieństwo cosinusowe (iloczyn skalarny).

    Wartości TF-IDF są nieujemne, więc wynik jest w [0, 1].
    """
    return float(min(1.0, max(0.0, a @ b)))


SIM_FN = {"numeric": sim_numeric, "category": sim_category, "set": sim_set, "text": sim_text}


def feature_sims(prep, feature_types, i, j):
    """Podobieństwa filmów na pozycjach i, j dla każdej cechy: {nazwa: wartość}."""
    return {name: SIM_FN[typ](prep[name][i], prep[name][j])
            for name, typ in feature_types.items()}


if __name__ == "__main__":
    from features import load_features
    from preprocess import prepare, FEATURE_TYPES

    # Testy własności: zakres [0, 1], symetria, film podobny do siebie
    assert sim_numeric(0.2, 0.2) == 1.0 and sim_numeric(0.0, 1.0) == 0.0
    assert sim_numeric(float("nan"), 0.3) == 0.5
    assert sim_set({1, 2}, {2, 3}) == 1 / 3 and sim_set(set(), {1}) == 0.0
    assert sim_category("en", "en") == 1.0 and sim_category("en", "ja") == 0.0
    assert sim_category(None, None) == 0.0

    def missing(v):
        return (v is None or (isinstance(v, float) and math.isnan(v))
                or (isinstance(v, (set, list)) and not v))

    features = load_features()
    movie_ids, prep = prepare(features)
    n = len(movie_ids)
    for name, typ in FEATURE_TYPES.items():
        for i in range(n):
            if not missing(prep[name][i]):   # brak danych to nie „identyczne”
                assert math.isclose(SIM_FN[typ](prep[name][i], prep[name][i]), 1.0), (name, i)
            for j in range(i + 1, n, 7):
                s = SIM_FN[typ](prep[name][i], prep[name][j])
                assert 0.0 <= s <= 1.0 and s == SIM_FN[typ](prep[name][j], prep[name][i])
    print("Testy własności: OK\n")

    pairs = [(158, 111), (158, 175), (175, 25), (55, 154), (1, 175)]
    names = list(FEATURE_TYPES)
    for a, b in pairs:
        s = feature_sims(prep, FEATURE_TYPES, a - 1, b - 1)
        print(f"{features[a]['title']} / {features[b]['title']}:")
        print("   " + ", ".join(f"{nm} {v:.2f}" for nm, v in s.items()))
