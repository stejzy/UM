"""Łączna miara podobieństwa filmów."""
import numpy as np

from preprocess import prepare, FEATURE_TYPES
from similarity import SIM_FN


def similarity_matrices(prep, feature_types=FEATURE_TYPES):
    """Zwraca (nazwy cech, S) gdzie S[f, i, j] = podobieństwo filmów i, j w cesze f."""
    names = list(feature_types)
    n = len(prep[names[0]])
    S = np.ones((len(names), n, n))
    for f, name in enumerate(names):
        sim, vals = SIM_FN[feature_types[name]], prep[name]
        for i in range(n):
            for j in range(i + 1, n):
                S[f, i, j] = S[f, j, i] = sim(vals[i], vals[j])
    return names, S


def off_diagonal(M):
    return M[~np.eye(M.shape[0], dtype=bool)]


def standardize(S):
    """Każda cecha: (S - średnia) / odchylenie, liczone poza przekątną."""
    Z = np.empty_like(S)
    for f in range(len(S)):
        vals = off_diagonal(S[f])
        Z[f] = (S[f] - vals.mean()) / vals.std()
    return Z


def combined(S, w, rescale=False):
    """Średnia ważona macierzy cech: sum(w_f * S_f) / sum(w_f).

    rescale=True przeskalowuje wynik do [0, 1] (potrzebne dla wariantu "std",
    bo po standaryzacji wartości mogą być ujemne). Przekątna = 1.
    """
    w = np.asarray(w, dtype=float)
    M = np.tensordot(w, S, axes=1) / w.sum()
    if rescale:
        vals = off_diagonal(M)
        M = (M - vals.min()) / (vals.max() - vals.min())
    np.fill_diagonal(M, 1.0)
    return M


def build(scale="raw", w=None):
    """Gotowa macierz podobieństw filmów 200x200 (pozycja i = movie_id i + 1)."""
    _, prep = prepare()
    names, S = similarity_matrices(prep)
    if scale == "std":
        S = standardize(S)
    w = np.ones(len(names)) if w is None else w
    return combined(S, w, rescale=(scale == "std"))


if __name__ == "__main__":
    import time
    from features import load_features

    features = load_features()
    movie_ids, prep = prepare(features)
    t = time.perf_counter()
    names, S = similarity_matrices(prep)
    print(f"Macierze podobieństw: {S.shape}, policzone w {time.perf_counter() - t:.1f} s\n")

    # Kontrola: symetria, zakres, przekątna
    assert np.allclose(S, S.transpose(0, 2, 1))
    assert S.min() >= 0 and S.max() <= 1
    assert all(np.allclose(np.diag(S[f]), 1) for f in range(len(names)))

    print("Typowe poziomy podobieństw cech (poza przekątną):")
    print(f"  {'cecha':<11}{'średnia':>9}{'odch.':>8}{'% zer':>8}")
    for f, name in enumerate(names):
        v = off_diagonal(S[f])
        print(f"  {name:<11}{v.mean():>9.3f}{v.std():>8.3f}{(v == 0).mean() * 100:>7.0f}%")

    Z = standardize(S)
    w = np.ones(len(names))
    variants = {"raw": combined(S, w), "std": combined(Z, w, rescale=True)}

    # Ile każda cecha "porusza" łączną miarą: odchylenie jej wkładu w sumę
    print("\nUdział cech w zróżnicowaniu łącznej miary (wagi = 1):")
    for label, M in [("raw", S), ("std", Z)]:
        spread = np.array([off_diagonal(M[f]).std() for f in range(len(names))])
        share = spread / spread.sum() * 100
        print(f"  {label}: " + ", ".join(f"{n} {s:.0f}%" for n, s in zip(names, share)))

    print("\nNajbardziej podobne filmy (wagi = 1):")
    for m in [158, 175, 55, 196]:
        i = movie_ids.index(m)
        print(f"  {features[m]['title']}")
        for label, M in variants.items():
            row = M[i].copy()
            row[i] = -np.inf
            top = np.argsort(-row)[:5]
            print(f"    {label}: " + "; ".join(
                f"{features[movie_ids[j]]['title'][:22]} ({row[j]:.2f})" for j in top))
