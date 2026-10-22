"""Szybka ocena kNN dla jednej osoby: funkcja fitness dla algorytmu genetycznego.

Zamiast pętli po filmach (evaluate.py) wszystko liczone jest macierzowo:
1. łączna macierz podobieństw filmów osoby = średnia wybranych cech,
2. jedno sortowanie każdego wiersza daje kolejność sąsiadów,
3. skumulowane sumy (np.cumsum) dają przewidywania dla wszystkich k naraz.

Wyniki są identyczne z evaluate.py (wariant "raw"), tylko liczone szybciej:
ok. 3x dla jednego k (tak liczy algorytm genetyczny) i ok. 10x dla wielu k naraz.
Zestaw cech to maska 0/1 (lub wagi) w kolejności cech z similarity_matrices().
"""
import numpy as np


def user_tensor(S, rated_idx):
    """Wycina z S (cechy, 200, 200) tylko filmy ocenione przez osobę: (cechy, n, n)."""
    # ascontiguousarray: tablica ułożona w pamięci ciągiem, inaczej combine() kopiowałoby
    # ją przy każdym wywołaniu (ok. 20x wolniej)
    return np.ascontiguousarray(S[:, rated_idx][:, :, rated_idx])


def combine(S_user, mask):
    """Średnia ważona macierzy cech (jak combined() w sim_matrix.py, wariant "raw")."""
    mask = np.asarray(mask, dtype=float)
    F, n, _ = S_user.shape
    return (mask @ S_user.reshape(F, n * n)).reshape(n, n) / mask.sum()


def _round(x):
    """Zaokrąglenie „szkolne” i obcięcie do 0-5 (jak round_rating() w knn.py)."""
    return np.clip(np.floor(x + 0.5), 0, 5).astype(int)


def predict_all_k(sim, r_nb, ks, method="wvote"):
    """Przewidywania dla wielu filmów testowych i wielu k naraz.

    sim:  (T, N) podobieństwa filmów testowych do N filmów-kandydatów na sąsiadów
          (-inf = film nie może być sąsiadem)
    r_nb: (N,) oceny kandydatów
    ks:   lista k
    Zwraca tablicę (len(ks), T) z ocenami 0-5.
    """
    ks = np.asarray(ks)
    if len(ks) == 1:
        return _predict_one_k(sim, r_nb, int(ks[0]), method)[None, :]
    kmax = ks.max()
    order = np.argsort(-sim, axis=1, kind="stable")[:, :kmax]
    s = np.take_along_axis(sim, order, axis=1)              # (T, kmax), malejąco
    R = r_nb[order]                                          # (T, kmax)
    kk = ks - 1

    if method in ("vote", "wvote"):
        onehot = np.zeros(R.shape + (6,))
        np.put_along_axis(onehot, R[..., None], 1.0, axis=2)
        if method == "wvote":
            votes = np.cumsum(onehot * s[..., None], axis=1)
        else:   # zwykłe głosowanie, remis rozstrzyga suma podobieństw (jak w knn.py)
            votes = (np.cumsum(onehot, axis=1)
                     + 1e-6 * np.cumsum(onehot * s[..., None], axis=1))
        return votes[:, kk, :].argmax(axis=2).T

    counts = np.arange(1, kmax + 1)
    mean = np.cumsum(R, axis=1) / counts
    if method == "mean":
        return _round(mean[:, kk]).T
    if method == "wmean":
        ws, wr = np.cumsum(s, axis=1), np.cumsum(s * R, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            wmean = np.where(ws > 0, wr / ws, mean)
        return _round(wmean[:, kk]).T
    raise ValueError(f"Nieznana metoda: {method}")


def _predict_one_k(sim, r_nb, k, method):
    """Szybsza ścieżka dla jednego k: zwykłe sumy zamiast sum skumulowanych."""
    order = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    s = np.take_along_axis(sim, order, axis=1)
    R = r_nb[order]
    if method in ("vote", "wvote"):
        votes = np.zeros((len(R), 6))
        for c in range(6):
            hit = R == c
            if method == "wvote":
                votes[:, c] = np.cumsum(np.where(hit, s, 0.0), axis=1)[:, -1]
            else:
                votes[:, c] = (hit.sum(axis=1)
                               + 1e-6 * np.cumsum(np.where(hit, s, 0.0), axis=1)[:, -1])
        return votes.argmax(axis=1)
    mean = np.cumsum(R, axis=1)[:, -1] / k
    if method == "mean":
        return _round(mean)
    if method == "wmean":
        ws, wr = np.cumsum(s, axis=1)[:, -1], np.cumsum(s * R, axis=1)[:, -1]
        with np.errstate(invalid="ignore", divide="ignore"):
            return _round(np.where(ws > 0, wr / ws, mean))
    raise ValueError(f"Nieznana metoda: {method}")


def loo_accuracy_all_k(S_user, ratings, mask, ks, method="wvote"):
    """Fitness: accuracy leave-one-out osoby dla każdego k z ks.

    S_user: (cechy, n, n) z user_tensor(); ratings: (n,) oceny osoby.
    Każdy film jest przewidywany z pozostałych n - 1 (sam nie jest swoim sąsiadem).
    """
    sim = combine(S_user, mask)
    np.fill_diagonal(sim, -np.inf)
    ks = np.minimum(ks, len(ratings) - 1)
    preds = predict_all_k(sim, ratings, ks, method)
    return (preds == ratings[None, :]).mean(axis=1)


def holdout_predict(S_user, ratings, train_pos, test_pos, mask, k, method="wvote"):
    """Przewidywania dla filmów test_pos, gdy sąsiadami mogą być tylko train_pos.

    Pozycje to indeksy w obrębie filmów osoby (0..n-1). Potrzebne w walidacji
    zagnieżdżonej (krok 9.5) i przy końcowej predykcji.
    """
    sim = combine(S_user, mask)[np.ix_(test_pos, train_pos)]
    k = min(k, len(train_pos))
    return predict_all_k(sim, ratings[train_pos], [k], method)[0]


if __name__ == "__main__":
    import time
    from data import load_all
    from evaluate import cross_val_predict, knn_predictor
    from knn import METHODS, user_data
    from preprocess import prepare
    from sim_matrix import combined, similarity_matrices

    _, train, _ = load_all()
    users = user_data(train)
    _, prep = prepare()
    names, S = similarity_matrices(prep)
    F = len(names)
    KS = [1, 3, 5, 10, 20, 45, 89]
    rng = np.random.default_rng(0)

    # 1. Zgodność z evaluate.py: losowe osoby, losowe zestawy cech, wszystkie metody i k
    checked = 0
    for user in rng.choice(list(users), 10, replace=False):
        rated_idx, ratings = users[user]
        S_user = user_tensor(S, rated_idx)
        mask = rng.integers(0, 2, F)
        mask[rng.integers(F)] = 1
        M = combined(S, mask)
        for method in METHODS:
            fast = loo_accuracy_all_k(S_user, ratings, mask, KS, method)
            for k in KS:                       # ścieżka dla jednego k też musi być zgodna
                single = loo_accuracy_all_k(S_user, ratings, mask, [k], method)[0]
                assert single == fast[KS.index(k)], (user, method, k)
            for k, acc in zip(KS, fast):
                slow = (cross_val_predict(knn_predictor(M, k, method), rated_idx, ratings)
                        == ratings).mean()
                assert abs(acc - slow) < 1e-9, (user, method, k, acc, slow)
                checked += 1
    print(f"Zgodność z evaluate.py: OK ({checked} porównań: 10 osób, losowe cechy, "
          f"{len(METHODS)} metody, {len(KS)} wartości k)")

    # 2. holdout_predict: zgodność z knn.predict
    from knn import predict
    user = list(users)[0]
    rated_idx, ratings = users[user]
    S_user = user_tensor(S, rated_idx)
    mask = np.ones(F)
    M = combined(S, mask)
    perm = rng.permutation(len(ratings))
    test_pos, train_pos = perm[:18], perm[18:]
    fast = holdout_predict(S_user, ratings, train_pos, test_pos, mask, 15, "wvote")
    slow = [predict(M, rated_idx[t], rated_idx[train_pos], ratings[train_pos], 15, "wvote")
            for t in test_pos]
    assert list(fast) == slow
    print("holdout_predict zgodne z knn.predict: OK")

    # 3. Szybkość
    tensors = {u: (user_tensor(S, idx), r) for u, (idx, r) in users.items()}
    t = time.perf_counter()
    for S_user, r in tensors.values():
        loo_accuracy_all_k(S_user, r, np.ones(F), KS)
    dt = time.perf_counter() - t
    print(f"\nSzybkość: 358 osób w {dt:.2f} s (jedno wywołanie fitness ≈ {dt / 358 * 1000:.2f} ms, "
          f"wszystkie {len(KS)} wartości k naraz)")
    t = time.perf_counter()
    for S_user, r in tensors.values():
        loo_accuracy_all_k(S_user, r, np.ones(F), [30])
    dt = time.perf_counter() - t
    print(f"Jedno k (tak liczy GA): 358 osób w {dt:.2f} s (≈ {dt / 358 * 1000:.2f} ms na wywołanie)")
    t = time.perf_counter()
    M = combined(S, np.ones(F))
    for user in list(users)[:20]:
        rated_idx, ratings = users[user]
        cross_val_predict(knn_predictor(M, 30, "wvote"), rated_idx, ratings)
    slow = (time.perf_counter() - t) / 20
    print(f"Dla porównania evaluate.py, jedno k: ≈ {slow * 1000:.1f} ms na osobę "
          f"(≈ {slow / (dt / 358):.0f}x wolniej)")
    gens, pop = 40, 30
    ga = dt * gens * pop / 60
    print(f"Szacowany czas GA (populacja {pop}, {gens} pokoleń, 358 osób): ≈ {ga:.1f} min; "
          f"walidacja zagnieżdżona (GA 5x): ≈ {5 * ga:.0f} min")
