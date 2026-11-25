import numpy as np

MEASURES = ("pearson", "cosine", "msd")
EPS = 1e-9


def common_counts(R):
    M = (~np.isnan(R)).astype(float)
    return M @ M.T


def similarity(R, measure="pearson"):
    M = (~np.isnan(R)).astype(float)
    X = np.nan_to_num(R)
    N = M @ M.T

    if measure == "pearson":
        Sx = X @ M.T
        Sxx = (X * X) @ M.T
        Sxy = X @ X.T
        num = N * Sxy - Sx * Sx.T
        var_x = N * Sxx - Sx ** 2
        den = np.sqrt(np.clip(var_x * var_x.T, 0, None))

    elif measure == "cosine":
        mu = np.nanmean(R, axis=1, keepdims=True)
        Z = np.nan_to_num(R - mu)
        num = Z @ Z.T
        norm2 = (Z * Z) @ M.T
        den = np.sqrt(norm2 * norm2.T)

    elif measure == "msd":
        Sxx = (X * X) @ M.T
        sq_diff = Sxx + Sxx.T - 2 * (X @ X.T)
        msd = np.divide(sq_diff, N, out=np.full_like(N, np.inf), where=N > 0)
        S = 1.0 / (1.0 + msd)
        np.fill_diagonal(S, 0.0)
        return S

    else:
        raise ValueError(f"nieznana miara: {measure!r}, dostępne: {MEASURES}")

    S = np.divide(num, den, out=np.zeros_like(num), where=den > EPS)
    S = np.clip(S, -1.0, 1.0)
    np.fill_diagonal(S, 0.0)
    return S


def adjust(S, N, lam=0.0, min_common=1):
    S = S * (N / (N + lam)) if lam > 0 else S.copy()
    S[N < min_common] = 0.0
    return S


if __name__ == "__main__":
    import time

    from data import load_all
    from matrix import build_matrix, make_index

    train, task = load_all()
    users, movies = make_index(train, task)
    R = build_matrix(train, users, movies)
    N = common_counts(R)

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    S = similarity(R, "pearson")
    print(f"pearson: macierz {S.shape} w {time.perf_counter() - t0:.3f} s")

    pairs = rng.choice(len(users), size=(2000, 2))
    max_err = 0.0
    for u, v in pairs:
        if u == v:
            continue
        common = ~np.isnan(R[u]) & ~np.isnan(R[v])
        x, y = R[u, common], R[v, common]
        if x.std() == 0 or y.std() == 0:
            expected = 0.0
        else:
            expected = np.corrcoef(x, y)[0, 1]
        max_err = max(max_err, abs(S[u, v] - expected))
    print(f"kontrola z np.corrcoef na 2000 losowych parach: max błąd {max_err:.2e}")
    assert max_err < 1e-9
    assert np.allclose(S, S.T), "macierz powinna być symetryczna"

    off = S[~np.eye(len(users), dtype=bool)]
    print(f"\nrozkład podobieństw (Pearson): min {off.min():.2f}, "
          f"średnia {off.mean():.2f}, max {off.max():.2f}")
    print(f"dodatnie: {(off > 0).mean():.1%} par | ujemne: {(off < 0).mean():.1%} | "
          f"zero: {(off == 0).mean():.1%}")
    for name in ("cosine", "msd"):
        o = similarity(R, name)[~np.eye(len(users), dtype=bool)]
        print(f"{name:>7}: min {o.min():.2f}, średnia {o.mean():.2f}, max {o.max():.2f}")

    u = 0
    S_shr = adjust(S, N, lam=25)
    top = np.argsort(-S[u])[:5]
    print(f"\nnajbardziej podobni do osoby {users[u]} (λ = 25):")
    print(" osoba   wspólne   sim   sim po shrinkage")
    for v in top:
        print(f"{users[v]:>6} {N[u, v]:>9.0f} {S[u, v]:>6.2f} {S_shr[u, v]:>13.2f}")
