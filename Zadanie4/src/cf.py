import numpy as np

from matrix import build_onehot_matrix, build_rating_matrix

GRADES = np.arange(6)


def to_rating(pred):
    return np.floor(np.round(np.clip(pred, 0, 5), 9) + 0.5).astype(int)


def normalize(Y, E, norm):
    if norm == "none":
        return Y.copy(), 0.0
    if norm == "movie":
        counts = np.maximum(E.sum(axis=1, keepdims=True), 1.0)
        if Y.ndim == 2:
            mean = Y.sum(axis=1, keepdims=True) / counts
            return (Y - mean) * E, mean
        mean = Y.sum(axis=2, keepdims=True) / counts[:, None, :]
        return (Y - mean) * E[:, None, :], mean
    if norm == "user":
        counts = np.maximum(E.sum(axis=0, keepdims=True), 1.0)
        if Y.ndim == 2:
            mean = Y.sum(axis=0, keepdims=True) / counts
            return (Y - mean) * E, mean
        mean = Y.sum(axis=0, keepdims=True) / counts[:, None, :]
        return (Y - mean) * E[:, None, :], mean
    raise ValueError(f"nieznana normalizacja: {norm!r}")


def fit_scalar(df, users, movies, n=10, lam=0.0, norm="movie", update="joint",
               lr=0.002, n_iters=600, seed=0):
    Y, E = build_rating_matrix(df, users, movies)
    Y_norm, mean = normalize(Y, E, norm)
    M, R = Y.shape

    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 0.05, (M, n))
    P = rng.normal(0.0, 0.05, (R, n))
    p0 = np.zeros(R)

    for _ in range(n_iters):
        if update == "joint":
            err = (X @ P.T + p0[None, :] - Y_norm) * E
            gX = err @ P + lam * X
            gP = err.T @ X + lam * P
            gp0 = err.sum(axis=0)
            X -= lr * gX
            P -= lr * gP
            p0 -= lr * gp0
        else:
            err = (X @ P.T + p0[None, :] - Y_norm) * E
            P -= lr * (err.T @ X + lam * P)
            p0 -= lr * err.sum(axis=0)
            err = (X @ P.T + p0[None, :] - Y_norm) * E
            X -= lr * (err @ P + lam * X)

    return X @ P.T + p0[None, :] + mean


def fit_onehot(df, users, movies, n=150, lam=4.5, norm="none", update="joint",
               lr=0.004, n_iters=400, seed=0):
    Y, E = build_onehot_matrix(df, users, movies)
    Y_norm, mean = normalize(Y, E, norm)
    M, C, R = Y.shape

    Y_flat = Y_norm.reshape(M * C, R)
    E_flat = np.repeat(E, C, axis=0)

    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 0.05, (M * C, n))
    P = rng.normal(0.0, 0.05, (R, n))
    p0 = np.zeros((C, R))

    for _ in range(n_iters):
        if update == "joint":
            pred = (X @ P.T).reshape(M, C, R) + p0[None, :, :]
            err = (pred.reshape(M * C, R) - Y_flat) * E_flat
            gX = err @ P + lam * X
            gP = err.T @ X + lam * P
            gp0 = err.reshape(M, C, R).sum(axis=0)
            X -= lr * gX
            P -= lr * gP
            p0 -= lr * gp0
        else:
            pred = (X @ P.T).reshape(M, C, R) + p0[None, :, :]
            err = (pred.reshape(M * C, R) - Y_flat) * E_flat
            P -= lr * (err.T @ X + lam * P)
            p0 -= lr * err.reshape(M, C, R).sum(axis=0)
            pred = (X @ P.T).reshape(M, C, R) + p0[None, :, :]
            err = (pred.reshape(M * C, R) - Y_flat) * E_flat
            X -= lr * (err @ P + lam * X)

    scores = (X @ P.T).reshape(M, C, R) + p0[None, :, :] + mean
    return scores


def predict_matrix(df, users, movies, mode="onehot", n=150, lam=4.5, norm="none",
                   update="joint", lr=None, n_iters=None, seed=0):
    if mode == "scalar":
        lr = 0.002 if lr is None else lr
        n_iters = 600 if n_iters is None else n_iters
        raw = fit_scalar(df, users, movies, n=n, lam=lam, norm=norm,
                         update=update, lr=lr, n_iters=n_iters, seed=seed)
        return to_rating(raw), raw
    if mode == "onehot":
        lr = (0.003 if lam == 0 else 0.004) if lr is None else lr
        n_iters = (500 if lam == 0 else 400) if n_iters is None else n_iters
        scores = fit_onehot(df, users, movies, n=n, lam=lam, norm=norm,
                            update=update, lr=lr, n_iters=n_iters, seed=seed)
        pred = np.argmax(scores, axis=1)
        return pred, pred.astype(float)
    raise ValueError(f"nieznany tryb: {mode!r}")
