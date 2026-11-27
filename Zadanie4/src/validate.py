import numpy as np

from cf import GRADES, predict_matrix, to_rating
from matrix import build_rating_matrix, make_index, positions

N_FOLDS = 5
METRICS = ("accuracy", "accuracy_1", "mae", "rmse")


def make_folds(train, n_folds=N_FOLDS, seed=0):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(train), dtype=int)
    for idx in train.groupby("user").indices.values():
        idx = rng.permutation(idx)
        fold[idx] = np.arange(len(idx)) % n_folds
    return fold


def scores(pred, truth, raw=None):
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    err = np.abs(pred - truth)
    raw_err = (np.clip(raw, 0, 5) - truth) if raw is not None else err
    return {
        "accuracy": float(np.mean(err == 0)),
        "accuracy_1": float(np.mean(err <= 1)),
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(raw_err ** 2))),
    }


def user_modes(Y, E):
    mu = Y.sum(axis=0) / np.maximum(E.sum(axis=0), 1.0)
    modes = np.empty(Y.shape[1], dtype=int)
    for r in range(Y.shape[1]):
        vals = Y[E[:, r] == 1, r]
        counts = np.array([(vals == c).sum() for c in GRADES])
        best = GRADES[counts == counts.max()]
        modes[r] = best[np.argmin(np.abs(best - mu[r]))]
    return modes


class Fold:
    def __init__(self, train, fold_ids, f, users, movies):
        self.fit = train[fold_ids != f]
        self.hidden = train[fold_ids == f]
        self.users = users
        self.movies = movies
        self.m_idx, self.r_idx = positions(self.hidden, users, movies)
        self.truth = self.hidden.rating.to_numpy()


def make_cv(train, task, n_folds=N_FOLDS, seed=0):
    users, movies = make_index(train, task)
    fold_ids = make_folds(train, n_folds, seed)
    return [Fold(train, fold_ids, f, users, movies) for f in range(n_folds)]


def summarize(per_fold):
    out = {}
    for key in per_fold[0]:
        vals = np.array([s[key] for s in per_fold])
        out[key] = float(vals.mean())
        if key in METRICS:
            out[f"{key}_std"] = float(vals.std())
    return out


def evaluate(folds, mode="onehot", n=150, lam=4.5, norm="none", update="joint",
             lr=None, n_iters=None, seed=0):
    per_fold = []
    for fold in folds:
        pred_mat, raw_mat = predict_matrix(
            fold.fit, fold.users, fold.movies,
            mode=mode, n=n, lam=lam, norm=norm, update=update,
            lr=lr, n_iters=n_iters, seed=seed,
        )
        per_fold.append(scores(
            pred_mat[fold.m_idx, fold.r_idx],
            fold.truth,
            raw_mat[fold.m_idx, fold.r_idx],
        ))
    return summarize(per_fold)


def evaluate_baseline(folds, rule):
    per_fold = []
    for fold in folds:
        Y, E = build_rating_matrix(fold.fit, fold.users, fold.movies)
        if rule == "mode":
            pred = user_modes(Y, E)[fold.r_idx]
            per_fold.append(scores(pred, fold.truth))
        elif rule == "user_mean":
            mu = Y.sum(axis=0) / np.maximum(E.sum(axis=0), 1.0)
            raw = mu[fold.r_idx]
            per_fold.append(scores(to_rating(raw), fold.truth, raw))
        elif rule == "movie_mean":
            mu = Y.sum(axis=1) / np.maximum(E.sum(axis=1), 1.0)
            raw = mu[fold.m_idx]
            per_fold.append(scores(to_rating(raw), fold.truth, raw))
        else:
            raise ValueError(f"nieznany baseline: {rule!r}")
    return summarize(per_fold)


if __name__ == "__main__":
    from data import load_all

    train, task = load_all()
    folds = make_cv(train, task, seed=0)

    def show(label, res):
        print(f"{label:<36} acc {res['accuracy']:.3f} ± {res['accuracy_std']:.3f} | "
              f"±1 {res['accuracy_1']:.3f} | MAE {res['mae']:.3f} | RMSE {res['rmse']:.3f}")

    show("baseline: średnia filmu", evaluate_baseline(folds, "movie_mean"))
    show("baseline: średnia osoby", evaluate_baseline(folds, "user_mean"))
    show("baseline: moda osoby", evaluate_baseline(folds, "mode"))
    show("CF scalar (n=4, lam=0)", evaluate(folds, mode="scalar", n=4, lam=0.0, norm="movie"))
    show("CF onehot (n=150, lam=4.5)", evaluate(folds, mode="onehot", n=150, lam=4.5, norm="none"))
