import numpy as np

from matrix import build_matrix, make_index, positions
from predict import predict, to_rating, user_modes
from similarity import adjust, common_counts, similarity

N_FOLDS = 5
METRICS = ("accuracy", "accuracy_1", "mae")


def make_folds(train, n_folds=N_FOLDS, seed=0):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(train), dtype=int)
    for idx in train.groupby("user").indices.values():
        idx = rng.permutation(idx)
        fold[idx] = np.arange(len(idx)) % n_folds
    return fold


def scores(pred, truth):
    err = np.abs(np.asarray(pred) - np.asarray(truth))
    return {"accuracy": float(np.mean(err == 0)),
            "accuracy_1": float(np.mean(err <= 1)),
            "mae": float(np.mean(err))}


class Fold:

    def __init__(self, train, fold_ids, f, users, movies):
        fit, hidden = train[fold_ids != f], train[fold_ids == f]
        self.R = build_matrix(fit, users, movies)
        self.N = common_counts(self.R)
        self.mu = np.nanmean(self.R, axis=1)
        self.rows, self.cols = positions(hidden, users, movies)
        self.truth = hidden.rating.to_numpy()
        self._raw = {}

    def similarity(self, measure="pearson", lam=0.0, min_common=1):
        if measure not in self._raw:
            self._raw[measure] = similarity(self.R, measure)
        return adjust(self._raw[measure], self.N, lam=lam, min_common=min_common)


def make_cv(train, task, n_folds=N_FOLDS, seed=0):
    users, movies = make_index(train, task)
    fold_ids = make_folds(train, n_folds, seed)
    return [Fold(train, fold_ids, f, users, movies) for f in range(n_folds)]


def evaluate(folds, measure="pearson", lam=0.0, min_common=1, k=20,
             aggregation="mean", centered=True, a=1.0, fallback="mode"):
    per_fold = []
    for fold in folds:
        S = fold.similarity(measure, lam, min_common)
        pred, fb = predict(fold.R, S, fold.rows, fold.cols, k=k, aggregation=aggregation,
                           centered=centered, a=a, fallback=fallback)
        s = scores(to_rating(pred), fold.truth)
        s["fallback_rate"] = float(fb.mean())
        per_fold.append(s)
    return summarize(per_fold)


def evaluate_baseline(folds, rule):
    per_fold = []
    for fold in folds:
        values = user_modes(fold.R) if rule == "mode" else fold.mu
        per_fold.append(scores(to_rating(values[fold.rows]), fold.truth))
    return summarize(per_fold)


def summarize(per_fold):
    out = {}
    for key in per_fold[0]:
        vals = np.array([s[key] for s in per_fold])
        out[key] = float(vals.mean())
        if key in METRICS:
            out[f"{key}_std"] = float(vals.std())
    return out


if __name__ == "__main__":
    from data import load_all

    train, task = load_all()

    fold_ids = make_folds(train, seed=0)
    per_user = train.assign(fold=fold_ids).groupby(["user", "fold"]).size()
    assert per_user.min() == per_user.max() == 18, "każda osoba: 18 ocen w każdym foldzie"
    assert (make_folds(train, seed=0) == fold_ids).all(), "podział powtarzalny dla seeda"
    print(f"foldy: {N_FOLDS}, w każdym {np.sum(fold_ids == 0)} zakrytych ocen, "
          f"każda osoba 18 zakrytych / 72 treningowe")

    folds = make_cv(train, task, seed=0)
    for fold in folds:
        assert np.isnan(fold.R[fold.rows, fold.cols]).all()
    print("kontrola: zakryte oceny nie trafiają do macierzy treningowej - OK\n")

    def show(label, res):
        print(f"{label:<42} acc {res['accuracy']:.3f} ± {res['accuracy_std']:.3f} | "
              f"±1 {res['accuracy_1']:.3f} | MAE {res['mae']:.3f}"
              + (f" | fallback {res['fallback_rate']:.1%}" if "fallback_rate" in res else ""))

    show("baseline: średnia osoby", evaluate_baseline(folds, "mean"))
    show("baseline: moda osoby", evaluate_baseline(folds, "mode"))
    show("kNN mean, pearson, k=20", evaluate(folds, k=20, aggregation="mean"))
    show("kNN vote, pearson, k=10, a=1", evaluate(folds, k=10, aggregation="vote", a=1.0))
