"""Etap 3: własna implementacja lasu losowego (zbiór drzew z tree.py).

Różnorodność drzew (punkt 5 treści zadania) pochodzi z dwóch losowań:
  * bootstrap: każde drzewo uczy się na n filmach wylosowanych ze zwracaniem
    z n filmów osoby (ok. 63% różnych filmów, reszta to filmy out-of-bag);
  * max_features: w każdym węźle drzewo rozważa tylko losowy podzbiór kolumn.

Przewidywanie: głosowanie większościowe drzew. Remis głosów rozstrzyga suma
rozkładów ocen z liści (pewność drzew), a dalszy remis - bliskość do średniej
oceny z głosów (tak samo jak remis w liściu drzewa).

Drzewo nr t losuje wszystko z generatora default_rng([seed, t]), więc las ze 100
drzew to dokładnie pierwsze 100 drzew lasu z 200 drzew o tym samym seed.
Dzięki temu jednym uczeniem sprawdzamy wiele rozmiarów lasu (predict_sizes).

Użycie:
    uv run python src/forest.py --user 1642
    uv run python src/forest.py --user 1642 --n-trees 200 --max-features 30
"""
import argparse

import numpy as np

from tree import N_CLASSES, DecisionTree


def vote(votes, proba):
    """Ocena z głosów drzew.

    votes - (m, N_CLASSES): ile drzew zagłosowało na każdą ocenę
    proba - (m, N_CLASSES): suma rozkładów ocen z liści tych drzew (do remisów)
    """
    out = np.empty(len(votes), dtype=int)
    ratings = np.arange(N_CLASSES)
    for j, (v, p) in enumerate(zip(votes, proba)):
        best = np.flatnonzero(v == v.max())
        if len(best) > 1:
            best = best[np.isclose(p[best], p[best].max())]
        if len(best) > 1:
            mean = np.dot(ratings, v) / v.sum()
            best = best[[np.argmin(np.abs(best - mean))]]
        out[j] = best[0]
    return out


def oob_accuracy(oob_pred, y):
    """Trafność OOB: tylko filmy, które ma kto ocenić (oob_pred >= 0)."""
    has = oob_pred >= 0
    return float(np.mean(oob_pred[has] == y[has])) if has.any() else float("nan")


class RandomForest:
    """Las losowy dla ocen 0-5.

    n_trees       - liczba drzew
    max_features  - ile losowych kolumn rozważa każdy węzeł (None = wszystkie, int, "sqrt")
    bootstrap     - True: każde drzewo na próbce bootstrap; False: na wszystkich filmach
    max_depth, min_samples_split, min_samples_leaf, min_gain - jak w DecisionTree
    seed          - ziarno (powtarzalność)

    Po fit(): oob_pred_ (ocena każdego filmu treningowego z głosów drzew, które go
    nie widziały; -1 gdy takich drzew nie ma) i oob_score_ (trafność OOB).
    """

    def __init__(self, n_trees=100, max_features="sqrt", bootstrap=True, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_gain=0.0, seed=0):
        self.n_trees = n_trees
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.seed = seed
        self.trees = []

    # --- uczenie

    def tree_params(self):
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf, "min_gain": self.min_gain,
                "max_features": self.max_features}

    def fit(self, X, y, feature_names=None):
        if feature_names is None and hasattr(X, "columns"):
            feature_names = list(X.columns)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n = len(y)
        assert X.ndim == 2 and n == len(X) and n > 0 and self.n_trees >= 1

        self.n_features_ = X.shape[1]
        self.trees = []
        self.in_bag_ = np.zeros((self.n_trees, n), dtype=int)   # ile razy film trafił do próbki
        for t in range(self.n_trees):
            rng = np.random.default_rng([self.seed, t])
            idx = rng.integers(0, n, size=n) if self.bootstrap else np.arange(n)
            tree = DecisionTree(**self.tree_params(), seed=int(rng.integers(2 ** 32)))
            tree.fit(X[idx], y[idx], feature_names)
            self.trees.append(tree)
            self.in_bag_[t] = np.bincount(idx, minlength=n)

        self.oob_pred_ = self.oob_predict_sizes(X, [self.n_trees])[0]
        self.oob_score_ = oob_accuracy(self.oob_pred_, y)
        return self

    # --- głosy drzew

    def tree_outputs(self, X):
        """Przewidywania (n_trees, m) i rozkłady ocen z liści (n_trees, m, N_CLASSES)."""
        assert self.trees, "najpierw wywołaj fit()"
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2 and X.shape[1] == self.n_features_, "zła liczba kolumn"
        preds = np.empty((len(self.trees), len(X)), dtype=int)
        proba = np.empty((len(self.trees), len(X), N_CLASSES))
        for t, tree in enumerate(self.trees):
            for j, x in enumerate(X):
                leaf = tree._leaf(x)
                preds[t, j] = leaf.prediction
                proba[t, j] = leaf.counts / leaf.n
        return preds, proba

    @staticmethod
    def _onehot(preds):
        return np.eye(N_CLASSES, dtype=int)[preds]            # (n_trees, m, N_CLASSES)

    def oob_predict_sizes(self, X, sizes):
        """Przewidywania OOB lasów z pierwszych k drzew (X = te same filmy co w fit).

        Film i oceniają tylko drzewa, w których próbce bootstrap go nie było;
        -1 = żadne z k drzew go nie pominęło. Zwraca (len(sizes), n).
        """
        assert len(X) == self.in_bag_.shape[1], "OOB liczy się na filmach z fit()"
        preds, proba = self.tree_outputs(X)
        oob = (self.in_bag_ == 0)[:, :, None]                  # (n_trees, n, 1)
        votes = np.cumsum(self._onehot(preds) * oob, axis=0)
        proba = np.cumsum(proba * oob, axis=0)
        out = np.full((len(sizes), len(X)), -1, dtype=int)
        for i, k in enumerate(sizes):
            has = votes[k - 1].sum(axis=1) > 0
            if has.any():
                out[i, has] = vote(votes[k - 1][has], proba[k - 1][has])
        return out

    # --- przewidywanie

    def predict(self, X):
        """Ocena (0-5) dla każdego wiersza X: głosowanie wszystkich drzew."""
        return self.predict_sizes(X, [len(self.trees)])[0]

    def predict_proba(self, X):
        """Udział głosów drzew na każdą ocenę (wiersze sumują się do 1)."""
        preds, _ = self.tree_outputs(X)
        return self._onehot(preds).mean(axis=0)

    def predict_sizes(self, X, sizes):
        """Przewidywania lasów złożonych z pierwszych k drzew, dla każdego k z sizes.

        Zwraca (len(sizes), len(X)). Las z k pierwszych drzew jest identyczny z lasem
        uczonym od razu z n_trees=k (to samo seed), co sprawdza test_forest.py.
        """
        assert all(1 <= k <= len(self.trees) for k in sizes)
        preds, proba = self.tree_outputs(X)
        votes = np.cumsum(self._onehot(preds), axis=0)
        proba = np.cumsum(proba, axis=0)
        return np.array([vote(votes[k - 1], proba[k - 1]) for k in sizes])

    # --- opis

    def summary(self):
        depths = [t.depth() for t in self.trees]
        leaves = [t.n_leaves() for t in self.trees]
        unique = (self.in_bag_ > 0).sum(axis=1)
        return {"n_trees": len(self.trees), "mean_depth": float(np.mean(depths)),
                "mean_leaves": float(np.mean(leaves)),
                "mean_unique_in_bag": float(np.mean(unique)),
                "oob_score": self.oob_score_}


# ---------------------------------------------------------------- podgląd dla jednej osoby

def parse_max_features(text):
    if text in ("None", "none", "all"):
        return None
    return text if text == "sqrt" else int(text)


def main():
    from data import load_all
    from matrix import load_matrix, user_data

    parser = argparse.ArgumentParser(description="Las losowy dla jednej osoby (Etap 3).")
    parser.add_argument("--user", type=int, default=1642)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--max-features", type=parse_max_features, default="sqrt",
                        help='"sqrt", liczba kolumn albo "None" (wszystkie)')
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    movies, train, task = load_all()
    X = load_matrix()
    X_train, y_train, X_task, ids = user_data(X, train, task, args.user)
    forest = RandomForest(n_trees=args.n_trees, max_features=args.max_features,
                          max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
                          min_samples_split=2 * args.min_samples_leaf, seed=args.seed)
    forest.fit(X_train, y_train)
    s = forest.summary()
    y = y_train.to_numpy()

    print(f"Osoba {args.user}: {len(y)} filmów treningowych, {X_train.shape[1]} kolumn, "
          f"max_features={args.max_features} -> {forest.trees[0]._n_candidates()} kolumn na węzeł")
    print(f"Las: {s['n_trees']} drzew, średnio {s['mean_depth']:.1f} głębokości, "
          f"{s['mean_leaves']:.1f} liści, {s['mean_unique_in_bag']:.1f} różnych filmów w próbce")
    print(f"Trafność na danych treningowych: {(forest.predict(X_train) == y).mean():.1%} "
          f"(to nie jest ocena jakości)")
    print(f"Trafność OOB: {s['oob_score']:.1%} "
          f"(każdy film oceniają tylko drzewa, które go nie widziały)")

    titles = movies.set_index("movie_id").title
    share = forest.predict_proba(X_task)
    print("\nPrzewidywania dla filmów z task.csv (id | ocena | głosy na 0-5 | tytuł):")
    for (mid, i), p, sh in zip(ids.items(), forest.predict(X_task), share):
        votes = " ".join(f"{round(v * s['n_trees']):>3}" for v in sh)
        print(f"  {i} | {p} | {votes} | {titles[mid]}")


if __name__ == "__main__":
    main()
