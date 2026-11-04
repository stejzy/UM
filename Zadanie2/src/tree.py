"""Etap 2: własna implementacja binarnego drzewa decyzyjnego (klasyfikacja ocen 0-5).

Decyzje projektowe (punkt 4 treści zadania):
  * zbiór warunków: "kolumna <= próg", progi w połowie między kolejnymi różnymi
    wartościami kolumny w węźle (dla kolumn 0/1 jedyny próg to 0.5);
    warunek spełniony -> lewe dziecko, niespełniony -> prawe;
  * wybór warunku: największy spadek indeksu Giniego (ważonego liczbą filmów);
  * zatrzymanie: węzeł czysty, max_depth, min_samples_split, min_samples_leaf,
    zysk mniejszy niż min_gain albo brak dozwolonego podziału.

Użycie:
    uv run python src/tree.py --user 1642
    uv run python src/tree.py --user 1642 --max-depth 3 --min-samples-leaf 8
"""
import argparse
from dataclasses import dataclass

import numpy as np

N_CLASSES = 6     # oceny 0, 1, 2, 3, 4, 5
EPS = 1e-12       # tolerancja na błędy zaokrągleń przy porównywaniu Giniego


# ---------------------------------------------------------------- węzeł i Gini

def gini(counts):
    """Indeks Giniego dla rozkładu ocen: 1 - suma p_i^2 (0 = wszystkie oceny takie same)."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    return float(1.0 - np.sum(p * p))


def leaf_prediction(counts):
    """Najczęstsza ocena; przy remisie ta bliższa średniej ocenie w węźle."""
    counts = np.asarray(counts)
    best = np.flatnonzero(counts == counts.max())
    if len(best) == 1:
        return int(best[0])
    mean = np.dot(np.arange(len(counts)), counts) / counts.sum()
    return int(best[np.argmin(np.abs(best - mean))])


@dataclass(eq=False)
class Node:
    """Węzeł drzewa. Liść ma feature = None; węzeł wewnętrzny ma warunek i dwoje dzieci."""
    counts: np.ndarray            # rozkład ocen w węźle, długość N_CLASSES
    prediction: int               # ocena zwracana, gdy węzeł jest liściem
    depth: int
    feature: int | None = None    # numer kolumny w warunku "X[:, feature] <= threshold"
    threshold: float | None = None
    gain: float = 0.0             # spadek Giniego dzięki temu podziałowi
    left: "Node | None" = None    # warunek spełniony (tak)
    right: "Node | None" = None   # warunek niespełniony (nie)

    @property
    def is_leaf(self):
        return self.feature is None

    @property
    def n(self):
        return int(self.counts.sum())

    @property
    def gini(self):
        return gini(self.counts)


# ---------------------------------------------------------------- wybór podziału

def best_split(X, y, features, min_samples_leaf=1):
    """Najlepszy warunek "X[:, f] <= próg" dla filmów w węźle.

    Dla każdej kolumny sortujemy filmy po jej wartości i przesuwamy próg od
    lewej do prawej: lewa strona to pierwsze i filmów, a rozkłady ocen obu
    stron dostajemy z sum skumulowanych (bez liczenia każdego progu od zera).

    Zwraca (feature, threshold, gain) albo None, gdy żaden podział nie jest dozwolony.
    Remisy: wygrywa pierwsza kolumna (wg kolejności w `features`) i najniższy próg.
    """
    n = len(y)
    if n < 2 * min_samples_leaf or n < 2:
        return None

    onehot = np.zeros((n, N_CLASSES))
    onehot[np.arange(n), y] = 1
    total = onehot.sum(axis=0)
    n_left = np.arange(1, n)              # lewa strona po podziale za i-tym filmem
    n_right = n - n_left
    size_ok = (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)

    best = None
    best_impurity = np.inf
    for f in features:
        order = np.argsort(X[:, f], kind="stable")
        xs = X[order, f]
        left = np.cumsum(onehot[order], axis=0)[:-1]   # rozkład ocen po lewej
        right = total - left
        # próg da się postawić tylko między różnymi wartościami
        valid = size_ok & (xs[1:] > xs[:-1])
        if not valid.any():
            continue
        gini_left = 1.0 - (left ** 2).sum(axis=1) / n_left ** 2
        gini_right = 1.0 - (right ** 2).sum(axis=1) / n_right ** 2
        impurity = (n_left * gini_left + n_right * gini_right) / n   # Gini ważony
        impurity[~valid] = np.inf
        i = int(np.argmin(impurity))
        if impurity[i] < best_impurity - EPS:
            best_impurity = impurity[i]
            best = (int(f), float((xs[i] + xs[i + 1]) / 2))

    if best is None:
        return None
    return best[0], best[1], gini(total) - float(best_impurity)


# ---------------------------------------------------------------- drzewo

class DecisionTree:
    """Binarne drzewo decyzyjne dla ocen 0-5.

    max_depth          - maksymalna głębokość (None = bez limitu); korzeń ma głębokość 0
    min_samples_split  - węzeł z mniejszą liczbą filmów zostaje liściem
    min_samples_leaf   - każde dziecko musi mieć co najmniej tyle filmów
    min_gain           - podział musi obniżyć Gini co najmniej o tyle
    max_features       - ile losowych kolumn rozważać w każdym węźle
                         (None = wszystkie, int, albo "sqrt"); używane przez las losowy
    seed               - ziarno losowania kolumn (powtarzalność wyników)
    """

    def __init__(self, max_depth=5, min_samples_split=10, min_samples_leaf=5,
                 min_gain=0.0, max_features=None, seed=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.max_features = max_features
        self.seed = seed
        self.root = None

    # --- uczenie

    def fit(self, X, y, feature_names=None):
        if feature_names is None and hasattr(X, "columns"):
            feature_names = list(X.columns)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        assert X.ndim == 2 and len(X) == len(y) and len(y) > 0
        assert y.min() >= 0 and y.max() < N_CLASSES, "oceny muszą być w zakresie 0-5"

        self.n_features_ = X.shape[1]
        self.feature_names_ = feature_names or [f"x{j}" for j in range(self.n_features_)]
        self.rng_ = np.random.default_rng(self.seed)
        self.root = self._build(X, y, depth=0)
        return self

    def _n_candidates(self):
        m = self.n_features_
        if self.max_features is None:
            return m
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(m)))
        return max(1, min(m, int(self.max_features)))

    def _candidate_features(self):
        k = self._n_candidates()
        if k == self.n_features_:
            return range(self.n_features_)
        return np.sort(self.rng_.choice(self.n_features_, size=k, replace=False))

    def _build(self, X, y, depth):
        counts = np.bincount(y, minlength=N_CLASSES)
        node = Node(counts=counts, prediction=leaf_prediction(counts), depth=depth)

        if (gini(counts) <= EPS                                        # (a) węzeł czysty
                or (self.max_depth is not None and depth >= self.max_depth)  # (b)
                or len(y) < self.min_samples_split):                   # (c)
            return node

        split = best_split(X, y, self._candidate_features(), self.min_samples_leaf)
        if split is None:                                              # (d) brak podziału
            return node
        feature, threshold, gain = split
        if gain < self.min_gain - EPS:                                 # (e) za mały zysk
            return node

        mask = X[:, feature] <= threshold
        node.feature, node.threshold, node.gain = feature, threshold, gain
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    # --- przewidywanie

    def _check_fitted(self, X):
        assert self.root is not None, "najpierw wywołaj fit()"
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2 and X.shape[1] == self.n_features_, "zła liczba kolumn"
        return X

    def _leaf(self, x):
        node = self.root
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict(self, X):
        """Ocena (0-5) dla każdego wiersza X."""
        X = self._check_fitted(X)
        return np.array([self._leaf(x).prediction for x in X], dtype=int)

    def predict_proba(self, X):
        """Rozkład ocen z liścia dla każdego wiersza X (wiersze sumują się do 1)."""
        X = self._check_fitted(X)
        leaves = [self._leaf(x) for x in X]
        return np.array([leaf.counts / leaf.n for leaf in leaves])

    # --- opis drzewa

    def nodes(self):
        """Wszystkie węzły (w kolejności: korzeń, lewe poddrzewo, prawe poddrzewo)."""
        stack, out = [self.root], []
        while stack:
            node = stack.pop()
            out.append(node)
            if not node.is_leaf:
                stack += [node.right, node.left]
        return out

    def depth(self):
        return max(node.depth for node in self.nodes())

    def n_leaves(self):
        return sum(node.is_leaf for node in self.nodes())

    def _describe(self, node):
        stats = f"n={node.n}, gini={node.gini:.3f}, oceny 0-5: {node.counts.tolist()}"
        if node.is_leaf:
            return f"ocena {node.prediction}  [{stats}]"
        return f"{self.feature_names_[node.feature]} <= {node.threshold:.10g} ?  [{stats}]"

    def to_text(self):
        """Drzewo jako tekst; 'tak' = warunek spełniony (lewo), 'nie' = prawo."""
        lines = [self._describe(self.root)]

        def walk(node, prefix):
            for label, child, last in (("tak", node.left, False), ("nie", node.right, True)):
                lines.append(f"{prefix}{'└─' if last else '├─'} {label}: {self._describe(child)}")
                if not child.is_leaf:
                    walk(child, prefix + ("   " if last else "│  "))

        if not self.root.is_leaf:
            walk(self.root, "")
        return "\n".join(lines)


# ---------------------------------------------------------------- podgląd dla jednej osoby

def main():
    from data import load_all
    from matrix import load_matrix, user_data

    parser = argparse.ArgumentParser(description="Drzewo decyzyjne dla jednej osoby (Etap 2).")
    parser.add_argument("--user", type=int, default=1642)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-split", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--min-gain", type=float, default=0.0)
    args = parser.parse_args()

    movies, train, task = load_all()
    X = load_matrix()
    X_train, y_train, X_task, ids = user_data(X, train, task, args.user)
    tree = DecisionTree(max_depth=args.max_depth, min_samples_split=args.min_samples_split,
                        min_samples_leaf=args.min_samples_leaf, min_gain=args.min_gain)
    tree.fit(X_train, y_train)

    print(f"Osoba {args.user}: {len(y_train)} filmów treningowych, {X_train.shape[1]} kolumn")
    print(f"Drzewo: głębokość {tree.depth()}, liście {tree.n_leaves()}\n")
    print(tree.to_text())

    acc = (tree.predict(X_train) == y_train.to_numpy()).mean()
    print(f"\nTrafność na danych treningowych: {acc:.1%} "
          f"(to nie jest ocena jakości; tę da walidacja w Etapie 4)")

    titles = movies.set_index("movie_id").title
    print("\nPrzewidywania dla filmów z task.csv (id | ocena | tytuł):")
    for (mid, i), p in zip(ids.items(), tree.predict(X_task)):
        print(f"  {i} | {p} | {titles[mid]}")


if __name__ == "__main__":
    main()
