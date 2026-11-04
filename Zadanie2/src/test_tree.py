"""Testy drzewa z Etapu 2 (krok 9). Plik niezależny: można go po prostu usunąć.

Uruchomienie:
    uv run python src/test_tree.py                          # bez porównania ze sklearn
    uv run --with scikit-learn python src/test_tree.py      # z porównaniem ze sklearn

`--with scikit-learn` instaluje sklearn tylko na czas tego uruchomienia,
nie zmienia pyproject.toml ani .venv projektu.
"""
import sys
import traceback

import numpy as np

from tree import N_CLASSES, DecisionTree, best_split, gini

try:
    from sklearn.tree import DecisionTreeClassifier
except ImportError:
    DecisionTreeClassifier = None


def split_gain(y, mask):
    """Spadek Giniego po podziale y na y[mask] (lewo) i y[~mask] (prawo)."""
    def g(part):
        return gini(np.bincount(part, minlength=N_CLASSES))
    n, nl = len(y), int(mask.sum())
    return g(y) - (nl * g(y[mask]) + (n - nl) * g(y[~mask])) / n


# ---------------------------------------------------------------- 1. Gini

def test_gini():
    assert gini([0, 0, 0, 0, 0, 7]) == 0.0
    assert abs(gini([0, 5, 5, 0, 0, 0]) - 0.5) < 1e-12
    # korzeń osoby 1642 (rozkład z Etapu 1): 0.73
    assert abs(gini([0, 11, 10, 18, 13, 38]) - 0.7336) < 1e-4
    return "gini([0,11,10,18,13,38]) = 0.7336"


# ---------------------------------------------------------------- 2. przykład z 10 filmami

def toy_data():
    """Przykład z wytłumaczenia Etapu 2: kolumny genre=Drama i year."""
    #           rating, drama, year
    films = [(1, 0, 1950), (5, 1, 1955), (3, 1, 1960), (4, 1, 1970), (1, 0, 1975),
             (5, 1, 1985), (1, 0, 1990), (4, 1, 1995), (2, 0, 2000), (5, 1, 2005)]
    y = np.array([f[0] for f in films])
    X = np.array([[f[1], f[2]] for f in films], dtype=float)
    return X, y


def test_toy_example():
    X, y = toy_data()
    assert abs(gini(np.bincount(y, minlength=N_CLASSES)) - 0.76) < 1e-12
    assert abs(split_gain(y, X[:, 0] <= 0.5) - 0.24333) < 1e-4      # genre=Drama
    assert abs(split_gain(y, X[:, 1] <= 1980) - 0.04) < 1e-12       # year <= 1980

    feature, threshold, gain = best_split(X, y, range(2), min_samples_leaf=1)
    assert feature == 0 and threshold == 0.5, (feature, threshold)
    assert abs(gain - 0.24333) < 1e-4

    tree = DecisionTree(max_depth=1, min_samples_split=2, min_samples_leaf=1)
    tree.fit(X, y, feature_names=["genre=Drama", "year"])
    assert tree.root.feature == 0
    assert tree.root.left.counts.tolist() == [0, 3, 1, 0, 0, 0]     # nie-dramaty: 1,1,1,2
    assert tree.root.right.counts.tolist() == [0, 0, 0, 1, 2, 3]    # dramaty: 3,4,4,5,5,5
    return f"korzeń: genre=Drama <= 0.5, zysk {gain:.4f} (year <= 1980: 0.04)"


# ---------------------------------------------------------------- 3. dane rozdzielne

def test_separable():
    rng = np.random.default_rng(0)
    X = rng.random((300, 4))
    y = (X[:, 0] > 0.5) * 3 + (X[:, 1] > 0.3) * 1 + (X[:, 2] > 0.8) * 1   # oceny 0-5
    tree = DecisionTree(max_depth=None, min_samples_split=2, min_samples_leaf=1).fit(X, y)
    acc = (tree.predict(X) == y).mean()
    assert acc == 1.0, acc
    return f"trafność treningowa 100%, głębokość {tree.depth()}, liście {tree.n_leaves()}"


# ---------------------------------------------------------------- 4. kryteria stopu

def test_max_depth_1():
    rng = np.random.default_rng(1)
    X, y = rng.random((100, 5)), rng.integers(0, 6, 100)
    tree = DecisionTree(max_depth=1, min_samples_split=2, min_samples_leaf=1).fit(X, y)
    assert tree.depth() == 1 and tree.n_leaves() == 2
    return "1 podział, 2 liście"


def test_stop_criteria():
    rng = np.random.default_rng(2)
    X, y = rng.random((400, 6)), rng.integers(0, 6, 400)
    checked = 0
    for max_depth, mss, msl, min_gain in [(4, 10, 5, 0.0), (None, 20, 7, 0.0),
                                          (6, 2, 1, 0.0), (None, 2, 1, 0.02)]:
        tree = DecisionTree(max_depth, mss, msl, min_gain).fit(X, y)
        for node in tree.nodes():
            assert node.n >= msl, "liść mniejszy niż min_samples_leaf"
            assert max_depth is None or node.depth <= max_depth
            if not node.is_leaf:
                assert node.n >= mss, "podzielony węzeł mniejszy niż min_samples_split"
                assert node.gain >= min_gain - 1e-12, "podział z zyskiem < min_gain"
                assert node.left.n + node.right.n == node.n
            checked += 1
    return f"sprawdzono {checked} węzłów w 4 konfiguracjach"


def test_pure_node_is_leaf():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    tree = DecisionTree(max_depth=None, min_samples_split=2, min_samples_leaf=1)
    tree.fit(X, np.full(10, 4))
    assert tree.root.is_leaf and tree.predict(X).tolist() == [4] * 10
    return "same oceny 4 -> jeden liść"


# ---------------------------------------------------------------- 5. best_split vs brute force

def test_best_split_bruteforce():
    rng = np.random.default_rng(3)
    for trial in range(30):
        n, m = rng.integers(10, 60), rng.integers(1, 6)
        X = rng.integers(0, 6, (n, m)).astype(float)       # dużo powtórzonych wartości
        y = rng.integers(0, 6, n)
        msl = int(rng.integers(1, 6))
        best = -np.inf
        for f in range(m):
            vals = np.unique(X[:, f])
            for t in (vals[:-1] + vals[1:]) / 2:
                mask = X[:, f] <= t
                if min(mask.sum(), (~mask).sum()) >= msl:
                    best = max(best, split_gain(y, mask))
        result = best_split(X, y, range(m), msl)
        if best == -np.inf:
            assert result is None
        else:
            f, t, gain = result
            assert abs(gain - best) < 1e-12, (trial, gain, best)
            assert abs(split_gain(y, X[:, f] <= t) - gain) < 1e-12
    return "30 losowych zbiorów: zysk = maksimum z pełnego przeszukania"


# ---------------------------------------------------------------- 6. prawdziwa osoba

def test_user_1642():
    from data import load_all
    from matrix import load_matrix, user_data

    movies, train, task = load_all()
    X_train, y_train, X_task, ids = user_data(load_matrix(), train, task, 1642)
    tree = DecisionTree().fit(X_train, y_train)
    pred = tree.predict(X_task)
    assert len(pred) == 30 and set(pred) <= set(range(6))
    proba = tree.predict_proba(X_task)
    assert np.allclose(proba.sum(axis=1), 1)
    acc = (tree.predict(X_train) == y_train.to_numpy()).mean()
    print("\n" + tree.to_text() + "\n")
    return (f"głębokość {tree.depth()}, liście {tree.n_leaves()}, "
            f"trafność treningowa {acc:.1%}, 30 przewidywań w zakresie 0-5")


# ---------------------------------------------------------------- 7. porównanie ze sklearn

def sk_counts(sk, s):
    """Rozkład ocen w węźle s drzewa sklearn (tree_.value to od 1.4 udziały, nie liczności)."""
    t = sk.tree_
    counts = np.zeros(N_CLASSES)
    counts[sk.classes_] = t.value[s][0] * t.weighted_n_node_samples[s]
    return np.rint(counts).astype(int)


def compare_with_sklearn(ours, sk, X, X32, y):
    """Porównuje oba drzewa węzeł po węźle na tych samych filmach.

    Zwraca (status, liczba porównanych węzłów); status:
      "ok"       - te same podziały filmów i rozkłady ocen w każdym węźle (kolumna może
                   być inna, jeśli dzieli filmy treningowe identycznie, np. year <= 1995
                   i budget <= 17.5 mln; na nowych filmach takie drzewa mogą się różnić)
      "remis"    - w którymś węźle wybrano inny warunek, ale o identycznym zysku
                   (np. dwie kolumny o tym samym zysku); dalej drzewa mogą się różnić
      "różnica"  - prawdziwa rozbieżność (inny zysk albo liść vs podział)
    """
    t = sk.tree_
    rank = {"ok": 0, "remis": 1, "różnica": 2}
    worst, compared = "ok", 0
    stack = [(ours.root, 0, np.arange(len(y)))]
    while stack:
        node, s, idx = stack.pop()
        compared += 1
        if node.counts.tolist() != sk_counts(sk, s).tolist():
            return "różnica", compared
        sk_leaf = t.children_left[s] == -1
        if node.is_leaf or sk_leaf:
            if node.is_leaf != sk_leaf:
                return "różnica", compared
            continue
        ours_mask = X[idx, node.feature] <= node.threshold
        sk_mask = X32[idx, t.feature[s]] <= t.threshold[s]
        if (ours_mask == sk_mask).all():
            stack += [(node.left, t.children_left[s], idx[ours_mask]),
                      (node.right, t.children_right[s], idx[~ours_mask])]
            continue
        g_ours, g_sk = split_gain(y[idx], ours_mask), split_gain(y[idx], sk_mask)
        status = "remis" if abs(g_ours - g_sk) < 1e-9 else "różnica"
        if rank[status] > rank[worst]:
            worst = status
    return worst, compared


def fit_both(X, y, max_depth, mss, msl):
    ours = DecisionTree(max_depth, mss, msl, min_gain=0.0).fit(X, y)
    sk = DecisionTreeClassifier(criterion="gini", max_depth=max_depth, min_samples_split=mss,
                                min_samples_leaf=msl, random_state=0).fit(X, y)
    return ours, sk


CONFIGS = [(None, 2, 1), (5, 10, 5), (3, 4, 2), (8, 6, 3)]   # max_depth, mss, msl


def random_data(seed, m):
    rng = np.random.default_rng(seed)
    # sklearn liczy na float32, więc dajemy obu drzewom te same wartości float32
    X = rng.random((200, m)).astype(np.float32).astype(np.float64)
    noise = rng.normal(0, 0.7, 200)
    y = np.clip(np.rint(5 * X[:, 0] + 2 * X[:, -1] - 1 + noise), 0, 5).astype(int)
    return X, y


def assert_identical(ours, sk):
    """Te same węzły w tej samej kolejności: kolumny, progi, rozkłady ocen, liście."""
    t = sk.tree_
    ours_nodes = ours.nodes()                       # ta sama kolejność co w sklearn
    assert len(ours_nodes) == t.node_count, (len(ours_nodes), t.node_count)
    for s, node in enumerate(ours_nodes):
        assert node.counts.tolist() == sk_counts(sk, s).tolist()
        if node.is_leaf:
            assert t.children_left[s] == -1
        else:
            assert node.feature == t.feature[s], (s, node.feature, t.feature[s])
            assert np.isclose(node.threshold, t.threshold[s], rtol=1e-6, atol=1e-9)
    return t.node_count


def test_sklearn_one_feature():
    """Jedna kolumna: sklearn nie ma czego losowo tasować, więc drzewa muszą być identyczne."""
    if DecisionTreeClassifier is None:
        return "POMINIĘTO (brak sklearn; uruchom z: uv run --with scikit-learn ...)"
    nodes = 0
    for seed in range(5):
        X, y = random_data(seed, m=1)
        for cfg in CONFIGS:
            nodes += assert_identical(*fit_both(X, y, *cfg))
    return f"20 drzew, {nodes} węzłów: identyczne kolumny, progi, rozkłady ocen i liście"


def test_sklearn_many_features():
    """8 kolumn ciągłych: drzewa identyczne albo różne tylko przez remis zysku."""
    if DecisionTreeClassifier is None:
        return "POMINIĘTO (brak sklearn; uruchom z: uv run --with scikit-learn ...)"
    stats = {"ok": 0, "remis": 0, "różnica": 0}
    for seed in range(5):
        X, y = random_data(seed, m=8)
        for cfg in CONFIGS:
            ours, sk = fit_both(X, y, *cfg)
            status, _ = compare_with_sklearn(ours, sk, X, X.astype(np.float32), y)
            stats[status] += 1
    assert stats["różnica"] == 0, stats
    return (f"20 drzew: te same podziały {stats['ok']}, różne tylko przez remis {stats['remis']}, "
            f"prawdziwe różnice {stats['różnica']}")


def test_sklearn_all_users():
    """Prawdziwe dane, wszystkie 358 osób, domyślne parametry drzewa."""
    if DecisionTreeClassifier is None:
        return "POMINIĘTO (brak sklearn; uruchom z: uv run --with scikit-learn ...)"
    from data import load_all
    from matrix import load_matrix, user_data

    movies, train, task = load_all()
    M = load_matrix()
    stats = {"ok": 0, "remis": 0, "różnica": 0}
    same = {"ok": 0, "remis": 0}       # zgodne przewidywania dla filmów z task.csv
    total = {"ok": 0, "remis": 0}
    for user in train.user.unique():
        X_train, y_train, X_task, ids = user_data(M, train, task, user)
        X, y = X_train.to_numpy(dtype=float), y_train.to_numpy()
        ours, sk = fit_both(X, y, max_depth=5, mss=10, msl=5)
        status, _ = compare_with_sklearn(ours, sk, X, X.astype(np.float32), y)
        stats[status] += 1
        if status in same:
            Xt = X_task.to_numpy(dtype=float)
            same[status] += int((ours.predict(Xt) == sk.predict(Xt.astype(np.float32))).sum())
            total[status] += len(Xt)
    assert stats["różnica"] == 0, stats

    def pct(s):
        return f"{same[s]}/{total[s]} ({same[s] / total[s]:.1%})" if total[s] else "-"
    return (f"358 osób: te same podziały filmów {stats['ok']}, różne tylko przez remis "
            f"{stats['remis']}, prawdziwe różnice {stats['różnica']}; zgodne przewidywania "
            f"dla task.csv: przy tych samych podziałach {pct('ok')}, przy remisach {pct('remis')}")


# ---------------------------------------------------------------- uruchamianie

TESTS = [test_gini, test_toy_example, test_separable, test_max_depth_1, test_stop_criteria,
         test_pure_node_is_leaf, test_best_split_bruteforce, test_user_1642,
         test_sklearn_one_feature, test_sklearn_many_features, test_sklearn_all_users]

if __name__ == "__main__":
    failed = 0
    for test in TESTS:
        try:
            info = test()
            print(f"[OK]   {test.__name__}: {info}")
        except Exception:
            failed += 1
            print(f"[BŁĄD] {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} testów przeszło")
    sys.exit(1 if failed else 0)
