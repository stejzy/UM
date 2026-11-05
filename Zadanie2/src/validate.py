"""Etap 4: narzędzia do oceny klasyfikatorów walidacją krzyżową, osobno dla każdej osoby.

Każda osoba ma 90 ocen w train.csv. Tasujemy je i dzielimy na N_FOLDS części;
w każdej rundzie uczymy na pozostałych częściach i przewidujemy tę jedną.
Po wszystkich rundach każdy film ma przewidywanie od modelu, który go nie widział.

Sztuczka dla drzewa: ograniczenie max_depth tylko obcina drzewo (pozostałe kryteria
stopu nie zależą od głębokości), więc uczymy jedno drzewo bez limitu i odczytujemy
przewidywania dla wielu głębokości naraz (predict_truncated).

Ten plik nie ma pętli po parametrach; korzysta z niego tune_tree.py.
"""
from concurrent.futures import as_completed

import numpy as np

from tree import N_CLASSES, DecisionTree

N_FOLDS = 5

_USERS = None     # {user: (X, y)}; w procesach pomocniczych ładowane przez init_worker()


# ---------------------------------------------------------------- dane

def load_users():
    """{user: (X, y)}: macierz cech 90 ocenionych filmów i oceny z train.csv (bez task.csv)."""
    from data import load_all
    from matrix import load_matrix, user_data

    movies, train, task = load_all()
    M = load_matrix()
    users = {}
    for user in train.user.unique():
        X_train, y_train, _, _ = user_data(M, train, task, user)
        users[int(user)] = (X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=int))
    return users


def get_users():
    global _USERS
    if _USERS is None:
        _USERS = load_users()
    return _USERS


def init_worker():
    """Initializer dla ProcessPoolExecutor: każdy proces sam wczytuje dane z dysku."""
    get_users()


def make_folds(n, user, n_folds=N_FOLDS, seed=0):
    """Podział indeksów 0..n-1 na n_folds części; inny dla każdej osoby, stały dla (seed, user)."""
    perm = np.random.default_rng([seed, int(user)]).permutation(n)
    return np.array_split(perm, n_folds)


# ---------------------------------------------------------------- miary

def scores(preds, ys):
    """Miary dla wszystkich osób. preds, ys: listy tablic (po jednej na osobę).

    accuracy   - średnia z trafności osób (miara główna, jak w Zadaniu 1)
    accuracy_1 - pomyłka o ±1 też liczy się jako trafienie
    mae        - średni błąd bezwzględny
    """
    acc = [np.mean(p == y) for p, y in zip(preds, ys)]
    acc1 = [np.mean(np.abs(p - y) <= 1) for p, y in zip(preds, ys)]
    mae = [np.mean(np.abs(p - y)) for p, y in zip(preds, ys)]
    return {"accuracy": float(np.mean(acc)), "accuracy_1": float(np.mean(acc1)),
            "mae": float(np.mean(mae))}


# ---------------------------------------------------------------- baseline'y

def most_frequent(y_train):
    """Najczęstsza ocena osoby (przy remisie wyższa), jak w Zadaniu 1."""
    counts = np.bincount(y_train, minlength=N_CLASSES)
    return int(N_CLASSES - 1 - counts[::-1].argmax())


def user_mean(y_train):
    """Średnia ocen osoby zaokrąglona do najbliższej oceny (0.5 w górę)."""
    return int(np.clip(np.floor(y_train.mean() + 0.5), 0, N_CLASSES - 1))


BASELINES = {"najczęstsza ocena osoby": most_frequent, "średnia ocen osoby": user_mean}


def baseline_scores(rule, users=None, n_folds=N_FOLDS, seed=0):
    """Walidacja krzyżowa baseline'u na tych samych częściach co drzewo."""
    users = users or get_users()
    preds, ys = [], []
    for user, (X, y) in users.items():
        pred = np.empty(len(y), dtype=int)
        for test in make_folds(len(y), user, n_folds, seed):
            mask = np.ones(len(y), dtype=bool)
            mask[test] = False
            pred[test] = rule(y[mask])
        preds.append(pred)
        ys.append(y)
    return scores(preds, ys)


# ---------------------------------------------------------------- drzewo obcięte do głębokości

def predict_truncated(tree, X, depths):
    """Przewidywania drzewa obciętego do każdej z głębokości (None = bez obcinania).

    Zwraca tablicę (len(depths), len(X)). Węzeł na głębokości d drzewa bez limitu
    ma te same filmy co liść drzewa uczonego z max_depth=d, więc przewidywanie jest
    identyczne (sprawdza to test_validate.py).
    """
    limits = [None if d is None else int(d) for d in depths]
    out = np.empty((len(depths), len(X)), dtype=int)
    for j, x in enumerate(np.asarray(X, dtype=float)):
        path, node = [tree.root], tree.root        # path[k] ma głębokość k
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
            path.append(node)
        for i, d in enumerate(limits):
            out[i, j] = path[-1 if d is None else min(d, len(path) - 1)].prediction
    return out


def truncated_size(tree, depth):
    """(głębokość, liczba liści) drzewa po obcięciu do `depth` (None = bez obcinania)."""
    nodes = tree.nodes()
    if depth is None:
        return tree.depth(), tree.n_leaves()
    leaves = sum((n.is_leaf and n.depth <= depth) or (not n.is_leaf and n.depth == depth)
                 for n in nodes)
    return min(tree.depth(), depth), leaves


# ---------------------------------------------------------------- walidacja drzewa

def cross_val(X, y, user, params, depths, n_folds=N_FOLDS, seed=0):
    """Walidacja krzyżowa drzewa jednej osoby, od razu dla wielu głębokości.

    params: parametry DecisionTree bez max_depth (min_samples_split, min_samples_leaf,
    min_gain). Zwraca słownik z tablicami o długości len(depths) (poza val_pred):
        val_pred  - (len(depths), 90): przewidywanie każdego filmu przez drzewo, które go nie widziało
        train_acc - średnia po rundach trafność na filmach, na których drzewo się uczyło
        depth, leaves - średnia faktyczna głębokość i liczba liści
    """
    assert params.get("max_features") is None, "obcinanie działa tylko bez losowania kolumn"
    n, k = len(y), len(depths)
    val_pred = np.full((k, n), -1, dtype=int)
    train_acc, depth, leaves = np.zeros(k), np.zeros(k), np.zeros(k)

    for test in make_folds(n, user, n_folds, seed):
        mask = np.ones(n, dtype=bool)
        mask[test] = False
        tree = DecisionTree(max_depth=None, **params).fit(X[mask], y[mask])
        val_pred[:, test] = predict_truncated(tree, X[test], depths)
        train_acc += (predict_truncated(tree, X[mask], depths) == y[mask]).mean(axis=1)
        for i, d in enumerate(depths):
            dd, ll = truncated_size(tree, d)
            depth[i] += dd
            leaves[i] += ll

    assert (val_pred >= 0).all()
    return {"val_pred": val_pred, "train_acc": train_acc / n_folds,
            "depth": depth / n_folds, "leaves": leaves / n_folds}


def _run_user(job):
    user, params, depths, n_folds, seed = job
    X, y = get_users()[user]
    return user, cross_val(X, y, user, params, depths, n_folds, seed)


def evaluate_tree(params, depths, n_folds=N_FOLDS, seed=0, executor=None, on_progress=None):
    """Walidacja drzewa dla wszystkich osób; zwraca listę wyników, jeden na głębokość.

    executor    - ProcessPoolExecutor(initializer=init_worker) albo None (jeden proces)
    on_progress - funkcja (ile_osób_gotowych, ile_wszystkich), wołana po każdej osobie
    """
    users = get_users()
    jobs = [(user, dict(params), list(depths), n_folds, seed) for user in users]
    results = {}
    if executor is None:
        for done, job in enumerate(jobs, 1):
            user, res = _run_user(job)
            results[user] = res
            if on_progress:
                on_progress(done, len(jobs))
    else:
        futures = [executor.submit(_run_user, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            user, res = future.result()
            results[user] = res
            if on_progress:
                on_progress(done, len(jobs))

    order = list(users)                      # stała kolejność osób, niezależna od procesów
    ys = [users[u][1] for u in order]
    rows = []
    for i, d in enumerate(depths):
        s = scores([results[u]["val_pred"][i] for u in order], ys)
        rows.append({
            "max_depth": d,
            "train_accuracy": float(np.mean([results[u]["train_acc"][i] for u in order])),
            "val_accuracy": s["accuracy"],
            "val_accuracy_1": s["accuracy_1"],
            "val_mae": s["mae"],
            "mean_depth": float(np.mean([results[u]["depth"][i] for u in order])),
            "mean_leaves": float(np.mean([results[u]["leaves"][i] for u in order])),
        })
    return rows
