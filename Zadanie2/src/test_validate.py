"""Testy walidacji z Etapu 4. Plik niezależny: można go po prostu usunąć.

Uruchomienie:
    uv run python src/test_validate.py
"""
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import validate as V
from tree import DecisionTree

DEPTHS = [1, 2, 3, 4, 5, 6, 8, None]


def some_users(k=12):
    users = V.get_users()
    return {u: users[u] for u in list(users)[:k]}


# ---------------------------------------------------------------- podział na części

def test_folds():
    a = V.make_folds(90, user=1642, seed=0)
    assert [len(f) for f in a] == [18] * 5
    assert sorted(np.concatenate(a).tolist()) == list(range(90)), "części nie pokrywają 90 filmów"
    b = V.make_folds(90, user=1642, seed=0)
    assert all((x == y).all() for x, y in zip(a, b)), "podział nie jest powtarzalny"
    c = V.make_folds(90, user=1642, seed=1)
    d = V.make_folds(90, user=1643, seed=0)
    assert not all((x == y).all() for x, y in zip(a, c)), "inny seed, a ten sam podział"
    assert not all((x == y).all() for x, y in zip(a, d)), "inna osoba, a ten sam podział"
    return "5 x 18 filmów, rozłączne, powtarzalne, inne dla innego seed i innej osoby"


# ---------------------------------------------------------------- obcinanie = max_depth

def test_truncation_equals_max_depth():
    checked = 0
    for user, (X, y) in some_users(10).items():
        for msl, gain in [(1, 0.0), (3, 0.0), (5, 0.01), (12, 0.03)]:
            params = {"min_samples_split": 2 * msl, "min_samples_leaf": msl, "min_gain": gain}
            deep = DecisionTree(max_depth=None, **params).fit(X[:72], y[:72])
            trunc = V.predict_truncated(deep, X, DEPTHS)
            for i, d in enumerate(DEPTHS):
                tree = DecisionTree(max_depth=d, **params).fit(X[:72], y[:72])
                assert (trunc[i] == tree.predict(X)).all(), (user, msl, gain, d)
                assert V.truncated_size(deep, d) == (tree.depth(), tree.n_leaves()), (user, d)
                checked += 1
    return f"{checked} drzew: przewidywania, głębokość i liście identyczne jak przy max_depth=d"


# ---------------------------------------------------------------- walidacja krzyżowa

def test_cross_val_no_leak():
    """Drzewo bez ograniczeń zapamiętuje filmy treningowe; walidacja musi to wykazać."""
    X, y = V.get_users()[1642]
    params = {"min_samples_split": 2, "min_samples_leaf": 1, "min_gain": 0.0}
    res = V.cross_val(X, y, 1642, params, [1, None])
    val_acc = (res["val_pred"] == y).mean(axis=1)
    assert res["train_acc"][1] > 0.95, res["train_acc"]
    assert val_acc[1] < res["train_acc"][1] - 0.3, "walidacja widzi filmy treningowe?"

    # ręcznie: film z części testowej przewidziany drzewem uczonym bez tej części
    test = V.make_folds(90, 1642)[0]
    mask = np.ones(90, dtype=bool)
    mask[test] = False
    manual = DecisionTree(max_depth=None, **params).fit(X[mask], y[mask]).predict(X[test])
    assert (res["val_pred"][1][test] == manual).all()
    return (f"bez limitu: trening {res['train_acc'][1]:.1%}, walidacja {val_acc[1]:.1%}; "
            f"przewidywania = ręcznie wyuczone drzewo")


# ---------------------------------------------------------------- baseline'y i miary

def test_baselines():
    assert V.most_frequent(np.array([3, 3, 5, 5, 1])) == 5     # remis -> wyższa
    assert V.most_frequent(np.array([2, 2, 2, 5])) == 2
    assert V.user_mean(np.array([4, 5])) == 5                   # 4.5 -> 5
    assert V.user_mean(np.array([3, 4, 4])) == 4                # 3.67 -> 4
    s = V.scores([np.array([1, 2, 3]), np.array([5, 5])], [np.array([1, 3, 5]), np.array([5, 0])])
    assert abs(s["accuracy"] - (1 / 3 + 1 / 2) / 2) < 1e-12
    assert abs(s["accuracy_1"] - (2 / 3 + 1 / 2) / 2) < 1e-12
    assert abs(s["mae"] - (1 + 5 / 2) / 2) < 1e-12
    return "najczęstsza (remis -> wyższa), średnia (0.5 w górę), accuracy/±1/MAE"


# ---------------------------------------------------------------- równoległość

def test_parallel_equals_serial():
    users = some_users(8)
    old = V._USERS
    V._USERS = users
    try:
        params = {"min_samples_split": 10, "min_samples_leaf": 5, "min_gain": 0.0}
        serial = V.evaluate_tree(params, DEPTHS, seed=0)
        with ProcessPoolExecutor(max_workers=2, initializer=V.init_worker) as ex:
            parallel = V.evaluate_tree(params, DEPTHS, seed=0, executor=ex)
    finally:
        V._USERS = old
    assert serial == parallel, "wynik zależy od liczby procesów"
    return "1 proces i 2 procesy dają identyczne wyniki"


TESTS = [test_folds, test_truncation_equals_max_depth, test_cross_val_no_leak,
         test_baselines, test_parallel_equals_serial]

if __name__ == "__main__":
    failed = 0
    for test in TESTS:
        try:
            print(f"[OK]   {test.__name__}: {test()}", flush=True)
        except Exception:
            failed += 1
            print(f"[BŁĄD] {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} testów przeszło")
    sys.exit(1 if failed else 0)
