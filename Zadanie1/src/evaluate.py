"""Ocena jakości przewidywań: walidacja krzyżowa osobno dla każdej osoby."""
import numpy as np

from knn import predict, round_rating


def cross_val_predict(predictor, rated_idx, ratings, folds=None, seed=0):
    """Przewidywanie każdej oceny osoby z pozostałych (nie widząc jej samej)."""
    n = len(rated_idx)
    if folds is None:
        splits = [[i] for i in range(n)]
    else:
        perm = np.random.default_rng(seed).permutation(n)
        splits = np.array_split(perm, folds)
    preds = np.empty(n, dtype=int)
    for test in splits:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test] = False      # filmy walidacyjne nie mogą być sąsiadami
        for t in test:
            preds[t] = predictor(rated_idx[t], rated_idx[train_mask], ratings[train_mask])
    return preds


def evaluate(predictor, users, folds=None, seed=0):
    """Wyniki dla wszystkich osób.

    Zwraca {"accuracy": średnia, "accuracy_1": średnia, "mae": średnia,
            "per_user": {user: accuracy}}.
    accuracy_1: jak accuracy, ale pomyłka o ±1 też liczy się jako trafienie
    (tylko do raportu, nie jest używana przy strojeniu).
    """
    per_user, per_user_1, errors = {}, [], []
    for user, (rated_idx, ratings) in users.items():
        preds = cross_val_predict(predictor, rated_idx, ratings, folds, seed)
        per_user[user] = (preds == ratings).mean()
        per_user_1.append((np.abs(preds - ratings) <= 1).mean())
        errors.append(np.abs(preds - ratings))
    return {"accuracy": float(np.mean(list(per_user.values()))),
            "accuracy_1": float(np.mean(per_user_1)),
            "mae": float(np.concatenate(errors).mean()),
            "per_user": per_user}


# --- Predyktory ---

def most_frequent(target, train_idx, train_ratings):
    """Baseline: najczęstsza ocena osoby (przy remisie wyższa)."""
    counts = np.bincount(train_ratings, minlength=6)
    return int(5 - counts[::-1].argmax())


def user_mean(target, train_idx, train_ratings):
    """Baseline: zaokrąglona średnia ocen osoby."""
    return round_rating(train_ratings.mean())


def knn_predictor(M, k, method):
    """Predyktor kNN dla danej macierzy podobieństw, k i sposobu łączenia."""
    return lambda target, train_idx, train_ratings: predict(
        M, target, train_idx, train_ratings, k, method)


if __name__ == "__main__":
    import time
    from data import load_all
    from knn import METHODS, user_data
    from sim_matrix import build

    _, train, _ = load_all()
    users = user_data(train)

    def show(label, predictor, folds=None):
        t = time.perf_counter()
        r = evaluate(predictor, users, folds)
        print(f"  {label:<26} accuracy {r['accuracy'] * 100:5.1f}%   MAE {r['mae']:.3f}"
              f"   ({time.perf_counter() - t:.1f} s)")
        return r

    print("Leave-one-out, wszystkie 358 osób:\n")
    print("Baseline:")
    show("najczęstsza ocena osoby", most_frequent)
    show("średnia ocen osoby", user_mean)

    k = 5
    for scale in ("raw", "std"):
        M = build(scale)
        print(f"\nkNN, k = {k}, wariant {scale}:")
        for method in METHODS:
            show(method, knn_predictor(M, k, method))

    print("\nPorównanie LOO z 5-krotną walidacją (kNN raw, k = 5, wvote):")
    M = build("raw")
    show("leave-one-out (1 film)", knn_predictor(M, k, "wvote"))
    show("5-krotna (18 filmów)", knn_predictor(M, k, "wvote"), folds=5)
