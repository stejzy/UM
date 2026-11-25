import numpy as np

from data import load_all
from matrix import build_matrix, make_index, positions
from predict import predict, to_rating, user_distribution, user_modes
from similarity import adjust, common_counts, similarity
from tune import params_of, run_grid
from validate import make_cv, scores

TEST_PER_USER, SEED = 18, 123

INNER_GRID = {"measure": ["pearson", "cosine"], "lam": [0, 50], "min_common": [1],
              "k": [5, 10, 20]}
INNER_AGG = (
    [{"aggregation": "mean", "centered": c, "a": None} for c in (True, False)]
    + [{"aggregation": "vote", "centered": c, "a": a}
       for c in (True, False) for a in (0.0, 1.0, 2.0, 3.0)]
)


def split_test(train, seed):
    rng = np.random.default_rng(seed)
    is_test = np.zeros(len(train), dtype=bool)
    for idx in train.groupby("user").indices.values():
        is_test[rng.choice(idx, TEST_PER_USER, replace=False)] = True
    return train[~is_test], train[is_test]


def fit_and_test(dev, test, users, movies, p):
    R = build_matrix(dev, users, movies)
    S = adjust(similarity(R, p["measure"]), common_counts(R), p["lam"], p["min_common"])
    rows, cols = positions(test, users, movies)
    pred, _ = predict(R, S, rows, cols, k=p["k"], aggregation=p["aggregation"],
                      centered=p["centered"], a=p["a"] or 0.0, fallback=p["fallback"])
    return to_rating(pred), test.rating.to_numpy()


def baselines(dev, test, users, movies):
    R = build_matrix(dev, users, movies)
    rows, _ = positions(test, users, movies)
    truth = test.rating.to_numpy()
    return {"moda osoby": scores(user_modes(R)[rows].astype(int), truth)["accuracy"],
            "średnia osoby": scores(to_rating(np.nanmean(R, 1)[rows]), truth)["accuracy"]}


def naive_vote(dev, test, users, movies, p):
    assert p["measure"] == "pearson" and p["aggregation"] == "vote"
    R = build_matrix(dev, users, movies)
    n_users = len(users)
    seen = ~np.isnan(R)
    mu = np.array([R[u, seen[u]].mean() for u in range(n_users)])
    dist = np.array([[np.mean(R[u, seen[u]] == c) for c in range(6)] for u in range(n_users)])

    S = np.zeros((n_users, n_users))
    for u in range(n_users):
        for v in range(u + 1, n_users):
            common = seen[u] & seen[v]
            x, y = R[u, common], R[v, common]
            r = 0.0 if x.std() == 0 or y.std() == 0 else np.corrcoef(x, y)[0, 1]
            n = common.sum()
            if n < p["min_common"]:
                r = 0.0
            S[u, v] = S[v, u] = r * n / (n + p["lam"]) if p["lam"] > 0 else r

    out = []
    for u, i in zip(*positions(test, users, movies)):
        raters = [v for v in range(n_users) if seen[v, i] and v != u]
        raters = sorted(raters, key=lambda v: -S[u, v])[:p["k"]]
        raters = [v for v in raters if S[u, v] > 0]
        if not raters:
            values, counts = np.unique(R[u, seen[u]], return_counts=True)
            best = values[counts == counts.max()]
            out.append(int(best[np.argmin(np.abs(best - mu[u]))]))
            continue
        total = sum(S[u, v] for v in raters)
        points = p["a"] * dist[u].copy()
        for v in raters:
            ballot = to_rating(np.array(mu[u] + R[v, i] - mu[v])) if p["centered"] else int(R[v, i])
            points[int(ballot)] += S[u, v] / total
        out.append(int(np.argmax(points)))
    return np.array(out)


def shuffle_within_user(train, seed):
    rng = np.random.default_rng(seed)
    out = train.copy()
    for idx in train.groupby("user").indices.values():
        out.iloc[idx, out.columns.get_loc("rating")] = rng.permutation(train.rating.to_numpy()[idx])
    return out


if __name__ == "__main__":
    train, task = load_all()
    users, movies = make_index(train, task)
    dev, test = split_test(train, SEED)
    print(f"podział: {len(dev)} ocen do tuningu (72/osobę), {len(test)} testowych (18/osobę)")

    folds = make_cv(dev, task, seed=0)
    results = run_grid(folds, INNER_GRID, INNER_AGG)
    best = params_of(results.iloc[0])
    cv_acc = results.iloc[0]["accuracy"]
    pred, truth = fit_and_test(dev, test, users, movies, best)
    test_acc = scores(pred, truth)["accuracy"]
    print(f"\n[1] wybrane w CV (tylko dane tuningowe): {best}")
    print(f"    accuracy w CV:            {cv_acc:.3f}")
    print(f"    accuracy na TEST (raz):   {test_acc:.3f}   <- dane nieużyte przy tuningu")
    for name, acc in baselines(dev, test, users, movies).items():
        print(f"    baseline {name:<16} {acc:.3f}")

    p_naive = dict(best) if best["measure"] == "pearson" and best["aggregation"] == "vote" \
        else {**best, "measure": "pearson", "aggregation": "vote"}
    vec, _ = fit_and_test(dev, test, users, movies, p_naive)
    naive = naive_vote(dev, test, users, movies, p_naive)
    print(f"\n[2] niezależna implementacja pętlami ({p_naive['measure']}, k={p_naive['k']}, "
          f"a={p_naive['a']}, centered={p_naive['centered']}):")
    print(f"    zgodność z kodem wektorowym: {np.mean(vec == naive):.1%} predykcji, "
          f"accuracy pętli {np.mean(naive == truth):.3f} vs wektorowo {np.mean(vec == truth):.3f}")

    shuffled = shuffle_within_user(train, SEED)
    s_dev, s_test = shuffled.loc[dev.index], shuffled.loc[test.index]
    s_pred, s_truth = fit_and_test(s_dev, s_test, users, movies, best)
    s_base = baselines(s_dev, s_test, users, movies)
    print("\n[3] oceny przetasowane w obrębie każdej osoby (sąsiedzi bez informacji):")
    print(f"    metoda:          {np.mean(s_pred == s_truth):.3f}")
    print(f"    baseline moda:   {s_base['moda osoby']:.3f}")
