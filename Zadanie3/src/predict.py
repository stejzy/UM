import numpy as np

AGGREGATIONS = ("mean", "vote")
FALLBACKS = ("mode", "mean")
GRADES = np.arange(6)


def round_half_up(x):
    return np.floor(np.round(x, 9) + 0.5)


def to_rating(pred):
    return round_half_up(np.clip(pred, 0, 5)).astype(int)


def user_distribution(R):
    rated = ~np.isnan(R)
    counts = np.stack([(R == c).sum(axis=1) for c in GRADES], axis=1)
    return counts / rated.sum(axis=1, keepdims=True)


def user_modes(R):
    mu = np.nanmean(R, axis=1)
    D = user_distribution(R)
    modes = np.empty(len(R))
    for u in range(len(R)):
        best = GRADES[D[u] == D[u].max()]
        modes[u] = best[np.argmin(np.abs(best - mu[u]))]
    return modes


def neighbour_weights(S, targets, raters, k):
    W = S[np.ix_(targets, raters)]
    W = np.where(W > 0, W, 0.0)
    if k < len(raters):
        drop = np.argpartition(-W, k, axis=1)[:, k:]
        np.put_along_axis(W, drop, 0.0, axis=1)
    return W


def predict(R, S, rows, cols, k=20, aggregation="mean", centered=True, a=1.0,
            fallback="mode"):
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"nieznana agregacja: {aggregation!r}, dostępne: {AGGREGATIONS}")
    if fallback not in FALLBACKS:
        raise ValueError(f"nieznany fallback: {fallback!r}, dostępne: {FALLBACKS}")
    rows, cols = np.asarray(rows), np.asarray(cols)
    mu = np.nanmean(R, axis=1)
    default = user_modes(R) if fallback == "mode" else mu
    D = user_distribution(R) if aggregation == "vote" else None

    pred = np.empty(len(rows))
    used_fallback = np.zeros(len(rows), dtype=bool)

    for i in np.unique(cols):
        q = np.flatnonzero(cols == i)
        targets = rows[q]
        raters = np.flatnonzero(~np.isnan(R[:, i]))

        W = neighbour_weights(S, targets, raters, k)
        den = W.sum(axis=1)
        ok = den > 0
        safe_den = np.where(ok, den, 1.0)

        if aggregation == "mean":
            values = R[raters, i] - mu[raters] if centered else R[raters, i]
            base = mu[targets] if centered else 0.0
            p = base + (W @ values) / safe_den
        else:
            if centered:
                dev = R[raters, i] - mu[raters]
                ballots = to_rating(mu[targets, None] + dev[None, :])
            else:
                ballots = np.broadcast_to(R[raters, i].astype(int), W.shape)
            points = np.stack([(W * (ballots == c)).sum(axis=1) for c in GRADES], axis=1)
            points = points / safe_den[:, None] + a * D[targets]
            p = GRADES[points.argmax(axis=1)].astype(float)

        pred[q] = np.where(ok, p, default[targets])
        used_fallback[q] = ~ok

    return pred, used_fallback


def explain(R, S, u, i, k=20, a=1.0, users=None, movies=None):
    mu = np.nanmean(R, axis=1)
    D = user_distribution(R)
    raters = np.flatnonzero(~np.isnan(R[:, i]))
    sims = S[u, raters]
    chosen = [j for j in np.argsort(-sims)[:k] if sims[j] > 0]
    name = (lambda x: users[x]) if users is not None else (lambda x: x)

    print(f"osoba {name(u)}, film {movies[i] if movies is not None else i}: "
          f"μ_u = {mu[u]:.2f}, film oceniło {len(raters)} osób, sąsiedzi: {len(chosen)}")
    if not chosen:
        print("  brak sąsiadów -> fallback")
        return
    print("  sąsiad    sim   ocena   μ_v   odchylenie   głos na")
    v, w = raters[chosen], sims[chosen]
    dev = R[v, i] - mu[v]
    ballots = to_rating(mu[u] + dev)
    for vv, ww, dd, bb in zip(v, w, dev, ballots):
        print(f"  {name(vv):>6} {ww:>6.2f} {R[vv, i]:>6.0f} {mu[vv]:>6.2f} {dd:>+10.2f} {bb:>9}")

    avg = (w * dev).sum() / w.sum()
    print(f"\n  [mean] średnie ważone odchylenie {avg:+.2f} -> {mu[u]:.2f} {avg:+.2f} "
          f"= {mu[u] + avg:.2f} -> ocena {to_rating(np.array(mu[u] + avg))}")

    neigh = np.array([w[ballots == c].sum() for c in GRADES]) / w.sum()
    total = neigh + a * D[u]
    print(f"  [vote] a = {a}")
    print("   ocena   sąsiedzi   + a·przyzwyczajenia   = razem")
    for c in GRADES:
        mark = "  <- wygrywa" if c == total.argmax() else ""
        print(f"   {c:>5} {neigh[c]:>10.2f} {a * D[u, c]:>21.2f} {total[c]:>9.2f}{mark}")


if __name__ == "__main__":
    from data import load_all
    from matrix import build_matrix, make_index, positions
    from similarity import similarity

    train, task = load_all()
    users, movies = make_index(train, task)
    R = build_matrix(train, users, movies)
    S = similarity(R, "pearson")
    rows, cols = positions(task, users, movies)
    mu = np.nanmean(R, axis=1)
    D = user_distribution(R)

    def naive(u, i, k, aggregation, centered, a):
        raters = np.flatnonzero(~np.isnan(R[:, i]))
        sims = S[u, raters]
        top = [j for j in np.argsort(-sims)[:k] if sims[j] > 0]
        w, v = sims[top], raters[top]
        if aggregation == "mean":
            if centered:
                return mu[u] + (w * (R[v, i] - mu[v])).sum() / w.sum()
            return (w * R[v, i]).sum() / w.sum()
        points = a * D[u].copy()
        for ww, vv in zip(w, v):
            ballot = int(to_rating(np.array(mu[u] + R[vv, i] - mu[vv]))) \
                if centered else int(R[vv, i])
            points[ballot] += ww / w.sum()
        return float(points.argmax())

    sample = np.random.default_rng(0).choice(len(rows), 2000, replace=False)
    for aggregation, centered, a in [("mean", True, 0), ("mean", False, 0),
                                     ("vote", True, 1.0), ("vote", False, 0.5),
                                     ("vote", True, 0.0)]:
        pred, fb = predict(R, S, rows, cols, k=10, aggregation=aggregation,
                           centered=centered, a=a)
        err = max(abs(pred[j] - naive(rows[j], cols[j], 10, aggregation, centered, a))
                  for j in sample if not fb[j])
        print(f"kontrola z pętlą ({aggregation}, centered={centered}, a={a}): "
              f"max błąd {err:.2e}")
        assert err < 1e-9

    print()
    explain(R, S, rows[0], cols[0], k=5, users=users, movies=movies)

    hidden = train.groupby("user", group_keys=False).sample(frac=0.2, random_state=42)
    fit = train.drop(hidden.index)
    R_fit = build_matrix(fit, users, movies)
    S_fit = similarity(R_fit, "pearson")
    h_rows, h_cols = positions(hidden, users, movies)
    truth = hidden.rating.to_numpy()

    def report(label, r):
        print(f"  {label:<38} acc {np.mean(r == truth):.3f} | "
              f"±1 {np.mean(np.abs(r - truth) <= 1):.3f}")

    print(f"\nszybki test: ukryte {len(hidden)} ocen (20% każdej osoby)")
    report("baseline: średnia osoby", to_rating(np.nanmean(R_fit, axis=1)[h_rows]))
    report("baseline: moda osoby", user_modes(R_fit)[h_rows].astype(int))
    for k in (10, 20):
        p, _ = predict(R_fit, S_fit, h_rows, h_cols, k=k, aggregation="mean")
        report(f"kNN mean, k={k}", to_rating(p))
        for a in (0.0, 1.0):
            p, _ = predict(R_fit, S_fit, h_rows, h_cols, k=k, aggregation="vote", a=a)
            report(f"kNN vote, k={k}, a={a}", to_rating(p))

    print()
    for aggregation in AGGREGATIONS:
        pred, fb = predict(R, S, rows, cols, k=10, aggregation=aggregation)
        ratings = to_rating(pred)
        dist = {int(c): int(n) for c, n in zip(*np.unique(ratings, return_counts=True))}
        print(f"task.csv [{aggregation}]: fallback {fb.sum()}, rozkład {dist}")
    dist = {int(c): int(n) for c, n in zip(*np.unique(train.rating, return_counts=True))}
    print(f"train.csv (dla porównania):  rozkład {dist}")
