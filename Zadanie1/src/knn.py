"""Predykcja oceny metodą kNN, osobno dla każdej osoby."""
import numpy as np

METHODS = ("vote", "wvote", "mean", "wmean")


def user_data(train):
    """{user: (pozycje ocenionych filmów, oceny)} z tabeli train."""
    return {u: (g.movie_id.to_numpy() - 1, g.rating.to_numpy())
            for u, g in train.groupby("user")}


def neighbours(M, target, rated_idx, k):
    """k najbardziej podobnych filmów do target spośród rated_idx.

    Zwraca (indeksy w rated_idx, ich podobieństwa), od najbardziej podobnego.
    """
    sims = M[target, rated_idx]
    order = np.argsort(-sims, kind="stable")[:k]
    return order, sims[order]


def round_rating(x):
    """Zaokrąglenie „szkolne” (2.5 -> 3) i obcięcie do 0-5."""
    return int(min(5, max(0, np.floor(x + 0.5))))


def aggregate(r, s, method):
    """Łączy oceny sąsiadów r (z podobieństwami s) w jedną ocenę 0-5."""
    if method == "mean":
        return round_rating(r.mean())
    if method == "wmean":
        return round_rating((s * r).sum() / s.sum()) if s.sum() > 0 else round_rating(r.mean())
    if method in ("vote", "wvote"):
        votes = np.zeros(6)
        np.add.at(votes, r, s if method == "wvote" else 1.0)
        # remis w zwykłym głosowaniu: wygrywa ocena bliższych sąsiadów
        if method == "vote":
            tie = np.zeros(6)
            np.add.at(tie, r, s)
            votes = votes + 1e-6 * tie
        return int(votes.argmax())
    raise ValueError(f"Nieznana metoda: {method}")


def predict(M, target, rated_idx, ratings, k, method="wmean"):
    """Przewidywana ocena (int 0-5) filmu na pozycji target dla jednej osoby."""
    order, s = neighbours(M, target, rated_idx, k)
    return aggregate(ratings[order], s, method)


if __name__ == "__main__":
    from data import load_all
    from sim_matrix import build

    movies, train, task = load_all()
    titles = dict(zip(movies.movie_id - 1, movies.title))
    users = user_data(train)
    M = build("raw")
    k = 5

    user = task.user.iloc[0]
    rated_idx, ratings = users[user]
    print(f"Osoba {user}: {len(rated_idx)} ocenionych filmów, "
          f"średnia ocen {ratings.mean():.2f}\n")

    for movie_id in task[task.user == user].movie_id.head(3):
        target = movie_id - 1
        order, s = neighbours(M, target, rated_idx, k)
        print(f"{titles[target]}: {k} najbliższych ocenionych filmów")
        for o, sim in zip(order, s):
            print(f"   {titles[rated_idx[o]][:40]:<40} podob. {sim:.2f}  ocena {ratings[o]}")
        preds = {m: predict(M, target, rated_idx, ratings, k, m) for m in METHODS}
        print("   przewidywania: " + ", ".join(f"{m} = {p}" for m, p in preds.items()) + "\n")

    # Kontrola na wszystkich wierszach task.csv
    for method in METHODS:
        preds = np.array([predict(M, row.movie_id - 1, *users[row.user], k, method)
                          for row in task.itertuples()])
        assert preds.dtype.kind == "i" and preds.min() >= 0 and preds.max() <= 5
        dist = {r: int((preds == r).sum()) for r in range(6)}
        print(f"task.csv, k={k}, {method:<5}: rozkład przewidywań {dist}")
