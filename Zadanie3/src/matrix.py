import numpy as np
import pandas as pd


def make_index(*frames):
    users = np.sort(pd.unique(pd.concat([f.user for f in frames])))
    movies = np.sort(pd.unique(pd.concat([f.movie_id for f in frames])))
    return users, movies


def positions(df, users, movies):
    u = df.user.to_numpy()
    m = df.movie_id.to_numpy()
    rows = np.searchsorted(users, u)
    cols = np.searchsorted(movies, m)
    assert (rows < len(users)).all() and (users[rows] == u).all(), "nieznana osoba"
    assert (cols < len(movies)).all() and (movies[cols] == m).all(), "nieznany film"
    return rows, cols


def build_matrix(df, users, movies):
    R = np.full((len(users), len(movies)), np.nan)
    rows, cols = positions(df, users, movies)
    R[rows, cols] = df.rating.to_numpy(dtype=float)
    return R


def rated_mask(R):
    return (~np.isnan(R)).astype(float)


def user_means(R):
    return np.nanmean(R, axis=1)


if __name__ == "__main__":
    from data import load_all

    train, task = load_all()
    users, movies = make_index(train, task)
    R = build_matrix(train, users, movies)
    M = rated_mask(R)
    mu = user_means(R)

    print(f"R: {R.shape[0]} osób × {R.shape[1]} filmów")
    print(f"wypełnienie: {M.mean():.1%} komórek ma ocenę, reszta to NaN")
    print(f"ocen na osobę (wiersz): {M.sum(1).min():.0f}-{M.sum(1).max():.0f}")
    print(f"ocen na film (kolumna): {M.sum(0).min():.0f}-{M.sum(0).max():.0f}")
    print(f"średnie osób mu_u: {mu.min():.2f}-{mu.max():.2f} (mediana {np.median(mu):.2f})")

    rows, cols = positions(train, users, movies)
    assert (R[rows, cols] == train.rating.to_numpy()).all()
    assert M.sum() == len(train)
    t_rows, t_cols = positions(task, users, movies)
    assert np.isnan(R[t_rows, t_cols]).all()
    print("kontrola pozycji: OK (train w komórkach, task w pustych komórkach)")

    N = M @ M.T
    off_diag = N[~np.eye(len(users), dtype=bool)]
    print(f"wspólne filmy dla par osób: min {off_diag.min():.0f}, "
          f"średnio {off_diag.mean():.1f}, max {off_diag.max():.0f}")

    print("\nFragment macierzy R (5 osób × 8 filmów):")
    print(pd.DataFrame(R[:5, :8], index=users[:5], columns=movies[:8]).to_string(na_rep="NaN"))
