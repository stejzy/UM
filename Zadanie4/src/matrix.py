import numpy as np
import pandas as pd


def make_index(*frames):
    users = np.sort(pd.unique(pd.concat([f.user for f in frames])))
    movies = np.sort(pd.unique(pd.concat([f.movie_id for f in frames])))
    return users, movies


def positions(df, users, movies):
    u = df.user.to_numpy()
    m = df.movie_id.to_numpy()
    r = np.searchsorted(users, u)
    c = np.searchsorted(movies, m)
    assert (r < len(users)).all() and (users[r] == u).all()
    assert (c < len(movies)).all() and (movies[c] == m).all()
    return c, r


def build_rating_matrix(df, users, movies):
    # Wiersze = filmy (M), kolumny = osoby (R), jak na wykładzie
    Y = np.zeros((len(movies), len(users)), dtype=float)
    E = np.zeros((len(movies), len(users)), dtype=float)
    m, r = positions(df, users, movies)
    Y[m, r] = df.rating.to_numpy(dtype=float)
    E[m, r] = 1.0
    return Y, E


def build_onehot_matrix(df, users, movies, n_classes=6):
    Y = np.zeros((len(movies), n_classes, len(users)), dtype=float)
    E = np.zeros((len(movies), len(users)), dtype=float)
    m, r = positions(df, users, movies)
    y = df.rating.to_numpy(dtype=int)
    Y[m, y, r] = 1.0
    E[m, r] = 1.0
    return Y, E
