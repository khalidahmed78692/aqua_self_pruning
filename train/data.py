import numpy as np

def make_spirals(n_per_class=300, classes=3, noise=0.18, seed=123):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class*classes, 2), dtype=np.float64)
    y = np.zeros(n_per_class*classes, dtype=np.int64)
    for c in range(classes):
        ix = slice(c*n_per_class, (c+1)*n_per_class)
        r = np.linspace(0.05, 1.0, n_per_class)
        theta = np.linspace(c*4.0, (c+1)*4.0, n_per_class) + rng.normal(0, noise, n_per_class)
        X[ix] = np.c_[r*np.sin(theta), r*np.cos(theta)]
        y[ix] = c
    perm = rng.permutation(len(y))
    return X[perm], y[perm]

def split_data(X, y, seed=999, train_fraction=0.8):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(len(y)*train_fraction)
    return X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]
