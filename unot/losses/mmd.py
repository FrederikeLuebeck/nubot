import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()


def mmd_distance_weigthed(x, y, w_x, gamma, w_y=None):
    p_x = np.array(w_x).astype("float64")
    p_x /= p_x.sum()

    if w_y is None:
        p_y = np.ones((len(y), 1)) / len(y)
    else:
        p_y = np.array(w_y).astype("float64")
        p_y /= p_y.sum()

    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return (
        (p_x * xx * p_x.T).sum()
        + (p_y * yy * p_y.T).sum()
        - 2 * (p_x * xy * p_y.T).sum()
    )


def resampled_mmd(x, y, w_x, gamma, f=2):
    p = np.array(w_x).astype("float64")
    p /= p.sum()
    p = p.squeeze(-1)

    x_rs = x.iloc[np.random.choice(len(x), size=f * len(x), p=p), :]
    return mmd_distance(x_rs, y, gamma)


def compute_scalar_mmd(target, transport, gammas=None):
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas)))


def compute_scalar_mmd_weighted(
    target, transport, w_transport, w_target=None, gammas=None
):
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance_weigthed(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return np.mean(
        list(
            map(lambda x: safe_mmd(transport, target, w_transport, x, w_target), gammas)
        )
    )
