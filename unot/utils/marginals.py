import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt


def check_marginal_constraint(target, T, n_bins=20):
    hist_target, edges = np.histogramdd(target.numpy(), bins=n_bins, density=True)
    hist_T, _ = np.histogramdd(T.numpy(), bins=edges, density=True)
    hist_T = np.nan_to_num(
        hist_T
    )  # hist_T contains NaN if there is no mass at all in the range of "edges"

    # for numerical stability
    eps = 0.0001
    hist_target += eps
    hist_T += eps

    # compare only at support of target
    support = hist_target > 0.0
    # dist = np.mean((hist_target[support] - hist_T[support]) ** 2)
    close = np.isclose(hist_target[support], hist_T[support], atol=0.1).mean()
    kl = np.sum(np.multiply(hist_target, (np.log(hist_target) - np.log(hist_T))))

    return close, kl
