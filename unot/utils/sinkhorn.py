import ot
import numpy as np


def sinkhorn_map(x, y, reg=0.1):
    if x.requires_grad:
        x = x.detach()
    if y.requires_grad:
        y = y.detach()
    gamma = ot.bregman.empirical_sinkhorn(x, y, reg=reg)
    v = gamma / np.sum(gamma, axis=1)
    T = np.matmul(v, y)
    return T
