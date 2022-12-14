import torch


def compute_w2_monge(source, T):
    """Compute 2-Wasserstein distance squared in Monge formulation

    Args:
        source (torch.tensor): source sample
        T (torch.tensor): transport map

    Returns:
        _type_: wasserstein distance
    """

    cost = 0.5 * torch.sum((source - T) * (source - T), dim=1)
    cost = cost.mean()

    return cost


def compute_w2_monge_scaled(source, T, eps):
    cost = 0.5 * torch.sum(eps * (source - T) * (source - T), dim=1)
    cost = cost.mean()

    return cost
