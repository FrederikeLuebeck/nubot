from unot import losses
from unot.utils.wasserstein import compute_w2_monge, compute_w2_monge_scaled


def compute_metrics(x, y, T, eps=None):
    """Compute metrics

    Args:
        x (torch.Tensor): source sample
        y (torch.Tensor): target sample
        T (torch.Tensor): transported sample
        eps (torch.Tensor): scaling factor. None if problem is balanced.

    Returns:
        dict: dictionary of metrics, to log with wandb.
    """
    x = x.detach()
    y = y.detach()
    T = T.detach()
    if eps is not None:
        eps = eps.detach()

    log_dict = {}

    # balanced
    log_dict["w2_monge"] = compute_w2_monge(source=x, T=T)
    log_dict["mmd"] = losses.compute_scalar_mmd(y, T)

    # unbalanced
    if eps is not None:
        log_dict["w2_monge_scaled"] = compute_w2_monge_scaled(source=x, T=T, eps=eps)
        log_dict["mmd_weighted"] = losses.compute_scalar_mmd_weighted(y, T, eps)

        log_dict["eps_mean"] = eps.mean()
        log_dict["eps_std"] = eps.std()

    return log_dict
