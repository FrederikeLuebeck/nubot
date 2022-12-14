import torch
from collections import namedtuple
from absl import flags
import ot
import matplotlib.pyplot as plt
import wandb
from pathlib import Path

from unot.networks.icnn import ICNN
from unot.networks.nicnn import NICNN
from unot.networks.nonnegNN import nonnegNN


FLAGS = flags.FLAGS

FGHPair = namedtuple("FGHPair", "f g h_fw h_bw")


def load_networks(config, **kwargs):
    """Loads the networks for NubOT model, that is:
    f, g: convex potentials (ICNNs)
    h_fw, h_bw: re-scaling functions, i.e., eta and zeta
    """

    def unpack_kernel_init_fxn(name="uniform", **kwargs):
        if name == "normal":

            def init(*args):
                return torch.nn.init.normal_(*args, **kwargs)

        elif name == "uniform":

            def init(*args):
                return torch.nn.init.uniform_(*args, **kwargs)

        else:
            raise ValueError

        return init

    kwargs.setdefault("hidden_units", [64] * 4)
    kwargs.update(dict(config.get("model", {})))

    kwargs.pop("name")

    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})
    hupd = kwargs.pop("h", {})

    fkwargs = kwargs.copy()
    fkwargs.update(fupd)
    fkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
        **fkwargs.pop("kernel_init_fxn")
    )
    fkwargs.pop("init")

    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
        **gkwargs.pop("kernel_init_fxn")
    )
    gkwargs.pop("init")

    hkwargs = {"input_dim": kwargs.get("input_dim")}
    hkwargs.update(hupd)
    hkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(**kwargs.get("kernel_init_fxn"))

    if config.model.init == "random":
        net = ICNN
    elif config.model.init == "identity":
        net = NICNN

    f = net(**fkwargs)
    g = net(**gkwargs)
    h_fw = nonnegNN(**hkwargs)
    h_bw = nonnegNN(**hkwargs)  # h_bw is exactly the same as h_fw

    if "verbose" in FLAGS and FLAGS.verbose:
        print(g)
        print(kwargs)

    return f, g, h_bw, h_fw


def load_opts(config, f, g, h_fw, h_bw):
    kwargs = dict(config.get("optim", {}))
    assert kwargs.pop("optimizer", "Adam") == "Adam"

    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})
    hupd = kwargs.pop("h", {})

    fkwargs = kwargs.copy()
    fkwargs.update(fupd)
    fkwargs["betas"] = (fkwargs.pop("beta1", 0.9), fkwargs.pop("beta2", 0.999))

    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["betas"] = (gkwargs.pop("beta1", 0.9), gkwargs.pop("beta2", 0.999))

    kwargs = dict(config.get("optim_h", {}))
    assert kwargs.pop("optimizer", "Adam") == "Adam"
    hkwargs = kwargs.copy()
    hkwargs.update(hupd)
    hkwargs["betas"] = (hkwargs.pop("beta1", 0.9), hkwargs.pop("beta2", 0.999))

    opts = FGHPair(
        f=torch.optim.Adam(f.parameters(), **fkwargs),
        g=torch.optim.Adam(g.parameters(), **gkwargs),
        h_fw=torch.optim.Adam(h_fw.parameters(), **hkwargs),
        h_bw=torch.optim.Adam(h_bw.parameters(), **hkwargs),
    )

    return opts


def load_nubot_model(config, restore=None, **kwargs):
    f, g, h_fw, h_bw = load_networks(config, **kwargs)
    opts = load_opts(config, f, g, h_fw, h_bw)

    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore)
        f.load_state_dict(ckpt["f_state"])
        opts.f.load_state_dict(ckpt["opt_f_state"])

        g.load_state_dict(ckpt["g_state"])
        opts.g.load_state_dict(ckpt["opt_g_state"])

        h_fw.load_state_dict(ckpt["h_fw_state"])
        opts.h_fw.load_state_dict(ckpt["opt_h_fw_state"])

        h_bw.load_state_dict(ckpt["h_bw_state"])
        opts.h_bw.load_state_dict(ckpt["opt_h_bw_state"])

        print("Model was restored.")

    return (f, g, h_fw, h_bw), opts


def compute_loss_g(f, g, source, w_yhat, transport=None):
    if transport is None:
        transport = g.transport(source)

    # compute loss terms for network g
    ft = f(transport)
    dot = torch.multiply(source, transport).sum(-1, keepdim=True)

    # weight loss of networks by weight of target points (`w_yhat` i.e., weights on yhat (i.e. on x))
    return ((ft - dot) * w_yhat).mean()


def compute_loss_f(f, g, source, target, w_yhat, w_xhat, transport=None):
    if transport is None:
        transport = g.transport(source)

    # compute loss terms for network f
    ft = f(transport)
    fy = f(target)

    # weight loss of networks by weight of respective target points
    return -(ft * w_yhat).mean() + (fy * w_xhat).mean()


def compute_loss_h(wtilde, what):
    # MSE loss on re-scaling functions
    loss = torch.nn.MSELoss()
    return loss(wtilde, what)


def compute_w2_distance(f, g, source, target, transport=None):
    """Makkuva eq. (6)"""
    if transport is None:
        transport = g.transport(source).squeeze()

    with torch.no_grad():
        Cpq = (source * source).sum(1, keepdim=True) + (target * target).sum(
            1, keepdim=True
        )
        Cpq = 0.5 * Cpq

        cost = (
            f(transport)
            - torch.multiply(source, transport).sum(-1, keepdim=True)
            - f(target)
            + Cpq
        )
        cost = cost.mean()
    return cost


def compute_weights_unbs(source, target, reg, reg_m):
    """
    Unbalanced Sinkhorn, returns normalized weights
    """
    a = torch.ones(len(source)) / len(source)
    b = torch.ones(len(target)) / len(target)
    M = ot.dist(source.detach(), target.detach())
    M /= M.max()

    gamma = ot.unbalanced.sinkhorn_unbalanced(
        a,
        b,
        M,
        reg,
        reg_m,
        method="sinkhorn",
        numItermax=1000,
    )
    # normalize so that the sum of all weights is equal to the number of points
    # otherwise the exact weight values depend heavily on reg and reg_m
    wtilde_yhat = torch.Tensor(torch.sum(gamma, dim=1) / torch.sum(gamma)).unsqueeze(-1)
    return wtilde_yhat * len(source)


def state_dict(f, g, h_fw, h_bw, opts, **kwargs):
    state = {
        "g_state": g.state_dict(),
        "f_state": f.state_dict(),
        "h_fw_state": h_fw.state_dict(),
        "h_bw_state": h_bw.state_dict(),
        "opt_g_state": opts.g.state_dict(),
        "opt_f_state": opts.f.state_dict(),
        "opt_h_fw_state": opts.h_fw.state_dict(),
        "opt_h_bw_state": opts.h_bw.state_dict(),
    }
    state.update(kwargs)

    return state
