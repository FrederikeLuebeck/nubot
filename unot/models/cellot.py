from pathlib import Path
import torch
import numpy as np
from collections import namedtuple
from absl import flags

from unot.networks.icnn import ICNN
from unot.networks.nicnn import NICNN

import wandb

FLAGS = flags.FLAGS

FGPair = namedtuple("FGPair", "f g")


def load_networks(config, **kwargs):
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

    # eg parameters specific to g are stored in config.model.g
    kwargs.pop("name")
    if "latent_dim" in kwargs:
        kwargs.pop("latent_dim")
    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})

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

    if config.model.init == "random":
        net = ICNN
    elif config.model.init == "identity":
        net = NICNN
    else:
        raise ValueError

    f = net(**fkwargs)
    g = net(**gkwargs)

    if "verbose" in FLAGS and FLAGS.verbose:
        print(g)
        print(kwargs)

    return f, g


def load_opts(config, f, g):
    kwargs = dict(config.get("optim", {}))
    assert kwargs.pop("optimizer", "Adam") == "Adam"

    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})

    fkwargs = kwargs.copy()
    fkwargs.update(fupd)
    fkwargs["betas"] = (fkwargs.pop("beta1", 0.9), fkwargs.pop("beta2", 0.999))

    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["betas"] = (gkwargs.pop("beta1", 0.9), gkwargs.pop("beta2", 0.999))

    opts = FGPair(
        f=torch.optim.Adam(f.parameters(), **fkwargs),
        g=torch.optim.Adam(g.parameters(), **gkwargs),
    )

    return opts


def load_cellot_model(config, restore=None, **kwargs):
    f, g = load_networks(config, **kwargs)
    opts = load_opts(config, f, g)

    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore)
        f.load_state_dict(ckpt["f_state"])
        opts.f.load_state_dict(ckpt["opt_f_state"])

        g.load_state_dict(ckpt["g_state"])
        opts.g.load_state_dict(ckpt["opt_g_state"])

    return (f, g), opts


def compute_loss_g(f, g, source, transport=None):
    if transport is None:
        transport = g.transport(source)

    ft = f(transport)
    dot = torch.multiply(source, transport).sum(-1, keepdim=True)

    wandb.log({"gl_ft": ft.mean(), "gl_dot": dot.mean()})

    # return f(transport) - torch.multiply(source, transport).sum(-1, keepdim=True)
    return torch.mean(ft) - torch.mean(dot)


def compute_loss_f(f, g, source, target, transport=None):
    if transport is None:
        transport = g.transport(source)

    ft = f(transport)
    fy = f(target)

    wandb.log({"fl_ft": ft.mean(), "fl_fy": fy.mean()})

    # return -f(transport) + f(target)
    return -torch.mean(ft) + torch.mean(fy)


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


def state_dict(f, g, opts, **kwargs):
    state = {
        "g_state": g.state_dict(),
        "f_state": f.state_dict(),
        "opt_g_state": opts.g.state_dict(),
        "opt_f_state": opts.f.state_dict(),
    }
    state.update(kwargs)

    return state


def loss_bw(f, g, x, y, w_x):
    """backward loss, weights on x"""
    x_hat = f.transport(y.requires_grad_(True))
    dot = torch.multiply(y, x_hat).sum(-1, keepdim=True)
    gt = g(x_hat)
    gx = g(x)

    wandb.log({"bw_dot": dot.mean(), "bw_gt": gt.mean(), "bw_gx": gx.mean()})

    return -torch.mean(dot) + torch.mean(gt) - torch.mean(w_x * gx)


def loss_fw(f, g, x, y, w_x):
    """forward loss, weights on x"""
    y_hat = g.transport(x.requires_grad_(True))
    dot = torch.multiply(x, y_hat).sum(-1, keepdim=True)
    ft = f(y_hat)
    fy = f(y)

    wandb.log({"fw_dot": dot.mean(), "fw_ft": ft.mean(), "fw_fy": fy.mean()})

    return -torch.mean(w_x * dot) + torch.mean(w_x * ft) - torch.mean(fy)


def get_w(x, config):
    if config.config_data.split("/")[-1] == "toy-unb.yaml":
        w = np.where(x[:, 0] <= 2.5, 0.5 / 0.6, 0.5 / 0.4)
    elif config.config_data.split("/")[-1] == "toy-three-two.yaml":
        c1 = np.where(x[:, 0] <= 3, 1, 0)
        c2 = np.where(x[:, 1] >= 5, 1, 0)
        mask = c1 & c2
        w = np.where(mask == 1, 0.01, 1.11)
    else:
        raise ValueError
    return torch.Tensor(w).unsqueeze(-1)


def analyze_loss_bw_weights(f, g, x, y, config):
    w_ones = torch.ones((x.size(0), 1))
    w_correct = get_w(x, config)

    loss_ones = loss_bw(f, g, x, y, w_ones)
    loss_w = loss_bw(f, g, x, y, w_correct)

    wandb.log({"loss_bw_ones": loss_ones})
    wandb.log({"loss_bw_w": loss_w})
    wandb.log({"loss_bw_ones_minus_w": loss_ones - loss_w})

    loss_ones = loss_fw(f, g, x, y, w_ones)
    loss_w = loss_fw(f, g, x, y, w_correct)

    wandb.log({"loss_fw_ones": loss_ones})
    wandb.log({"loss_fw_w": loss_w})
    wandb.log({"loss_fw_ones_minus_w": loss_ones - loss_w})
