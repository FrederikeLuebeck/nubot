import torch
from torch import nn
from torch import autograd
from pathlib import Path
from collections import namedtuple
import numpy as np

from unot.networks.nn import NN
from unot.networks.nonnegNN import nonnegNN


DGPair = namedtuple("DGPair", "D G")


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

    kwargs.update(dict(config.get("model", {})))
    kwargs.pop("name")

    dupd = kwargs.pop("D", {})
    gupd = kwargs.pop("G", {})

    dkwargs = kwargs.copy()
    dkwargs.update(dupd)
    if kwargs.get("kernel_init_fxn", None) is not None:
        dkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
            **dkwargs.pop("kernel_init_fxn")
        )

    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    if kwargs.get("kernel_init_fxn", None) is not None:
        gkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
            **gkwargs.pop("kernel_init_fxn")
        )

    D = Discriminator(**dkwargs)
    G = Generator(**gkwargs)
    return D, G


def load_opts(config, D, G):
    kwargs = dict(config.get("optim", {}))
    assert kwargs.pop("optimizer", "Adam") == "Adam"

    dupd = kwargs.pop("D", {})
    gupd = kwargs.pop("G", {})

    dkwargs = kwargs.copy()
    dkwargs.update(dupd)
    dkwargs["betas"] = (dkwargs.pop("beta1", 0.9), dkwargs.pop("beta2", 0.999))

    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["betas"] = (gkwargs.pop("beta1", 0.9), gkwargs.pop("beta2", 0.999))

    opts = DGPair(
        D=torch.optim.Adam(D.parameters(), **dkwargs),
        G=torch.optim.Adam(G.parameters(), **gkwargs),
    )

    return opts


def load_gan_model(config, restore=None, **kwargs):
    D, G = load_networks(config, **kwargs)
    opts = load_opts(config, D, G)

    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore)
        D.load_state_dict(ckpt["D_state"])
        opts.D.load_state_dict(ckpt["opt_D_state"])

        G.load_state_dict(ckpt["G_state"])
        opts.G.load_state_dict(ckpt["opt_G_state"])
        print("Model was restored from checkpoint.")

    return (D, G), opts


class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim=10,
        hidden_units=[64, 64],
        activation="relu",
        kernel_init_fxn=None,
    ):
        super(Discriminator, self).__init__()

        # standard feed-forward NN with output dimension 1
        self.net = NN(
            input_dim=input_dim,
            hidden_units=hidden_units,
            out_dim=1,
            activation=activation,
            kernel_init_fxn=kernel_init_fxn,
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(
        self,
        input_dim=10,
        hidden_units=[64, 64],
        activation="relu",
        sigmoid=False,
        sigmoid_max=1,
        hidden_units_scaling=[32, 32],
        kernel_init_fxn=None,
    ):
        super(Generator, self).__init__()

        # standard feed-forward NN
        self.net = NN(
            input_dim=input_dim,
            hidden_units=hidden_units,
            out_dim=input_dim,
            activation=activation,
            kernel_init_fxn=kernel_init_fxn,
        )

        self.sigmoid = sigmoid
        self.sigmoid_max = sigmoid_max

        self.scaling = nonnegNN(
            input_dim,
            hidden_units_scaling,
            activation,
            sigmoid,
            sigmoid_max,
            kernel_init_fxn,
        )

    def forward(self, x):
        return self.net(x), self.scaling(x)


LOG2 = torch.from_numpy(np.ones(1) * np.log(2)).float()
logsigmoid = nn.LogSigmoid()


def compute_loss_D(D, source, target, transported, eps, psi, gradient_penalty, lambG):
    """
    Discriminator Loss
    eps * D(T) - psi*(D(y))
    where psi is the convex conjugate of the psi-divergence
    """
    psi_star = PSI_FUNS.get(psi)
    act = js_act if psi == "js" or psi == "paper" else identity_act

    if gradient_penalty:
        gp = gradient_penalty_D(D, source, target, lambG)
    else:
        gp = 0

    if psi == "paper":
        # not sure about the minus
        # they seem to have -eps*(ln2+ln(.)-D(T)) -(ln2-ln(.))
        # but I would say - (eps*(ln2+ln(.)-D(T)) -(ln2-ln(.))) (?)
        Dt = D(transported)
        Dy = D(target)

        # exactly as in their code
        their_code = -torch.mean(
            eps * (LOG2.expand_as(Dt) + logsigmoid(Dt) - Dt)
        ) - torch.mean(LOG2.expand_as(Dy) + logsigmoid(Dy))

        # should be equal
        # after a few iterations, not equal anymore (but first 4 digits still equal)
        # v2 = -torch.mean(eps * (act(Dt) - Dt)) - torch.mean(psi_star(act(Dy)))
        # if v2.isnan():
        #     print("v2 is nan.")

        # how it would make more sense
        # v3 = -torch.mean(eps * (act(Dt) - Dt)) + torch.mean(psi_star(act(Dy)))

        return their_code + gp

    return (
        -(torch.mean(eps * act(D(transported))) - torch.mean(psi_star(act(D(target)))))
        + gp
    )


def compute_loss_G(
    D,
    G,
    source,
    transported,
    eps,
    c1,
    c2,
    psi,
    lamb0,
    lamb1,
    lamb2,
    gradient_penalty,
    lambG2,
):
    """
    Generator Loss
    c1(source, transported) * eps + c2(eps) + eps * D(transported)
    """
    if gradient_penalty:
        gp = gradient_penalty_G_rho(
            G, source, source[torch.randperm(source.size(0))], lambG2
        )
    else:
        gp = 0

    c1 = C1_FUNS.get(c1)
    c2 = C2_FUNS.get(c2)
    act = js_act if psi == "js" or psi == "paper" else identity_act

    if psi == "paper":
        Dt = D(transported)
        mse = nn.MSELoss(reduction="none")
        their_code = (
            lamb0 * torch.mean(eps * torch.sum(mse(transported, source), dim=1))
            + lamb1 * torch.mean((eps - 1) * torch.log(eps))
            + lamb2 * torch.mean(eps * (LOG2.expand_as(Dt) + logsigmoid(Dt) - Dt))
        )

        return their_code + gp

    return (
        lamb0 * torch.mean(eps * c1(source, transported))
        + lamb1 * torch.mean(c2(eps))
        + lamb2 * torch.mean(eps * act(D(transported)))
    ) + gp


def state_dict(D, G, opts, **kwargs):
    state = {
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "opt_G_state": opts.G.state_dict(),
        "opt_D_state": opts.D.state_dict(),
    }
    state.update(kwargs)

    return state


def mse(x, y):
    return torch.sum((x - y) * (x - y), dim=1, keepdim=True)  # or mean?


def kl(s):
    return s * torch.log(s) - s + 1


def pearson(s):
    return (s - 1) * (s - 1)


def hellinger(s):
    return (torch.sqrt(s) - 1) ** 2


def js(s):
    return s * torch.log(s) - (s + 1) * torch.log((s + 1) / 2)


def kl_cjg(s):
    return torch.exp(s - 1)


def pearson_cjg(s):
    return s * s / 4 + s


def paper(s):
    # c2 used in code to paper
    return (s - 1) * torch.log(s)


def identity(s):
    # conjugate function used in code to paper EQ
    # identity
    return s


def js_cjg(s):
    return -torch.log(2 - torch.exp(s))


def js_act(s):
    return torch.log(torch.tensor(2.0)) - torch.log(1 + torch.exp(-s))


def identity_act(s):
    return s


def paper_cjg(s):
    return identity_act(s)


C1_FUNS = {"mse": mse}

C2_FUNS = {
    "kl": kl,
    "pearson": pearson,
    "hellinger": hellinger,
    "js": js,
    "paper": paper,
}

PSI_FUNS = {
    "kl": kl_cjg,
    "pearson": pearson_cjg,
    "identity": identity,
    "js": js_cjg,
    "paper": paper_cjg,
}


def gradient_penalty_D(D, real, fake, lambG):
    alpha = torch.rand(real.size(0), 1)
    alpha = alpha.expand(real.size())

    interp = alpha * real + ((1 - alpha) * fake)
    interp.requires_grad_(True)

    # gradient of D(interp) at interp
    d_interp = D(interp)
    (grad,) = autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones(d_interp.size()),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )

    grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean() * lambG
    return grad_penalty


def gradient_penalty_G_rho(G, real, fake, lambG2):
    alpha = torch.rand(real.size(0), 1)
    alpha = alpha.expand(real.size())

    interp = alpha * real + ((1 - alpha) * fake)
    interp.requires_grad_(True)

    # gradient of D(interp) at interp
    _, g_interp = G(interp)
    (grad,) = autograd.grad(
        outputs=g_interp,
        inputs=interp,
        grad_outputs=torch.ones(g_interp.size()),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )

    grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean() * lambG2
    return grad_penalty
