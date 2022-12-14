import numpy as np
import torch
from torch import autograd
from torch import nn

from unot.networks.layers import NonNegativeLinear, PosDefDense

# import unot.networks

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
}


class NICNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_units,
        activation="LeakyReLU",
        softplus_W_kernels=False,
        softplus_beta=1,
        std=0.1,
        fnorm_penalty=0,
        kernel_init_fxn=None,
        **kwargs
    ):

        super(NICNN, self).__init__()
        self.fnorm_penalty = fnorm_penalty
        self.softplus_W_kernels = softplus_W_kernels

        if isinstance(activation, str):
            activation = ACTIVATIONS[
                activation.lower().replace("_", "")
            ]  # unot.networks.
        self.sigma = activation

        units = hidden_units + [1]

        if self.softplus_W_kernels:

            def Linear(*args, **kwargs):
                return NonNegativeLinear(*args, **kwargs, beta=softplus_beta)

            # this function should be inverse map of function used in PositiveDense layers
            rescale = lambda x: np.log(np.exp(x) - 1)
        else:
            Linear = nn.Linear
            rescale = lambda x: x

        # keep track of previous size to normalize accordingly
        normalization = 1
        self.wz = list()
        for idim, odim in zip([input_dim] + units[1:-1], units[1:]):
            _wz = Linear(idim, odim, bias=False)
            nn.init.constant_(_wz.weight, rescale(1.0 / normalization))
            self.wz.append(_wz)
            normalization = odim
        self.wz = nn.ModuleList(self.wz)

        # first square layer, initialized to identity
        # TODO: could be replaced by a transport map between Gaussians
        _wx = PosDefDense(input_dim, input_dim, bias=True)
        nn.init.eye_(_wx.weight)
        nn.init.zeros_(_wx.bias)
        self.wx = [_wx]
        # subsequent layers reinjected into convex functions
        for odim in units[1:]:
            _wx = nn.Linear(input_dim, odim, bias=True)
            kernel_init_fxn(_wx.weight)
            nn.init.zeros_(_wx.bias)
            self.wx.append(_wx)
        self.wx = nn.ModuleList(self.wx)

    def forward(self, x):
        z = self.wx[0](x)
        z = 0.5 * (x * z)

        for wz, wx in zip(self.wz[:-1], self.wx[1:-1]):
            z = self.sigma(0.2)(wz(z) + wx(x))

        y = self.wz[-1](z) + self.wx[-1](x)

        return y

    def transport(self, x):
        assert x.requires_grad

        (output,) = autograd.grad(
            self.forward(x),
            x,
            create_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((x.size()[0], 1), device=x.device).float(),
        )
        return output

    def clamp_w(self):
        if self.softplus_W_kernels:
            return

        for w in self.wz:
            w.weight.data = w.weight.data.clamp(min=0)
        return

    def penalize_w(self):
        return self.fnorm_penalty * sum(
            map(lambda x: torch.nn.functional.relu(-x.weight).norm(), self.wz)
        )
