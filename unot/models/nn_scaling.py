import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from unot.networks import ACTIVATIONS


class NN_scaling(nn.Module):
    def __init__(
        self,
        in_dim=10,
        hidden_dims=[5, 5],
        out_dim=10,
        activation="relu",
        init=0.4,
        joint=True,
        sigmoid=True,
        sigmoid_max=1,
    ):
        super(NN_scaling, self).__init__()

        self.activation = ACTIVATIONS[activation.lower()]
        self.sigmoid_max = sigmoid_max
        self.sigmoid = sigmoid
        if sigmoid:
            self.sigma = nn.Sigmoid()
        else:
            self.sigma = nn.Softplus()
        self.joint = joint

        if self.joint:
            dims = [in_dim] + hidden_dims + [out_dim + 1]

            self.lins = nn.ModuleList(
                [nn.Linear(i, o, bias=True) for i, o in zip(dims[:-1], dims[1:])]
            )
        else:
            dims = [in_dim] + hidden_dims + [out_dim]
            self.lins = nn.ModuleList(
                [nn.Linear(i, o, bias=True) for i, o in zip(dims[:-1], dims[1:])]
            )

            self.scaling = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, 1),
                self.sigma,
            )

        kernel_init_fxn = torch.nn.init.uniform_
        if kernel_init_fxn is not None:
            for layer in self.lins:
                kernel_init_fxn(layer.weight, a=-init, b=init)

    def forward(self, x):

        z = x

        for lin in self.lins[:-1]:
            z = self.activation(lin(z))
        z = self.lins[-1](z)

        if self.joint:
            out_z = z[:, :-1]
            eps = z[:, -1]
            # restrict output to be greater than zero
            eps = self.sigma(eps).unsqueeze(-1)
        else:
            out_z = z
            eps = self.scaling(x)

        if self.sigmoid:
            eps = eps * self.sigmoid_max

        return out_z, eps


class scaling(nn.Module):
    def __init__(
        self,
        in_dim=10,
        hidden_dims=[5, 5],
        activation="relu",
        init=0.4,
        sigmoid=True,
        sigmoid_max=1,
    ):
        super(scaling, self).__init__()

        self.activation = ACTIVATIONS[activation.lower()]
        self.sigmoid_max = sigmoid_max
        self.sigmoid = sigmoid
        if sigmoid:
            self.sigma = nn.Sigmoid()
        else:
            self.sigma = nn.Softplus()

        dims = [in_dim] + hidden_dims + [1]
        self.lins = nn.ModuleList(
            [nn.Linear(i, o, bias=True) for i, o in zip(dims[:-1], dims[1:])]
        )

        kernel_init_fxn = torch.nn.init.uniform_
        if kernel_init_fxn is not None:
            for layer in self.lins:
                kernel_init_fxn(layer.weight, a=-init, b=init)

    def forward(self, x):

        z = x

        for lin in self.lins[:-1]:
            z = self.activation(lin(z))
        z = self.lins[-1](z)
        z = self.sigma(z)

        if self.sigmoid:
            z = z * self.sigmoid_max

        return z
