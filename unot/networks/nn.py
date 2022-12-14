import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from unot.networks import ACTIVATIONS


class NN(nn.Module):
    def __init__(
        self,
        input_dim=10,
        hidden_units=[5, 5],
        out_dim=10,
        activation="relu",
        init=0.4,
        softplus=False,
        kernel_init_fxn=None,
    ):
        super(NN, self).__init__()

        self.activation = ACTIVATIONS[activation.lower()]
        self.softplus = softplus

        if self.softplus:
            self.sigma = nn.Softplus()

        dims = [input_dim] + hidden_units + [out_dim]

        self.lins = nn.ModuleList(
            [nn.Linear(i, o, bias=True) for i, o in zip(dims[:-1], dims[1:])]
        )

        if kernel_init_fxn is not None:
            for layer in self.lins:
                kernel_init_fxn(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):

        z = x

        for lin in self.lins[:-1]:
            z = self.activation(lin(z))

        z = self.lins[-1](z)

        if self.softplus:
            # restrict output to be greater than zero
            z = self.sigma(z)

        return z
