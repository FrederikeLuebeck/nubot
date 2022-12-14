import torch
import torch.nn as nn

from unot.networks import ACTIVATIONS


class nonnegNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_units,
        activation="LeakyReLU",
        sigmoid=True,
        sigmoid_max=1,
        kernel_init_fxn=None,
    ):
        """
        Fully connected NN with 1-dimensional non-negative output.

        Args:
            input_dim (int): input dimension
            hidden_units (list): hidden dimensions
            activation (str, optional): activation function. Defaults to "LeakyReLU".
            sigmoid (bool, optional): If True, output actication is sigmoid If False, output actication is Softplus. Defaults to True.
            sigmoid_max (int, optional): If Sigmoid=True, then output is multiplied by sigmoid_max to restrict output to (0, sigmoid_max). Defaults to 1.
            kernel_init_fxn (_type_, optional): _description_. Defaults to None.
        """
        super(nonnegNN, self).__init__()

        self.activation = ACTIVATIONS[activation.lower()]
        self.sigmoid_max = sigmoid_max
        self.sigmoid = sigmoid
        if sigmoid:
            self.sigma = nn.Sigmoid()
        else:
            self.sigma = nn.Softplus()

        dims = [input_dim] + hidden_units + [1]
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
        z = self.sigma(z)

        if self.sigmoid:
            z = z * self.sigmoid_max

        return z
