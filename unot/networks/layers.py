import torch
from torch import nn


class NonNegativeLinear(nn.Linear):
    def __init__(self, *args, beta=1.0, **kwargs):
        super(NonNegativeLinear, self).__init__(*args, **kwargs)
        self.beta = beta

    def forward(self, x):
        return nn.functional.linear(x, self.kernel(), self.bias)

    def kernel(self):
        return nn.functional.softplus(self.weight, beta=self.beta)


class PosDefDense(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(PosDefDense, self).__init__(*args, **kwargs)

    def forward(self, x):
        kernel = nn.functional.linear(x, self.weight, None)
        return nn.functional.linear(
            kernel, torch.transpose(self.weight, 0, 1), self.bias
        )
