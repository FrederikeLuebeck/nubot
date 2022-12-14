import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU(),
    "leakyrelu": nn.LeakyReLU(0.2),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "prelu": nn.PReLU(),
}
