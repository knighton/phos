import torch
from torch import nn
from torch.nn import Parameter as P


def inverse_sigmoid(x):
    if isinstance(x, torch.Tensor):
        assert x.numel() == 1
    elif hasattr(x, '__len__'):
        x = torch.Tensor(x)
    else:
        x = torch.Tensor([x])
    assert 0 < x < 1
    return -(1 / x - 1).log()


class Skip(nn.Module):
    def __init__(self, *inner, rate=0.05):
        super().__init__()
        self.inner = nn.Sequential(*inner)
        self.raw_rate = P(inverse_sigmoid(rate))

    def forward(self, x):
        rate = self.raw_rate.sigmoid()
        return (1 - rate) * x + rate * self.inner(x)
