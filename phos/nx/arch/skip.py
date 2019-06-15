from torch import Tensor
from torch.nn import Parameter as P

from ..module import Module
from .sequence import Sequence


def inverse_sigmoid(x):
    if isinstance(x, Tensor):
        assert x.numel() == 1
    elif hasattr(x, '__len__'):
        x = Tensor(x)
    else:
        x = Tensor([x])
    assert 0 < x < 1
    return -(1 / x - 1).log()


class Skip(Module):
    def __init__(self, *inner, rate=0.05):
        super().__init__()
        self.inner = Sequence(*inner)
        self.raw_rate = P(inverse_sigmoid(rate))

    def forward_inner(self, x, x_loss):
        rate = self.raw_rate.sigmoid()
        y, y_loss = self.inner(x, x_loss)
        z = (1 - rate) * x + rate * y
        if x_loss is None:
            if y_loss is None:
                z_loss = None
            else:
                z_loss = rate * y_loss
        else:
            if y_loss is None:
                z_loss = (1 - rate) * x_loss
            else:
                z_loss = (1 - rate) * x_loss + rate * y_loss
        return z, z_loss

    def summary_inner(self, num_percentiles):
        return {
            'rate': self.raw_rate.sigmoid().item(),
            'inner': self.inner.summary(num_percentiles),
        }
