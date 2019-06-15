import sys
import torch

from .module import Module


class TorchWrapper(Module):
    """
    Phos API shim around an inner PyTorch nn.Module.
    """

    def __init__(self, inner):
        """
        Initialize, instantiating the wrapped inner module.
        """
        super().__init__()
        self.inner = inner

    def forward_inner(self, x, x_loss):
        """
        Forward with auxiliary loss.
        """
        y = self.inner(x)
        y_loss = x_loss
        return y, y_loss

    def summary_inner(self, num_percentiles):
        """
        Summarize.
        """
        return {
            'inner': self.inner.__class__.__name__,
        }


class Builder(object):
    def __init__(self, klass):
        self.klass = klass

    def __call__(self, *args, **kwargs):
        inner = self.klass(*args, **kwargs)
        return TorchWrapper(inner)


# Find all the torch.nn modules and wrap them into this module.
this = sys.modules[__name__]
for k in sorted(dir(torch.nn)):
    v = getattr(torch.nn, k)
    if not isinstance(v, type):
        continue
    if not issubclass(v, torch.nn.Module):
        continue
    if k == 'Module':
        continue
    setattr(this, k, Builder(v))
