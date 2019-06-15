from torch.nn import ModuleList

from ..module import Module


class Sequence(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = ModuleList()
        for layer in layers:
            assert isinstance(layer, Module)
            self.layers.append(layer)

    def forward_inner(self, x, x_loss):
        for layer in self.layers:
            x, x_loss = layer(x, x_loss)
        return x, x_loss

    def summary_inner(self, num_percentiles):
        xx = []
        for layer in self.layers:
            x = layer.summary(num_percentiles)
            xx.append(x)
        return {
            'layers': xx,
        }
