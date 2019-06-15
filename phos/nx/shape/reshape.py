from ..module import Module


class Reshape(Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward_inner(self, x, x_loss):
        shape = (x.shape[0],) + self.shape
        return x.view(*shape), x_loss

    def summary_inner(self, num_percentiles):
        return {
            'shape': self.shape,
        }
