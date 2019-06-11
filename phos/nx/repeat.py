from torch import nn


class Repeat(nn.Module):
    def __init__(self, count, new_layer):
        super().__init__()
        layers = [new_layer() for i in range(count)]
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)
