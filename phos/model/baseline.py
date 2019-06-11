from ..nx import *
from .base import ModelBuilder


class ConvBlock(Sequential):
    def __init__(self, channels, height, width):
        super().__init__(
            ReLU(),
            Conv2d(channels, channels, 3, 1, 1),
            BatchNorm2d(channels),
        )


class DenseBlock(Sequential):
    def __init__(self, channels, height, width):
        dim = channels * height * width
        super().__init__(
            Flatten(),
            ReLU(),
            Linear(dim, dim),
            BatchNorm1d(dim),
            Reshape(channels, height, width),
        )


# All conv blocks.
AllConvBaseline = ModelBuilder(ConvBlock)

# First half conv, last half dense.
blocks = [ConvBlock] * 3 + [DenseBlock] * 3
HalfConvBaseline = ModelBuilder(blocks)
