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
            Dropout(),
            Linear(dim, dim),
            BatchNorm1d(dim),
            Reshape(channels, height, width),
        )


x = [ConvBlock] * 5 + [DenseBlock] * 1
baseline_5conv_1dense = ModelBuilder(x)

x = [ConvBlock] * 4 + [DenseBlock] * 2
baseline_4conv_2dense = ModelBuilder(x)

x = [ConvBlock] * 3 + [DenseBlock] * 3
baseline_3conv_3dense = ModelBuilder(x)
