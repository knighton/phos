from ...nx import *


class ModelChassis(Sequential):
    """
    Standardized convolutional classifier frame that custom blocks are plugged into.
    """

    def __init__(self, builder, in_channels, in_height, in_width, out_classes,
                 blocks_per_stage, block_channels):
        """
        Initialize with (a) conv/dense/etc block defs and (b) model dimensions.

        builder           Specification providing plug-in block definitions.
        in_channels       Model input channels (eg, 3 colors).
        in_height         Model input height (eg, 32 pixels).
        in_width          Model input width (eg, 32 pixels).
        out_classes       Model output classes (eg, 10 object categories).
        blocks_per_stage  Number of blocks inside each model stage (eg, 3 blocks).
        block_channels    Internal dimensionality (eg, 128 conv channels).
        """

        def new_stage(hw):
            block_class = builder.get_block_class(hw)
            new_block = lambda: Skip(block_class(block_channels, hw, hw))
            return Skip(Repeat(blocks_per_stage, new_block))

        def new_pool(size):
            return MaxPool2d(size)

        layers = Sequential(
            Conv2d(in_channels, block_channels, 3, 1, 1),
            BatchNorm2d(block_channels),
        )

        for hw in [32, 16, 8, 4, 2, 1]:
            layers.append(new_stage(hw))
            if 1 < hw:
                layers.append(new_pool(2))

        layers += [
            Flatten(),
            ReLU(),
            Dropout(),
            Linear(block_channels, out_classes),
        )

        super().__init__(*layers)

    def blurb(self, num_percentiles):
        return None
