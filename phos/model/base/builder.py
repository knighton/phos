from .chassis import ModelChassis
  

class ModelBuilder(object):
    """
    Block definitions to plug into a "standardized" image classifier frame.
    """

    # Tensor size (height and width) per stage, in execution order.
    sizes = 32, 16, 8, 4, 2, 1

    # Size -> index of corresponding block class.
    size2index = {}
    for i, size in enumerate(sizes):
        size2index[size] = i

    @classmethod
    def normalize_block_classes(cls, x):
        """
        Normalize the different ways to pass block classes into init.
        """
        if isinstance(x, (tuple, list)):
            blocks = x
            assert len(blocks) == 6
        elif isinstance(x, dict):
            sizes = sorted(x, reverse=True)
            assert sizes == sizes
            blocks = []
            for size in sizes:
                block = x[size]
                blocks.append(block)
        else:
            block = x
            blocks = (block,) * 6
        return tuple(blocks)

    def __init__(self, blocks):
        """
        Initialize, normalizing block classes to a list ordered by size.
        """
        self.block_classes = self.normalize_block_classes(blocks)

    def get_block_class(self, size):
        """
        Get the type of block to instantiate for the given size.
        """
        index = self.size2index[size]
        return self.block_classes[index]

    def __call__(self, *args, **kwargs):
        """
        Instantiate a model frame, inserting our blocks, with the given kwargs.
        """
        return ModelChassis(self, *args, **kwargs)
