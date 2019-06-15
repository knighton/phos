import torch


def dtype_to_str(x):
    assert isinstance(x, torch.dtype)
    return str(x)[6:]


class TensorForm(object):
    """
    Tensor shape and dtype.
    """

    @classmethod
    def of_tensor(cls, x):
        shape = tuple(x.shape[1:])
        dtype = x.dtype
        return cls(shape, dtype)

    def __init__(self, shape, dtype):
        assert isinstance(shape, tuple)
        assert isinstance(dtype, torch.dtype)
        self.shape = shape
        self.dtype = dtype

    def accepts(self, x):
        return x.shape[1:] == self.shape and x.dtype == self.dtype

    def dump(self):
        return {
            'shape': self.shape,
            'dtype': dtype_to_str(self.dtype),
        }
