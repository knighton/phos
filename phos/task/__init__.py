from .cifar10 import load_cifar10


# Name -> Dataset load function.
_NAME2LOAD = {
    'cifar10': load_cifar10,
}


def load_task(name):
    """
    Load the task by name.

    Returns (in shape, out shape), (train dataset, val dataset).
    """
    load = _NAME2LOAD[name]
    return load()
