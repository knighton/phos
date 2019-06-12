from .cifar import load_cifar10, load_cifar100


# Name -> Dataset load function.
_NAME2LOAD = {
    'cifar10': load_cifar10,
    'cifar100': load_cifar100,
}


def load_dataset(name):
    """
    Load the dataset by name.

    Returns (in shape, out shape), (train dataset, val dataset).
    """
    load = _NAME2LOAD[name]
    return load()
