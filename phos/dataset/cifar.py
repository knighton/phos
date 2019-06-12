from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms as tf


def _load_cifar(num_classes, klass):
    """
    Load CIFAR-N.
    """
    shapes = (3, 32, 32), (num_classes,)

    train = tf.Compose([
        tf.ToTensor(),
    ])
    train_dataset = klass(root='./data', train=True, download=True, transform=train)

    val = tf.Compose([
        tf.ToTensor(),
    ])
    val_dataset = klass(root='./data', train=False, download=True, transform=val)

    datasets = train_dataset, val_dataset

    return shapes, datasets


def load_cifar10():
    return _load_cifar(10, CIFAR10)


def load_cifar100():
    return _load_cifar(100, CIFAR100)
