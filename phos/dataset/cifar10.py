from torchvision.datasets import CIFAR10
from torchvision import transforms as tf


def load_cifar10():
    """
    Load CIFAR10.
    """
    shapes = (3, 32, 32), (10,)
    train_transform = tf.Compose([
        tf.ToTensor(),
    ])
    train_dataset = CIFAR10(root='./data', train=True, download=True,
                            transform=train_transform)
    val_transform = tf.Compose([
        tf.ToTensor(),
    ])
    val_dataset = CIFAR10(root='./data', train=False, download=True,
                          transform=val_transform)
    datasets = train_dataset, val_dataset
    return shapes, datasets
