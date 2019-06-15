from time import time
import torch
from torch.nn import functional as F


def compute_accuracy(y_pred, y_true):
    """
    Compute classification accuracy.
    """
    y_pred_classes = y_pred.max(1)[1]
    return (y_pred_classes == y_true).type(torch.float32).mean().item()


def train_on_batch(model, x, y_true, optimizer, results=None):
    """
    Train on a single batch, returning loss/acc/time.
    """
    optimizer.zero_grad()

    t = time()
    y_pred = model(x)
    loss = F.cross_entropy(y_pred, y_true)
    forward = time() - t

    t = time()
    loss.backward()
    backward = time() - t

    optimizer.step()

    accuracy = compute_accuracy(y_pred, y_true)

    return loss.item(), accuracy, forward, backward


def validate_on_batch(model, x, y_true):
    """
    Validate on a single batch, returning loss/acc/time.
    """
    t = time()
    y_pred = model(x)
    loss = F.cross_entropy(y_pred, y_true)
    forward = time() - t

    accuracy = compute_accuracy(y_pred, y_true)

    return loss.item(), accuracy, forward
