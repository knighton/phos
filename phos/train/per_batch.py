from time import time
import torch
from torch.nn import functional as F


def compute_accuracy(y_pred, y_true):
    """
    Compute classification accuracy.
    """
    y_pred_classes = y_pred.max(1)[1]
    return (y_pred_classes == y_true).type(torch.float32).mean().item()


def forward_on_batch(model, x, y_true):
    """
    Helper method to forward propagate through a classifier.
    """
    t = time()
    y_pred, aux_loss = model(x)
    loss = F.cross_entropy(y_pred, y_true)
    if aux_loss is not None:
        loss = loss + aux_loss
    forward_time = time() - t
    return y_pred, loss, forward_time


def train_on_batch(model, x, y_true, optimizer):
    """
    Train on a single batch, returning loss/acc/time.
    """
    optimizer.zero_grad()
    y_pred, loss, forward_time = forward_on_batch(model, x, y_true)
    t = time()
    loss.backward()
    backward_time = time() - t
    optimizer.step()
    accuracy = compute_accuracy(y_pred, y_true)
    return loss.item(), accuracy, forward_time, backward_time


def validate_on_batch(model, x, y_true):
    """
    Validate on a single batch, returning loss/acc/time.
    """
    y_pred, loss, forward_time = forward_on_batch(model, x, y_true)
    accuracy = compute_accuracy(y_pred, y_true)
    return loss.item(), accuracy, forward_time
