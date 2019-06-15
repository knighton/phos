import numpy as np

from ..spy import SpyList
from .batch_interleaver import each_batch
from .per_batch import train_on_batch, validate_on_batch


def fit_on_batch(epoch, batch, model, is_training, x, y_true, optimizer, spies):
    """
    Perform one batch of model training.

    We take the slight overhead of model mode switching here, so that users can call
    train_on_batch/validate_on_batch repeatedly without it in other use cases.

    Likewise, knowledge of spies and the concept of where we are in training
    (epoch/batch) ends here.
    """
    if is_training:
        spies.on_train_on_batch_begin(epoch, batch)
        model.train()
        args = train_on_batch(model, x, y_true, optimizer)
        spies.on_train_on_batch_end(epoch, batch, *args)
    else:
        spies.on_validate_on_batch_begin(epoch, batch)
        model.eval()
        args = validate_on_batch(model, x, y_true)
        spies.on_validate_on_batch_end(epoch, batch, *args)


def fit_on_epoch(epoch, train_loader, val_loader, model, optimizer, num_train,
                 num_val, use_cuda, spies):
    """
    Perform one epoch of model training.
    """
    spies.on_epoch_begin(epoch, model)

    for batch, is_training, x, y_true in each_batch(
            train_loader, val_loader, num_train, num_val, use_cuda):
        fit_on_batch(epoch, batch, model, is_training, x, y_true, optimizer, spies)

    spies.on_epoch_end(epoch, model)


def fit(train_loader, val_loader, model, optimizer, num_epochs,
        train_batches_per_epoch, val_batches_per_epoch, use_cuda, spies):
    """
    Train a model.  Uses the provided data loaders, model, optimizer, etc.
    """
    # Their final batch may be truncated.  We require batches to be the same size.
    # This assumes both the train and val data loaders shuffle their samples.
    assert train_batches_per_epoch < len(train_loader)
    assert val_batches_per_epoch < len(val_loader)

    # Make the 0+ spies callable as one for tidyness.
    spies = SpyList(*spies)

    # Note fit begin.
    spies.on_fit_begin()

    # Perform each epoch of training.
    for epoch in range(num_epochs):
        fit_on_epoch(epoch, train_loader, val_loader, model, optimizer,
                     train_batches_per_epoch, val_batches_per_epoch, use_cuda, spies)

    # Note fit end.
    spies.on_fit_end()
