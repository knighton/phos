import numpy as np
import torch

from ..spy import SpyList
from .per_batch import train_on_batch, validate_on_batch


class Trainer(object):
    def __init__(self, train_loader, val_loader, model, optimizer, num_epochs,
                 train_batches_per_epoch, val_batches_per_epoch, use_cuda, spies):
        """
        Initialize with training configuration.
        """
        # Their final batch may be truncated.  We require batches to be the same
        # size.  This assumes both the train and val data loaders shuffle.
        assert train_batches_per_epoch < len(train_loader)
        assert val_batches_per_epoch < len(val_loader)

        # Configuration.
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.train_batches_per_epoch = train_batches_per_epoch
        self.val_batches_per_epoch = val_batches_per_epoch
        self.use_cuda = use_cuda
        self.spies = SpyList(*spies)

        # Progress.
        self.epoch = 0  # Used by save_checkpoint/load_checkpoint.
        self.batch = 0
        self.train_batch = 0
        self.val_batch = 0

    def save_checkpoint(self, f):
        """
        Save to a checkpoint file.
        """
        x = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(x, f)

    def load_checkpoint(self, f):
        """
        Load from a checkpoint file.
        """
        x = torch.load(f)
        self.epoch = x['epoch']
        self.model.load_state_dict(x['model'])
        self.optimizer.load_state_dict(x['optimizer'])

    def each(self, xx):
        """
        Make a generator out of a sequence.
        """
        for x in xx:
            yield x

    def next_sample(self, each_split_batch):
        """
        Pull the next sample out of a split and optionally put it on the GPU.
        """
        x, y_true = next(each_split_batch)
        if self.use_cuda:
            x = x.cuda()
            y_true = y_true.cuda()
        return x, y_true

    def each_batch(self):
        """
        Iterate training and validation sets interspersed.
        """
        each_train_batch = self.each(self.train_loader)
        each_val_batch = self.each(self.val_loader)

        splits = [1] * self.train_batches_per_epoch + \
            [0] * self.val_batches_per_epoch
        np.random.shuffle(splits)

        for is_training in splits:
            if is_training:
                each_split_batch = each_train_batch
            else:
                each_split_batch = each_val_batch
            x, y = self.next_sample(each_split_batch)
            yield is_training, x, y

    def fit_on_batch(self, is_training, x, y):
        """
        Fit the model on one batch.

        We take the slight overhead of model mode switching here, so that users can
        call train_on_batch/validate_on_batch repeatedly without it in other use
        cases.

        Likewise, knowledge of spies and the concept of where we are in training
        (epoch/batch) ends here.
        """
        self.spies.on_fit_on_batch_begin(self)
        if is_training:
            self.spies.on_train_on_batch_begin(self)
            self.model.train()
            args = train_on_batch(self.model, x, y, self.optimizer)
            self.spies.on_train_on_batch_end(self, *args)
        else:
            self.spies.on_validate_on_batch_begin(self)
            self.model.eval()
            args = validate_on_batch(self.model, x, y)
            self.spies.on_validate_on_batch_end(self, *args)
        self.spies.on_fit_on_batch_end(self)

    def increment_batch(self, is_training):
        """
        Increment batch progress.
        """
        self.batch += 1
        if is_training:
            self.train_batch += 1
        else:
            self.val_batch += 1

    def fit_on_epoch(self):
        """
        Fit the model for one epoch.
        """
        self.spies.on_fit_on_epoch_begin(self)
        for is_training, x, y in self.each_batch():
            self.fit_on_batch(is_training, x, y)
            self.increment_batch(is_training)
        self.spies.on_fit_on_epoch_end(self)

    def fit(self):
        """
        Fit the model.
        """
        self.spies.on_fit_begin(self)
        while self.epoch < self.num_epochs:
            self.fit_on_epoch()
            self.epoch += 1
        self.spies.on_fit_end(self)
