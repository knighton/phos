import json
import numpy as np
import os

from .util.batch_result_buffer import BatchResultBuffer
from .spy import Spy


class SaveModelDir(Spy):
    """
    Populates a model directory with artifacts of its run.

    /
        summary/
            0.json
            1.json
            ...
        stat/
            epoch/
                0/
                    train_accuracy.npy
                    ...
                    val_loss.npy
                1/
                    ...
                ...
            batch_train_accuracy.npy
            ...
            epoch_val_loss.npy
    """

    def __init__(self, model_dir, summary_percentiles):
        self.model_dir = model_dir
        self.summary_percentiles = summary_percentiles

    def on_fit_begin(self, trainer):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        d = '%s/summary/' % self.model_dir
        if not os.path.exists(d):
            os.makedirs(d)

        d = '%s/stat/epoch/' % self.model_dir
        if not os.path.exists(d):
            os.makedirs(d)

    def on_fit_on_epoch_begin(self, trainer):
        x = trainer.model.summary(self.summary_percentiles)
        f = '%s/summary/%d.json' % (self.model_dir, trainer.epoch)
        text = json.dumps(x, sort_keys=True)
        with open(f, 'w') as out:
            out.write(text)

        self.buffer = BatchResultBuffer()

    def on_train_on_batch_end(self, trainer, *args):
        self.buffer.train.add(*args)

    def on_validate_on_batch_end(self, trainer, *args):
        self.buffer.val.add(*args)

    def each_split_attribute(self):
        for split in ['train', 'val']:
            for attribute in ['loss', 'accuracy', 'forward', 'backward']:
                if split == 'val' and attribute == 'backward':
                    continue
                yield split, attribute

    def on_fit_on_epoch_end(self, trainer):
        d = '%s/stat/epoch/%d/' % (self.model_dir, trainer.epoch)
        os.makedirs(d)

        stats = self.buffer.dump()
        for split, attribute in self.each_split_attribute():
            x = stats[split][attribute]
            x = np.array(x, np.float32)
            f = '%s/stat/epoch/%d/%s_%s.npy' % \
                (self.model_dir, trainer.epoch, split, attribute)
            x.tofile(f)

    def on_fit_end(self, trainer):
        for split, attribute in self.each_split_attribute():
            xx = []
            for i in range(trainer.num_epochs):
                f = '%s/stat/epoch/%d/%s_%s.npy' % \
                    (self.model_dir, i, split, attribute)
                x = np.fromfile(f, np.float32)
                xx.append(x)
            x = np.stack(xx, 0)

            f = '%s/stat/batch_%s_%s.npy' % (self.model_dir, split, attribute)
            x.tofile(f)

            f = '%s/stat/epoch_%s_%s.npy' % (self.model_dir, split, attribute)
            x = x.mean(1)
            x.tofile(f)

        f = '%s/done.txt' % self.model_dir
        with open(f, 'w') as out:
            out.write('')
