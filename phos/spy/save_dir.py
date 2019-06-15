import json
import numpy as np
import os

from .spy import Spy
from .util.batch_result_buffer import BatchResultBuffer


class SaveDir(Spy):
    """
    Saves training results to a directory of JSON dumps and numpy arrays.
    """

    def __init__(self, model_dir, blurb_percentiles):
        self.model_dir = model_dir
        self.blurb_percentiles = blurb_percentiles

    def on_fit_begin(self):
        os.makedirs(self.model_dir)
        self.buffer = BatchResultBuffer()
        d = '%s/blurb/' % self.model_dir
        os.makedirs(d)
        for resolution in [1, 10, 100, 1000]:
            d = '%s/result/%d/' % (self.model_dir, resolution)
            os.makedirs(d)

    def on_epoch_begin(self, epoch, model):
        x = model.blurb(self.blurb_percentiles)
        f = '%s/blurb/%d.json' % (self.model_dir, epoch)
        text = json.dumps(x, sort_keys=True)
        with open(f, 'w') as out:
            out.write(text)

    def on_train_on_batch_end(self, epoch, batch, *args):
        self.buffer.train.add(*args)

    def on_validate_on_batch_end(self, epoch, batch, *args):
        self.buffer.val.add(*args)

    def each_mode_attribute(self):
        for split in ['train', 'val']:
            for attribute in ['loss', 'accuracy', 'forward', 'backward']:
                if split == 'val' and attribute == 'backward':
                    continue
                yield split, attribute

    def on_fit_end(self):
        lists = self.buffer.dump()
        for split, attribute in self.each_mode_attribute():
            # Per batch (eg, 50,000 data points).
            x = lists[split][attribute]
            x = np.array(x, np.float32)
            f = '%s/result/1/%s_%s.npy' % (self.model_dir, split, attribute)
            x.tofile(f)

            # Per deca-batch (eg, 5,000 data points).
            x = x.reshape(-1, 10)
            x = x.mean(1)
            f = '%s/result/10/%s_%s.npy' % (self.model_dir, split, attribute)
            x.tofile(f)

            # Per centi-batch (eg, 500 data points).
            x = x.reshape(-1, 10)
            x = x.mean(1)
            f = '%s/result/100/%s_%s.npy' % (self.model_dir, split, attribute)
            x.tofile(f)

            # Per kilo-batch (eg, 50 data points).
            x = x.reshape(-1, 10)
            x = x.mean(1)
            f = '%s/result/1000/%s_%s.npy' % (self.model_dir, split, attribute)
            x.tofile(f)
