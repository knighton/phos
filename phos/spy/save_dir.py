import json
import numpy as np
import os

from .spy import Spy
from .util.batch_result_buffer import BatchResultBuffer()


class SaveDir(Spy):
    """
    Saves training results to a directory of JSON dumps and numpy arrays.
    """

    def __init__(self, run_dir, blurb_percentiles):
        self.run_dir = run_dir
        self.blurb_percentiles = blurb_percentiles

    def on_fit_begin(self):
        os.makedirs(self.run_dir)
        self.buffer = BatchResultBuffer()
        d = '%s/blurb/' % self.run_dir
        os.makedirs(d)
        for window in [1, 10, 100]:
            d = '%s/result/%d/' % (self.run_dir, window)
            os.makedirs(d)

    def on_epoch_begin(self, epoch, model):
        x = model.blurb(self.blurb_percentiles)
        f = '%s/blurb/%d.json' % (self.run_dir, epoch)
        text = json.dumps(x, sort_keys=True)
        with open(f, 'w') as out:
            out.write(text)

    def on_train_on_batch_end(self, epoch, batch, *args):
        self.buffer.train.add(*args)

    def on_validate_on_batch_end(self, epoch, batch, *args):
        self.buffer.val.add(*args)

    def each_mode_attribute(self):
        for mode in ['train', 'val']:
            for attribute in ['loss', 'accuracy', 'forward', 'backward']:
                if mode == 'val' and attribute == 'backward':
                    continue
                yield mode, attribute

    def on_fit_end(self):
        lists = self.buffer.dump()
        for mode, attribute in self.each_mode_attribute():
            # Per batch (eg, 20,000 data points).
            x = lists[mode][attribute]
            x = np.array(x, np.float32)
            f = '%s/result/1/%s_%s.npy' % (self.run_dir, mode, attribute)
            x.tofile(f)

            # Per deca-batch (eg, 2,000 data points).
            x = x.reshape(-1, 10)
            x = x.mean(1)
            f = '%s/result/10/%s_%s.npy' % (self.run_dir, mode, attribute)
            x.tofile(f)

            # Per centi-batch (eg, 200 data points).
            x = x.reshape(-1, 10)
            x = x.mean(1)
            f = '%s/result/100/%s_%s.npy' % (self.run_dir, mode, attribute)
            x.tofile(f)
