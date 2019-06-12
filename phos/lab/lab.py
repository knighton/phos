import json
import numpy as np
import os
from shutil import rmtree
from torch import optim
from torch.utils.data import DataLoader

from ..dataset import load_dataset
from ..fit.session import fit
from ..model import each_model_name, get_model
from ..spy import RowPerEpoch, SaveDir
from ..util.flag_parsing import parse_class_and_kwargs
from .settings import LabSettings


class Lab(object):
    """
    A collection of comparable models trained on a hyperparameter grid.

    Sits atop a lab directory on disk.  Locking is accomplished by 'done' files.
    Highly advanced.  Gets job done.
    """

    def __init__(self, lab_dir):
        self.lab_dir = lab_dir

        f = '%s/settings.json' % lab_dir
        x = json.load(open(f))
        self.settings = LabSettings(x)

    def is_model_done(self, model):
        f = '%s/model/%s/done.txt' % (self.lab_dir, model)
        return os.path.exists(f)

    def get_models(self):
        d = '%s/model/' % self.lab_dir
        models_with_dirs = set(os.listdir(d))

        todo = []
        part = []
        done = []

        for model in each_model_name():
            if model not in models_with_dirs:
                todo.append(model)
            elif not self.is_model_done(model):
                part.append(model)
            else:
                done.append(model)

        return {
            'todo': todo,
            'part': part,
            'done': done,
        }

    def get_done_models(self):
        x = self.get_models()
        return x['done']

    def is_run_done(self, model, run_id):
        f = '%s/model/%s/run/%d/done.txt' % (self.lab_dir, model, run_id)
        return os.path.exists(f)

    def get_runs(self):
        todo = []
        part = []
        done = []

        x = self.get_models()

        for model in x['todo']:
            for run_id in range(self.settings.num_grid_combos):
                todo.append((model, run_id))

        for model in x['part']:
            d = '%s/model/%s/run/' % (self.lab_dir, model)
            runs_with_dirs = set(map(int, os.listdir(d)))
            for run_id in range(self.settings.num_grid_combos):
                pair = model, run_id
                if run_id not in runs_with_dirs:
                    todo.append(pair)
                elif not self.is_run_done(model, run_id):
                    part.append(pair)
                else:
                    done.append(pair)

        for model in x['done']:
            for run_id in range(self.settings.num_grid_combos):
                assert self.is_run_done(model, run_id)
                done.append((model, run_id))

        return {
            'todo': todo,
            'part': part,
            'done': done,
        }

    def get_done_runs(self):
        x = self.get_runs()
        return x['done']

    def fit(self, dataset, loader_cpus, use_cuda, model_name, blocks_per_stage,
            block_channels, optimizer, num_epochs, train_batches_per_epoch,
            val_batches_per_epoch, batch_size, blurb_percentiles, lab_dir, run_id):
        """
        Fit a model from flags, saving to the lab.
        """
        # First, check if already done.  If not done, wipe the preexisting.
        run_dir = '%s/model/%s/run/%d/' % (lab_dir, model_name, run_id)
        done_filename = '%s/done.txt' % run_dir
        if os.path.exists(done_filename):
            return
        if os.path.exists(run_dir):
            rmtree(run_dir)

        # Load the dataset.
        (in_shape, out_shape), (train_dataset, val_dataset) = load_dataset(dataset)
        in_channels, in_height, in_width = in_shape
        out_classes, = out_shape

        # Instantiate data loaders for the dataset.  Both are shuffled.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=loader_cpus)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=loader_cpus)

        # Model parameters are derived from both dataset dimensions and input args.
        model_class = get_model(model_name)
        model_kwargs = {
            'in_channels': in_channels,
            'in_height': in_height,
            'in_width': in_width,
            'out_classes': out_classes,
            'blocks_per_stage': blocks_per_stage,
            'block_channels': block_channels,
        }
        model = model_class(**model_kwargs)
        if use_cuda:
            model.cuda()

        # Create the optimizer according to the parsed optimizer flag.
        opt_class, opt_kwargs = parse_class_and_kwargs(optim, optimizer)
        optimizer = opt_class(model.parameters(), **opt_kwargs)

        # Add spies for notifying the user of progress and updating the lab.
        row_per_epoch = RowPerEpoch()
        save_dir = SaveDir(run_dir, blurb_percentiles)
        spies = row_per_epoch, save_dir

        # Now, train the model.
        fit(train_loader, val_loader, model, optimizer, num_epochs,
            train_batches_per_epoch, val_batches_per_epoch, use_cuda, spies)

        # Note done.
        with open(done_filename, 'w') as out:
            out.write('')

    def add_run(self, model, run_id):
        x = {
            'lab_dir': self.lab_dir,
            'model_name': model,
            'run_id': run_id,
        }
        x.update(self.settings.grid_combos[run_id])
        x.update(self.settings.fixed)
        self.fit(**x)

    def update(self):
        x = self.get_runs()
        while True:
            for model, run_id in x['todo']:
                self.add_run(model, run_id)
            else:
                break

    def query_results(self, x):
        models = x.get('model')
        if not modeis:
            z = self.get_models()
            models = z['part'] + z['done']
        elif not isinstance(models, list):
            models = [models]

        run_ids = self.settings.query_runs(x)

        resolutions = x.get('resolution', [1, 10, 100])

        splits = x.get('split', ['train', 'val'])

        attributes = x.get('attribute', ['loss', 'accuracy', 'forward', 'backard'])

        pairs = []
        for model in models:
            d = '%s/model/%s/' % (self.lab_dir, 'model')
            if not os.path.exists(d):
                continue
            for run_id in run_ids:
                if not self.is_run_done(model, run_id):
                    continue
                for resolution in resolutions:
                    for split in splits:
                        for attribute in attributes:
                            if split == 'val' and attribute == 'backard':
                                continue
                            f = '%s/model/%s/run/%d/result/%d/%s_%s.npy' % \
                                (self.lab_dir, model, run_id, resolution, split,
                                 attribute)
                            x = np.fromfile(f, np.float32)
                            key = {
                                'model': model,
                                'run_id': run_id,
                                'resolution': resolution,
                                'split': split,
                                'attribute': attribute,
                            }
                            value = x.tolist()
                            pairs.append((key, value))
        return pairs
