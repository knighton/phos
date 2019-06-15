#!/usr/bin/env python3

from argparse import ArgumentParser
import json
import os
from shutil import rmtree
from torch import optim
from torch.utils.data import DataLoader

from ..dataset import load_dataset
from ..model import each_model_name, get_model
from ..spy import Checkpoint, LogRowPerEpoch, SaveModelDir
from ..train import Trainer
from ..util.flag_parsing import parse_class_and_kwargs


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--dir', type=str, default='data/bench/example/')
    return a.parse_args()


def add_model_to_benchmark(bench_dir, settings, model_name):
    f = '%s/model/%s/done.txt' % (bench_dir, model_name)
    if os.path.exists(f):
        return

    x = settings

    (in_shape, out_shape), (train_dataset, val_dataset) = load_dataset(x['dataset'])
    in_channels, in_height, in_width = in_shape
    out_classes, = out_shape

    train_loader = DataLoader(train_dataset, batch_size=x['batch_size'],
                              shuffle=True, num_workers=x['loader_cpus'])
    val_loader = DataLoader(val_dataset, batch_size=x['batch_size'], shuffle=True,
                            num_workers=x['loader_cpus'])

    model_class = get_model(model_name)
    model_kwargs = {
        'in_channels': in_channels,
        'in_height': in_height,
        'in_width': in_width,
        'out_classes': out_classes,
        'blocks_per_stage': x['blocks_per_stage'],
        'block_channels': x['block_channels'],
    }
    model = model_class(**model_kwargs)
    if x['use_cuda']:
        model.cuda()

    optimizer_class, optimizer_kwargs = parse_class_and_kwargs(optim, x['optimizer'])
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    model_dir = '%s/model/%s/' % (bench_dir, model_name)
    checkpoint_file = '%s/checkpoint.chk' % model_dir
    checkpoint = Checkpoint(checkpoint_file)
    log_row_per_epoch = LogRowPerEpoch()
    save_model_dir = SaveModelDir(model_dir, x['summary_percentiles'])
    spies = checkpoint, log_row_per_epoch, save_model_dir

    trainer = Trainer(train_loader, val_loader, model, optimizer, x['num_epochs'],
                      x['train_batches_per_epoch'], x['val_batches_per_epoch'],
                      x['use_cuda'], spies)
    trainer.fit()

    
def update_benchmark(bench_dir):
    f = '%s/settings.json' % bench_dir
    settings = json.load(open(f))
    for model in each_model_name():
        add_model_to_benchmark(bench_dir, settings, model)


def main(flags):
    update_benchmark(flags.dir)


if __name__ == '__main__':
    main(parse_flags())
