from argparse import ArgumentParser
import json
import os

from ..model import each_model_name
from .settings import LabSettings


def parse_flags():
    """
    Parse flags.
    """
    a = ArgumentParser()
    a.add_argument('--dir', type=str, default='data/lab/example/')
    return a.parse_args()


def add_experiment(dataset, loader_cpus, use_cuda, model_name, blocks_per_stage,
                   block_channels, optimizer, num_epochs, train_batches_per_epoch,
                   val_batches_per_epoch, batch_size, blurb_percentiles, lab_dir,
                   run_id):
    """
    Fit a model from flags, saving to the lab.
    """
    pass


def add_model(lab_dir, settings, model_name):
    """
    Fit the model against a hyperparamter grid, saving to the lab.
    """
    # Skip if already done.
    f = '%s/model/%s/done.txt' % (lab_dir, model_name)
    if os.path.exists(f):
        return
    d = '%s/model/%s/' % (lab_dir, model_name)
    os.makedirs(d)

    # Try the model with each combination of training settings.
    for run_id, kwargs in enumerate(settings.each_experiment()):
        x = {
            'lab_dir': lab_dir,
            'model_name': model_name,
            'run_id': run_id,
        }
        kwargs.update(x)
        add_experiment(**kwargs)

    # Then note that we have completed the running of this model.
    with open(f, 'w') as out:
        out.write('')


def update_lab(lab_dir):
    """
    Update the lab with any new models that have been added since last run.
    """
    # Load lab settings.
    f = '%s/settings.json' % lab_dir
    x = json.load(open(f))
    settings = LabSettings(x)

    # For each model, grid fit according to settings, saving to the lab.
    for model_name in each_model_name():
        add_model(lab_dir, settings, model_name)


def main(flags):
    """
    Update lab given by flag.
    """
    update_lab(flags.dir)


if __name__ == '__main__':
    main(parse_flags())
