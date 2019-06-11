from argparse import ArgumentParser
import json
import os
from shutil import rmtree
from torch import optim
from torch.utils.data import DataLoader

from ..dataset import load_dataset
from ..fit import fit
from ..model import each_model_name, get_model
from ..spy import RowPerEpoch, SaveDir
from ..util.flag_parsing import parse_class_and_kwargs
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
