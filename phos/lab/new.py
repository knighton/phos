from argparse import ArgumentParser
import json
import os
from shutil import rmtree

from .settings import LabSettings


def parse_flags():
    """
    Parse flags.
    """
    a = ArgumentParser()
    a.add_argument('--settings', type=str, default='phos/lab/example.json')
    a.add_argument('--dir', type=str, default='data/lab/example/')
    a.add_argument('--force', type=int, default=0)
    return a.parse_args()


def create_lab(settings, lab_dir, force):
    """
    Set up a lab directory containing settings, then populate the experiments.
    """
    # Create the new lab directory.
    if force:
        if os.path.exists(lab_dir):
            rmtree(lab_dir)
    else:
        assert not os.path.exists(lab_dir)
    os.makedirs(lab_dir)

    # Save the normalized settings to the lab dir.
    f = '%s/settings.json' % lab_dir
    x = settings.dump()
    text = json.dumps(x, sort_keys=True)
    with open(f, 'w') as out:
        out.write(text)

    # Make the lab models dir.
    d = '%s/model/' % lab_dir
    os.makedirs(d)

    # Now, grid-fit each model according to the settings.
    # update_lab(lab_dir)  TODO


def main(flags):
    """
    Create lab according to the flags.
    """
    # Load the grid search settings.
    x = json.load(open(flags.settings))

    # Validate and normalize the settings.
    settings = LabSettings(x)

    # Create the new lab directory.
    create_lab(settings, flags.dir, flags.force)


if __name__ == '__main__':
    main(parse_flags())
