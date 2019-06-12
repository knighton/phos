from argparse import ArgumentParser
import json
import os
from shutil import rmtree

from .lab import Lab
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


def main(flags):
    """
    Set up a lab directory containing settings, then populate the experiments.
    """
    # Validate the settings file.
    settings_text = open(flags.settings).read()
    x = json.loads(settings_text)
    LabSettings(x)

    # Create the new lab directory.
    if flags.force:
        if os.path.exists(flags.dir):
            rmtree(flags.dir)
    else:
        assert not os.path.exists(flags.dir)
    os.makedirs(flags.dir)

    # Copy the validated settings file into the lab dir.
    f = '%s/settings.json' % flags.dir
    with open(f, 'w') as out:
        out.write(settings_text)

    # Make the lab models dir.
    d = '%s/model/' % flags.dir
    os.makedirs(d)

    # Now we can create a Lab and hand off control to it.
    lab = Lab(flags.dir)
    lab.update()


if __name__ == '__main__':
    main(parse_flags())
