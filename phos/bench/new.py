#!/usr/bin/env python3
from argparse import ArgumentParser
import json
import os
from shutil import rmtree

from .add_models import add_models_to_benchmark


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--settings', type=str, default='phos/bench/settings.json')
    a.add_argument('--settings_spec', type=str,
                   default='phos/bench/settings_spec.json')
    a.add_argument('--dir', type=str, default='data/bench/example/')
    a.add_argument('--force', type=int, default=0)
    return a.parse_args()


def validate_settings(x, spec):
    assert sorted(x) == sorted(spec)
    name2class = {
        'bool': bool,
        'int': int,
        'str': str,
    }
    for k, v in x.items():
        klass = name2class[spec[k]]
        assert isinstance(v, klass)


def create_benchmark(settings_file, settings_spec_file, bench_dir, force):
    settings = json.load(open(settings_file))
    settings_spec = json.load(open(settings_spec_file))
    validate_settings(settings, settings_spec)

    if force:
        if os.path.exists(bench_dir):
            rmtree(bench_dir)
    else:
        assert not os.path.exists(bench_dir)
    os.makedirs(bench_dir)

    f = '%s/settings.json' % bench_dir
    with open(f, 'w') as out:
        json.dump(settings, out)

    d = '%s/model/' % bench_dir
    os.makedirs(d)

    add_models_to_benchmark(bench_dir)


def main(flags):
    create_benchmark(flags.settings, flags.settings_spec, flags.dir, flags.force)


if __name__ == '__main__':
    main(parse_flags())
