from argparse import ArgumentParser
import json

from .settings import LabSettings


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--settings', type=str, required=True)
    return a.parse_args()


def main(flags):
    x = json.load(open(flags.settings))
    LabSettings(x)


if __name__ == '__main__':
    main(parse_flags())
