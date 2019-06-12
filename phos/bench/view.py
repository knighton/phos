#!/usr/bin/env python3

import itertools
import json
import numpy as np
import os


RESOLUTIONS = 1, 10, 100
SPLITS = 'train', 'val'
ATTRIBUTES = 'loss', 'accuracy', 'forward', 'backward'


def each_done_model(bench_dir):
    d = '%s/model/' % bench_dir
    for model in sorted(os.listdir(d)):
        f = '%s/model/%s/done.txt' % (bench_dir, model)
        if os.path.exists(f):
            yield model


def get(k2v, k, vv):
    x = k2v.get(k)
    if x is None:
        return vv
    elif isinstance(x, list):
        if x:
            return x
        else:
            return vv
    else:
        return [x]


def product(models, resolutions, splits, attributes):
    for model, resolution, split, attribute in itertools.product(
            models, resolutions, splits, attributes):
        if split == 'val' and attribute == 'backward':
            continue
        yield model, resolution, split, attribute


def each_query_result(bench_dir, x):
    done_models = list(each_done_model(bench_dir))
    models = get(x, 'model', done_models)
    models = sorted(filter(lambda s: s in done_models, models))

    resolutions = get(x, 'resolution', RESOLUTIONS)
    splits = get(x, 'split', SPLITS)
    attributes = get(x, 'attribute', ATTRIBUTES)

    for model, resolution, split, attribute in product(models, resolutions, splits,
                                                       attributes):
        k = {
            'model': model,
            'resolution': resolution,
            'split': split,
            'attribute': attribute,
        }

        f = '%s/model/%s/result/%d/%s_%s.npy' % \
            (bench_dir, model, resolution, split, attribute)
        x = np.fromfile(f, np.float32)
        v = x.tolist()

        yield k, v


def get_blurb(bench_dir, model, epoch):
    f = '%s/model/%s/blurb/%d.json' % (bench_dir, model, epoch)
    return json.load(open(f))
