#!/usr/bin/env python3

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


def each_query_result(bench_dir, x):
    done_models = list(each_done_model(bench_dir))
    models = get(x, 'model', done_models)
    models = sorted(filter(lambda s: s in done_models, models))
    resolutions = get(x, 'resolution', RESOLUTIONS)
    splits = get(x, 'split', SPLITS)
    attributes = get(x, 'attribute', ATTRIBUTES)

    for model in models:
        for resolution in resolutions:
            for split in splits:
                for attribute in attributes:
                    if split == 'val' and attribute == 'backward':
                        continue

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


"""
d = 'data/bench/example/'
x = {
    'resolution': 100,
    'split': ['train', 'val'],
    'attribute': None,
}
for k, v in each_query_result(d, x):
    print(k, v)
"""