import itertools
import json
import numpy as np
import os


_RESOLUTIONS = 1, 10, 100, 1000
_SPLITS = 'train', 'val'
_ATTRIBUTES = 'loss', 'accuracy', 'forward', 'backward'


_STATIC_QUERY_OPTIONS = {
    'resolution': _RESOLUTIONS,
    'split': _SPLITS,
    'attribute': _ATTRIBUTES,
}


def get_settings(bench_dir):
    f = '%s/settings.json' % bench_dir
    return json.load(open(f))


def get_done_models(bench_dir):
    d = '%s/model/' % bench_dir
    models = []
    for model in sorted(os.listdir(d)):
        f = '%s/model/%s/done.txt' % (bench_dir, model)
        if os.path.exists(f):
            models.append(model)
    return models


def get_static_query_options():
    return _STATIC_QUERY_OPTIONS


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


def query_results(bench_dir, x):
    models_done = get_done_models(bench_dir)
    models = get(x, 'model', models_done)
    models = sorted(filter(lambda s: s in models_done, models))

    resolutions = get(x, 'resolution', _RESOLUTIONS)
    splits = get(x, 'split', _SPLITS)
    attributes = get(x, 'attribute', _ATTRIBUTES)

    pairs = []
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

        pairs.append((k, v))

    return pairs


def get_blurb(bench_dir, model, epoch):
    f = '%s/model/%s/blurb/%d.json' % (bench_dir, model, epoch)
    return json.load(open(f))
