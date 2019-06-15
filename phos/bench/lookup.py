import itertools
import json
import numpy as np
import os


_RESOLUTIONS = 1, 10, 100, 1000
_SPLITS = 'train', 'val'
_ATTRIBUTES = 'loss', 'accuracy', 'forward', 'backward'


_STATIC_QUERY_OPTIONS = {
    'split': _SPLITS,
    'attribute': _ATTRIBUTES,
}


def get_static_query_options():
    return _STATIC_QUERY_OPTIONS


def get_settings(bench_dir):
    f = '%s/settings.json' % bench_dir
    return json.load(open(f))


def get_models(bench_dir):
    d = '%s/model/' % bench_dir
    return sorted(os.listdir(d))


def get_done_models(bench_dir):
    d = '%s/model/' % bench_dir
    models = []
    for model in sorted(os.listdir(d)):
        f = '%s/model/%s/done.txt' % (bench_dir, model)
        if os.path.exists(f):
            models.append(model)
    return models


def get_query_param(k2v, k, vv):
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


def each_query_result(models, splits, attributes):
    for split in splits:
        for model in models:
            for attribute in attributes:
                if split == 'val' and attribute == 'backward':
                    continue
                yield model, split, attribute


def query_stats(bench_dir, x):
    models_done = get_models(bench_dir)
    models = get_query_param(x, 'model', models_done)
    models = sorted(filter(lambda s: s in models_done, models))

    splits = get_query_param(x, 'split', _SPLITS)
    attributes = get_query_param(x, 'attribute', _ATTRIBUTES)

    pairs = []
    for model, split, attribute in each_query_result(models, splits, attributes):
        k = {
            'model': model,
            'split': split,
            'attribute': attribute,
        }

        f = '%s/model/%s/stat/epoch_%s_%s.npy' % (bench_dir, model, split, attribute)
        if os.path.exists(f):
            x = np.fromfile(f, np.float32)
        else:
            d = '%s/model/%s/stat/epoch/' % (bench_dir, model)
            ss = os.listdir(d)
            epochs = sorted(map(int, ss))
            assert epochs == list(range(len(epochs)))
            xx = []
            for i in epochs:
                f = '%s/model/%s/stat/epoch/%d/%s_%s.npy' % \
                    (bench_dir, model, i, split, attribute)
                x = np.fromfile(f, np.float32)
                xx.append(x.mean())
            x = np.array(xx, np.float32)
        v = x.tolist()

        pairs.append((k, v))

    return pairs


def get_summary(bench_dir, model, epoch):
    f = '%s/model/%s/summary/%d.json' % (bench_dir, model, epoch)
    if not os.path.exists(f):
        return None
    return json.load(open(f))
