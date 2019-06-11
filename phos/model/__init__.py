import sys

from .baseline import *
  

def each_model():
    from .base.builder import ModelBuilder
    this = sys.modules[__name__]
    for k in sorted(dir(this)):
        v = getattr(this, k)
        if isinstance(v, ModelBuilder):
            yield k, v


def each_model_name():
    for k, _ in each_model():
        yield k


def get_model(model_name):
    this = sys.modules[__name__]
    return getattr(this, model_name)
