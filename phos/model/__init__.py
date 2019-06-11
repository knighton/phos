from .baseline import *
  

def each_model():
    import sys
    from .base.builder import ModelBuilder
    this = sys.modules[__name__]
    for k in sorted(dir(this)):
        v = getattr(this, k)
        if isinstance(v, ModelBuilder):
            yield k, v
