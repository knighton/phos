from itertools import product
  

class LabSettings(object):
    """
    Collection of model training options.

    This class:
        (a) strictly validates the user-producted JSON config, and
        (b) allows iterating over all the combinations of hyperparameters for grid
            search.
    """

    # List of (key, value type).
    option_specs = (
        ('dataset', str),
        ('loader_cpus', int),
        ('use_cuda', bool),
        ('blocks_per_stage', int),
        ('block_channels', int),
        ('optimizer', str),
    )

    # List of (key, value type).
    fixed_specs = (
        ('num_epochs', int),
        ('train_batches_per_epoch', int),
        ('val_batches_per_epoch', int),
        ('batch_size', int),
        ('blurb_percentiles', int),
    )

    @classmethod
    def normalize_option_values(cls, x, klass):
        """
        Normalize one or more option value to a list of items of the given type.
        """
        if isinstance(x, klass):
            x = [x]
        else:
            assert isinstance(x, list)
            for item in x:
                assert isinstance(item, klass)
            assert len(set(x)) == len(x)
        return x

    @classmethod
    def normalize_options(cls, x, option_specs):
        """
        Normalize and check all the option lists.
        """
        keys = set(map(lambda pair: pair[0], option_specs))
        for key in x:
            assert key in keys
        k2vv = {}
        vvv = []
        for key, klass in option_specs:
            values = cls.normalize_option_values(x[key], klass)
            k2vv[key] = values
            vvv.append(values)
        return k2vv, vvv

    @classmethod
    def validate_fixed(cls, x, fixed_specs):
        """
        Check all the fixed settings.
        """
        keys = set(map(lambda pair: pair[0], fixed_specs))
        for key in x:
            assert key in keys
        for key, klass in fixed_specs:
            value = x[key]
            assert isinstance(value, klass)
        return x

    def __init__(self, x):
        """
        Initialize given JSON dict settings.
        """
        self.options_k2vv, self.options_vvv = \
            self.normalize_options(x['options'], self.option_specs)
        self.fixed_k2v = self.validate_fixed(x['fixed'], self.fixed_specs)

    def dump(self):
        """
        Dump to JSON dict.
        """
        return {
            'options': self.options_k2vv,
            'fixed': self.fixed_k2v,
        }

    def each_experiment(self):
        """
        Iterate over each possible combination of settings.
        """
        for options_vv in product(*self.options_vvv):
            k2v = {}
            for (key, _), value in zip(self.option_specs, options_vv):
                k2v[key] = value
            k2v.update(self.fixed_k2v)
            yield k2v
