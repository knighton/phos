from itertools import product


class LabSettings(object):
    """
    Collection of model training options.  Some are grid searched, some are solo.
    """

    @classmethod
    def normalize_values(cls, x):
        """
        Normalize a grid parameter value list.

        Usually just one option, making it cleaner to define sans list, but they are
        all lists for the product().
        """
        if isinstance(x, list):
            return sorted(x)
        else:
            return [x]

    def __init__(self, x):
        """
        Validates and indexes a lab settings JSON object.
        """
        # Verify section names.
        assert tuple(sorted(x)) == ('fixed', 'grid')

        # Mapping of grid key -> values.
        #
        # For looking up the possible values of a field.
        self.grid = {}
        pairs = []
        for key, values in x['grid'].items():
            values = self.normalize_values(values)
            self.grid[key] = values
            pairs.append((key, values))
        pairs.sort()

        # Pair of lists: keys and lists of possible values.
        #
        # For iterating the grid fields in a fixed order.
        self.grid_keys = []
        self.grid_value_lists = []
        for key, values in pairs:
            self.grid_keys.append(key)
            self.grid_value_lists.append(values)

        # A list of dicts, each mapping key -> selected value.
        #
        # There is a dict for every possible combination of param values.  We fit a
        # model on each one.  Run IDs <-> these param dicts.  We typically refer to
        # runs by their run IDs for simplicity.
        self.grid_combos = []
        for selected_grid_values in product(*self.grid_value_lists):
            combo = dict(zip(self.grid_keys, selected_grid_values))
            self.grid_combos.append(combo)

        # Number of grid combos.
        self.num_grid_combos = len(self.grid_combos)

        # Mapping of key -> value.
        self.fixed = x['fixed']

    def query_runs(self, x):
        """
        Convert query -> run IDs.

        A query is a collection of optional grid value filters.  These filters
        together either accept or reject each run.
        """
        # Extract just the list of query filters that correspond to grid hyperparams
        # (as opposed to filtering by model name, resolution, split, attribute, etc).
        # Normalize expecteds into grid value lists.
        filters = []
        for k, v in x.items():
            if k not in self.grid:
                continue
            if v is None:
                vv = self.grid[k]
            elif isinstance(v, list):
                assert v
                vv = v
            else:
                vv = [v]
            filters.append((k, vv))

        # Now iterate over each combination of parameter values (runs), checking for
        # matches..
        #
        # Could be done better, but will not be the bottleneck for reasonable sizes
        # of hyperparameter grid.
        run_ids = []
        for run_id, grid_combo in enumerate(self.grid_combos):
            ok = True
            for k, vv in filters:
                if grid_combo[k] not in vv:
                    ok = False
                    break
            run_ids.append(run_id)

        return run_ids
