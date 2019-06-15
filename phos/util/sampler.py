import numpy as np
  
from .summary_statistics import numpy_to_summary


class Sampler(object):
    """
    Collection of probabilistically updated values.

    Used to compte online summary statistics easily.
    """

    def __init__(self, size=1000, momentum=0.01, dtype=np.float32):
        """
        Initialize to an empty array of sample slots.

        Each update will overwrite a decreasing fraction of the sample until
        reaching the base rate.

        Moving summary statistics are just summary statistics on the values that
        have survived in the sample.
        """
        assert isinstance(size, int)
        assert 1 <= size
        assert 0 < momentum < 1

        self.size = size
        self.momentum = momentum
        self.dtype = dtype

        self.sample = np.zeros(size, dtype)
        self.num_updates = 0

    @classmethod
    def compute_num_overwrites(cls, size, momentum, num_updates):
        """
        Compute the number of sample slots to overwrite with the new value.

        You start by ovewriting proportionally to how many values you have seen
        so far (all the values, then half of them, then one third, then ...).
        The sample gradually stabilizes as these overwrites continue to get
        sparser until reaching the base ovewrwrite rate (say, one percent).

        This logic is separated out here in order to make subclassing easier.
        """
        rate = 1 - momentum
        cross_over = int(1 / rate)
        if num_updates < cross_over:
            count = int(size / (num_updates + 1))
        else:
            count = int(size * rate)
        return count

    def update(self, x):
        """
        Receive one new value.
        """
        num_overwrites = self.compute_num_overwrites(self.size, self.momentum,
                                                     self.num_updates)
        victim_indices = np.random.choice(self.size, num_overwrites, False)
        self.sample[victim_indices] = x
        self.num_updates += 1

    def update_many(self, xx):
        """
        Receive multiple new values.
        """
        for x in xx:
            self.update(x)

    def summary(self, num_percentiles=20):
        """
        Get moving summary statistics of the values.
        """
        return numpy_to_summary(self.sample, num_percentiles)
