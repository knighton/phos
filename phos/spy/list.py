from .spy import Spy
  

class SpyList(Spy):
    """
    A list of spies, called as one.  Recursive.

    This is used to keep the training code tidy.
    """

    def __init__(self, *spies):
        self.spies = []
        for spy in spies:
            assert isinstance(spy, Spy)
            self.spies.append(spy)

    def apply(self, func):
        for spy in self.spies:
            func(spy)

    def on_fit_begin(self, *args, **kwargs):
        self.apply(lambda x: x.on_fit_begin(*args, **kwargs))

    def on_epoch_begin(self, *args, **kwargs):
        self.apply(lambda x: x.on_epoch_begin(*args, **kwargs))

    def on_train_on_batch_begin(self, *args, **kwargs):
        self.apply(lambda x: x.on_train_on_batch_begin(*args, **kwargs))

    def on_train_on_batch_end(self, *args, **kwargs):
        self.apply(lambda x: x.on_train_on_batch_end(*args, **kwargs))

    def on_validate_on_batch_begin(self, *args, **kwargs):
        self.apply(lambda x: x.on_validate_on_batch_begin(*args, **kwargs))

    def on_validate_on_batch_end(self, *args, **kwargs):
        self.apply(lambda x: x.on_validate_on_batch_end(*args, **kwargs))

    def on_epoch_end(self, *args, **kwargs):
        self.apply(lambda x: x.on_epoch_end(*args, **kwargs))

    def on_fit_end(self, *args, **kwargs):
        self.apply(lambda x: x.on_fit_end(*args, **kwargs))
