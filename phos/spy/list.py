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

    def on_fit_begin(self, trainer):
        for spy in self.spies:
            spy.on_fit_begin(trainer)

    def on_fit_on_epoch_begin(self, trainer):
        for spy in self.spies:
            spy.on_fit_on_epoch_begin(trainer)

    def on_fit_on_batch_begin(self, trainer):
        for spy in self.spies:
            spy.on_fit_on_batch_begin(trainer)

    def on_train_on_batch_begin(self, trainer):
        for spy in self.spies:
            spy.on_train_on_batch_begin(trainer)

    def on_train_on_batch_end(self, trainer, *args):
        for spy in self.spies:
            spy.on_train_on_batch_end(trainer, *args)

    def on_validate_on_batch_begin(self, trainer):
        for spy in self.spies:
            spy.on_validate_on_batch_begin(trainer)

    def on_validate_on_batch_end(self, trainer, *args):
        for spy in self.spies:
            spy.on_validate_on_batch_end(trainer, *args)

    def on_fit_on_batch_end(self, trainer):
        for spy in self.spies:
            spy.on_fit_on_batch_end(trainer)

    def on_fit_on_epoch_end(self, trainer):
        for spy in self.spies:
            spy.on_fit_on_epoch_end(trainer)

    def on_fit_end(self, trainer):
        for spy in self.spies:
            spy.on_fit_end(trainer)
