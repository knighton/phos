import os

from .spy import Spy


class Checkpoint(Spy):
    """
    Checkpoints/resumes from checkpoint model training.

    Warning: this must come before any other Spies, because it sets the epoch during
    on_fit_begin() if a checkpoint exists.
    """

    def __init__(self, path):
        self.path = path

    def on_fit_begin(self, trainer):
        if os.path.exists(self.path):
            trainer.load_checkpoint(self.path)

    def on_fit_on_epoch_begin(self, trainer):
        trainer.save_checkpoint(self.path)

    def on_fit_end(self, trainer):
        trainer.save_checkpoint(self.path)
