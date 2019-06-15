class Spy(object):
    """
    Callbacks doing arbitrary things surrounding model training.
    """

    def on_fit_begin(self, trainer):
        pass

    def on_fit_on_epoch_begin(self, trainer):
        pass

    def on_fit_on_batch_begin(self, trainer):
        pass

    def on_train_on_batch_begin(self, trainer):
        pass

    def on_train_on_batch_end(self, trainer, *args):
        pass

    def on_validate_on_batch_begin(self, trainer):
        pass

    def on_validate_on_batch_end(self, trainer, *args):
        pass

    def on_fit_on_batch_end(self, trainer):
        pass

    def on_fit_on_epoch_end(self, trainer):
        pass

    def on_fit_end(self, trainer):
        pass
