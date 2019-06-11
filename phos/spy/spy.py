class Spy(object):
    """
    Callbacks doing arbitrary things surrounding model training.
    """

    def on_fit_begin(self):
        """
        Handle fit begin.
        """
        pass

    def on_epoch_begin(self, epoch, model):
        """
        Handle epoch begin.
        """
        pass

    def on_train_on_batch_begin(self, epoch, batch):
        """
        Handle train on batch begin.
        """
        pass

    def on_train_on_batch_end(self, epoch, batch, loss, accuracy, forward, backward):
        """
        Handle the results of one training batch.
        """
        pass

    def on_validate_on_batch_begin(self, epoch, batch):
        """
        Handle validate on batch begin.
        """
        pass

    def on_validate_on_batch_end(self, epoch, batch, loss, accuracy, forward,
                                 backward):
        """
        Handle the results of one validation batch.
        """
        pass

    def on_epoch_end(self, epoch, model):
        """
        Handle epoch end.
        """
        pass

    def on_fit_end(self):
        """
        Handle fit end.
        """
        pass
