from .spy import Spy
from .util.batch_result_buffer import BatchResultBuffer


mean = lambda xx: sum(xx) / len(xx)


class LogRowPerEpoch(Spy):
    """
    Writes a results row after each epoch, forming a happy little ASCII table.

    Format:
        * epoch
        * loss (train, val, difference)
        * accuracy % (train, val, difference)
        * time ms (train forward, train backward, val forward)
    """

    def __init__(self, filename=None):
        if filename is None:
            filename = '/dev/stdout'
        self.filename = filename

    def on_fit_on_epoch_begin(self, trainer):
        """
        Handle epoch begin.
        """
        self.buffer = BatchResultBuffer()

    def on_train_on_batch_end(self, trainer, *args):
        """
        Handle the results of one training batch.
        """
        self.buffer.train.add(*args)

    def on_validate_on_batch_end(self, trainer, *args):
        """
        Handle the results of one validation batch.
        """
        self.buffer.val.add(*args)

    def on_fit_on_epoch_end(self, trainer):
        """
        Handle epoch end.
        """
        x = self.buffer.dump()

        t = x['train']
        t_loss = mean(t['loss'])
        t_acc = mean(t['accuracy']) * 100
        t_fwd = mean(t['forward'])
        t_fwd = int(t_fwd * 1000)
        t_bwd = mean(t['backward'])
        t_bwd = int(t_bwd * 1000)

        v = x['val']
        v_loss = mean(v['loss'])
        v_acc = mean(v['accuracy']) * 100
        v_fwd = mean(v['forward'])
        v_fwd = int(v_fwd * 1000)

        d_loss = t_loss - v_loss
        d_acc = t_acc - v_acc

        epoch_part = '%6d' % trainer.epoch
        loss_part = '%6.2f %6.2f : %6.2f' % (t_loss, v_loss, d_loss)
        acc_part = '%6.2f%% %6.2f%% : %6.2f%%' % (t_acc, v_acc, d_acc)
        time_part = '%4d %4d : %4d' % (t_fwd, t_bwd, v_fwd)
        parts = epoch_part, loss_part, acc_part, time_part
        line = ' | '.join(parts)
        line = '| %s |\n' % line

        with open(self.filename, 'w') as out:
            out.write(line)
