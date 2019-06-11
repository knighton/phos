class SplitBatchResultBuffer(object):
    """
    One split of BatchResultBuffer.

    Validation batches do not have a backward pass, thus the handling of 'backward'.
    """

    def __init__(self):
        self.loss = []
        self.accuracy = []
        self.forward = []
        self.backward = []

    def add(self, loss, accuracy, forward, backward=None):
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.forward.append(forward)
        if backward is not None:
            self.backward.append(backward)

    def dump(self):
        x = {
            'loss': self.loss,
            'accuracy': self.accuracy,
            'forward': self.forward,
        }
        if self.backward:
            x['backward'] = self.backward
        return x

    def clear(self):
        self.loss = []
        self.accuracy = []
        self.forward = []
        self.backward = []


class BatchResultBuffer(object):
    """
    Buffers training results for sending.
    """

    def __init__(self):
        self.train = SplitBatchResultBuffer()
        self.val = SplitBatchResultBuffer()

    def dump(self):
        return {
            'train': self.train.dump(),
            'val': self.val.dump(),
        }

    def clear(self):
        self.train.clear()
        self.val.clear()
