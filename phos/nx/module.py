import numpy as np
from time import time
from torch import nn
from torch import Tensor

from ..util.sampler import Sampler
from ..util.tensor_form import TensorForm


class Module(nn.Module):
    """
    Phos abstract base module class.

    To subclass, provide your own:
    * forward_inner(x, x_loss) -> y, y_loss
    * summary_inner(num_percentiles) -> dict
    """

    def __init__(self, timer_size=1000, timer_momentum=0.98, timer_dtype=np.float32):
        super().__init__()

        self.nxmod_x_form = None
        self.nxmod_y_form = None

        timer = lambda: Sampler(timer_size, timer_momentum, timer_dtype)
        self.nxmod_train_timer = timer()
        self.nxmod_val_timer = timer()

    def nxmod_check_input(self, x):
        if not isinstance(x, Tensor):
            assert self.nxmod_x_form is None
            return

        if self.nxmod_x_form is None:
            self.nxmod_x_form = TensorForm.of_tensor(x)
        else:
            assert self.nxmod_x_form.accepts(x)

    def nxmod_check_output(self, y):
        if not isinstance(y, Tensor):
            assert self.nxmod_y_form is None
            return

        if self.nxmod_y_form is None:
            self.nxmod_y_form = TensorForm.of_tensor(y)
        else:
            assert self.nxmod_y_form.accepts(y)

    def forward_inner(self, x, x_loss):
        raise NotImplementedError

    def forward(self, x, x_loss=None):
        self.nxmod_check_input(x)

        t = time()
        y, y_loss = self.forward_inner(x, x_loss)
        t = time() - t

        if self.training:
            self.nxmod_train_timer.update(t)
        else:
            self.nxmod_val_timer.update(t)

        self.nxmod_check_output(y)

        return y, y_loss

    def __call__(self, x, x_loss=None):
        return self.forward(x, x_loss)

    def summary_inner(self, num_percentiles):
        raise NotImplementedError

    def summary(self, num_percentiles):
        if self.nxmod_x_form:
            x_form = self.nxmod_x_form.dump()
        else:
            x_form = None

        if self.nxmod_y_form:
            y_form = self.nxmod_y_form.dump()
        else:
            y_form = None

        train_timer = self.nxmod_train_timer.summary(num_percentiles)
        val_timer = self.nxmod_val_timer.summary(num_percentiles)

        body = self.summary_inner(num_percentiles)

        return {
            'type': self.__class__.__name__,
            'x_form': x_form,
            'y_form': y_form,
            'train_timer': train_timer,
            'val_timer': val_timer,
            'body': body,
        }
