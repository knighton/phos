import requests
from time import time

from .spy import Spy
from .util.batch_result_buffer import BatchResultBuffer


class Uploader(Spy):
    """
    POSTs training results to a server.
    """

    def __init__(self, blurb_upload_path, blurb_percentiles, results_upload_path,
                 results_upload_interval):
        """
        Initialize with upload paths and other configuration.
        """
        # Config.
        self.blurb_upload_path = blurb_upload_path
        self.blurb_percentiles = blurb_percentiles
        self.results_upload_path = results_upload_path
        self.results_upload_interval = results_upload_interval

        # When results were last uploaded.
        self.last_sent_results = None

        # Buffer storing fit results (loss/acc/time) before they are uploaded.
        self.buffer = BatchResultBuffer()

    def on_epoch_begin(self, model):
        """
        Send a model blurb.
        """
        if not self.blurb_upload_path:
            return
        x = model.blurb(self.blurb_percentiles)
        requests.post(self.blurb_upload_path, x)

    def flush_results(self):
        """
        Flush the buffered results.
        """
        if not self.results_upload_path:
            return
        then = time()
        x = self.buffer.dump()
        requests.post(self.results_upload_path, x)
        self.buffer.clear()
        self.last_sent_results = then

    def maybe_flush_results(self):
        """
        Flush the buffered results if they're getting stale.
        """
        if self.last_sent_results is None:
            self.last_sent_results = time()
        elif self.last_sent_results + self.results_upload_interval <= time():
            self.flush_results()

    def on_train_on_batch_end(self, *args):
        """
        Cache/send the results of one training batch.
        """
        if not self.results_upload_path:
            return
        self.buffer.train.add(*args)
        self.maybe_flush_results()

    def on_validate_on_batch_end(self, *args):
        """
        Cache/send the results of one validation batch.
        """
        if not self.results_upload_path:
            return
        self.buffer.val.add(*args)
        self.maybe_flush_results()

    def on_epoch_end(self, epoch, batch):
        """
        Force flush the remaining results (in case this is the last epoch).
        """
        self.flush_results()
