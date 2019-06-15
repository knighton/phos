## bench

Filesystem
----------

```
(benchmark dir)/
    settings.json
    model/
        (model name)/
            blurb/
                0.json
                1.json
                ...
            done.txt
            result/
                1/
                    train_loss.npy
                    train_accuracy.npy
                    train_forward.npy
                    train_backward.npy
                    val_loss.npy
                    val_accuracy.npy
                    val_forward.npy
                10/
                    "
                100/
                    "
```

* `blurb/` stores model blurbs dumped at the begining of each epoch.

* Resolution (batches per statistic)
  * **1** Each batch (usually impractical).
  * **10** Groups of batches (high resolution).
  * **100** Mini-epochs (low resolution).

* Split
  * **train** Training split.
  * **val** Validation split.

* Attribute
  * **loss** Cross-entropy.
  * **accuracy** Classification accuracy.
  * **forward** Forward time.
  * **backward** Backward time.
