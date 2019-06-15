## bench

```
benchmark_name/
    model/
        model_name/
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
        ...
    settings.json
```

* **Benchmark directory**
    * **model/** stores each training run by model name.
    * **settings.json** stores this benchmark's training configuration.
* **Model directory**
    * **blurb/** stores model blurbs dumped at the begining of each epoch.
    * **done.txt** is touched when the run is finished.
    * **result/** stores training results over batches.
* **Results directory**
    * **Resolution** Batches per saved statistic (takes the mean).
        * **1** Each batch (usually impractical).
        * **10** Groups of batches (high resolution).
        * **100** Mini-epochs (low resolution).
    * **Split** Dataset split.
        * **train** Training split.
        * **val** Validation split.
    * **Attribute** Collected statistic type.
        * **loss** Cross-entropy.
        * **accuracy** Classification accuracy.
        * **forward** Forward time.
        * **backward** Backward time.
