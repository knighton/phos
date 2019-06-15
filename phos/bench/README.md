## bench

Filesystem
----------

```
(lab dir)/
    settings.json
    model/
        (model name)/
            status.txt
            run/
                0/
                1/
                ...
        ...
```

```            
(run id)/
    blurb/
        0.json
        1.json
        ...
    status.txt
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
    settings.json
```
