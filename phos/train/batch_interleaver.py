import numpy as np


def each(xx):
    """
    Make a generator out of a sequence.
    """
    for x in xx:
        yield x


def next_sample(each_split_batch, use_cuda):
    """
    Pull the next sample out of a split and optionally put it on the GPU.
    """
    x, y_true = next(each_split_batch)
    if use_cuda:
        x = x.cuda()
        y_true = y_true.cuda()
    return x, y_true


def each_batch(train_loader, val_loader, num_train_batches, num_val_batches,
               use_cuda):
    """
    Iterate training and validation sets interspersed.
    """
    each_train_batch = each(train_loader)
    each_val_batch = each(val_loader)

    splits = [1] * num_train_batches + [0] * num_val_batches
    np.random.shuffle(splits)

    for batch_id, is_training in enumerate(splits):
        if is_training:
            each_split_batch = each_train_batch
        else:
            each_split_batch = each_val_batch
        x, y_true = next_sample(each_split_batch, use_cuda)
        yield batch_id, is_training, x, y_true
