import numpy as np
  

def sorted_to_median(x):
    n = len(x)
    if n % 2:
        x = x[n // 2]
    else:
        a = x[n // 2 - 1]
        b = x[n // 2]
        x = (a + b) / 2
    return float(x)


def sorted_to_percentiles(x, num_percentiles):
    count = len(x)
    percentiles = []
    for i in range(num_percentiles):
        index = count * i // num_percentiles
        pct = x[index]
        percentiles.append(pct)
    percentiles.append(x[count - 1])
    return list(map(float, percentiles))


def sorted_to_summary(x, num_percentiles):
    mean = float(np.mean(x))
    std = float(np.std(x))
    minimum = float(x[0])
    maximum = float(x[-1])
    median = sorted_to_median(x)
    percentiles = sorted_to_percentiles(x, num_percentiles)
    return {
        'mean': mean,
        'std': std,
        'min': minimum,
        'max': maximum,
        'median': median,
        'percentiles': percentiles,
    }


def numpy_to_summary(x, num_percentiles, dtype=None):
    x = x.flatten()
    if dtype:
        x = x.astype(dtype)
    x = sorted(x)
    return sorted_to_summary(x, num_percentiles)


def tensor_to_summary(x, num_percentiles, dtype=None):
    x = x.detach()
    if x.is_cuda:
        x = x.cpu()
    x = x.numpy()
    return numpy_to_summary(x, num_percentiles, dtype)
