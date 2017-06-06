import numpy as np


def generate_data(d1=10, d2=10, d3=10, m=5, n=4):
    At_ = np.random.rand(d1, d2, d3, m).astype(np.float64)
    At = At_ / np.sum(At_, axis=(0, 1, 2))
    Bt_ = np.random.rand(d1, d2, d3, n).astype(np.float64)
    Bt = Bt_ / np.sum(Bt_, axis=(0, 1, 2))

    return At, Bt


def generate_multiple(d1=10, d2=10, d3=10, m=5, nb=5):
    At = np.random.rand(d1, d2, d3, m, nb).astype(np.float64)
    At = At / np.sum(At, axis=(0, 1, 2))
    return At


def generate_single(d1=10, d2=10, d3=10):
    a, b = generate_data(d1, d2, d3, 1, 1)
    # return a.squeeze(), b.squeeze()
    return a, b


def generate_nice(d1=10, d2=10, d3=10, m=4, nb=3, thresh=1e-3):
    y = generate_multiple(d1, d2, d3, m, nb)
    y[y < thresh] = 0
    y = y/np.sum(y, axis=(0,1,2))
    print(np.sum(y, axis=(0,1,2)))
    return y

