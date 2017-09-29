import chainer.cuda as cuda
import numpy as np


def prepare_gradient(x):
    xp = cuda.get_array_module(x)
    x.grad = xp.ones(x.shape, dtype=np.float64)


# deprecated, for numpy only implementation

def f(x, h):
    return x.dot(h)


def xi(x, hx, hy, hz):
    return (((x.dot(hz)).transpose([0, 3, 1, 2])
             .dot(hy)).transpose([0, 3, 1, 2]).dot(hx)).transpose([0, 3, 1, 2])
