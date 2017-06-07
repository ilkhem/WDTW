"""
Define utility functions to work with for sinkhorn and stdw distances. Make sure that:
    - each fuction supports numpy and cupy for both CPU/GPU support
    - if numpy-cupy generic code is not possible, write different functions for CPU and GPU support
"""
import chainer.cuda as cuda
import numpy as np
from chainer.functions.array.reshape import reshape
from chainer.functions.array.transpose import transpose
from chainer.functions.math.matmul import matmul


def f(x, h):
    xp = cuda.get_array_module(x)
    return xp.reshape(xp.dot(h, xp.reshape(x, (h.shape[1], int(x.size / h.shape[1])))), x.shape)


def xi(x, hx, hy, hz):
    xp = cuda.get_array_module(x)
    return f(xp.transpose(f(xp.transpose(f(xp.transpose(x, [1, 0, 2, 3]),
                                           hy), [2, 0, 1, 3]),
                            hz), [2, 1, 0, 3]),
             hx)


def f_chainer(x, h):
    return reshape(matmul(h, reshape(x, (h.shape[1], int(x.size / h.shape[1])))), x.shape)


def xi_chainer(x, hx, hy, hz):
    res = f_chainer(transpose(f_chainer(transpose(f_chainer(transpose(x, [1, 0, 2, 3]),
                                                            hy), [2, 0, 1, 3]),
                                        hz), [2, 1, 0, 3]),
                    hx)

    return res


def prepare_gradient(x):
    xp = cuda.get_array_module(x)
    x.grad = xp.ones(x.shape, dtype=np.float64)
