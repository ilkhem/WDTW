"""
Define utility functions to work with for sinkhorn and stdw distances. Make sure that:
    - each fuction supports numpy and cupy for both CPU/GPU support
    - if numpy-cupy generic code is not possible, write different functions for CPU and GPU support
"""
import numpy as np
import chainer.cuda as cuda
from chainer.functions.array.transpose import transpose
from chainer.functions.array.reshape import reshape
from chainer.functions.math.matmul import matmul
from chainer import Function


def f(x, h):
    """ 
    detect whether arrays are on CPU or GPU. If CUDA and thus CuPy is not installed/configured correctly, x and h are
    assumed to be numpy.ndarrays, and thus numpy is used. Else, cupy.get_array_module will retrieve the correct type.
    """
    xp = cuda.get_array_module(x)
    return xp.reshape(xp.dot(h, xp.reshape(x, (h.shape[1], int(x.size / h.shape[1])))), x.shape)


def xi(x, hx, hy, hz):
    """ 
    detect whether arrays are on CPU or GPU. If CUDA and thus CuPy is not installed/configured correctly, x and h are
    assumed to be numpy.ndarrays, and thus numpy is used. Else, cupy.get_array_module will retrieve the correct type.
    """
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


class Xi2(Function):
    """implement the kernel convolution. define a new dot24"""

    def __init__(self, hx, hy, hz):
        self.hx = hx
        self.hy = hy
        self.hz = hz

    def forward(self, inputs):
        x, = inputs
        xp = cuda.get_array_module(x)
        y = xp.dot(self.hy, x.transpose([1, 0, 2, 3]))
        y = xp.dot(self.hz, y.transpose([2, 0, 1, 3]))
        y = xp.dot(self.hx, y.transpose([2, 1, 0, 3]))
        return y,

    def backward(self, inputs, grad_outputs):
        go, = grad_outputs
        x, = input

class Dot4d2d(Function):
    def forward(self, inputs):
        # print('Dot4d2d forward')
        a, b = inputs
        assert a.shape[-1] == b.shape[0]
        xp = cuda.get_array_module(a)
        return xp.dot(a, b),

    def backward(self, inputs, grad_outputs):
        # print('Dot4d2d backward')
        go, = grad_outputs
        # print('go', go.shape)
        a, b = inputs
        # print('a', a.shape)
        # print('b', b.shape)
        assert a.shape[-1] == b.shape[0]
        xp = cuda.get_array_module(a)
        ga = xp.dot(go, xp.transpose(b))
        # print('ga', ga.shape)
        return ga, xp.zeros_like(b, dtype=np.float64)


def dot42(a, b):
    return Dot4d2d()(a, b)
