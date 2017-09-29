import numpy as np
from chainer import Function
from chainer import cuda
from chainer.utils import type_check


class Conv(Function):
    def __init__(self, hx, hy, hz):
        self.hx = hx
        self.hy = hy
        self.hz = hz

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types
        type_check.expect(x_type.ndim == 4)
        type_check.expect(x_type.dtype.kind == 'f')
        type_check.expect(x_type.shape[1:] ==
                          (self.hx.shape[0], self.hy.shape[0], self.hz.shape[0]))

    @property
    def label(self):
        return 'Convolution'

    def forward(self, inputs):
        x, = inputs
        return (((x
                  .dot(self.hz)).transpose([0, 3, 1, 2])
                 .dot(self.hy)).transpose([0, 3, 1, 2])
                .dot(self.hx)).transpose([0, 3, 1, 2]),

    def backward(self, inputs, grad_outputs):
        go, = grad_outputs
        x, = inputs
        return (((go
                  .dot(self.hz)).transpose([0, 3, 1, 2])
                 .dot(self.hy)).transpose([0, 3, 1, 2])
                .dot(self.hx)).transpose([0, 3, 1, 2]),


def convol(x, hx, hy, hz):
    return Conv(hx, hy, hz)(x)


class Dot4d2d(Function):
    def __init__(self, b):
        self.b = b

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        a_type, = in_types
        type_check.expect(a_type.ndim == 4)
        type_check.expect(a_type.dtype.kind == 'f')

    @property
    def label(self):
        return 'Dot4D2D'

    def forward(self, inputs):
        a, = inputs
        assert a.shape[0] == self.b.shape[0]  # correspondance of dimension for application of dot
        return (a.transpose([1, 2, 3, 0]).dot(self.b)).transpose([3, 0, 1, 2]),

    def backward(self, inputs, grad_outputs):
        go, = grad_outputs
        a, = inputs
        assert a.shape[0] == self.b.shape[0]
        return (go.transpose([1, 2, 3, 0])).dot(self.b.transpose()).transpose([3, 0, 1, 2]),


def dot42(a, b):
    return Dot4d2d(b)(a)


def _prepare_gradient(x):
    xp = cuda.get_array_module(x)
    x.grad = xp.ones(x.shape, dtype=np.float64)
