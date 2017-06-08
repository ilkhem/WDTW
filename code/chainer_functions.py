import chainer.cuda as cuda
import numpy as np
from chainer import Function
from chainer.utils import type_check


class Conv(Function):
    def __init__(self, h, axes):
        self.h = h
        self.axes = axes
        if axes is None:
            self.axes = [0, 1, 2, 3]

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types
        type_check.expect(x_type.ndim == 4)
        if self.axes:
            type_check.expect(x_type.ndim == len(self.axes))
        type_check.expect(x_type.dtype.kind == 'f')

    @property
    def label(self):
        return 'Convolution'

    def forward(self, inputs):
        x, = inputs
        xp = cuda.get_array_module(x)
        assert x.size % self.h.shape[1] == 0
        x_shape = x.shape
        return xp.reshape(xp.dot(self.h,
                                 xp.reshape(x.transpose(self.axes),
                                            (self.h.shape[1], int(x.size / self.h.shape[1])))),
                          x.transpose(self.axes).shape),

    def backward(self, inputs, grad_outputs):
        go, = grad_outputs
        x, = inputs
        xp = cuda.get_array_module(x)
        # assert x.shape == go.shape
        # print(go.shape)
        inv_axes = self.axes
        if self.axes:
            axes = [ax % len(self.axes) for ax in self.axes]
            inv_axes = list(np.argsort(axes))
        print(self.axes)
        print(inv_axes)
        return xp.reshape(xp.dot(self.h.transpose(),
                                 xp.reshape(go,
                                            (self.h.shape[1], int(go.size / self.h.shape[1])))),
                          x.transpose(self.axes).shape).transpose(inv_axes),


def conv(x, h, axes=None):
    return Conv(h, axes)(x)


def kernel_conv(x, hx, hy, hz):
    return conv(conv(conv(x, hy, [1, 0, 2, 3]), hz, [2, 0, 1, 3]), hx, [2, 1, 0, 3])


class Dot4d2d(Function):
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        a_type, b_type = in_types
        type_check.expect(a_type.ndim == 4)
        type_check.expect(b_type.ndim == 2)
        type_check.expect(
            a_type.dtype.kind == 'f',
            a_type.dtype == b_type.dtype
        )

    @property
    def label(self):
        return 'Dot4D2D'

    def forward(self, inputs):
        # print('Dot4d2d forward')
        a, b = inputs
        assert a.shape[-1] == b.shape[0]
        return a.dot(b),

    def backward(self, inputs, grad_outputs):
        # print('Dot4d2d backward')
        go, = grad_outputs
        # print('go', go.shape)
        a, b = inputs
        # print('a', a.shape)
        # print('b', b.shape)
        assert a.shape[-1] == b.shape[0]
        xp = cuda.get_array_module(a)
        ga = go.dot(b.transpose())
        # print('ga', ga.shape)
        return ga, xp.zeros_like(b, dtype=np.float64)


def dot42(a, b):
    return Dot4d2d()(a, b)
