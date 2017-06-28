"""
Compute Sinkhorn distances between 2 time series of shapes
Each shape is of dimension d1*d2*d3, lives in R^3
"""

import numpy as np
from chainer import Variable
from chainer import cuda
from chainer.functions.math.sum import sum

from _utils import xi, prepare_gradient
from chainer_functions import dot42, kernel_conv


def _kernels(xp, d1, d2, d3, mu=0.5, p_exp=2, min_thresh=1e-150):
    X1, Y1 = xp.meshgrid(xp.arange(0, d1), xp.arange(0, d1))
    X2, Y2 = xp.meshgrid(xp.arange(0, d2), xp.arange(0, d2))
    X3, Y3 = xp.meshgrid(xp.arange(0, d3), xp.arange(0, d3))

    hx = xp.exp(-xp.power(xp.abs(X1 - Y1), p_exp) / (2 * mu ** p_exp))
    hx[hx < min_thresh] = min_thresh
    hy = xp.exp(-xp.power(xp.abs(X2 - Y2), p_exp) / (2 * mu ** p_exp))
    hy[hy < min_thresh] = min_thresh
    hz = xp.exp(-xp.power(xp.abs(X3 - Y3), p_exp) / (2 * mu ** p_exp))
    hz[hz < min_thresh] = min_thresh
    kx = xp.multiply(hx, xp.power(xp.abs(X1 - Y1), p_exp))
    kx[kx < min_thresh] = min_thresh
    ky = xp.multiply(hy, xp.power(xp.abs(X2 - Y2), p_exp))
    ky[ky < min_thresh] = min_thresh
    kz = xp.multiply(hz, xp.power(xp.abs(X3 - Y3), p_exp))
    kz[kz < min_thresh] = min_thresh

    return hx, hy, hz, kx, ky, kz


def sinkhorn(a, b, n_iter=100, mu=0.45, tau=1, min_thresh=1e-100, p_exp=2):
    """"""

    xp = cuda.get_array_module(a)

    assert a.shape[:3] == b.shape[:3]  # make sure a and b are of same dimensions.
    d1, d2, d3 = a.shape[:3]

    if a.ndim == 3 and b.ndim == 4:
        n = b.shape[3]
        m = 1
        s = n
        A = xp.concatenate([a.reshape((*a.shape, 1))] * n, axis=-1)
        B = b
    elif a.ndim == 3 and b.ndim == 3:
        m = 1
        n = 1
        A = a.reshape((*a.shape, 1))
        B = b.reshape((*b.shape, 1))
        s = 1
    elif a.ndim == 4 and b.ndim == 4:
        m = a.shape[3]
        n = b.shape[3]
        s = m * n
        A = xp.repeat(a, n, axis=-1)
        B = xp.concatenate([b] * m, axis=-1)
    else:
        raise Exception('check input dimensions')

    hx, hy, hz, kx, ky, kz = _kernels(xp, d1, d2, d3, mu=mu, p_exp=p_exp, min_thresh=min_thresh)

    u = xp.ones((d1, d2, d3, s)) / (d1 * d2 * d3 * s)
    for i in range(n_iter):
        u = xp.divide(A, xi(xp.divide(B, xi(u, hx, hy, hz)) ** tau, hx, hy, hz)) ** tau
        if i % 5 == 0:
            print('iteration %d: Umax = %f' % (i, xp.max(u)))
    v = xp.divide(B, xi(u, hx, hy, hz)) ** tau

    v_tilde = xi(v, kx, hy, hz) + xi(v, hx, ky, hz) + xi(v, hx, hy, kz)

    d = xp.sum(xp.multiply(u, v_tilde), axis=(0, 1, 2))

    return d.reshape((m, n))


def sinkhorn_chainer(a, b, n_iter=50, mu=0.45, tau=1, min_thresh=1e-100, p_exp=2):
    """
    a, b: instances of chainer.Variable
    This is the same as sinkhorn routine using Chainer to be able to compute gradient with respect to a.
    """

    xp = cuda.get_array_module(a)

    assert a.shape[:3] == b.shape[:3]  # make sure a and b are of same dimensions.
    d1, d2, d3 = a.shape[:3]

    if a.ndim == 3 and b.ndim == 4:
        n = b.shape[3]
        m = 1
        a = a.reshape((*a.shape, 1))
        e = Variable(xp.concatenate([xp.eye(a.shape[-1], dtype=np.float64)] * n, axis=-1))
        A = dot42(a, e)
        B = b
    elif a.ndim == 3 and b.ndim == 3:
        m = 1
        n = 1
        A = a.reshape((*a.shape, 1))
        B = b.reshape((*b.shape, 1))
    elif a.ndim == 4 and b.ndim == 4:
        m = a.shape[3]
        n = b.shape[3]
        c = Variable(xp.repeat(xp.eye(a.shape[-1], dtype=np.float64), n, axis=-1))
        e = Variable(xp.concatenate([xp.eye(b.shape[-1], dtype=np.float64)] * m, axis=-1))
        A = dot42(a, c)
        B = dot42(b, e)
    else:
        raise Exception('check input dimensions')
    s = m * n

    hx, hy, hz, kx, ky, kz = _kernels(xp, d1, d2, d3, mu=mu, p_exp=p_exp, min_thresh=min_thresh)

    u = Variable(xp.ones((d1, d2, d3, s), dtype=np.float64))
    for i in range(n_iter):
        if tau == 1:  # reduce memory footprint by not taking the power of tau
            u = A / kernel_conv(B / kernel_conv(u, hx, hy, hz), hx, hy, hz)
        else:
            u = (A / kernel_conv((B / kernel_conv(u, hx, hy, hz)) ** tau, hx, hy, hz)) ** tau
        if i % 5 == 0:
            print('iteration %d: Umax = %f' % (i, xp.max(u.data)))
    if tau == 1:
        v = B / kernel_conv(u, hx, hy, hz)
    else:
        v = (B / kernel_conv(u, hx, hy, hz)) ** tau

    v_tilde = kernel_conv(v, kx, hy, hz) + kernel_conv(v, hx, ky, hz) + kernel_conv(v, hx, hy, kz)

    d = sum(u * v_tilde, axis=(0, 1, 2))

    return d.reshape((m, n))


def sinkhorn_fb(a, b, **kwargs):
    xp = cuda.get_array_module(a)
    m = 1
    if a.ndim == 4:
        m = a.shape[3]
    n = 1
    if b.ndim == 4:
        n = b.shape[3]

    av = Variable(a)
    bv = Variable(b)
    print('forward')
    d = sinkhorn_chainer(av, bv, **kwargs)

    M = cuda.to_cpu(d.data)

    # The actual Jacobian Matrix is of size [(d1*d2*d3)*m]*mn, but since grad_{x_k} d(x_i,y_j) != 0 iif k == i, it is
    # a sparse Matrix and can thus be reduced to size [(d1*d2*d3)*m]*n, but omitting the n(m-1) zeros in each row.
    J = xp.empty(shape=(*a.shape[:3], m, n))
    print('backward')
    for j in range(n):
        d_ = d[:, j]
        av.cleargrad()
        prepare_gradient(d_)
        d_.backward()
        J[:, :, :, :, j] = av.grad.reshape((*a.shape[:3], m))

    return M, J


if __name__ == '__main__':
    print('debugging sinkhorn')

    from generate_data import generate_nice
    from _utils import prepare_gradient

    y = generate_nice()

    x0 = y[:, :, :, :, 0]
    x1 = y[:, :, :, :, 1]

    x0v = Variable(y[:, :, :, :, 0])
    x1v = Variable(y[:, :, :, :, 1])

    # x00v = Variable(x0[:, :, :, 0])
    # x01v = Variable(x0[:, :, :, 0].reshape((*x0.shape[:3], 1)))
    #
    # d_ref = sinkhorn(x0, x1)
    # d0 = sinkhorn(x0[:, :, :, 0], x1)
    # d00 = sinkhorn(x0[:, :, :, 0].reshape((*x0.shape[:3], 1)), x1)

    # d_ref1 = sinkhorn_chainer(x0v, x1v)
    # d1 = sinkhorn_chainer(x00v, x1v)
    # d11 = sinkhorn_chainer(x01v, x1v)

    d = sinkhorn_chainer(x0v, x1v).reshape((4, 4))
    prepare_gradient(d)
    d.backward()

    x00g = x0v.grad

    x0v.cleargrad()

    d0 = d[0, 1]
    d1 = d[1, 1]
    x0v.cleargrad()
    d0.backward()
    x01g = x0v.grad
    x0v.cleargrad()
    d1.backward()
    x11g = x0v.grad

    print('first test')
