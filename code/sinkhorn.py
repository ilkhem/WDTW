"""
Compute Sinkhorn distances between 2 time series of shapes
Each shape is of dimension d1*d2*d3, lives in R^3
"""

import numpy as np
from chainer import Variable
from chainer import cuda
from chainer.functions.math.sum import sum

from chainer_functions import dot42, convol, _prepare_gradient


def _kernels(xp, d1, d2, d3, mu=0.5, p_exp=2., min_thresh=1e-150):
    # gamma = 2 * mu ** p_exp
    gamma = mu

    x1, y1 = xp.meshgrid(xp.arange(0, d1), xp.arange(0, d1))
    x2, y2 = xp.meshgrid(xp.arange(0, d2), xp.arange(0, d2))
    x3, y3 = xp.meshgrid(xp.arange(0, d3), xp.arange(0, d3))
    mx = xp.power(xp.abs(x1 - y1), p_exp)
    my = xp.power(xp.abs(x2 - y2), p_exp)
    mz = xp.power(xp.abs(x3 - y3), p_exp)

    hx = xp.exp(-mx / gamma)
    hy = xp.exp(-my / gamma)
    hz = xp.exp(-mz / gamma)
    kx = xp.multiply(hx, mx)
    ky = xp.multiply(hy, my)
    kz = xp.multiply(hz, mz)

    # numerical stability trick
    hx[hx < min_thresh] = min_thresh
    hy[hy < min_thresh] = min_thresh
    hz[hz < min_thresh] = min_thresh
    kx[kx < min_thresh] = min_thresh
    ky[ky < min_thresh] = min_thresh
    kz[kz < min_thresh] = min_thresh

    return hx, hy, hz, kx, ky, kz


def sinkhorn(a, b, hx, hy, hz, kx, ky, kz, n_iter, tau=1.):
    xp = cuda.get_array_module(a)

    if a.ndim == 3 and b.ndim == 3:  # shape to shape
        m = 1
        n = 1
        A = a.reshape((1, *a.shape))
        B = b.reshape((1, *b.shape))
    elif a.ndim == 3 and b.ndim == 4:  # shape to series
        m = 1
        n = b.shape[0]
        e = xp.concatenate([xp.eye(m, dtype=np.float64)] * n, axis=-1)
        A = dot42(a.reshape((1, *a.shape)), e)
        B = b

    elif a.ndim == 4 and b.ndim == 4:  # series to series
        m = a.shape[0]
        n = b.shape[0]
        c = xp.repeat(xp.eye(m, dtype=np.float64), n, axis=-1)  # m * s
        e = xp.concatenate([xp.eye(n, dtype=np.float64)] * m, axis=-1)  # n * s
        A = dot42(a, c)
        B = dot42(b, e)
    else:
        raise Exception('check input dimensions')
    s = m * n

    assert A.shape == B.shape  # make sure a and b are of same dimensions.
    d1, d2, d3 = A.shape[-3:]  # first dimension is for time

    u = Variable(xp.ones((s, d1, d2, d3), dtype=np.float64))
    for i in range(n_iter):
        if tau == 1:  # reduce memory footprint by not taking the power of tau
            u = A / convol(B / convol(u, hx, hy, hz), hx, hy, hz)
        else:
            u = (A / convol((B / convol(u, hx, hy, hz)) ** tau, hx, hy, hz)) ** tau
        if i % 50 == 0:
            print('iteration %d: Umax = %f' % (i, xp.max(u.data)))
    if tau == 1:
        v = B / convol(u, hx, hy, hz)
    else:
        v = (B / convol(u, hx, hy, hz)) ** tau

    v_tilde = convol(v, kx, hy, hz) + convol(v, hx, ky, hz) + convol(v, hx, hy, kz)

    d = sum(u * v_tilde, axis=(1, 2, 3))

    return d.reshape((m, n))


def sinkhorn_fb(a, b, n_iter=100, mu=0.5, tau=1., p_exp=2., min_thresh=1e-150):
    # do a forward and then backward sinkhorn pass
    # this function takes care of transforming a and b to chainer.Variables, so they just need to be numpy/cupy arrays
    xp = cuda.get_array_module(a)

    m = 1
    if a.ndim == 4:
        m = a.shape[0]
    n = 1
    if b.ndim == 4:
        n = b.shape[0]
    d1, d2, d3 = a.shape[-3:]

    hx, hy, hz, kx, ky, kz = _kernels(xp, d1, d2, d3, mu, p_exp, min_thresh)

    av = Variable(a)
    bv = Variable(b)

    print('\n** sinkhorn forward pass **')
    d = sinkhorn(av, bv, hx, hy, hz, kx, ky, kz, n_iter, tau)
    M = cuda.to_cpu(d.data)

    # The actual Jacobian Matrix is of size [m*(d1*d2*d3)]*mn, but since grad_{x_k} d(x_i,y_j) != 0 iif k == i, it is
    # a sparse Matrix and can thus be reduced to size [m*(d1*d2*d3)]*mn, by omitting the n(m-1) zeros in each row.
    print('\n** sinkhorn backward pass **')
    J = xp.empty(shape=(m, *a.shape[-3:], n))
    for j in range(n):
        print('\t* sinkhorn forward pass: ', j, 'th gradient *')

        d_ = d[:, j]
        av.cleargrad()
        _prepare_gradient(d_)
        d_.backward()
        J[:, :, :, :, j] = av.grad.reshape((m, *a.shape[-3:]))

    return M, J
