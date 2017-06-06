"""
Compute Sinkhorn distances between 2 time series of shapes
Each shape is of dimension d1*d2*d3, lives in R^3
"""

import numpy as np
from _utils import xi, xi_chainer, dot42
import chainer
from chainer import cuda
from chainer import Variable
from chainer.functions.math.exponential import exp
from chainer.functions.math.sum import sum
from chainer.functions.array.reshape import reshape


def sinkhorn(At_, Bt_, verbose=1, n_iter=100, mu=0.45, tau=1, min_thresh=1e-100, p_exp=2, cpu=False):
    if cuda.available and not cpu:  # if CUDA is installed on the machine, use it.
        if isinstance(At_.data, cuda.cupy.ndarray):  # if At_ is on GPU
            # make sure it is on the same device as Bt_
            assert At_.data.device.id == Bt_.data.device.id
            # use that Device as main device
            cuda.get_device_from_id(At_.data.device.id).use()
            # set pointers
            At = At_
            Bt = Bt_
        else:  # At_ is a CPU array
            At = cuda.to_gpu(At_, device=2)
            Bt = cuda.to_gpu(Bt_, device=2)
            cuda.get_device_from_id(2).use()
    else:  # CUDA is not installed, run on CPU
        # set pointers
        At = At_
        Bt = Bt_
    xp = cuda.get_array_module(At)

    # make sure At and Bt are of same dimensions.
    assert At.shape[:3] == Bt.shape[:3]

    d1, d2, d3 = At.shape[:3]
    m = At.shape[3]
    n = Bt.shape[3]
    S = m * n

    # define pair-wise L2 distances as a gaussian kernel convolution
    X1, Y1 = xp.meshgrid(xp.arange(0, d1), xp.arange(0, d1))
    X2, Y2 = xp.meshgrid(xp.arange(0, d2), xp.arange(0, d2))
    X3, Y3 = xp.meshgrid(xp.arange(0, d3), xp.arange(0, d3))

    Hx = xp.exp(-xp.power(xp.abs(X1 - Y1), p_exp) / (2 * mu ** p_exp))
    Hx[Hx < min_thresh] = min_thresh
    Hy = xp.exp(-xp.power(xp.abs(X2 - Y2), p_exp) / (2 * mu ** p_exp))
    Hy[Hy < min_thresh] = min_thresh
    Hz = xp.exp(-xp.power(xp.abs(X3 - Y3), p_exp) / (2 * mu ** p_exp))
    Hz[Hz < min_thresh] = min_thresh
    Kx = xp.multiply(Hx, xp.power(xp.abs(X1 - X1), p_exp))
    Kx[Kx < min_thresh] = min_thresh
    Ky = xp.multiply(Hy, xp.power(xp.abs(X2 - Y2), p_exp))
    Ky[Ky < min_thresh] = min_thresh
    Kz = xp.multiply(Hz, xp.power(xp.abs(X3 - Y3), p_exp))
    Kz[Kz < min_thresh] = min_thresh
    if verbose > 2:
        print('Hx', Hx)

    U = xp.ones((d1, d2, d3, S)) / (d1 * d2 * d3 * S)
    V = xp.ones((d1, d2, d3, S)) / (d1 * d2 * d3 * S)
    U_old = xp.zeros_like(U)

    A = xp.repeat(At, n, axis=-1)
    B = xp.concatenate([Bt] * m, axis=-1)

    for i in range(n_iter):
        if verbose > 0 and i % 10 == 0:
            print('iteration %d' % i)
        U_old = U
        U = xp.divide(A, xi(V, Hx, Hy, Hz)) ** tau
        V = xp.divide(B, xi(U, Hx, Hy, Hz)) ** tau
        if verbose > 2:
            print(np.linalg.norm(U - U_old))
    V_tilde = xi(V, Kx, Hy, Hz) + xi(V, Hx, Ky, Hz) + xi(V, Hx, Hy, Kz)

    d = xp.sum(xp.multiply(U, V_tilde), axis=(0, 1, 2))

    return xp.reshape(d, (m, n))


def sinkhorn_chainer(At_, Bt_, verbose=1, n_iter=50, mu=0.45, tau=1, min_thresh=1e-100, p_exp=2, cpu=False):
    """
    At, Bt: instances of chainer.Variable
    This is the same as sinkhorn routine using Chainer to be able to compute gradient with respect to At.
    cpu: Force work to be done on cpu
    """
    if cuda.available and not cpu:  # if CUDA is installed on the machine, use it.
        if cuda.get_array_module(At_).__name__ == 'cupy':  # if At_ is on GPU
            # make sure it is on the same device as Bt_
            assert At_.data.device.id == Bt_.data.device.id
            # use that Device as main device
            cuda.get_device_from_id(At_.data.device.id).use()
            # set pointers
            At = At_
            Bt = Bt_
        else:  # At_ is a CPU array
            At = cuda.to_gpu(At_, device=2)
            Bt = cuda.to_gpu(Bt_, device=2)
            cuda.get_device_from_id(2).use()
    else:  # CUDA is not installed, run on CPU
        # set pointers
        At = At_
        Bt = Bt_
    xp = cuda.get_array_module(At)

    # make sure At and Bt are of same dimensions.
    if Bt.ndim < 4:
        Bt = reshape(Bt, (*Bt.shape, 1))
    try:
        assert At.shape[:-1] == Bt.shape[:-1]
    except AssertionError:
        At = reshape(At, (*At.shape, 1))
    d1, d2, d3 = At.shape[:3]
    m = At.shape[3]
    n = Bt.shape[3]
    S = m * n

    # Prepare convolution kernels
    X1, Y1 = xp.meshgrid(xp.arange(0, d1), xp.arange(0, d1))
    X1 = Variable(X1.astype(np.float64))
    Y1 = Variable(Y1.astype(np.float64))
    X2, Y2 = xp.meshgrid(xp.arange(0, d2), xp.arange(0, d2))
    X2 = Variable(X2.astype(np.float64))
    Y2 = Variable(Y2.astype(np.float64))
    X3, Y3 = xp.meshgrid(xp.arange(0, d3), xp.arange(0, d3))
    X3 = Variable(X3.astype(np.float64))
    Y3 = Variable(Y3.astype(np.float64))

    Hx = exp(-(X1 - Y1).__abs__() ** p_exp / (2 * mu ** p_exp))
    Hx.data[Hx.data < min_thresh] = min_thresh
    Hy = exp(-(X2 - Y2).__abs__() ** p_exp / (2 * mu ** p_exp))
    Hy.data[Hy.data < min_thresh] = min_thresh
    Hz = exp(-(X3 - Y3).__abs__() ** p_exp / (2 * mu ** p_exp))
    Hz.data[Hz.data < min_thresh] = min_thresh
    Kx = Hx * ((X1 - Y1).__abs__() ** p_exp)
    Kx.data[Kx.data < min_thresh] = min_thresh
    Ky = Hy * ((X2 - Y2).__abs__() ** p_exp)
    Ky.data[Ky.data < min_thresh] = min_thresh
    Kz = Hz * ((X3 - Y3).__abs__() ** p_exp)
    Kz.data[Kz.data < min_thresh] = min_thresh

    if verbose == -2:
        print(Hx.data)

    # initialize for sinkhorn iterations
    if S > 1:
        c = Variable(xp.repeat(xp.eye(At.shape[-1], dtype=np.float64), n, axis=-1))
        e = Variable(xp.concatenate([xp.eye(Bt.shape[-1], dtype=np.float64)] * m, axis=-1))
        A = dot42(At, c)
        B = dot42(Bt, e)
    else:
        A = At
        B = Bt
    u = Variable(xp.ones((d1, d2, d3, S), dtype=np.float64))
    for i in range(n_iter):
        u = A / xi_chainer(B / xi_chainer(u, Hx, Hy, Hz), Hx, Hy, Hz)
        if i % 5 == 0:
            print('iteration %d: Umax = %f' % (i + 1, xp.max(u.data)))
    v = B / xi_chainer(u, Hx, Hy, Hz)
    v_tilde = xi_chainer(v, Kx, Hy, Hz) + xi_chainer(v, Hx, Ky, Hz) + xi_chainer(v, Hx, Hy, Kz)
    d = sum(u * v_tilde, axis=(0, 1, 2))
    return reshape(d, (m, n))


def sinkhorn_mp(a, B, mu=0.5, p_exp=2, n_iter=50, min_thresh=1e-100):
    d1, d2, d3, n = B.shape
    xp = cuda.get_array_module(a)

    # Prepare convolution kernels
    X1, Y1 = xp.meshgrid(xp.arange(0, d1), xp.arange(0, d1))
    X1 = Variable(X1.astype(np.float64))
    Y1 = Variable(Y1.astype(np.float64))
    X2, Y2 = xp.meshgrid(xp.arange(0, d2), xp.arange(0, d2))
    X2 = Variable(X2.astype(np.float64))
    Y2 = Variable(Y2.astype(np.float64))
    X3, Y3 = xp.meshgrid(xp.arange(0, d3), xp.arange(0, d3))
    X3 = Variable(X3.astype(np.float64))
    Y3 = Variable(Y3.astype(np.float64))

    Hx = exp(-(X1 - Y1).__abs__() ** p_exp / (2 * mu ** p_exp))
    Hx.data[Hx.data < min_thresh] = min_thresh
    Hy = exp(-(X2 - Y2).__abs__() ** p_exp / (2 * mu ** p_exp))
    Hy.data[Hy.data < min_thresh] = min_thresh
    Hz = exp(-(X3 - Y3).__abs__() ** p_exp / (2 * mu ** p_exp))
    Hz.data[Hz.data < min_thresh] = min_thresh
    Kx = Hx * ((X1 - Y1).__abs__() ** p_exp)
    Kx.data[Kx.data < min_thresh] = min_thresh
    Ky = Hy * ((X2 - Y2).__abs__() ** p_exp)
    Ky.data[Ky.data < min_thresh] = min_thresh
    Kz = Hz * ((X3 - Y3).__abs__() ** p_exp)
    Kz.data[Kz.data < min_thresh] = min_thresh

    A = reshape(a, (*a.shape, 1))
    # print(A.shape)
    e = Variable(xp.concatenate([xp.eye(A.shape[-1], dtype=np.float64)] * n, axis=-1))
    A = dot42(A, e)
    # print(A.shape)
    u = Variable(xp.ones(B.shape, dtype=np.float64))
    for i in range(n_iter):
        u = A / xi_chainer(B / xi_chainer(u, Hx, Hy, Hz), Hx, Hy, Hz)
        if i%5 ==0:
            print('iteration %d: Umax = %f' % (i + 1, xp.max(u.data)))
    v = B / xi_chainer(u, Hx, Hy, Hz)
    v_tilde = xi_chainer(v, Kx, Hy, Hz) + xi_chainer(v, Hx, Ky, Hz) + xi_chainer(v, Hx, Hy, Kz)
    d = sum(u * v_tilde, axis=(0, 1, 2))
    return d


def sinkhorn_modified():
    a = np.random.rand(100,100,100)
    with chainer.no_backprop_mode():
        B = np.random.rand(100,100,100,3)

    p_exp = 1.2
    mu = 1
    min_thresh = 1e-100
    n_iter = 50

    d1, d2, d3, n = B.shape
    xp = cuda.get_array_module(a)

    # Prepare convolution kernels

    with chainer.no_backprop_mode():
        X1, Y1 = xp.meshgrid(xp.arange(0, d1), xp.arange(0, d1))
        X1 = Variable(X1.astype(np.float64))
        Y1 = Variable(Y1.astype(np.float64))
        X2, Y2 = xp.meshgrid(xp.arange(0, d2), xp.arange(0, d2))
        X2 = Variable(X2.astype(np.float64))
        Y2 = Variable(Y2.astype(np.float64))
        X3, Y3 = xp.meshgrid(xp.arange(0, d3), xp.arange(0, d3))
        X3 = Variable(X3.astype(np.float64))
        Y3 = Variable(Y3.astype(np.float64))
        Hx = exp(-(X1 - Y1).__abs__() ** p_exp / (2 * mu ** p_exp))
        Hx.data[Hx.data < min_thresh] = min_thresh
        Hy = exp(-(X2 - Y2).__abs__() ** p_exp / (2 * mu ** p_exp))
        Hy.data[Hy.data < min_thresh] = min_thresh
        Hz = exp(-(X3 - Y3).__abs__() ** p_exp / (2 * mu ** p_exp))
        Hz.data[Hz.data < min_thresh] = min_thresh
        Kx = Hx * ((X1 - Y1).__abs__() ** p_exp)
        Kx.data[Kx.data < min_thresh] = min_thresh
        Ky = Hy * ((X2 - Y2).__abs__() ** p_exp)
        Ky.data[Ky.data < min_thresh] = min_thresh
        Kz = Hz * ((X3 - Y3).__abs__() ** p_exp)
        Kz.data[Kz.data < min_thresh] = min_thresh

    A = reshape(a, (*a.shape, 1))
    # print(A.shape)

    with chainer.no_backprop_mode():
        e = Variable(xp.concatenate([xp.eye(A.shape[-1], dtype=np.float64)] * n, axis=-1))
    A = dot42(A, e)
    # print(A.shape)
    u = Variable(xp.ones(B.shape, dtype=np.float64))
    for i in range(n_iter):
        u = A / xi_chainer(B / xi_chainer(u, Hx, Hy, Hz), Hx, Hy, Hz)
        if i % 5 == 0:
            print('iteration %d: Umax = %f' % (i + 1, xp.max(u.data)))
    v = B / xi_chainer(u, Hx, Hy, Hz)
    v_tilde = xi_chainer(v, Kx, Hy, Hz) + xi_chainer(v, Hx, Ky, Hz) + xi_chainer(v, Hx, Hy, Kz)
    d = sum(u * v_tilde, axis=(0, 1, 2))
    return d

if __name__ == '__main__':
    from generate_data import generate_nice
    import time
    test = 0
    print('\tstarting Sinkhorn')
    if test==0:
        y = generate_nice(100, 100, 100, 3, 2, 1e-10)
        y0 = Variable(y[:, :, :, :, 0])
        x0 = Variable(y[:, :, :, 0, 0])
        y1 = Variable(y[:, :, :, :, 1])
        x1 = Variable(y[:, :, :, 0, 1])

        st = time.time()
        d = sinkhorn_mp(x1, y0, n_iter=50, mu=1, p_exp=50)
        print('Elapsed time for sinkhorn_mp: %f' % (time.time() - st))
    #
    # st = time.time()
    # d0 = sinkhorn_chainer(x1, x0, n_iter=100, mu=1, cpu=True)
    # print(d0.data)
    # d0.backward()
    # print(x1.grad)
    # print('Elapsed time for sinkhorn_chainer: %f' % (time.time() - st))

    else:
        sinkhorn_modified()
