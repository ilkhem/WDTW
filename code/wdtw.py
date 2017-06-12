import chainer.cuda as cuda
import numpy as np
from chainer import Variable
from scipy.optimize import minimize

from _utils import prepare_gradient
from sdtw import soft_dtw_grad, soft_dtw, soft_dtw_sec_grad
from sinkhorn import sinkhorn_chainer


def worker(a, b, **kwargs):
    xp = cuda.get_array_module(a)
    m = 1
    if a.ndim == 4:
        m = a.shape[3]
    n = 1
    if b.ndim == 4:
        n = b.shape[3]

    av = Variable(a)
    bv = Variable(b)

    d = sinkhorn_chainer(av, bv, **kwargs)

    M = d.data

    # The actual Jacobian Matrix is of size [(d1*d2*d3)*m]*mn, but since grad_{x_k} d(x_i,y_j) != 0 iif k == i, it is
    # a sparse Matrix and can thus be reduced to size [(d1*d2*d3)*m]*n, but omitting the n(m-1) zeros in each row.
    J = xp.empty(shape=(*a.shape[:3], m, n))

    for j in range(n):
        d_ = d[:, j]
        av.cleargrad()
        prepare_gradient(d_)
        d_.backward()
        J[:, :, :, :, j] = av.grad

    return M, J


# step 1
def sinkhorn_dist_grad(At_, Bt_, **kwargs):
    xp = cuda.get_array_module(At_)

    m = 1
    if At_.ndim == 4:
        m = At_.shape[3]
    n = 1
    if Bt_.ndim == 4:
        n = Bt_.shape[3]

    At = Variable(At_)
    Bt = Variable(Bt_)

    print('Computing Sinkhorn distances...')
    d = sinkhorn_chainer(At, Bt, **kwargs)
    M = d.data

    print('Computing gradients...')
    G = xp.zeros((*At.shape, Bt.shape[-1]), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            print(' element (%d, %d)' % (i, j))
            At.cleargrad()
            d_ = d[i, j]
            d_.backward()
            G[:, :, :, i, j] = At.grad[:, :, :, i]
    return M, G


# step 4
def _loss(X_, Y_, verbose=1, beta=1, weights=None, cpu=False, **kwargs):
    """CUDA ready: implementation based on NumPy/CuPy"""
    if cuda.available and not cpu:  # if CUDA is installed on the machine, use it.
        gpu = True
        if cuda.get_array_module(X_).__name__ == 'cupy':  # if At_ is on GPU
            # make sure it is on the same device as Bt_
            assert X_.data.device.id == Y_.data.device.id
            # use that Device as main device
            cuda.get_device_from_id(X_.data.device.id).use()
            # set pointers
            X = X_
            Y = Y_
            print(X.device.id)
        else:  # At_ is a CPU array
            print('Copying to GPU.')
            X = cuda.to_gpu(X_, device=2)
            Y = cuda.to_gpu(Y_, device=2)
            cuda.get_device_from_id(2).use()
    else:  # CUDA is not installed, run on CPU
        gpu = False
        # set pointers
        X = X_
        Y = Y_
    xp = cuda.get_array_module(X)

    # Compute objective value and grad at Z.
    X = X.reshape(Y.shape[:-1])
    d1, d2, d3, m = X.shape
    nb = Y.shape[-1]
    if weights is None:
        weights = np.ones(nb)
    G = xp.zeros((d1, d2, d3, m))
    d = 0
    for i in range(nb):
        if verbose > 1:
            print('computing distances and gradients for shape %d' % (i + 1))
        M_gpu, G_gpu = sinkhorn_dist_grad(X, Y[:, :, :, :, i], verbose=verbose, **kwargs)
        if gpu:
            if verbose > 1:
                print('_ copying to cpu')
            M_cpu = cuda.to_cpu(M_gpu)
        else:
            M_cpu = M_gpu
        d_cpu, D_cpu = soft_dtw(M_cpu, beta, verbose=verbose)
        D_bar_cpu = soft_dtw_sec_grad(D_cpu, beta, verbose=verbose)
        if gpu:
            if verbose > 1:
                print('_ copying to gpu')
            D_bar_gpu = cuda.to_gpu(D_bar_cpu, device=G_gpu.device.id)
        else:
            D_bar_gpu = D_bar_cpu

        final_G_gpu = soft_dtw_grad(D_bar_gpu, G_gpu, verbose=verbose)
        G += weights[i] * final_G_gpu
        d += weights[i] * d_cpu
    return d, G


def gradient_descent(x, grad, lr, norm=True):
    assert x.shape == grad.shape
    xp = cuda.get_array_module(x)
    _x = xp.multiply(x, xp.exp(-1. * lr * grad))
    if norm:
        return _x / xp.sum(_x, axis=(0, 1, 2))
    else:
        return _x


# Deprecated
def compute_wass_dist_grad_chainer(At_, Bt_, verbose=1, **kwargs):
    """CUDA ready: implementation based on NumPy/CuPy"""
    xp = cuda.get_array_module(At_)
    d1, d2, d3 = At_.shape[:-1]
    m = At_.shape[-1]
    n = Bt_.shape[-1]
    At = Variable(At_)
    Bt = Variable(Bt_)

    if verbose > 0:
        print('Computing Sinkhorn distances...')
    d = sinkhorn_chainer(At, Bt, verbose=verbose, **kwargs)

    if verbose > 0:
        print('Computing gradients...')
    G = Variable(xp.zeros((*At.shape, Bt.shape[-1]), dtype=np.float64))
    for i in range(m):
        for j in range(n):
            if verbose > 1:
                print(' element (%d, %d)' % (i, j))
            At.cleargrad()
            d_ = d[i, j]
            d_.backward()
            G.data[:, :, :, i, j] = At.grad[:, :, :, i]
    return d, G


# Deprecated
def barycenter(Y, X_init, verbose=1, gpu=False, method="L-BFGS-B", beta=1.0, tol=1e-3, weights=None, max_iter=50):
    # DEPRECATED
    # xp = cuda.get_array_module(Y)
    # if gpu and xp.__name__ == 'numpy':
    #     if verbose > 0:
    #         print('_ copying to gpu')
    #     X_init = cuda.to_gpu(X_init, device=2)
    #     Y = cuda.to_gpu(Y, device=2)
    #     cuda.get_device_from_id(2).use()
    #     xp = cuda.get_array_module(Y)

    X_init = np.ravel(X_init)
    nb = Y.shape[-1]

    if weights is None:
        weights = np.ones(nb)

    def loss(X):
        return _loss(X, Y, verbose=verbose, gpu=gpu, beta=beta, weights=weights)

    # The function works with vectors so we need to vectorize Z_init.
    res = minimize(loss, X_init, method=method, jac=True,
                   tol=tol, options=dict(maxiter=max_iter, disp=100))

    return np.exp(res.x)


if __name__ == "__main__":
    print('slt cv?')

    from generate_data import generate_data

    At_, Bt_ = generate_data(10, 10, 10, 5, 4)

    M, G = sinkhorn_dist_grad(At_, Bt_, verbose=2)

    print(M.shape)
    print(G.shape)

    d, D = soft_dtw(M)
    D_bar = soft_dtw_sec_grad(D)
    print(D_bar.shape)
    final_grad = soft_dtw_grad(D_bar, G)
    print(final_grad.shape)
