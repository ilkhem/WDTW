import dtw_fast
import numpy as np
from chainer import cuda

from sinkhorn import sinkhorn_fb


def _soft_dtw(M, beta=1.0):
    """not CUDA ready: implementation based on NumPy/Cython"""
    # produces the actual soft-DTW distance, and the matrix used for the iterative process
    print('\n** sDTW forward pass **')
    m, n = M.shape
    D = np.zeros((m + 2, n + 2), dtype=np.float64)  # We need +2 later for the gradient computation.
    dtw_fast.soft_dtw(M, D, beta=beta)
    return D[m, n], D


def _soft_dtw_grad(D, beta=1.0):
    """not CUDA ready: implementation based on NumPy/Cython"""
    # iteratively compute the gradient of soft-DTW with respect to the pair-wise (sinkhorn) distance
    # matrix produced by sinkhorn
    print('\n** sDTW backward pass **')
    m, n = D.shape
    m -= 2
    n -= 2
    D_bar = np.zeros_like(D, dtype=np.float64)
    dtw_fast.soft_dtw_grad(m, n, D, D_bar, beta)
    return D_bar[1:-1, 1:-1]


def _wdtw_chain_rule(D_bar, J):
    """CUDA ready: implementation based on NumPy/CuPy"""
    # applies the chain rule to compute the derivatives of WDTW with respect to the initial input a
    print('\n** WDTW chain-ruling **')
    xp = cuda.get_array_module(J)
    m, d1, d2, d3, n = J.shape
    assert D_bar.shape == (m, n)
    G = xp.stack([J[i].dot(D_bar[i]) for i in range(m)], axis=0)
    # final_G = xp.zeros((d1, d2, d3, m), dtype=np.float64)
    # for i in range(m):
    #     final_G[:, :, :, i] = xp.sum(D_bar[i] * G[:, :, :, i, :], axis=-1)
    assert G.shape == (m, d1, d2, d3)
    return G


def wdtw(a, b, n_iter=100, mu=0.5, tau=1., beta=1., p_exp=2., min_thresh=1e-150):
    M, J = sinkhorn_fb(a, b, n_iter, mu, tau, p_exp, min_thresh)
    # print(M)
    # print(J)
    d, D = _soft_dtw(M, beta)
    D_bar = _soft_dtw_grad(D, beta)
    G = _wdtw_chain_rule(D_bar, J)
    # print(d)
    # print(G)
    return d, G


def _gradient_descent_update(x, grad, lr, norm=True):
    assert x.shape == grad.shape
    xp = cuda.get_array_module(x)
    _x = xp.multiply(x, xp.exp(-1. * lr * grad))
    if norm:
        return _x / xp.sum(_x, axis=(1, 2, 3))
    else:
        return _x


def _single_gradient_step(x, ys, n_sinkhorn_steps, mu, tau, beta):
    # ys is a list of multiple time-series y_j with lengths n_j
    energy = 0
    gradient = np.zeros_like(x)
    for y in ys:
        d, G = wdtw(x, y, n_sinkhorn_steps, mu, tau, beta)
        energy += d
        gradient += G
    assert gradient.shape == x.shape
    return energy, gradient


def gradient_descent(x, ys, n_sinkhorn_steps, n_gradient_steps, lr, mu, tau, beta, norm=True, stop_b=True):
    old_e = np.inf
    for _ in range(n_gradient_steps):
        e, g = _single_gradient_step(x, ys, n_sinkhorn_steps, mu, tau, beta)
        print('\t\t Gradient Iteration %d: Energy = %f' % (_ + 1, e))
        if stop_b and e > old_e:
            print('stopping at iteration %d' % (_ + 1))
            return x
        old_e = e
        x = _gradient_descent_update(x, g, lr, norm)

    return x
