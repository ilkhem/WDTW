import chainer.cuda as cuda
import dtw_fast
import numpy as np


def soft_dtw(M, beta=1.0):
    """not CUDA ready: implementation based on NumPy/Cython"""
    # produces the actual soft-DTW distance, and the matrix used for the iterative process
    print('\n** sDTW forward pass **')
    m, n = M.shape
    D = np.zeros((m + 2, n + 2), dtype=np.float64)  # We need +2 later for the gradient computation.
    dtw_fast.soft_dtw(M, D, beta=beta)
    return D[m, n], D


def soft_dtw_grad(D, beta=1.0):
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


def wdtw_chain_rule(D_bar, J):
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
