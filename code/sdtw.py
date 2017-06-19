import chainer.cuda as cuda
import dtw_fast
import numpy as np


def soft_dtw(M, beta=1.0):
    """not CUDA ready: implementation based on NumPy/Cython"""
    print('Computing soft_dtw distance')
    m, n = M.shape
    D = np.zeros((m + 2, n + 2), dtype=np.float64)  # We need +2 later for the gradient computation.
    dtw_fast.soft_dtw(M, D, beta=beta)
    return D[m, n], D


def soft_dtw_grad(D, beta=1.0):
    """not CUDA ready: implementation based on NumPy/Cython"""
    m, n = D.shape
    m -= 2
    n -= 2
    D_bar = np.zeros_like(D, dtype=np.float64)
    dtw_fast.soft_dtw_grad(m, n, D, D_bar, beta)
    return D_bar[1:-1, 1:-1]


def chain_rule(D_bar, G):
    """CUDA ready: implementation based on NumPy/CuPy"""
    xp = cuda.get_array_module(G)
    d1, d2, d3, m, n = G.shape
    assert D_bar.shape == (m, n)
    final_G = xp.zeros((d1, d2, d3, m), dtype=np.float64)
    for i in range(m):
        final_G[:, :, :, i] = xp.sum(D_bar[i] * G[:, :, :, i, :], axis=-1)
    return final_G
