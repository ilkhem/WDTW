import numpy as np
from _utils import xi

from autograd import grad

At_ = np.random.rand(10, 10, 10, 5).astype(np.float32)
At = At_ / np.sum(At_, axis=(0, 1, 2))
Bt_ = np.random.rand(10, 10, 10, 4).astype(np.float32)
Bt = Bt_ / np.sum(Bt_, axis=(0, 1, 2))


def sinkhorn_autograd(At, Bt):
    n_iter = 100
    mu = 0.45
    min_thresh = 1e-150
    p_exp = 1.2

    d1, d2, d3 = At.shape[:3]

    m = At.shape[3]
    n = Bt.shape[3]
    S = m * n

    # Sinkhorn algorithm #

    # define pair-wise L2 distances as a gaussian kernel convolution
    X, Y = np.meshgrid(np.arange(0, d1), np.arange(0, d1))
    Hx = np.exp(-np.power(np.abs(X - Y), p_exp) / (2 * mu ** p_exp))
    Hx[Hx < min_thresh] = min_thresh

    X, Y = np.meshgrid(np.arange(0, d2), np.arange(0, d2))
    Hy = np.exp(-np.power(np.abs(X - Y), p_exp) / (2 * mu ** p_exp))
    Hy[Hy < min_thresh] = min_thresh

    X, Y = np.meshgrid(np.arange(0, d3), np.arange(0, d3))
    Hz = np.exp(-np.power(np.abs(X - Y), p_exp) / (2 * mu ** p_exp))
    Hz[Hz < min_thresh] = min_thresh

    U = np.ones((d1, d2, d3, S))
    V = np.ones((d1, d2, d3, S))

    c = np.repeat(np.eye(At_.shape[-1], dtype=np.float32), n, axis=-1)
    A = np.dot(At, c)
    e = np.concatenate([np.eye(Bt_.shape[-1], dtype=np.float32)] * m, axis=-1)
    B = np.dot(Bt, e)

    for i in range(n_iter):
        if i % 10 == 0:
            print('iteration %d' % i)
        U = np.divide(A, xi(V, Hx, Hy, Hz))
        V = np.divide(B, xi(U, Hx, Hy, Hz))

    # U and V are now computed. Proceed to computing pairwise distances using <D(U)KD(V), M>.
    # In our Case, M is the L_2 distance in R^3

    # Define K_tilde kernels
    X, Y = np.meshgrid(np.arange(0, d1), np.arange(0, d1))
    Kx = np.multiply(np.exp(-np.power(np.abs(X - Y), p_exp) / (2 * mu ** p_exp)), np.power(np.abs(X - Y), p_exp))
    Kx[Kx < min_thresh] = min_thresh

    X, Y = np.meshgrid(np.arange(0, d2), np.arange(0, d2))
    Ky = np.multiply(np.exp(-np.power(np.abs(X - Y), p_exp) / (2 * mu ** p_exp)), np.power(np.abs(X - Y), p_exp))
    Ky[Ky < min_thresh] = min_thresh

    X, Y = np.meshgrid(np.arange(0, d3), np.arange(0, d3))
    Kz = np.multiply(np.exp(-np.power(np.abs(X - Y), p_exp) / (2 * mu ** p_exp)), np.power(np.abs(X - Y), p_exp))
    Kz[Kz < min_thresh] = min_thresh

    V_tilde = xi(V, Kx, Hy, Hz) + xi(V, Hx, Ky, Hz) + xi(V, Hx, Hy, Kz)

    d = np.sum(np.multiply(U, V_tilde), axis=(0, 1, 2))

    return d


sgradient = grad(sinkhorn_autograd)

print(sgradient(At, Bt))