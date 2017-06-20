import numpy as np
import chainer.cuda as cuda
from chainer import Variable
from chainer.functions.math.exponential import exp
from chainer.functions.math.logarithm_1p import log1p
from chainer.functions.math.sum import sum



def softmin_chainer(a, b, c, beta=1.0):
    xp = cuda.get_array_module(a)
    s = Variable(xp.zeros(1))[0]
    if a.data < b.data:
        if a.data < c.data:
            s += exp(-(b - a) * beta)
            s += exp(-(c - a) * beta)
            return -1 / beta * log1p(s) + a
        else:
            s += exp(-(a - c) * beta)
            s += exp(-(b - c) * beta)
            return -1 / beta * log1p(s) + c
    else:
        if b.data < c.data:
            s += exp(-(a - b) * beta)
            s += exp(-(c - b) * beta)
            return -1 / beta * log1p(s) + b
        else:
            s += exp(-(a - c) * beta)
            s += exp(-(b - c) * beta)
            return -1 / beta * log1p(s) + c


def _e_chainer(a, b, c, beta=1.0):
    b = (a - b) * beta
    c = (a - c) * beta

    if b.data > 30 or c.data > 30:
        return 0

    if b.data < -20:
        exp_b = 0
    else:
        exp_b = exp(b)

    if c.data < -20:
        exp_c = 0
    else:
        exp_c = exp(c)

    return 1. / (1 + exp_b + exp_c)


def soft_dtw_chainer(M, beta=1.0, verbose=1):
    xp = cuda.get_array_module(M)
    if verbose > 0:
        print('chainer sdtw')
    m, n = M.shape
    D = xp.zeros((m + 2, n + 2))

    for i in range(m + 2):
        D[i, 0] = xp.inf
    for j in range(n + 2):
        D[0, j] = xp.inf
    D[0, 0] = 0
    D = Variable(D)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            D[i, j] = M[i - 1, j - 1] + \
                      softmin_chainer(D[i - 1, j], D[i - 1, j - 1], D[i, j - 1], beta)

    return D[m, n], D


def soft_dtw_sec_grad_chainer(D, beta=1.0, verbose=1):
    xp = cuda.get_array_module(D)
    if verbose > 0:
        print('chainer sdtw_sec_grad')
    m, n = D.shape
    m -= 2
    n -= 2

    D_bar = xp.zeros_like(D, dtype=np.float64)

    for i in range(m + 2):
        D[i, m+1] = xp.inf
    for j in range(n + 2):
        D[n+1, j] = xp.inf

    D_bar[m+1, n+1] = 1.0

    D_bar = Variable(D_bar)

    for j in reversed(range(1, n+1)):
        for i in reversed(range(1, m+1)):
            D_bar[i, j] += D_bar[i+1,j] * _e_chainer(D[i,j], D[i, j-1], D[i+1,j-1], beta)
            D_bar[i, j] += D_bar[i+1,j+1] * _e_chainer(D[i,j], D[i,j+1], D[i+1,j], beta)
            D_bar[i, j] += D_bar[i,j+1] * _e_chainer(D[i,j], D[i-1,j+1], D[i-1, j], beta)

    return D_bar[1:-1, 1:-1]


def soft_dtw_grad(D_bar, G, verbose=1):
    xp = cuda.get_array_module(G)
    if verbose > 0:
        print('Computing final gradient')
    d1, d2, d3, m, n = G.shape
    assert D_bar.shape == (m, n)
    final_G = Variable(xp.zeros((d1, d2, d3, m), dtype=np.float64))
    for i in range(m):
        final_G.data[:, :, :, i] = sum(D_bar[i] * G[:, :, :, i, :], axis=-1)
    return final_G


if __name__ == '__main__':
    from generate_data import generate_data
    from wdtw import compute_wass_dist_grad_chainer
    a,b = generate_data()
    M, G = compute_wass_dist_grad_chainer(a,b, verbose=2)
    print(type(M))
    d, D = soft_dtw_chainer(M)
