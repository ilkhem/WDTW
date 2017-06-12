# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
np.import_array()

from libc.float cimport DBL_MAX
from libc.math cimport exp, log1p
from libc.string cimport memset


cdef _init(np.ndarray[double, ndim=2] D,
           int m,
           int n,
           int left,
           int right):

    cdef int i, j

    for i in range(m + 2):
        D[i, right] = DBL_MAX

    for j in range(n + 2):
        D[left, j] = DBL_MAX


cdef inline double softmin(double a,
                           double b,
                           double c,
                           double beta):
    cdef double s = 0

    if a < b:
        if a < c:
            s += exp(-(b - a) * beta)
            s += exp(-(c - a) * beta)
            return -1/ beta * log1p(s) + a
        else:
            s += exp(-(a - c) * beta)
            s += exp(-(b - c) * beta)
            return -1/ beta * log1p(s) + c
    else:
        if b < c:
            s += exp(-(a - b) * beta)
            s += exp(-(c - b) * beta)
            return -1/ beta * log1p(s) + b
        else:
            s += exp(-(a - c) * beta)
            s += exp(-(b - c) * beta)
            return -1/ beta * log1p(s) + c


def soft_dtw(np.ndarray[double, ndim=2] M,
             np.ndarray[double, ndim=2] D,
             double beta):

    cdef int m = M.shape[0]
    cdef int n = M.shape[1]

    cdef int i, j

    memset(<void*>D.data, 0, (m+2) * (n+2) * sizeof(double))
    _init(D, m, n, 0, 0)
    D[0, 0] = 0

    for i in xrange(1, m + 1):
        for j in xrange(1, n + 1):
            # M is indexed starting from 0.
            D[i, j] = M[i-1,j-1] + softmin(D[i-1,j], D[i-1,j-1], D[i, j-1],
                                           beta)


cdef inline double _e(double a,
                      double b,
                      double c,
                      double beta):

    cdef double exp_b, exp_c

    b = (a - b) * beta
    c = (a - c) * beta

    if b > 30 or c > 30:
        return 0

    if b < -20:
        exp_b = 0
    else:
        exp_b = exp(b)

    if c < -20:
        exp_c = 0
    else:
        exp_c = exp(c)

    return 1. / (1 + exp_b + exp_c)


def soft_dtw_grad(int m,
                      int n,
                      np.ndarray[double, ndim=2] D,
                      np.ndarray[double, ndim=2] D_bar,
                      double beta):

    cdef int i, j

    _init(D, m, n, m+1, n+1)
    memset(<void*>D_bar.data, 0, (m+2) * (n+2) * sizeof(double))
    D_bar[m+1, n+1] = 1.0

    for j in reversed(range(1, n+1)):
        for i in reversed(range(1, m+1)):
            D_bar[i, j] += D_bar[i+1,j] * _e(D[i,j], D[i, j-1], D[i+1,j-1], beta)
            D_bar[i, j] += D_bar[i+1,j+1] * _e(D[i,j], D[i,j+1], D[i+1,j], beta)
            D_bar[i, j] += D_bar[i,j+1] * _e(D[i,j], D[i-1,j+1], D[i-1, j], beta)

