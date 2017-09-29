import numpy as np

from plot import plot2D
from wdtw import gradient_descent

if __name__ == '__main__':
    d1 = 3
    d2 = 3
    d3 = 3
    m = 3
    n1 = 4
    n2 = 3
    mu = 1
    tau = 1
    beta = 2.

    n_iter = 150
    n_g_iter = 150
    lr = 0.01

    a = np.zeros((n1, d1, d2, d3))
    a[0, 0, 0, 0] = 1
    a[0, 2, 2, 2] = 1
    a[1, 0, 0, 2] = 1
    a[1, 2, 2, 0] = 1
    a[2, 0, 0, 0] = 1
    a[2, 2, 2, 2] = 1
    a[3, 0, 0, 2] = 1
    a[3, 2, 2, 0] = 1
    a /= 2

    b = np.zeros((n2, d1, d2, d3))
    b[0, 0, 0, 2] = 1
    b[0, 2, 2, 0] = 1
    b[1, 0, 0, 0] = 1
    b[1, 2, 2, 2] = 1
    b[2, 0, 0, 2] = 1
    b[2, 2, 2, 0] = 1
    # b[3, 0, 0, 0] = 1
    # b[3, 2, 2, 2] = 1
    b /= 2

    ys = [a, b]

    x = np.ones((m, d1, d2, d3))
    x /= np.sum(x, axis=(1, 2, 3))

    xf = gradient_descent(x, ys, n_iter, n_g_iter, lr, mu, tau, beta, stop_b=False)
    print('**** instant 1 ****')
    print(ys[0][0])
    print('*** ***')
    print(xf[0])
    print('*** ***')
    print(ys[1][0])
    print('**** instant 2 ****')
    print(ys[0][1])
    print('*** ***')
    print(xf[1])
    print('*** ***')
    print(ys[1][1])
    print('**** instant 3 ****')
    print(ys[0][2])
    print('*** ***')
    print(xf[2])
    print('*** ***')
    print(ys[1][2])
    # d, G = wdtw(a, b, n_iter, mu, tau=tau, beta=beta)
    plot2D(ys[0], 'a433b')
    plot2D(ys[1], 'b433b')
    plot2D(xf, 'x433b')
