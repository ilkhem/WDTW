import numpy as np
from _utils import xi, xi_chainer, prepare_gradient, dot42
from autograd import grad
import chainer.cuda as cuda
from chainer import Variable
from chainer.functions.math.exponential import exp
from chainer.functions.math.sum import sum
import time

At_ = np.random.rand(10, 10, 10, 5).astype(np.float32)
At_ = At_ / np.sum(At_, axis=(0, 1, 2))
Bt_ = np.random.rand(10, 10, 10, 4).astype(np.float32)
Bt_ = Bt_ / np.sum(Bt_, axis=(0, 1, 2))

At = Variable(At_)
Bt = Variable(Bt_)

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
X1, Y1 = np.meshgrid(np.arange(0, d1, dtype=np.float32), np.arange(0, d1, dtype=np.float32))
X1 = Variable(X1.astype(np.float32))
Y1 = Variable(Y1.astype(np.float32))
Hx = exp(-(X1 - Y1).__abs__() ** p_exp / (2 * mu ** p_exp))
Hx.data[Hx.data < min_thresh] = min_thresh

X2, Y2 = np.meshgrid(np.arange(0, d2), np.arange(0, d2))
X2 = Variable(X2.astype(np.float32))
Y2 = Variable(Y2.astype(np.float32))
Hy = exp(-(X2 - Y2).__abs__() ** p_exp / (2 * mu ** p_exp))
Hy.data[Hy.data < min_thresh] = min_thresh

X3, Y3 = np.meshgrid(np.arange(0, d3, dtype=np.float32), np.arange(0, d3, dtype=np.float32))
X3 = Variable(X3.astype(np.float32))
Y3 = Variable(Y3.astype(np.float32))
Hz = exp(-(X3 - Y3).__abs__() ** p_exp / (2 * mu ** p_exp))
Hz.data[Hz.data < min_thresh] = min_thresh

U = Variable(np.ones((d1, d2, d3, S), dtype=np.float32))
V = Variable(np.ones((d1, d2, d3, S), dtype=np.float32))

c = Variable(np.repeat(np.eye(At_.shape[-1], dtype=np.float32), n, axis=-1))
A = dot42(At, c)
e = Variable(np.concatenate([np.eye(Bt_.shape[-1], dtype=np.float32)] * m, axis=-1))
B = dot42(Bt, e)

for i in range(n_iter):
    if i % 10 == 0:
        print('iteration %d' % i)
    U = A / xi_chainer(V, Hx, Hy, Hz)
    V = B / xi_chainer(U, Hx, Hy, Hz)


# U and V are now computed. Proceed to computing pairwise distances using <D(U)KD(V), M>.
# In our Case, M is the L_2 distance in R^3

# Define K_tilde kernels
Kx = (exp(-(X1 - Y1).__abs__() ** p_exp / (2 * mu ** p_exp))) * ((X1 - Y1).__abs__() ** p_exp)

Ky = (exp(-(X2 - Y2).__abs__() ** p_exp / (2 * mu ** p_exp))) * ((X2 - Y2).__abs__() ** p_exp)

Kz = (exp(-(X3 - Y3).__abs__() ** p_exp / (2 * mu ** p_exp))) * ((X3 - Y3).__abs__() ** p_exp)

V_tilde = xi_chainer(V, Kx, Hy, Hz) + xi_chainer(V, Hx, Ky, Hz) + xi_chainer(V, Hx, Hy, Kz)

d = sum(U * V_tilde, axis=(0, 1, 2))

prepare_gradient(d)
d.backward()
print(At.grad)
