from wdtw import _loss, gradient_descent
from generate_data import generate_multiple, generate_nice
import numpy as np
from _utils import prepare_gradient
import chainer.cuda as cuda
from chainer import Variable
from sinkhorn import sinkhorn, sinkhorn_chainer

# y = generate_nice()
# x = np.mean(y, axis=-1)

y1 = np.zeros(
    (3, 3, 3, 2))
y1[0, 0, 0, 0] = 1
y1[0, 0, 0, 1] = 1
# y1[1, 1, 1, 0] = 1
y1 /= np.sum(y1, axis=(0, 1, 2))
y2 = np.zeros((3, 3, 3, 2))
# y2[0, 1, 0, 0] = 1
y2[2, 2, 2, 0] = 1
y2[2, 2, 2, 1] = 1
y2 /= np.sum(y2, axis=(0, 1, 2))

y = np.zeros((3, 3, 3, 2, 2))
y[:, :, :, :, 0] = y1
y[:, :, :, :, 1] = y2
x = np.ones((3, 3, 3, 2), dtype=np.float64)
x /= np.sum(x, axis=(0, 1, 2))


x1 = np.ones((3, 3, 3, 2), dtype=np.float64)
x1 /= np.sum(x1, axis=(0, 1, 2))

y3 = np.arange(1, 27 * 2 + 1, dtype=np.float64).reshape((3, 3, 3, 2))
y3 /= np.sum(y3, axis=(0, 1, 2))
#
# x0_ = Variable(x0)
# y0_ = Variable(y0[:, :, :, :, 0].reshape(x0.shape))
x1_ = Variable(x1)
y1_ = Variable(y1)
y3_ = Variable(y3)
x_ = Variable(x)
y2_ = Variable(y2)

if cuda.available:
    print('Cuda available, copying to GPU')
    x = cuda.to_gpu(x)
    y = cuda.to_gpu(y)
    print(x.device.id)
xp = cuda.get_array_module(x)
print(y.shape)
for i in range(50):
    # print('\t\titeration %d' % i)
    f, g = _loss(x, y, verbose=-2, n_iter=100, mu=0.8, p_exp=1.5, tau=1)
    print('iteration %d: \tEnergy: %f' % (i + 1, f))
    x = gradient_descent(x, g, lr=0.5)
    # print(np.round(x0 / np.max(x0), 3))

print(x)

print('done')
#
# d = sinkhorn_chainer(x_, y2_, verbose=1, n_iter=100, p_exp=2, tau=1, mu=0.5)
# prepare_gradient(d)
# d.backward()
# print(x_.grad)
# print(np.max(x_.grad), np.min(x_.grad))
# print("done")
