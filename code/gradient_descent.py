from chainer import cuda

def gradient_descent(x, grad, lr, norm=True):
    assert x.shape == grad.shape
    xp = cuda.get_array_module(x)
    _x = xp.multiply(x, xp.exp(-1. * lr * grad))
    if norm:
        return _x / xp.sum(_x, axis=(0, 1, 2))
    else:
        return _x