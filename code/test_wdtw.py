import numpy as np
import chainer.cuda as cuda
from wdtw import barycenter, _loss
import time

from generate_data import generate_multiple

if __name__ == "__main__":
    Y = generate_multiple(3, 3, 3, 2, 3)
    print(Y.shape)
    x_init = np.mean(Y, axis=-1)


    start = time.time()
    # d, G = _loss(x_init, Y, verbose=2, gpu=True)
    bar = barycenter(Y, x_init, gpu=False, verbose=0)
    print('Elapsed time: %fs' % (time.time() - start))
    print(type(bar))
    print(bar.shape)
