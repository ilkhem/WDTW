from multiprocessing import Process, Pipe

import numpy as np

from generate_data import generate_nice
from sdtw import soft_dtw, soft_dtw_sec_grad
from wdtw import worker


def f(p_id, pipe, a, b, **kwargs):
    print('process', p_id)
    M, J = worker(a, b, **kwargs)
    print('sending to main', p_id)
    pipe.send(M)
    print('waiting to recv', p_id)
    D_bar = pipe.recv()
    print('computing grad', p_id)
    grad = J.dot(D_bar)
    print('grad for process done', p_id)
    # print(grad)


if __name__ == '__main__':
    s1, r1 = Pipe()
    s2, r2 = Pipe()

    y = generate_nice(50, 50, 50, 4, 2, 1e-6)

    a = y[:, :, :, :, 0]
    b = y[:, :, :, :, 1]
    a1 = a[:, :, :, 0]
    a2 = a[:, :, :, 1]

    p1 = Process(target=f, args=('1', r1, a1, b,), kwargs={})
    p2 = Process(target=f, args=('2', r2, a2, b,), kwargs={})
    p1.start()
    p2.start()
    print('main process recv')
    M1 = s1.recv()
    M2 = s2.recv()

    M = np.concatenate([M1, M2])
    print('computing sdtw')
    d, D = soft_dtw(M)
    D_bar = soft_dtw_sec_grad(D)
    print('sending to children')
    s1.send(D_bar[0])
    s2.send(D_bar[1])
    print('sending done, waiting for grads')
    p1.join()
    p2.join()
