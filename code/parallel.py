from multiprocessing import Process, Pipe
from chainer import cuda
import numpy as np

from generate_data import generate_nice
from sdtw import soft_dtw, soft_dtw_sec_grad
from wdtw import worker, gradient_descent


def f(p_id, device, pipe, a, b, **kwargs):
    cuda.get_device_from_id(device).use()
    print('copying to gpu')
    ag = cuda.to_gpu(a, device=device)
    bg = cuda.to_gpu(b, device=device)
    print('process', p_id)
    M, J = worker(ag, bg, **kwargs)
    print('sending to main', p_id)
    pipe.send(M)
    print('waiting to recv', p_id)
    D_bar = cuda.to_gpu(pipe.recv(), device=device)
    print('computing grad', p_id)
    grad = J.dot(D_bar)
    print(J.shape)
    print(D_bar.shape)
    print(grad.shape)
    print('updating', p_id)
    ag = gradient_descent(ag, grad, 0.1)


if __name__ == '__main__':
    s1, r1 = Pipe()
    s2, r2 = Pipe()

    y = generate_nice(80, 80, 80, 4, 2, 1e-6)

    a = y[:, :, :, :, 0]
    b = y[:, :, :, :, 1]
    a1 = a[:, :, :, 0]
    a2 = a[:, :, :, 1]

    p1 = Process(target=f, args=('1', 2, r1, a1, b,), kwargs={})
    p2 = Process(target=f, args=('2', 6, r2, a2, b,), kwargs={})
    p1.start()
    p2.start()
    print('main process recv')
    M1 = s1.recv()
    M2 = s2.recv()

    M = np.concatenate([M1, M2])
    print(M)
    print(type(M))
    print('computing sdtw')
    d, D = soft_dtw(M)
    D_bar = soft_dtw_sec_grad(D)
    print(D_bar)
    print('sending to children')
    s1.send(D_bar[0])
    s2.send(D_bar[1])
    print('sending done, waiting for grads')
    p1.join()
    p2.join()
