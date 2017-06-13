from multiprocessing import Process, Pipe, Queue

import numpy as np
from chainer import cuda

from generate_data import generate_nice
from sdtw import soft_dtw, soft_dtw_sec_grad
from wdtw import sinkhorn_fb, gradient_descent

GPU_COUNT = 7


def worker_cpu(id, pipe, out_q, a, b, **kwargs):
    print('process', id)
    if a.ndim == 3:
        a = a.reshape((*a.shape, 1))

    M, J = sinkhorn_fb(a, b, **kwargs)
    print('sending to main', id)
    pipe.send(M)
    print('waiting to recv', id)
    D_bar = pipe.recv()
    print('computing grad', id)
    grad = J.dot(D_bar)
    print('updating', id)
    out_q.put({id: gradient_descent(a, grad, 0.1)})

    return a


def worker_gpu(pid, pipe, out_q, a, b, **kwargs):
    print('process', pid)

    cuda.get_device_from_id(pid).use()
    print('copying to gpu')
    ag = cuda.to_gpu(a, device=pid)
    if ag.ndim == 3:
        ag = ag.reshape((*a.shape, 1))
    bg = cuda.to_gpu(b, device=pid)

    M, J = sinkhorn_fb(ag, bg, **kwargs)
    print('sending to main', pid)
    pipe.send(M)
    print('waiting to recv', pid)
    D_bar = cuda.to_gpu(pipe.recv(), device=pid)
    print('computing grad', pid)
    grad = J.dot(D_bar)
    print('updating', pid)
    out_q.put({pid: gradient_descent(ag, grad, 0.1)})


def _single_gradient_step(x, y):
    """
    Version 0:
        - assume len(x) <= 7
        - assume f/b computation of sinkhorn(x_i, y) has low enough memory footprint to fit in 12Gb (Lowest GPU memory)
    """

    out_q = Queue()
    procs = []
    p_pipes = []
    c_pipes = []

    n = min(x.shape[3], GPU_COUNT)
    # x_gpu = []
    # y_gpu = []
    # for i in range(n):
    #     print('copying to gpu')
    #     x_gpu.append(cuda.to_gpu(x[:, :, :, i], device=i))
    #     y_gpu.append(cuda.to_gpu(y, device=i))

    for i in range(n):
        # spawn a Process and a Pipe per GPU
        parent_pipe, child_pipe = Pipe()
        p_pipes.append(parent_pipe)
        c_pipes.append(child_pipe)
        # p = Process(target=worker_gpu, args=(i, child_pipe, out_q, x_gpu[i], y_gpu[i],), kwargs={})
        p = Process(target=worker_gpu, args=(i, child_pipe, out_q, x[:, :, :, i], y,), kwargs={})
        procs.append(p)
        p.start()

    M = np.concatenate([ppe.recv() for ppe in p_pipes])
    d, D = soft_dtw(M)
    D_bar = soft_dtw_sec_grad(D)
    for i in range(n):
        p_pipes[i].send(D_bar[i])

    results = {}

    for i in range(GPU_COUNT):
        results.update(out_q.get())

    for p in procs:
        p.join()

    return results


if __name__ == '__main__':
    # s1, r1 = Pipe()
    # s2, r2 = Pipe()
    import sys

    d1 = int(sys.argv[1])

    y = generate_nice(d1, d1, d1, 4, 4, 1e-6)

    results = _single_gradient_step(y[:, :, :, :, 0], y[:, :, :, :, 1])
    #
    # a = y[:, :, :, :, 0]
    # b = y[:, :, :, :, 1]
    # a1 = a[:, :, :, 0]
    # a2 = a[:, :, :, 1]
    #
    # out_q = Queue()
    #
    # p1 = Process(target=worker_cpu, args=(1, r1, a1, b,), kwargs={})
    # p2 = Process(target=worker_cpu, args=(2, r2, a2, b,), kwargs={})
    #
    # p1 = Process(target=worker_gpu, args=(1, r1, out_q, a1, b,), kwargs={})
    # p2 = Process(target=worker_gpu, args=(2, r2, out_q, a2, b,), kwargs={})
    #
    # p1.start()
    # p2.start()
    # print('main process recv')
    # M1 = s1.recv()
    # M2 = s2.recv()
    #
    # M = np.concatenate([M1, M2])
    # print('computing sdtw')
    # d, D = soft_dtw(M)
    # D_bar = soft_dtw_sec_grad(D)
    # print('sending to children')
    # s1.send(D_bar[0])
    # s2.send(D_bar[1])
    # print('sending done, waiting for grads')
    # p1.join()
    # p2.join()
