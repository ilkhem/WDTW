from multiprocessing import Process, Pipe, Queue

import numpy as np
from chainer import cuda

from generate_data import generate_nice
from sinkhorn import sinkhorn_fb
from sdtw import soft_dtw, soft_dtw_grad
from wdtw import gradient_descent

GPU_COUNT = 7
GPU_PRIORITY = [2, 6, 0, 1, 4, 5, 3]
GPU_PRIORITY_REVERSE = list(np.argsort(GPU_PRIORITY))


def worker_cpu(pid, pipe, out_q, a, b, **kwargs):
    print('process', id)

    M, J = sinkhorn_fb(a, b, **kwargs)
    print('sending to main', pid)
    pipe.send(M)
    print('waiting to recv', pid)
    D_bar = pipe.recv()
    print('computing grad', pid)
    grad = J.dot(D_bar)
    print('updating', pid)
    out_q.put({id: grad})
    print('\t\tout of process', pid)

    return a


def worker_gpu(pid, i, j, pipe, out_q, a, b, **kwargs):
    print('process', pid)

    cuda.get_device_from_id(pid).use()
    print('copying to gpu')
    ag = cuda.to_gpu(a, device=pid)
    bg = cuda.to_gpu(b, device=pid)

    M, J = sinkhorn_fb(ag, bg, **kwargs)
    print('sending to main', pid)
    pipe.send(M)
    print('waiting to recv', pid)
    D_bar = pipe.recv()
    print('computing grad', pid)
    grad = cuda.to_cpu(J).dot(D_bar)
    print('updating', pid)
    out_q.put({(i, j): grad})
    print('\t\tout of process', pid)


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
        p = Process(target=worker_gpu, args=(GPU_PRIORITY[i], child_pipe, out_q, x[:, :, :, i], y,), kwargs={})
        procs.append(p)
        p.start()

    M = np.concatenate([ppe.recv() for ppe in p_pipes])
    d, D = soft_dtw(M)
    D_bar = soft_dtw_grad(D)
    for i in range(n):
        p_pipes[i].send(D_bar[i])

    results = {}

    for i in range(n):
        results.update(out_q.get())

    for p in procs:
        p.join()

    return results


def barycenter_0(x, y_list, n_grad_iter=20, lr=0.1):
    """
    Computes the barycenter of a list of shape-timeseries
    :param x: initial barycenter
    :param y_list: list of shape-timeseries of which we need to compute the barycenter
    :return: barycenter of y
    """

    y_lengths = [y.shape[3] for y in y_list]
    j_total = sum(y_lengths)
    # decide on the number of Processes to be spawn (parallel GPUs), this should be less than GPU_COUNT
    n = min(x.shape[3] * j_total, GPU_COUNT)
    big_y = np.concatenate([y for y in y_list], axis=-1)

    y_step = 3
    x_step = 1
    current_x_loop = 0


    for k in range(n_grad_iter):
        print('----------- iteration %d -----------' % k)
        grads = np.empty((x.shape[3], j_total, *x.shape[:3]))
        for i in range(x.shape[3]):
            for j in range(0, j_total, n):
                out_q = Queue()
                procs = []
                p_pipes = []
                c_pipes = []

                for g in range(n):
                    # spawn a Process and a Pipe per GPU
                    parent_pipe, child_pipe = Pipe()
                    p_pipes.append(parent_pipe)
                    c_pipes.append(child_pipe)
                    # p = Process(target=worker_gpu, args=(i, child_pipe, out_q, x_gpu[i], y_gpu[i],), kwargs={})
                    p = Process(target=worker_gpu,
                                args=(
                                GPU_PRIORITY[g], i, j + g, child_pipe, out_q, x[:, :, :, i], big_y[:, :, :, j + g]),
                                kwargs={})
                    procs.append(p)
                    p.start()

                M = np.concatenate([ppe.recv() for ppe in p_pipes])
                d, D = soft_dtw(M)
                D_bar = soft_dtw_grad(D)
                for g in range(n):
                    p_pipes[g].send(D_bar[g])

                results = {}

                for g in range(n):
                    results.update(out_q.get())

                for p in procs:
                    p.join()

                for g in range(n):
                    grads[g] += results[GPU_PRIORITY_REVERSE[g]]

        for i in range(x.shape[3]):
            x[:, :, :, i] = gradient_descent(x[:, :, :, i], grads[i], lr=0.1)

    return x


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
    # D_bar = soft_dtw_grad(D)
    # print('sending to children')
    # s1.send(D_bar[0])
    # s2.send(D_bar[1])
    # print('sending done, waiting for grads')
    # p1.join()
    # p2.join()
