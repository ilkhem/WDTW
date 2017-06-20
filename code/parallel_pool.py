from multiprocessing import Pool, Lock

import numpy as np
from chainer import cuda

from sinkhorn import sinkhorn_fb
from sdtw import soft_dtw, soft_dtw_grad
from gradient_descent import gradient_descent

GPUCOUNT = 7
GPU_PRIORITY = [2, 6, 0, 1, 4, 5, 3]


def prepare_data(X, Y_list):
    # example of the iterable: [(x, y, gpuid #the id of the gpu, i #index of x in X,  j #index of y in Y,
    # k #index of Y in Y_list), for Y in Y_list, for x in X, for y in Y, for gpuid in range(GPUCOUNT)]
    args_list = []

    gpuid = 0
    for i in range(X.shape[3]):
        for k, Y in enumerate(Y_list):
            for j in range(Y.shape[3]):
                args_list.append((X[:, :, :, i], Y[:, :, :, j], GPU_PRIORITY[gpuid], i, j, k))
                gpuid += 1
                if gpuid > GPUCOUNT - 1:
                    gpuid = 0
    return args_list


def worker(x, y, gpuid, i, j, k):
    print('\tWorking on gpu %d for indices (%d, %d, %d)' % (gpuid, i, j, k))
    l_ = locks[gpuid]
    with l_:  # lock access to GPU gpuid
        cuda.get_device_from_id(gpuid).use()  # current process will use GPU gpuid
        xg = cuda.to_gpu(x, device=gpuid)  # copy data to current GPU
        yg = cuda.to_gpu(y, device=gpuid)  # copy data to current GPU
        Mg, Jg = sinkhorn_fb(xg, yg)  # compute sinkhorn distances, and gradient with respect to x
        M, J = cuda.to_cpu(Mg), cuda.to_cpu(Jg)
    return {(i, j, k): (M, J)}  # return a dict


# access to GPUs should be locked: only one process can access to a GPU at a time
def init_locks(l):
    global locks
    locks = l


def _single_gradient_step(X, Y_list):
    x_shape = X.shape
    y_shapes = [Y.shape for Y in Y_list]
    args_list = prepare_data(X, Y_list)

    locks = []
    for _ in range(GPUCOUNT):
        locks.append(Lock())
    pool = Pool(maxtasksperchild=1, processes=GPUCOUNT, initializer=init_locks, initargs=(locks,))
    res = pool.starmap(worker, args_list, chunksize=1)
    pool.close()
    pool.join()

    M_glob = [np.empty((x_shape[3], y_shape[3])) for y_shape in y_shapes]
    J_glob = [np.empty((*x_shape, y_shape[3])) for y_shape in y_shapes]

    for r in res:
        i, j, k = list(r.keys())[0]
        M, J = list(r.values())[0]
        M_glob[k][i, j] = M
        J_glob[k][:, :, :, i, j] = J

    final_gradient = np.empty(x_shape)
    for M, J in zip(M_glob, J_glob):
        d, D = soft_dtw(M)
        D_bar = soft_dtw_grad(D)
        final_gradient += J.dot(D_bar)

    return final_gradient


def main(X, Y_list, lr, n_g_iter):

    x = X

    for _ in n_g_iter:
        g = _single_gradient_step(x, Y_list)
        x = gradient_descent(x, g, lr=lr, norm=True)

    return x



if __name__ == '__main__':

    import sys
    from generate_data import generate_nice

    try:
        d1 = int(sys.argv[1])
    except:
        d1 = 80

    y_ = generate_nice(d1, d1, d1, 4, 5, 1e-6)
    X = y_[:, :, :, :, 0]
    Y_list = [y_[:, :, :, :, i] for i in range(1, y_.shape[4])]

    res_dict = _single_gradient_step(X, Y_list)
    print('done')
