from multiprocessing import Pool, Lock

from chainer import cuda

from wdtw import sinkhorn_fb

GPUCOUNT = 7
GPU_PRIORITY = [2, 6, 0, 1, 4, 5, 3]


def prepare_data(X, Y_list):
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
    print('Working on gpu %d for indices (%d, %d, %d)' % (gpuid, i, j, k))
    l_ = lock[gpuid]
    with l_:  # lock access to GPU gpuid
        cuda.get_device_from_id(gpuid).use()  # current process will use GPU gpuid
        xg = cuda.to_gpu(x, device=gpuid)  # copy data to current GPU
        yg = cuda.to_gpu(y, device=gpuid)  # copy data to current GPU
        M, J = sinkhorn_fb(xg, yg)  # compute sinkhorn distances, and gradient with respect to x
    return {(i, j, k): (M, J)}  # return a dict


# access to GPUs should be locked: only one process can access to a GPU at a time.
# lock = [None] * GPUCOUNT


def init_locks(l):
    global lock
    lock = l


# example of the iterable: [(x, y, gpuid #the id of the gpu, i #index of x in X,  j #index of y in Y,
# k #index of Y in Y_list), for Y in Y_list, for x in X, for y in Y, for gpuid in range(GPUCOUNT)]

def main(args_list):
    lock = [Lock()] * GPUCOUNT
    pool = Pool(processes=GPUCOUNT, maxtasksperchild=4, initializer=init_locks, initargs=(lock,))
    res = pool.starmap_async(worker, args_list, chunksize=1)
    pool.close()
    pool.join()

    res_dict = {}
    for r in res:
        res_dict.update(r)

    return res_dict


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

    args_list = prepare_data(X, Y_list)
    res_dict = main(args_list)
    print('done')
