from sinkhorn import sinkhorn_chainer
from generate_data import generate_data
from chainer import Variable

if __name__ == "__main__":
    x, y = generate_data()
    y = x + 5
    y /= y.sum(axis=(0,1,2))
    d = sinkhorn_chainer(Variable(x), Variable(x))
    print(d.data)

    d1 = sinkhorn_chainer(x, y)
    print(d1.data)
