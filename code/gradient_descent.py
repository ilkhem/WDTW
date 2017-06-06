import numpy as np


def proj_histo(histo):
    """
    HISTO
    :param histo: 
    :return: 
    """
    grid_size = len(histo)
    projection = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size - i):
            k = int((grid_size-1-i+j)/2)
            projection[grid_size - 1 - k, k] += histo[i,j]
    return projection


def proj_histo_transposed(histo):
    """
    LA TRANSPOSE DU HISTO FDP
    :param histo: 
    :return: 
    """
    # We need the transposed version since it is used in gradient descent
    # It correspond to the "endomorphisme adjoint" of the projection R
    n = len(histo)
    projection = np.zeros((n, n))
    for i in range(n):
        j = n - i - 1 # Only admissible E_ij, others are send to 0
        for p in range(n):
            q = 2 *j + 1 - n + p
            if 0 <= q < n-1:
                projection[p,q] += histo[i,j]
                projection[p,q+1] += histo[i,j]

    return projection


def compute_gradient(alpha, beta):
    return alpha + proj_histo_transposed(beta)


def parameter_update(m, grad, learning_rate, exponential=True):
    '''
        :param m: Current estimation of optimum
        :param grad: Gradient at local position
        :param learning_rate: the learning rate.
        :param exponential: Mention if we perform local grad update or exponential one
        :return: new estimation of m and its projection Rm
    '''
    if exponential:
        m_updated = np.multiply(m, np.exp(- learning_rate * grad))
        Rm_updated = proj_histo(m_updated)

        # To ensure the sum is one
        # total_mass = np.sum(m_updated + Rm_updated)
        # m_updated /= total_mass
        # Rm_updated /= total_mass

    else:
        m_updated = m - learning_rate * grad
        Rm_updated = proj_histo(m_updated)
    return m_updated, Rm_updated


def test_projection():
    n = 10
    A = np.zeros((n,n))
    for i in range(n):
        #for j in range(n):
        #    if i <= n - j - 1:
        #        A[i,j] = i + j
        A[i,0] = 1
        A[i,3] = 2
    print(A)
    print(proj_histo_transposed(A))

# test_projection()

def check_gradient(grad, f, m, h):
    # grad is supposed to be the gradient of f in m
    #   and h to be small
    e = f(m + h) - f(m) - np.dot(grad, h)
    return np.linalg.norm(e) / grad.size
