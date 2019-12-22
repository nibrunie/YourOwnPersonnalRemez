import bigfloat
import random
import fpylll

import numpy as np
import matplotlib.pyplot as plt


def cheb_root(degree, index, interval=(-1, 1)):
    """ Return the index root of a chebyshev polynomial of degree degree
        linearly mapped onto interval

        :param degree: Chebyshev polynomial degree
        :type degree: int
        :param index: index of the root to be returned
        :type index: int
        :return: index-th root of the degree-th Chebyshev polynomial
        :retype: bigfloat.BigFloat
    """
    lo, hi = interval
    size = hi - lo
    return (lo + hi) / 2 + size * 0.5 * bigfloat.cos(bigfloat.const_pi() * (index + 0.5) / bigfloat.BigFloat(degree))

def cheb_extrema(degree, index, interval=(-1, 1)):
    """ Return the index-th extrema of a chebyshev polynomial of degree degree
        linearly mapped onto interval

        :param degree: Chebyshev polynomial degree
        :type degree: int
        :param index: index of the extrema to be returned
        :type index: int
        :return: index-th extrema of the degree-th Chebyshev polynomial
        :retype: bigfloat.BigFloat
    """
    lo, hi = interval
    size = hi - lo
    return (lo + hi) / 2 + size * 0.5 * bigfloat.cos(bigfloat.const_pi() * index / bigfloat.BigFloat(degree))

def get_random_interval_pt(interval):
    """ generate a random point within interval (tuple (lo, hi))

        :param interval: value where random value must be picked
        :type interval: tuple(float, float)
        :return: random value in [lo, hi]
        :rtype: float
    """
    lo, hi = interval
    size = hi - lo
    return lo + size * random.random()


def generate_approx_cvp(function, interval, NUM_POINT=100, poly_degree=4, epsilon=0.01, precision=53):
    """ Using Closest Vector Problem, generates a polynomial approximation of
        degree  poly_degree of function over interval """
    # point value to minimize polynomial - function distance
    # are chebyshev extrema
    input_value = [cheb_extrema(NUM_POINT, i, interval) for i in range(NUM_POINT)]
    input_value = sorted(input_value)

    # as tanh is increasing we can get min/max
    # by looking at function value at interval bounds
    min_value = function(input_value[0])
    max_value = function(input_value[-1])

    factor = 2**precision / min_value

    # local conversion to/from integer functions
    def int_conv(x):
        return int(x * factor)
    def back_conv(x):
        return x / factor

    target_vector = [function(x) for i, x in enumerate(input_value)]

    matrix = fpylll.IntegerMatrix(poly_degree + 1, NUM_POINT)
    np_matrix = np.zeros((poly_degree + 1, NUM_POINT))

    # each column contains the i-th power of the input row
    for row in range(NUM_POINT):
        for col in range(poly_degree+1):
            coeff = int_conv(input_value[row]**col)
            matrix[col, row] = coeff
            np_matrix[col, row] = coeff

    reduced_matrix = fpylll.IntegerMatrix(matrix)

    # reducing matrix to simplify CVP search
    fpylll.LLL.reduction(reduced_matrix)

    conv_target = [int_conv(v) for v in target_vector]

    closest_vector = fpylll.CVP.closest_vector(reduced_matrix, conv_target)
    print("distance : ", back_conv(max(abs(a - b) for a, b in zip(closest_vector, conv_target))))

    #poly_coeff = [back_conv(v) for v in closest_vector]

    b = np.array(closest_vector)
    M = np_matrix.transpose()

    poly_coeff = [v for v in np.linalg.lstsq(M, b)[0]]

    return Polynomial(poly_coeff)


def generate_approx_remez(function, interval, poly_degree=4, epsilon=0.01, precision=53, num_iter=1):
    NUM_POINT = poly_degree + 1
    input_value = [cheb_extrema(NUM_POINT, i, interval) for i in range(NUM_POINT)]
    input_value = sorted(input_value)

    for iter_id in range(num_iter):
        target_vector = np.asarray([function(x) + (-1)**(i+1) * epsilon for i, x in enumerate(input_value)], dtype='float')

        # current problem definition is using float or exact values

        np_matrix = np.zeros((NUM_POINT, poly_degree + 1), dtype='float')
        for row in range(NUM_POINT):
            for col in range(poly_degree +1):
                np_matrix[row][col] = input_value[row]**col

        lstsq_solution = np.linalg.solve(np_matrix, target_vector)
        poly_coeff = [v for v in lstsq_solution]

        poly = Polynomial(poly_coeff)
        if iter_id + 1 < num_iter:
            extremas = find_extremas(poly - func, interval, min_dist=0.0001, delta=1e-8)
            input_value = [interval[0]] + sorted(extremas) + [interval[1]]
    return poly

class Function:
    def __init__(self, func):
        self.func = func

    def __call__(self, x):
        return self.func(x)

    def __sub__(self, func):
        return Function(lambda x: (self(x) - func(x)))
    def __add__(self, func):
        return Function(lambda x: (self(x) + func(x)))
    def __abs__(self):
        return Function(lambda x: (abs(self(x))))

    def derivate(self, u=0.0001):
        return Function(lambda x: (self(x+u) - self(x)) / u)

class Polynomial(Function):
    def __init__(self, coeff_vector):
        self.coeff_vector = coeff_vector

    def eval(self, x):
        """ Evaluate polynomial defined by poly_coeff list of
            coefficient numerical value at value """
        acc = 0
        for i, c in enumerate(self.coeff_vector):
            acc += c * x**i
        return acc

    def __call__(self, x):
        return self.eval(x)

    def derivate(self, u = None):
        return Polynomial([v * (i + 1) for i, v in enumerate(self.coeff_vector[1:])])

def eval_poly_vs_fct(poly, function, test_values):
    diff = max(abs(poly(v) - function(v)) for v in test_values)
    return diff


def find_zeros(fct, interval, start_pts=None, min_dist=0.01, delta=0.00001):
    start_u = min_dist
    lo, hi = interval
    size = hi - lo
    x = lo
    zeros = []
    while x <= hi:
        u = start_u
        while abs(fct(x)) > delta and x < hi:
            if fct(x+u) * fct(x) < 0:
                # opposite sign means at least one zero in between
                # because we assume fct is contiguous
                u /= 2.0
            else:
                x += u
        if abs(fct(x)) < delta:
            zeros.append(x)
        x += start_u
    return zeros

def find_extremas(fct, interval, start_pts=None, min_dist=0.01, delta=0.00001):
    derivative = fct.derivate()
    return find_zeros(derivative, interval, start_pts, min_dist, delta)

def dirty_supnorm(fct, interval, min_dist=0.01, delta=0.000001):
    return max(abs(fct(v)) for v in list(interval) + find_extremas(fct, interval, min_dist=min_dist, delta=delta))


if __name__ == "__main__":
    func = Function(lambda x: bigfloat.cos(x))
    interval_lo, interval_hi = 0.0, 0.125
    interval = interval_lo, interval_hi
    NUM_TEST_PTS = 10

    POLY_DEGREE = 5


    # remez method
    poly_remez_1 = generate_approx_remez(func, interval, poly_degree=POLY_DEGREE, epsilon=1e-6)
    poly_remez_3 = generate_approx_remez(func, interval, poly_degree=POLY_DEGREE, epsilon=1e-6, num_iter=3)
    poly_remez_5 = generate_approx_remez(func, interval, poly_degree=POLY_DEGREE, epsilon=1e-6, num_iter=5)

    print("max_diff for remez 1 is ", dirty_supnorm(poly_remez_1 - func, interval))
    print("max_diff for remez 3 is ", dirty_supnorm(poly_remez_3 - func, interval))
    print("max_diff for remez 5 is ", dirty_supnorm(poly_remez_5 - func, interval))



    # generating coefficients of polynomial approximation
    poly_cvp_1 = generate_approx_cvp(func, interval, NUM_POINT=200, precision=60, poly_degree=POLY_DEGREE)
    poly_cvp_2 = generate_approx_cvp(func, interval, NUM_POINT=120, precision=40, poly_degree=POLY_DEGREE)
    print("max_diff for poly_cvp_1 is", dirty_supnorm(poly_cvp_1 - func, interval))
    print("max_diff for poly_cvp_2 is", dirty_supnorm(poly_cvp_2 - func, interval))

    # graphical representation
    fig = plt.figure()  # an empty figure with no axes
    fig.suptitle('No axes on this figure')  # Add a title so we know which it is

    x = np.linspace(interval_lo, interval_hi, 100)
    tanh_y = np.array([func(v) for v in x])
    poly_cvp_1_y = np.array([poly_cvp_1(v) for v in x])
    poly_cvp_2_y = np.array([poly_cvp_2(v) for v in x])
    poly_remez_1_y = np.array([poly_remez_1(v) for v in x])
    poly_remez_3_y = np.array([poly_remez_3(v) for v in x])
    poly_remez_5_y = np.array([poly_remez_5(v) for v in x])


    plt.plot(x, tanh_y, label='tanh')
    plt.plot(x, poly_cvp_1_y, label='poly_cvp_1')
    plt.plot(x, poly_cvp_2_y, label='poly_cvp_2')
    plt.plot(x, poly_remez_1_y, label='poly_remez_1')
    plt.plot(x, poly_remez_3_y, label='poly_remez_3')
    plt.plot(x, poly_remez_5_y, label='poly_remez_5')
    # plt.plot(x, error_y, label='error')

    plt.title("Simple Plot")

    plt.legend()

    plt.show()



