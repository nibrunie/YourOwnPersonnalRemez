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


def generate_approx(function, interval, NUM_POINT=100, poly_degree=4, epsilon=0.01, precision=53):
    """ Using Closest Vector Problem, generates a polynomial approximation of
        degree  poly_degree of function over interval """
    # point value to minimize polynomial - function distance
    # are chebyshev extrema
    input_value = [cheb_extrema(NUM_POINT, i, interval) for i in range(NUM_POINT)]
    input_value = sorted(input_value)
    print("input_value={}".format(input_value))

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
    print(matrix)

    reduced_matrix = fpylll.IntegerMatrix(matrix)

    # reducing matrix to simplify CVP search
    fpylll.LLL.reduction(reduced_matrix)
    print(reduced_matrix)

    conv_target = [int_conv(v) for v in target_vector]
    print("conv_target=", conv_target)

    closest_vector = fpylll.CVP.closest_vector(reduced_matrix, conv_target)
    print("closest vector: ", closest_vector)
    print("distance : ", back_conv(max(abs(a - b) for a, b in zip(closest_vector, conv_target))))

    #poly_coeff = [back_conv(v) for v in closest_vector]

    b = np.array(closest_vector)
    M = np_matrix.transpose()
    print("M=", M)
    print("b=", b)

    poly_coeff = [v for v in np.linalg.lstsq(M, b)[0]]
    print("poly_coeff: ", poly_coeff)

    return Polynomial(poly_coeff)



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
            print("zero found at {}".format(x))
            zeros.append(x)
        x += start_u
    return zeros

def find_extremas(fct, interval, start_pts=None, min_dist=0.01, delta=0.00001):
    derivative = fct.derivate()
    return find_zeros(derivative, interval, start_pts, min_dist, delta)


if __name__ == "__main__":
    func = Function(lambda x: bigfloat.cos(x))
    interval_lo, interval_hi = 0.0, 0.125
    interval = interval_lo, interval_hi
    # generating coefficients of polynomial approximation
    poly = generate_approx(func, interval, NUM_POINT=120, precision=60, poly_degree=8)

    # evaluating polynomial approximation on random points
    NUM_TEST_PTS = 10
    print("testing on {} random points on the interval".format(NUM_TEST_PTS))
    max_diff = eval_poly_vs_fct(poly, func, (get_random_interval_pt(interval) for i in range(NUM_TEST_PTS)))
    print("max_diff is {}".format(max_diff))

    zeros = find_zeros(poly - func, interval, min_dist=0.0001, delta=1e-10)
    print("zeros=", zeros)
    extremas = find_extremas((poly - func), interval, min_dist=0.0001, delta=1e-10)
    print("extremas=", extremas)
    max_diff = max(abs((poly - func)(x)) for x in extremas)
    print("max_diff=", max_diff)
    max_diff = max(abs((poly - func)(x)) for x in [interval_lo, interval_hi])
    print("max_diff=", max_diff)

    # graphical representation
    fig = plt.figure()  # an empty figure with no axes
    fig.suptitle('No axes on this figure')  # Add a title so we know which it is

    x = np.linspace(interval_lo, interval_hi, 100)
    tanh_y = np.array([func(v) for v in x])
    poly_y = np.array([poly(v) for v in x])
    error_y = tanh_y - poly_y


    plt.plot(x, tanh_y, label='tanh')
    plt.plot(x, poly_y, label='poly')
    # plt.plot(x, error_y, label='error')

    plt.title("Simple Plot")

    plt.legend()

    plt.show()



