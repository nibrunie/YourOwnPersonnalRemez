import bigfloat
import random
import fpylll

import numpy as np


def cheb_root(degree, index, interval=(-1, 1)):
    """ Return the index root of a chebyshev polynomial of degree degree
        linearly mapped onto interval

        :param degree: Chebyshev polynomial degree
        :type degree: int
        :param index: index of the root to be returned
        :type index: int
        :return: index-th root of the degree-th Chebyshev polynomial
        :rtype: bigfloat.BigFloat
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
        :rtype: bigfloat.BigFloat
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

class PolyConditionner:
    """ Conditionning object to shape a polynomial object """
    def get_index_list(self):
        """ Return the ordered list of indexes of non-zero
            polynomial coefficient """
        raise NotImplementedError

    def get_max_index(self):
        """ Return the maximal index of a non-zero coefficient """
        raise NotImplementedError

    def build_poly_from_coeff_list(self, coeff_list, zero=0.0):
        """ Build a polynomial from a sparse list of coefficients
            assuming they are mapped to the conditionner indexes """
        dense_list = [zero] * (self.get_max_index() + 1)
        print(len(coeff_list), self.get_index_list(), self.get_max_index())
        for index, value in zip(self.get_index_list(), coeff_list):
            dense_list[index] = value
        return Polynomial(dense_list)

class PolyDegreeConditionner(PolyConditionner):
    """ Conditionnal object to define a polynomial shape from its degree """
    def __init__(self, poly_degree):
        self.poly_degree = poly_degree

    def get_index_list(self):
        return list(range(self.poly_degree+1))
    def get_max_index(self):
        return self.poly_degree

class PolyIndexListConditionner(PolyConditionner):
    """ Conditionnal object to define a polynomial shape from a list of
        coefficient indexes """
    def __init__(self, poly_index_list):
        self.poly_index_list = list(poly_index_list)
        self.poly_degree = max(self.poly_index_list)
    def get_index_list(self):
        return self.poly_index_list
    def get_max_index(self):
        return self.poly_degree


def generate_approx_cvp(function, interval, NUM_POINT=100, poly_conditionner=None, epsilon=0.01, precision=53):
    """ Using Closest Vector Problem, generates a polynomial approximation of
        degree  poly_degree of function over interval """
    poly_conditionner = poly_conditionner or PolyDegreeConditionner(4)
    # point value to minimize polynomial - function distance
    # are chebyshev extrema
    input_value = [cheb_extrema(NUM_POINT, i, interval) for i in range(NUM_POINT)]
    input_value = sorted(input_value)

    target_vector = [function(x) for i, x in enumerate(input_value)]

    min_value = min(abs(v) for v in target_vector)

    factor = 2**precision / min_value

    # local conversion to/from integer functions
    def int_conv(x):
        return int(x * factor)
    def back_conv(x):
        return x / factor

    poly_index_list = poly_conditionner.get_index_list()
    NUM_POLY_INDEX = len(poly_index_list)

    matrix = fpylll.IntegerMatrix(NUM_POLY_INDEX, NUM_POINT)
    np_matrix = np.zeros((NUM_POLY_INDEX, NUM_POINT))

    # each column contains the i-th power of the input row
    for row in range(NUM_POINT):
        for col, power in enumerate(poly_index_list):
            coeff = int_conv(input_value[row]**power)
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

    return poly_conditionner.build_poly_from_coeff_list(poly_coeff)


def generate_approx_remez(function, interval, poly_conditionner=None, epsilon=0.01, precision=53, num_iter=1):
    """ Using Remez method find an approximation polynoial of function over
        interval whose degree is poly_defree and whose absolute error is less
        or equal to epsilon """
    poly_conditionner = poly_conditionner or PolyDegreeConditionner(4)
    poly_degree = poly_conditionner.get_max_index()
    poly_index_list = poly_conditionner.get_index_list()
    POLY_SIZE = len(poly_index_list)

    NUM_POINT = poly_degree + 1
    input_value = [cheb_extrema(NUM_POINT, i, interval) for i in range(NUM_POINT)]
    input_value = sorted(input_value)

    for iter_id in range(num_iter):
        target_vector = np.asarray([function(x) + (-1)**(i+1) * epsilon for i, x in enumerate(input_value)], dtype='float')

        # current problem definition is using float or exact values

        np_matrix = np.zeros((NUM_POINT, POLY_SIZE), dtype='float')
        for row in range(NUM_POINT):
            for col, power in enumerate(poly_index_list):
                np_matrix[row][col] = input_value[row]**power

        if POLY_SIZE == poly_degree + 1:
            lstsq_solution = np.linalg.solve(np_matrix, target_vector)
        else:
            lstsq_solution = np.linalg.lstsq(np_matrix, target_vector)
        poly_coeff = [v for v in lstsq_solution]
        print("remez poly_coeff=", poly_coeff)

        poly = poly_conditionner.build_poly_from_coeff_list(poly_coeff)
        if iter_id + 1 < num_iter:
            extremas = find_extremas(poly - function, interval, min_dist=0.0001, delta=1e-8)
            input_value = [interval[0]] + sorted(extremas) + [interval[1]]
    return poly

def generate_approx_remez_cvp(function, interval, poly_degree=4, epsilon=0.01, precision=53, num_iter=1):
    """ Using Remez method find an approximation polynoial of function over
        interval whose degree is poly_defree and whose absolute error is less
        or equal to epsilon """
    NUM_POINT = poly_degree + 1
    input_value = [cheb_extrema(NUM_POINT, i, interval) for i in range(NUM_POINT)]
    input_value = sorted(input_value)

    for iter_id in range(num_iter):
        target_vector = np.asarray([function(x) + (-1)**(i) * epsilon for i, x in enumerate(input_value)], dtype='float')

        min_value = min(abs(v) for v in target_vector)
        factor = 2**precision / min_value
        def int_conv(x):
            return int(x * factor)
        def back_conv(x):
            return x / factor

        fplll_matrix = fpylll.IntegerMatrix(poly_degree + 1, NUM_POINT)
        np_matrix = np.zeros((NUM_POINT, poly_degree + 1), dtype='float')
        for row in range(NUM_POINT):
            for col in range(poly_degree +1):
                # np matrix MUST be transposed wrt fplll matriix
                np_matrix[row][col] = int_conv(input_value[row]**col)
                fplll_matrix[col, row] = int_conv(input_value[row]**col)

        reduced_matrix = fpylll.IntegerMatrix(fplll_matrix)

        # reducing matrix to simplify CVP search
        #fpylll.LLL.reduction(reduced_matrix)

        conv_target = [int_conv(v) for v in target_vector]

        closest_vector = fpylll.CVP.closest_vector(reduced_matrix, conv_target)
        b = np.array(closest_vector)
        M = np_matrix
        #M = np_matrix.transpose()
        poly_coeff = [v for v in np.linalg.lstsq(M, b)[0]]
        print("remez_cvp poly_coeff=", poly_coeff)
        #poly_coeff = [back_conv(v) for v in closest_vector]

        poly = Polynomial(poly_coeff)
        if iter_id + 1 < num_iter:
            extremas = find_extremas(poly - func, interval, min_dist=0.0001, delta=1e-8)
            print(input_value)
            print(extremas)
            input_value = [interval[0]] + sorted(extremas) + [interval[1]]
            print(len(input_value), NUM_POINT)
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


