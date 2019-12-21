import bigfloat
import random
import fpylll

import numpy as np
import matplotlib.pyplot as plt

function = lambda x: bigfloat.tanh(x)


poly_degree = 8
NUM_POINT = 10
interval_lo = 1.0
interval_hi = 1.05

epsilon = 0.01

interval_size = interval_hi - interval_lo

def cheb_root(degree, index):
    """ Return the index root of a chebyshev polynomial of degree degree

        :param degree: Chebyshev polynomial degree
        :type degree: int
        :param index: index of the root to be returned
        :type index: int
        :return: index-th root of the degree-th Chebyshev polynomial
        :retype: bigfloat.BigFloat
    """
    return (interval_lo + interval_hi) / 2 + interval_size * 0.5 * bigfloat.cos(bigfloat.const_pi() * (index + 0.5) / bigfloat.BigFloat(degree))
def cheb_extrema(degree, index):
    """ Return the index-th extrream of a chebyshev polynomial of degree degree

        :param degree: Chebyshev polynomial degree
        :type degree: int
        :param index: index of the extrema to be returned
        :type index: int
        :return: index-th extrema of the degree-th Chebyshev polynomial
        :retype: bigfloat.BigFloat
    """
    return (interval_lo + interval_hi) / 2 + interval_size * 0.5 * bigfloat.cos(bigfloat.const_pi() * index / bigfloat.BigFloat(degree))

def get_random_interval_pt():
    """ generate a random point within [interval_lo, interval_hi]

        :return: random value in [interval_lo, interval_hi]
        :rtype: float
    """
    return interval_lo + interval_size * random.random()

#input_value = sorted([get_random_interval_pt() for i in range(NUM_POINT)])
input_value = [cheb_extrema(NUM_POINT, i) for i in range(NUM_POINT)]
input_value = sorted(input_value)
print("input_value={}".format(input_value))

# as tanh is increasing we can get min/max
# by looking at function value at interval bounds
min_value = function(input_value[0])
max_value = function(input_value[-1])

precision = 24
factor = 2**precision / min_value

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

def vector_eval(vector, value):
    acc = 0
    for i, c in enumerate(vector):
        acc += c * value**i
    return acc

diff = max(vector_eval(poly_coeff, v) - function(v) for v in input_value)
print("max_diff is {}".format(diff))
NUM_TEST_PTS = 10
print("testing on {} random points on the interval".format(NUM_TEST_PTS))
for i in range(NUM_TEST_PTS):
    pt = get_random_interval_pt()
    diff = max(diff, vector_eval(poly_coeff, pt) - function(pt))
print("max_diff is {}".format(diff))



# graphical representation
fig = plt.figure()  # an empty figure with no axes
fig.suptitle('No axes on this figure')  # Add a title so we know which it is

x = np.linspace(interval_lo, interval_hi, 100)
tanh_y = np.array([function(v) for v in x])
poly_y = np.array([vector_eval(poly_coeff, v) for v in x])
error_y = tanh_y - poly_y


plt.plot(x, tanh_y, label='tanh')
plt.plot(x, poly_y, label='poly')
plt.plot(x, error_y, label='error')

plt.title("Simple Plot")

plt.legend()

plt.show()



