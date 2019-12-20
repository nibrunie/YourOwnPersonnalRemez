import bigfloat
import random
import fpylll

import numpy as np
import matplotlib.pyplot as plt

function = lambda x: bigfloat.tanh(x)


for i in range(10):
    value = random.random()
    print(value, function(value))

poly_degree = 8
NUM_POINT = poly_degree + 1
interval_lo = 1.0
interval_hi = 1.05

interval_size = interval_hi - interval_lo

def get_random_interval_pt():
    return interval_lo + interval_size * random.random()

input_value = sorted([get_random_interval_pt() for i in range(NUM_POINT)])
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

target_vector = [function(x) for x in input_value]

matrix = fpylll.IntegerMatrix(NUM_POINT, poly_degree+1)
np_matrix = np.zeros((NUM_POINT, poly_degree+1))

# each column contains the i-th power of the input row
for row in range(NUM_POINT):
    for col in range(poly_degree+1):
        coeff = int_conv(input_value[row]**col)
        matrix[row, col] = coeff
        np_matrix[row, col] = coeff
print(matrix)

reduced_matrix = fpylll.IntegerMatrix(matrix)

fpylll.LLL.reduction(reduced_matrix)
print(reduced_matrix)

conv_target = [int_conv(v) for v in target_vector]

closest_vector = fpylll.CVP.closest_vector(reduced_matrix, conv_target)
print("closest vector: ", closest_vector)
print("distance : ", back_conv(max(abs(a - b) for a, b in zip(closest_vector, conv_target))))

#poly_coeff = [back_conv(v) for v in closest_vector]

b = np.array(closest_vector)
poly_coeff = [v for v in np.linalg.solve(np_matrix, b)]
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



