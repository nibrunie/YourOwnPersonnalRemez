from cvp_approx import (
    Function,
    generate_approx_remez_cvp, generate_approx_remez,
    generate_approx_cvp, dirty_supnorm,
    PolyIndexListConditionner, PolyDegreeConditionner,
)

import numpy as np
import matplotlib.pyplot as plt
import bigfloat

if __name__ == "__main__":
    func = Function(lambda x: bigfloat.tanh(x))
    interval_lo, interval_hi = 0.01, 0.125
    interval = interval_lo, interval_hi
    NUM_TEST_PTS = 10

    POLY_DEGREE = 8

    # remez method
    poly_remez_1 = generate_approx_remez(func, interval, poly_conditionner=PolyDegreeConditionner(POLY_DEGREE), epsilon=1e-6)
    poly_remez_3 = generate_approx_remez(func, interval, poly_conditionner=PolyDegreeConditionner(POLY_DEGREE), epsilon=1e-6, num_iter=3)
    poly_remez_5 = generate_approx_remez(func, interval, poly_conditionner=PolyDegreeConditionner(POLY_DEGREE), epsilon=1e-6, num_iter=5)

    print("max_diff for remez 1 is ", dirty_supnorm(poly_remez_1 - func, interval))
    print("max_diff for remez 3 is ", dirty_supnorm(poly_remez_3 - func, interval))
    print("max_diff for remez 5 is ", dirty_supnorm(poly_remez_5 - func, interval))



    # generating coefficients of polynomial approximation
    poly_cvp_1 = generate_approx_cvp(func, interval, NUM_POINT=200, precision=60, poly_conditionner=PolyIndexListConditionner(range(0, POLY_DEGREE+1, 2)))
    poly_cvp_2 = generate_approx_cvp(func, interval, NUM_POINT=120, precision=60, poly_conditionner=PolyIndexListConditionner(range(0, POLY_DEGREE+1, 2)))
    print("max_diff for poly_cvp_1 is", dirty_supnorm(poly_cvp_1 - func, interval))
    print("max_diff for poly_cvp_2 is", dirty_supnorm(poly_cvp_2 - func, interval))


    #poly_remez_cvp = generate_approx_remez_cvp(func, interval, poly_degree=POLY_DEGREE, epsilon=1e-6, num_iter=1)
    #print("max_diff for poly_remez_cvp is", dirty_supnorm(poly_remez_cvp - func, interval))

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
    #poly_remez_cvp_y = np.array([poly_remez_cvp(v) for v in x])

    #error_y = np.array([(poly_remez_cvp - func)(v) for v in x])


    plt.plot(x, tanh_y, label='tanh')
    plt.plot(x, poly_cvp_1_y, label='poly_cvp_1')
    plt.plot(x, poly_cvp_2_y, label='poly_cvp_2')
    plt.plot(x, poly_remez_1_y, label='poly_remez_1')
    plt.plot(x, poly_remez_3_y, label='poly_remez_3')
    plt.plot(x, poly_remez_5_y, label='poly_remez_5')
    #plt.plot(x, poly_remez_cvp_y, label='poly_remez_cvp')
    #plt.plot(x, error_y, label='error')

    plt.title("Simple Plot")

    plt.legend()

    plt.show()



