import sys
import bigfloat
import argparse
import numpy as np
import matplotlib.pyplot as plt


from cvp_approx import (
    Function,
    PolyIndexListConditionner, PolyDegreeConditionner,
    dirty_supnorm,
    generate_approx_remez, generate_approx_cvp,
    generate_approx_remez_cvp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command-line interface for Your Own Personnal Remez')
    parser.add_argument('--function', action='store',
                        default=bigfloat.tanh, type=(lambda s: eval(s, globals())),
                        help='function to approximate')
    parser.add_argument('--method', action='store',
                        default="remez", choices=["remez", "cvp", "remez_cvp"],
                        help='method for the approximation')
    parser.add_argument("--interval", action="store",
                       default=(0,1.0), type=(lambda s: [float(v) for v in s.split(',')]),
                        help='approximation interval')
    parser.add_argument("--epsilon", action="store",
                       default=1e-6,
                       help="target error")
    parser.add_argument("--plot", action="store_const", default=False, const="True",
                        help="plot function and approximation")
    parser.add_argument("--plot-error", action="store_const", default=False, const="True",
                        help="plot error")
    parser.add_argument('--degree', action='store',
                        default=4,
                        help='polynomial degree')
    parser.add_argument('--num-iter', action='store',
                        default=1,
                        help='number of iteration (for iterative methods)')
    parser.add_argument('--num-pts', action='store',
                        default=100,
                        help='number of iteration (for iterative methods)')
    parser.add_argument('--precision', action='store',
                        default=60,
                        help='precision')
    parser.add_argument('--num-plot-points', action='store',
                        default=100,
                        help='number of points in plots')
    parser.add_argument('--index-list', action='store',
                        default=None, type=(lambda s: [int(v) for v in s.split(',')]),
                        help='polynomial coefficient index list (overloads degree)')

    args = parser.parse_args(sys.argv[1:])

    func = Function(lambda x: args.function(x))
    interval = args.interval
    NUM_PLOT_PTS = args.num_plot_points

    poly_conditioner = None

    if not args.index_list is None:
        poly_conditioner = PolyIndexListConditionner(args.index_list)
    else:
        poly_conditioner = PolyDegreeConditionner(args.degree)

    if args.method == "remez":
        poly = generate_approx_remez(func, interval, poly_conditioner, args.epsilon, num_iter=args.num_iter)
    elif args.method == "remez_cvp":
        pass
    elif args.method == "cvp":
        poly = generate_approx_cvp(func, interval, NUM_POINT=args.num_pts, precision=args.precision, poly_conditionner=poly_conditioner)
    else:
        raise NotImplementedError

    # error
    max_diff = dirty_supnorm(poly - func, interval)
    print("max absolute diff is ", max_diff)

    if args.plot or args.plot_error:
        # graphical representation
        fig = plt.figure()  # an empty figure with no axes
        fig.suptitle('Subtitle')  # Add a title so we know which it is

        x = np.linspace(interval[0], interval[1], NUM_PLOT_PTS)

        if args.plot:
            func_y = np.array([func(v) for v in x])
            poly_y = np.array([poly(v) for v in x])
            plt.plot(x, func_y, label='func')
            plt.plot(x, poly_y, label='poly')
        if args.plot_error:
            error_y = np.array([(poly - func)(v) for v in x])
            plt.plot(x, error_y, label='error')

        plt.title("Simple Plot")
        plt.legend()
        plt.show()

