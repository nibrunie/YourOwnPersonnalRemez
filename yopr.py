import sys
import bigfloat
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json


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
                       default=1e-6, type=float,
                       help="target error")
    parser.add_argument("--plot", action="store_const", default=False, const="True",
                        help="plot function and approximation")
    parser.add_argument("--plot-error", action="store_const", default=False, const="True",
                        help="plot error")
    parser.add_argument('--degree', action='store',
                        default=4, type=int,
                        help='polynomial degree')
    parser.add_argument('--num-iter', action='store',
                        default=1,
                        type=int,
                        help='number of iteration (for remez methods)')
    parser.add_argument('--num-pts', action='store',
                        default=100,
                        help='number of points (for cvp methods)')
    parser.add_argument('--precision', action='store',
                        default=60,
                        help='precision')
    parser.add_argument('--num-plot-points', action='store',
                        default=100,
                        help='number of points in plots')
    parser.add_argument('--index-list', action='store',
                        default=None, type=(lambda s: [int(v) for v in s.split(',')]),
                        help='polynomial coefficient index list (overloads degree)')
    parser.add_argument('--dump-axf-approx', action='store',
                        default=None, type=str,
                        help='if set dump approximation in output file in AXF format')

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
        raise NotImplementedError
    elif args.method == "cvp":
        poly = generate_approx_cvp(func, interval, NUM_POINT=args.num_pts, precision=args.precision, poly_conditionner=poly_conditioner)
    else:
        raise NotImplementedError

    print("poly is {}".format(poly))
    # error
    max_diff = dirty_supnorm(poly - func, interval)
    print("max absolute diff is ", max_diff)

    if args.dump_axf_approx:
        with open(args.dump_axf_approx, "w") as out_stream:
            axf_dict = {
                "bound_high": str(interval[1]),
                "bound_low": str(interval[0]),
                "class": "!PieceWiseApprox",
                # TODO/FIXME: only an estimation, not a true error bound
                "error_bound": str(max_diff),
                "even": False,
                "odd": False,
                "max_degree": poly_conditioner.get_max_index(),
                "num_intervals": 1,
                "precision": "float", # TODO/FIXME
                "tag": "",
                "approx_list": {
                    "absolute": True,
                    # TODO/FIXME: only an estimation, not a true error bound
                    "approx_error": str(max_diff),
                    "class": "!SimplePolyApprox",
                    "degree_list": poly_conditioner.get_index_list(),
                    "format_list": ["float"] * len(poly_conditioner.get_index_list()),
                    "function": str(func),
                    "interval": "[{};{}]".format(interval[0], interval[1]),
                    "poly": {
						("%d" % i): str(c) for i, c in enumerate(poly.coeff_vector) 
                    }
                }
            }
            out_stream.write(json.dumps([axf_dict], sort_keys=True, indent=4))

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

