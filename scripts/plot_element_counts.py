"""Measure vector_add latency versus number of elements."""

from timing_common import argument_parser, integer_list, positive_int, run


def main():
    parser = argument_parser(__doc__, "results/vector_add/00_vector_add/elements_sweep")
    parser.add_argument("--n", type=integer_list,
                        default=[2**exponent for exponent in range(10, 26)],
                        help="Comma-separated element counts (default: powers of two from 2^10 to 2^25)")
    parser.add_argument("--threads", type=positive_int, default=128,
                        help="Fixed threads per block (default: 128)")
    args = parser.parse_args()
    run(parser, args, args.n, [args.threads], "n")


if __name__ == "__main__":
    main()
