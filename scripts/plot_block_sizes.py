"""Measure vector_add latency versus threads per block."""

from timing_common import argument_parser, integer_list, positive_int, run


def main():
    parser = argument_parser(__doc__, "results/vector_add/00_vector_add/threads_sweep")
    parser.add_argument("--n", type=positive_int, default=100000,
                        help="Fixed element count (default: 100000)")
    parser.add_argument("--threads", type=integer_list, default=[64, 128, 256, 512, 1024],
                        help="Comma-separated block sizes (default: 64,128,256,512,1024)")
    args = parser.parse_args()
    run(parser, args, [args.n], args.threads, "threads")


if __name__ == "__main__":
    main()
