"""Measure naive matrix multiplication versus 2D thread block shape."""

import argparse
import csv
import math
from pathlib import Path
import random
import re
import statistics
import subprocess

from timing_common import plt, positive_int


def block_list(value):
    shapes = []
    for part in value.split(","):
        match = re.fullmatch(r"\s*([0-9]+)x([0-9]+)\s*", part)
        if not match or min(map(int, match.groups())) <= 0:
            raise argparse.ArgumentTypeError("expected positive block shapes, e.g. 16x16,32x8")
        shape = tuple(map(int, match.groups()))
        if shape not in shapes:
            shapes.append(shape)
    return shapes


def collect(binary, destination, shapes, samples):
    rng = random.Random(0)
    with destination.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["sample", "M", "N", "K", "block_x", "block_y", "elapsed_ms"])
        for sample in range(1, samples + 1):
            order = list(shapes)
            rng.shuffle(order)  # Avoid always testing the same shape on a colder GPU.
            for x, y in order:
                result = subprocess.run([str(binary), str(x), str(y)], text=True,
                                        capture_output=True, check=True)
                config = re.search(r"^M=(\d+) N=(\d+) K=(\d+) block_x=(\d+) block_y=(\d+)$",
                                   result.stdout, re.MULTILINE)
                timing = re.search(r"^([0-9.]+) \(ms\) elapsed$", result.stdout, re.MULTILINE)
                if not config or not timing:
                    raise ValueError(f"Unexpected benchmark output: {result.stdout!r}")
                m, n, k, actual_x, actual_y = map(int, config.groups())
                elapsed = float(timing[1])
                if (actual_x, actual_y) != (x, y) or not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError("Benchmark returned an unexpected shape or invalid timing")
                writer.writerow([sample, m, n, k, x, y, elapsed])
                output.flush()
                print(f"Sample {sample}/{samples}, {x}x{y}: {elapsed:.3f} ms", flush=True)


def analyze(source, output_dir):
    groups = {}
    matrices = set()
    with source.open(newline="") as data:
        for row in csv.DictReader(data):
            m, n, k, x, y = (int(row[field]) for field in ("M", "N", "K", "block_x", "block_y"))
            elapsed = float(row["elapsed_ms"])
            if min(m, n, k, x, y) <= 0 or not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Matrix dimensions, block dimensions, and timings must be positive")
            matrices.add((m, n, k))
            groups.setdefault((x, y), []).append(elapsed)
    if len(matrices) != 1:
        raise ValueError("Expected samples for exactly one matrix size")
    m, n, k = matrices.pop()
    rows = []
    for (x, y), timings in sorted(groups.items(), key=lambda item: (math.prod(item[0]), item[0])):
        rates = [2.0 * m * n * k / (elapsed * 1e9) for elapsed in timings]
        rows.append(dict(M=m, N=n, K=k, block_x=x, block_y=y, threads_per_block=x*y,
                         samples=len(timings), mean_ms=statistics.mean(timings),
                         std_ms=statistics.stdev(timings) if len(timings) > 1 else None,
                         mean_tflops=statistics.mean(rates),
                         std_tflops=statistics.stdev(rates) if len(rates) > 1 else None))
    with (output_dir / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        deviation = f"{row['std_ms']:.3f}" if row["std_ms"] is not None else "N/A"
        print(f"{row['block_x']}x{row['block_y']} ({row['threads_per_block']} threads): "
              f"mean={row['mean_ms']:.3f} ms, sample std={deviation} ms, "
              f"{row['mean_tflops']:.6f} TFLOPS ({row['samples']} samples)")
    best = min(rows, key=lambda row: row["mean_ms"])
    print(f"Fastest measured mean: {best['block_x']}x{best['block_y']}")

    labels = [f"{row['block_x']}×{row['block_y']}\n{row['threads_per_block']} threads" for row in rows]
    for metric, unit, suffix in [("ms", "GPU time per launch (ms)", "timings"),
                                 ("tflops", "FP32 throughput (TFLOPS)", "tflops")]:
        fig, ax = plt.subplots(figsize=(max(9, len(rows) * 0.9), 5.5))
        positions = list(range(len(rows)))
        ax.bar(positions, [row[f"mean_{metric}"] for row in rows], color="tab:blue")
        for pos, row in enumerate(rows):
            if row[f"std_{metric}"] is not None:
                ax.errorbar(pos, row[f"mean_{metric}"], yerr=row[f"std_{metric}"],
                            fmt="none", ecolor="black", capsize=4)
        ax.set_xticks(positions, labels, rotation=45, ha="right")
        ax.set_xlabel("Block shape (x × y), sorted by total threads")
        ax.set_ylabel(unit)
        error_note = "Error bars: ±1 sample standard deviation" if any(
            row["samples"] > 1 for row in rows) else "One sample per shape; no error bars"
        ax.set_title(f"Naive matmul: M={m}, N={n}, K={k}\n{error_note}")
        ax.set_ylim(bottom=0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"block_shape_{suffix}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/matmul_fat_matrices_00_naive"))
    parser.add_argument("--blocks", type=block_list,
                        default=block_list("8x8,16x8,8x16,16x16,32x8,8x32,32x16,16x32,32x32"),
                        help="Comma-separated x-by-y block shapes (e.g. 16x16,32x8)")
    parser.add_argument("--samples", type=positive_int, default=3,
                        help="Independent runs per shape (default: 3)")
    parser.add_argument("--input", type=Path, help="Replot samples.csv without accessing the GPU")
    parser.add_argument("--output-dir", type=Path, default=Path("results/matmul/fat_matrices/00_naive/blocks"))
    args = parser.parse_args()
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        source = args.input or args.output_dir / "samples.csv"
        if args.input is None:
            collect(args.binary.resolve(), source, args.blocks, args.samples)
        analyze(source, args.output_dir)
    except subprocess.CalledProcessError as error:
        parser.exit(1, f"Benchmark failed: {error.cmd}\n{error.stderr}")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
