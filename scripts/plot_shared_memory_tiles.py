"""Plot shared-memory matmul latency for different tile configurations at M=N=K=4096."""

import argparse
import csv
from pathlib import Path
import random
import re
import statistics
import subprocess

from matplotlib.patches import Patch

from timing_common import plt, positive_int


def tile_list(text):
    try:
        tiles = list(dict.fromkeys(tuple(map(int, item.split('x'))) for item in text.split(',')))
        if any(len(tile) != 3 or any(v <= 0 or 4096 % v for v in tile)
               or tile[0] * tile[1] > 1024 for tile in tiles):
            raise ValueError
        return tiles
    except ValueError:
        raise argparse.ArgumentTypeError('Use MxNxK tiles dividing 4096 with M*N <= 1024')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles', type=tile_list,
                        default=[(m, n, k) for m, n in [(4, 16), (4, 32), (8, 32), (16, 16), (32, 8),
                                                       (16, 32), (32, 16), (32, 32)]
                                 for k in [8, 16, 32]],
                        help='Comma-separated BLOCK_MxBLOCK_NxBLOCK_K configurations')
    parser.add_argument('--samples', type=positive_int, default=3)
    parser.add_argument('--input', type=Path, help='Replot saved samples without GPU work')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('results/matmul/fat_matrices/01_shared_memory/tile_configurations'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.input or args.output_dir / 'samples.csv'
    if args.input is None:
        rng = random.Random(0)
        with source.open('w', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(['sample', 'M', 'N', 'K', 'BLOCK_M', 'BLOCK_N', 'BLOCK_K', 'elapsed_ms'])
            for sample in range(1, args.samples + 1):
                order = args.tiles.copy()
                rng.shuffle(order)
                for tile in order:
                    result = subprocess.run(
                        ['./build/matmul_fat_matrices_01_shared_memory',
                         '4096', '4096', '4096', *map(str, tile)],
                        check=True, capture_output=True, text=True)
                    match = re.search(r'^([0-9.]+) \(ms\) elapsed$', result.stdout, re.MULTILINE)
                    if match is None:
                        raise ValueError(f'Unexpected benchmark output: {result.stdout}')
                    elapsed = float(match[1])
                    writer.writerow([sample, 4096, 4096, 4096, *tile, elapsed])
                    output.flush()
                    print(f'Sample {sample}/{args.samples}, tile {tile}: {elapsed:.3f} ms', flush=True)
    groups = {}
    with source.open(newline='') as data:
        for row in csv.DictReader(data):
            tile = tuple(int(row[key]) for key in ['BLOCK_M', 'BLOCK_N', 'BLOCK_K'])
            if (int(row['M']), int(row['N']), int(row['K'])) != (4096, 4096, 4096):
                raise ValueError('Expected M=N=K=4096')
            groups.setdefault(tile, []).append(float(row['elapsed_ms']))
    summary = [(tile, len(values), statistics.mean(values),
                statistics.stdev(values) if len(values) > 1 else 0.0)
               for tile, values in sorted(groups.items())]
    with (args.output_dir / 'summary.csv').open('w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['BLOCK_M', 'BLOCK_N', 'BLOCK_K', 'samples', 'mean_ms', 'std_ms'])
        writer.writerows([*tile, count, mean, std] for tile, count, mean, std in summary)
    fig, ax = plt.subplots(figsize=(max(10, len(summary) * 0.7), 6))
    depths = sorted({tile[2] for tile in groups})
    colors = {k: plt.get_cmap('tab10')(i % 10) for i, k in enumerate(depths)}
    for pos, (tile, count, mean, std) in enumerate(summary):
        ax.bar(pos, mean, yerr=std, capsize=3, color=colors[tile[2]])
        ax.annotate(f'{mean:,.0f}', (pos, mean + std), xytext=(0, 4),
                    textcoords='offset points', ha='center', fontsize=8)
        print(f'Tile {tile}: {mean:.3f} ± {std:.3f} ms ({count} samples)')
    ax.legend(handles=[Patch(facecolor=colors[k], label=f'BLOCK_K={k}') for k in depths])
    ax.set_xticks(range(len(summary)), ['×'.join(map(str, r[0])) for r in summary],
                  rotation=45, ha='right')
    ax.set_xlabel('BLOCK_M × BLOCK_N × BLOCK_K')
    ax.set_ylabel('Kernel latency (ms)')
    ax.set_title('Shared-memory matmul: M = N = K = 4096\nMean latency; error bars: ±1 sample standard deviation')
    ax.set_ylim(0, max(r[2] + r[3] for r in summary) * 1.18)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / 'tile_latency.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
