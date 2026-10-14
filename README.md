## Build

Requires CMake 3.20+, a CUDA toolkit and host compiler with C++20 `std::source_location`
support (tested with CUDA 12.9 / GCC 13), and an NVIDIA GPU with a working driver.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

Pass the element count and threads per block as positive integers:

```bash
./build/vector_add 100000 256
```

The baseline is `src/vector_add/00_vector_add.cu`. A separate vectorized
implementation in `src/vector_add/01_vector_add_float4.cu` processes four elements
per thread using aligned `float4` loads/stores and handles the remaining 1–3
elements with scalar accesses:

```bash
./build/vector_add_float4 100000 256
```

Both executables use the same warm-up, cache eviction, timing, and correctness
checks. For `float4`, the grid contains `ceil(ceil(n / 4) / threads)` blocks.
Use `--binary build/vector_add_float4` with either plotting script to benchmark it.

`src/vector_add/02_vector_add_unrolled.cu` provides scalar unrolled variants with
2, 4, or 8 elements per thread, selected at compile time by CMake:

```bash
./build/vector_add_unrolled2 100000 256
./build/vector_add_unrolled4 100000 256
./build/vector_add_unrolled8 100000 256
```

Each block covers `threads * elements_per_thread` elements. Within each unrolled
iteration, adjacent threads access adjacent elements; bounds checks cover partial
blocks. The kernel calculates all per-thread sums before storing them, exposing
independent work while using more registers. These executables retain the baseline
measurement and correctness checks and support the plotting scripts' `--binary`
option.

To compare individual launches with nvprof:

```bash
nvprof --print-gpu-trace ./build/vector_add 100000 256
nvprof --print-gpu-trace ./build/vector_add_float4 100000 256
nvprof --print-gpu-trace ./build/vector_add_unrolled4 100000 256
```

Compare the **second** vector-add invocation, after cache eviction; the first is
the warm-up. Collect hardware metrics separately, if driver permissions allow:

```bash
nvprof --kernels '::vector_add:2' --metrics inst_executed_global_loads,inst_executed_global_stores ./build/vector_add 100000 256
nvprof --kernels '::vector_add_float4:2' --metrics inst_executed_global_loads,inst_executed_global_stores ./build/vector_add_float4 100000 256
```

The program warms up vector addition, reads a separate scratch buffer sized at
four times the GPU's L2 cache, then measures one vector-add launch with CUDA events
and checks the result. It prints the elapsed microseconds to stdout; errors go to
stderr. Allocation, memory transfers, warm-up, cache eviction, and result checking
are outside the timed region. Scratch-buffer eviction is a practical heuristic,
not a guaranteed hardware cache flush. Small kernels remain sensitive to event
timing resolution and scheduling overhead.
Invalid launch configurations are reported by CUDA.
Eviction reads the scratch buffer without modifying it to avoid leaving dirty
scratch cache lines that could add writeback traffic during the measured launch.

## Statistics and plots

Results mirror the source tree, with the `.cu` extension removed and an
experiment directory underneath. For example,
`src/matmul/fat_matrices/01_shared_memory.cu` has profiling results in
`results/matmul/fat_matrices/01_shared_memory/nvprof/`.

```text
results/
├── vector_add/
│   ├── 00_vector_add/{threads_sweep,elements_sweep,nvprof}/
│   ├── 01_vector_add_float4/comparison/
│   └── 02_vector_add_unrolled/comparison/
└── matmul/fat_matrices/
    ├── 00_naive/{blocks,blocks_replot,simplified_check,nvprof}/
    ├── 01_shared_memory/nvprof/
    └── nvprof/                     # Shared profiler capability inventory
```

Comparison directories retain all variants measured in the same experiment.
Results are local, Git-ignored artifacts. Plotting defaults target the baseline
source; when selecting another `--binary`, also set `--output-dir` to its source
directory and experiment name (for example,
`results/vector_add/01_vector_add_float4/threads_sweep`). Use separate experiment
names to preserve earlier runs.

The two plotting scripts share measurement and plotting helpers in
`scripts/timing_common.py`. Each runs the program once per configuration per sample and
calculates means and, when multiple samples are collected, sample standard deviations:

```bash
export UV_CACHE_DIR=.cache/uv
uv sync --locked
MPLCONFIGDIR=.cache/matplotlib uv run --locked python scripts/plot_block_sizes.py --output-dir results/vector_add/00_vector_add/threads_sweep
MPLCONFIGDIR=.cache/matplotlib uv run --locked python scripts/plot_element_counts.py --output-dir results/vector_add/00_vector_add/elements_sweep
```

Python is pinned in `.python-version`; `uv.lock` pins all Python dependencies.
Run `uv sync --locked` after cloning to recreate the environment. To intentionally
update dependencies, run `uv lock --upgrade` and commit the updated lockfile.

The default is 10 samples per configuration. The block-size sweep uses
64, 128, 256, 512, and 1024 threads per block at 100,000 elements. The element-count
sweep uses 128 threads per block and powers of two from 2^10 (1,024) through
2^25 (33,554,432) elements. Element-count plots use a base-2 logarithmic x-axis;
the latency plot also uses a logarithmic y-axis (log-log).

Each output directory contains `samples.csv`, `summary.csv`, and separate latency
and effective memory bandwidth PNG plots (`*_timings.png` and `*_bandwidth.png`).
Use `--samples` to set the number of independent program runs. The scripts print
the average and sample standard deviation for each configuration.
For compatibility with saved CSVs, new measurements retain `timing_iterations`
(always 1) and `average_us` (the single measured launch time). Older batch timing
CSVs can still be replotted. Each sample starts a fresh process and CUDA context.
Error bars show sample standard deviation across runs, not confidence intervals.
With `--samples 1`,
error bars are omitted and standard deviation fields in `summary.csv` are blank.

Effective memory bandwidth counts two float32 reads and one float32 write per
element: `bandwidth_GB/s = 12 * n / (elapsed_us * 1000)` (decimal GB/s).
The scripts calculate bandwidth for each sample, then report its mean and sample
standard deviation in `mean_bandwidth_gbps` and `std_bandwidth_gbps` in `summary.csv`.
Bandwidth error bars use these transformed samples. This measures useful bytes
per elapsed time; it does not measure physical DRAM traffic or guarantee peak
DRAM bandwidth, particularly for small inputs or cached historical measurements.

Bandwidth plots include a dashed theoretical VRAM bandwidth line at 112 GB/s for
this machine's GTX 1050 Ti ([manufacturer specifications](https://www.pny.com/File%20Library/Company/Support/Product%20Brochures/GeForce%20Graphics/English/PNY-NVIDIA-GeForce-GTX-1050Ti-4GB.pdf)).
Use `--peak-bandwidth <GB/s>` for a different GPU or memory clock configuration;
the reference value is configured, not automatically detected.

New sample CSVs record `cache_mode` as `eviction`. Output writes can still remain
buffered in L2 during a short launch; the large-input plateau is more useful for
estimating sustained DRAM bandwidth.

Replot saved samples without accessing the GPU:

```bash
MPLCONFIGDIR=.cache/matplotlib uv run --locked python scripts/plot_element_counts.py --input results/vector_add/00_vector_add/elements_sweep/samples.csv --output-dir results/vector_add/00_vector_add/elements_sweep
```

## Naive matrix multiplication block sweep

The matmul example multiplies fixed 4096×4096 FP32 matrices. Pass the block's
x and y dimensions as two positive integers (default: `32 32`):

```bash
./build/matmul_fat_matrices_00_naive 32 8
```

CUDA reports an error if the block exceeds the GPU's launch limits.
Each run performs one untimed warm-up and one timed kernel launch,
then prints the matrix and block dimensions, elapsed milliseconds, and TFLOPS
(`2*M*N*K / (elapsed_ms * 1e9)`). Inputs are initialized to zero directly on the
GPU. Allocation and initialization are outside the timing.

Sweep block shapes and plot their mean latency and FP32 throughput:

```bash
MPLCONFIGDIR=.cache/matplotlib uv run --locked python scripts/plot_matmul_block_sizes.py
MPLCONFIGDIR=.cache/matplotlib uv run --locked python scripts/plot_matmul_block_sizes.py --blocks 8x32,16x16,32x8 --samples 5 --output-dir results/matmul/fat_matrices/00_naive/blocks_256_threads
```

The default sweep covers nine shapes with 64–1024 threads and three independent
program runs per shape. Shapes are tested in a reproducibly shuffled order each
round to reduce ordering bias. Comparing shapes with the same total thread count
helps reveal the effect of arranging threads along matrix rows and columns.
This benchmark retains natural cache behavior; it does not explicitly evict caches.

Results go to `results/matmul/fat_matrices/00_naive/blocks/`: `samples.csv`, `summary.csv`,
`block_shape_timings.png`, and `block_shape_tflops.png`. Error bars show ±1 sample
standard deviation across runs, not confidence intervals. TFLOPS are calculated
per sample from elapsed time before averaging. With `--samples 1`, standard
deviations are left blank and error bars are omitted. The fastest measured mean
is printed; close results may require more samples to distinguish reliably.

Replot saved data without running GPU work:

```bash
MPLCONFIGDIR=.cache/matplotlib uv run --locked python scripts/plot_matmul_block_sizes.py --input results/matmul/fat_matrices/00_naive/blocks/samples.csv --output-dir results/matmul/fat_matrices/00_naive/blocks_replot
```

## Shared-memory matrix multiplication

Pass matrix dimensions followed by tile dimensions:

```bash
./build/matmul_fat_matrices_01_shared_memory 4096 4096 4096 16 32 8
```

Arguments are `M N K BLOCK_M BLOCK_N BLOCK_K`; defaults are
`4096 4096 4096 32 32 32`. Each matrix dimension must be divisible by its
corresponding tile dimension. A block uses `BLOCK_M * BLOCK_N` threads and
`4 * BLOCK_K * (BLOCK_M + BLOCK_N)` bytes of dynamic shared memory; CUDA reports
configurations exceeding device limits. Tile dimensions may differ.

The program prints the configuration, latency in milliseconds, and TFLOPS after
one warm-up and one measured launch. Tile sizes are runtime values, so changing
them requires no recompilation. Future sweeps can save results under
`results/matmul/fat_matrices/01_shared_memory/`.

## IDE

The root `compile_commands.json` symlink provides compiler settings.
[`.clangd`](.clangd) filters NVCC-only flags for clangd; restart the language
server if stale errors remain. It also explicitly enables C++20 for CUDA files
so `std::source_location` is available when clangd uses a fallback compile command.

CMake's cache and compile database contain absolute paths. After moving or
renaming the project directory, regenerate them (requires CMake 3.24+):

```bash
cmake --fresh -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Then restart clangd or reopen the editor to clear stale diagnostics.
