#include <cuda_runtime.h>

#include <charconv>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <source_location>
#include <string_view>

void cuda_check(cudaError_t error,
                const std::source_location location = std::source_location::current()) {
    if (error != cudaSuccess) {
        std::cerr << location.file_name() << ':' << location.line() << ": "
                  << cudaGetErrorString(error) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

int positive_int(std::string_view text) {
    int value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value <= 0) {
        std::cerr << "Dimensions must be positive integers.\n";
        std::exit(EXIT_FAILURE);
    }
    return value;
}

__global__ void matmul(const float *a, const float *b, float *c, const int M, const int N,
                       const int K, const int BLOCK_K) {
    const int BLOCK_M = blockDim.y, BLOCK_N = blockDim.x;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t x = blockIdx.x * blockDim.x + tx;
    const size_t y = blockIdx.y * blockDim.y + ty;

    extern __shared__ float tiles[];
    float *a_block = tiles;
    float *b_block = tiles + BLOCK_M * BLOCK_K;

    float sum = 0.0f;
    for (int i = 0; i < K; i += BLOCK_K) {
        // Cooperatively load tiles even when their dimensions differ.
        for (int k = tx; k < BLOCK_K; k += BLOCK_N)
            a_block[ty * BLOCK_K + k] = a[y * K + i + k];
        for (int k = ty; k < BLOCK_K; k += BLOCK_M)
            b_block[k * BLOCK_N + tx] = b[size_t(i + k) * N + x];
        __syncthreads();

        for (int k = 0; k < BLOCK_K; ++k)
            sum += a_block[ty * BLOCK_K + k] * b_block[k * BLOCK_N + tx];
        __syncthreads();
    }

    c[y * N + x] = sum;
}

int main(int argc, char **argv) {
    const bool help = argc == 2 && std::string_view(argv[1]) == "--help";
    if (help || (argc != 1 && argc != 7)) {
        std::cout << "Usage: " << argv[0] << " [M N K BLOCK_M BLOCK_N BLOCK_K]\n"
                  << "Defaults: 4096 4096 4096 32 32 32\n"
                  << "M, N, K must be divisible by BLOCK_M, BLOCK_N, BLOCK_K.\n";
        return help ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    const int M = argc == 7 ? positive_int(argv[1]) : 4096;
    const int N = argc == 7 ? positive_int(argv[2]) : 4096;
    const int K = argc == 7 ? positive_int(argv[3]) : 4096;
    const int BLOCK_M = argc == 7 ? positive_int(argv[4]) : 32;
    const int BLOCK_N = argc == 7 ? positive_int(argv[5]) : 32;
    const int BLOCK_K = argc == 7 ? positive_int(argv[6]) : 32;
    if (M % BLOCK_M || N % BLOCK_N || K % BLOCK_K) {
        std::cerr << "Matrix dimensions must be divisible by tile dimensions.\n";
        return EXIT_FAILURE;
    }
    const size_t a_bytes = size_t(M) * K * sizeof(float);
    const size_t b_bytes = size_t(K) * N * sizeof(float);
    const size_t c_bytes = size_t(M) * N * sizeof(float);
    const size_t shared_bytes = (size_t(BLOCK_M) + BLOCK_N) * BLOCK_K * sizeof(float);

    float *d_a, *d_b, *d_c;
    cuda_check(cudaMalloc(&d_a, a_bytes));
    cuda_check(cudaMalloc(&d_b, b_bytes));
    cuda_check(cudaMalloc(&d_c, c_bytes));
    cuda_check(cudaMemset(d_a, 0, a_bytes));
    cuda_check(cudaMemset(d_b, 0, b_bytes));

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start));
    cuda_check(cudaEventCreate(&stop));

    const dim3 threads_per_block(BLOCK_N, BLOCK_M);
    const dim3 num_blocks(1 + (N - 1) / BLOCK_N, 1 + (M - 1) / BLOCK_M);

    // Warm up the kernel before measuring a single launch.
    matmul<<<num_blocks, threads_per_block, shared_bytes>>>(d_a, d_b, d_c, M, N, K, BLOCK_K);
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());

    cuda_check(cudaEventRecord(start));
    matmul<<<num_blocks, threads_per_block, shared_bytes>>>(d_a, d_b, d_c, M, N, K, BLOCK_K);
    cuda_check(cudaGetLastError());
    cuda_check(cudaEventRecord(stop));
    cuda_check(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop));
    // Count each multiply-add as two floating-point operations.
    const double tflops = (2.0 * M * N * K) / (elapsed_ms * 1.0e9);
    std::cout << "M=" << M << " N=" << N << " K=" << K << " BLOCK_M=" << BLOCK_M
              << " BLOCK_N=" << BLOCK_N << " BLOCK_K=" << BLOCK_K << '\n';
    std::cout << std::fixed << std::setprecision(6) << elapsed_ms << " (ms) elapsed\n";
    std::cout << std::setprecision(6) << tflops << " TFLOPS\n";

    cuda_check(cudaEventDestroy(start));
    cuda_check(cudaEventDestroy(stop));
    cuda_check(cudaFree(d_a));
    cuda_check(cudaFree(d_b));
    cuda_check(cudaFree(d_c));

    return EXIT_SUCCESS;
}
