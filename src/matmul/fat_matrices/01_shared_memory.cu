#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <source_location>

void cuda_check(cudaError_t error,
                const std::source_location location = std::source_location::current()) {
    if (error != cudaSuccess) {
        std::cerr << location.file_name() << ':' << location.line() << ": "
                  << cudaGetErrorString(error) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void matmul(const float *a, const float *b, float *c, const int M, const int N,
                       const int K) {
    static_assert(BLOCK_M > 0 && BLOCK_N > 0 && BLOCK_K > 0);
    static_assert(BLOCK_M == BLOCK_K && BLOCK_N == BLOCK_K,
                  "One load per thread requires equal tile dimensions.");

    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float a_block[BLOCK_M][BLOCK_K];
    __shared__ float b_block[BLOCK_K][BLOCK_N];

    float sum = 0.0f;
    for (int i = 0; i < K; i += BLOCK_K) {
        a_block[threadIdx.y][threadIdx.x] = a[y * K + (i + threadIdx.x)];
        b_block[threadIdx.y][threadIdx.x] = b[(i + threadIdx.y) * N + x];
        __syncthreads();

        for (int j = 0; j < BLOCK_K; ++j) {
            sum += a_block[threadIdx.y][j] * b_block[j][threadIdx.x];
        }
        __syncthreads();
    }

    c[y * N + x] = sum;
}

int main() {
    constexpr auto M = 4096, N = 4096, K = 4096;
    constexpr auto BLOCK_M = 32, BLOCK_N = 32, BLOCK_K = 32;
    static_assert(M % BLOCK_M == 0 && N % BLOCK_N == 0 && K % BLOCK_K == 0,
                  "Matrix dimensions must be divisible by tile dimensions.");
    constexpr size_t a_bytes = M * K * sizeof(float);
    constexpr size_t b_bytes = K * N * sizeof(float);
    constexpr size_t c_bytes = M * N * sizeof(float);

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
    matmul<BLOCK_M, BLOCK_N, BLOCK_K><<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, M, N, K);
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());

    cuda_check(cudaEventRecord(start));
    matmul<BLOCK_M, BLOCK_N, BLOCK_K><<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, M, N, K);
    cuda_check(cudaGetLastError());
    cuda_check(cudaEventRecord(stop));
    cuda_check(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop));
    // Count each multiply-add as two floating-point operations.
    const double tflops = (2.0 * M * N * K) / (elapsed_ms * 1.0e9);
    std::cout << "M=" << M << " N=" << N << " K=" << K << " BLOCK_M = " << BLOCK_M << " BLOCK_N = " << BLOCK_N << " BLOCK_K = " << BLOCK_K << "\n";
    std::cout << std::fixed << std::setprecision(3) << elapsed_ms << " (ms) elapsed\n";
    std::cout << std::setprecision(6) << tflops << " TFLOPS\n";

    cuda_check(cudaEventDestroy(start));
    cuda_check(cudaEventDestroy(stop));
    cuda_check(cudaFree(d_a));
    cuda_check(cudaFree(d_b));
    cuda_check(cudaFree(d_c));

    return EXIT_SUCCESS;
}
