#include "../common/solver.hpp"
#include <algorithm>
#include <string>
#include <vector>
#include <cuda_runtime.h>

static std::string X_global;
static std::string Y_global;
static std::vector<std::vector<int>> dp_table;
static char* d_X = nullptr;
static char* d_Y = nullptr;
static int* d_dp = nullptr;
static int m = 0;
static int n = 0;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__
void lcs_diagonal_kernel(const char* X, const char* Y, int* dp, int m, int n, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = max(1, d - n + 1);
    int end = min(d, m);
    int length = end - start + 1;
    if (idx < length) {
        int i = start + idx;
        int j = d - i + 1;
        int ij = i*(n+1) + j;
        int im1_jm1 = (i-1)*(n+1) + (j-1);
        int im1_j   = (i-1)*(n+1) + j;
        int i_jm1   = i*(n+1) + (j-1);
        if (X[i-1] == Y[j-1]) {
            dp[ij] = dp[im1_jm1] + 1;
        } else {
            dp[ij] = max(dp[im1_j], dp[i_jm1]);
        }
    }
}

void init(const std::string &X_input, const std::string &Y_input) {
    X_global = X_input;
    Y_global = Y_input;
    m = (int)X_global.size();
    n = (int)Y_global.size();
    dp_table.resize(m + 1, std::vector<int>(n + 1, 0));
    CUDA_CHECK(cudaMalloc((void**)&d_X, m * sizeof(char)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, n * sizeof(char)));
    CUDA_CHECK(cudaMalloc((void**)&d_dp, (m+1)*(n+1)*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_X, X_global.data(), m*sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, Y_global.data(), n*sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dp, 0, (m+1)*(n+1)*sizeof(int)));
}

int compute_lcs() {
    int blockSize = 256;
    for (int d = 1; d <= m + n - 1; ++d) {
        int start = max(1, d - n + 1);
        int end = min(d, m);
        int length = (end >= start) ? (end - start + 1) : 0;
        if (length > 0) {
            int gridSize = (length + blockSize - 1) / blockSize;
            lcs_diagonal_kernel<<<gridSize, blockSize>>>(d_X, d_Y, d_dp, m, n, d);
        }
    }
    std::vector<int> dp_host((m+1)*(n+1));
    CUDA_CHECK(cudaMemcpy(dp_host.data(), d_dp, (m+1)*(n+1)*sizeof(int), cudaMemcpyDeviceToHost));
    int result = dp_host[m*(n+1) + n];
    for (int i = 0; i <= m; ++i)
        for (int j = 0; j <= n; ++j)
            dp_table[i][j] = dp_host[i*(n+1) + j];
    return result;
}

void free_memory() {
    if (d_X) cudaFree(d_X);
    if (d_Y) cudaFree(d_Y);
    if (d_dp) cudaFree(d_dp);
    dp_table.clear();
}
