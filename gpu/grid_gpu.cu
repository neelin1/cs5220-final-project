#include "../common/solver.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

static std::string X_global;
static std::string Y_global;
static int *d_table = nullptr;
static int *d_P = nullptr;
static size_t table_size = 0;
static size_t P_size = 0;

void init(const std::string &X_input, const std::string &Y_input) {
    X_global = X_input;
    Y_global = Y_input;
    size_t n = X_input.size();
    size_t m = Y_input.size();

    table_size = (n + 1) * (m + 1) * sizeof(int);
    P_size = 256 * (m + 1) * sizeof(int); 

    cudaMalloc(&d_table, table_size);
    cudaMalloc(&d_P, P_size);

    cudaMemset(d_table, 0, table_size);
}

__global__ void preprocess_P_kernel(const char *Y, int *P, size_t m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 256) return;

    for (int j = 1; j <= m; ++j) {
        if (Y[j - 1] == idx)
            P[idx * (m + 1) + j] = j;
        else
            P[idx * (m + 1) + j] = P[idx * (m + 1) + j - 1];
    }
}

__global__ void lcs_kernel(const char *X, const char *Y, int *table, int *P, size_t n, size_t m) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > m) return;

    for (int i = 1; i <= n; ++i) {
        int char_idx = X[i - 1];
        int P_val = P[char_idx * (m + 1) + j];
        int val1 = table[(i - 1) * (m + 1) + j];
        int val2 = (P_val > 0) ? table[(i - 1) * (m + 1) + P_val - 1] + 1 : 0;
        table[i * (m + 1) + j] = max(val1, val2);
    }
}

int compute_lcs() {
    size_t n = X_global.size();
    size_t m = Y_global.size();

    char *d_X, *d_Y;
    cudaMalloc(&d_X, n * sizeof(char));
    cudaMalloc(&d_Y, m * sizeof(char));
    cudaMemcpy(d_X, X_global.c_str(), n * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y_global.c_str(), m * sizeof(char), cudaMemcpyHostToDevice);
    preprocess_P_kernel<<<256, 256>>>(d_Y, d_P, m);
    cudaDeviceSynchronize();
    for (int i = 1; i <= n; ++i) {
        lcs_kernel<<<(m + 255) / 256, 256>>>(d_X, d_Y, d_table, d_P, n, m);
        cudaDeviceSynchronize();
    }

    int *host_table = new int[table_size / sizeof(int)];
    cudaMemcpy(host_table, d_table, table_size, cudaMemcpyDeviceToHost);

    int result = host_table[n * (m + 1) + m];
    delete[] host_table;

    cudaFree(d_X);
    cudaFree(d_Y);
    return result;
}

void free_memory() {
    cudaFree(d_table);
    cudaFree(d_P);
}
