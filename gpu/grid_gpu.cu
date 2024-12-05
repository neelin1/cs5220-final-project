#include "../common/solver.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>

static std::string X_global;
static std::string Y_global;

// Mapping for the full alphabet: A-Z followed by a-z
// A-Z: 0-25, a-z: 26-51
__host__ __device__ inline int char2idx(char c) {
    if (c >= 'A' && c <= 'Z') {
        return c - 'A';
    } else if (c >= 'a' && c <= 'z') {
        return 26 + (c - 'a');
    } else {
        return -1; // Invalid character
    }
}

static int *d_dp_prev = nullptr;
static int *d_dp_curr = nullptr;
static int *d_P = nullptr;  // P array on device

static std::vector<int> h_P; // P array on host
static size_t n, m;          // lengths of X and Y

// Build P on CPU:
// P[c, j] = last occurrence of character c up to position j in Y (1-based index), 0 if none.
// c ranges over full alphabet: size = 52.
// Size of P: ALPHABET_SIZE * (m+1)
static void build_P(const std::string &Y) {
    m = Y.size();
    const int ALPHABET_SIZE = 52;
    h_P.resize(ALPHABET_SIZE * (m + 1), 0);

    std::vector<int> last_occ(ALPHABET_SIZE, 0);
    for (int j = 1; j <= (int)m; j++) {
        int idx = char2idx(Y[j-1]);
        last_occ[idx] = j;
        for (int c = 0; c < ALPHABET_SIZE; c++) {
            h_P[c*(m+1) + j] = last_occ[c];
        }
    }
}

// GPU kernel to compute one row of DP:
__global__ void lcs_kernel(const int *dp_prev, int *dp_curr, const int *P, const char *X, const char *Y, int i, int m) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1; // 1-based indexing
    if (j > m) return;

    char x_c = X[i-1];
    char y_c = Y[j-1];

    if (x_c == y_c) {
        // S[i,j] = S[i-1,j-1] + 1
        dp_curr[j] = dp_prev[j-1] + 1;
    } else {
        int c_idx = char2idx(x_c);
        int prev_occ = P[c_idx*(m+1) + j];

        if (prev_occ == 0) {
            // S[i,j] = max(S[i-1,j], 0)
            int val = dp_prev[j]; 
            dp_curr[j] = (val > 0) ? val : 0;
        } else {
            // S[i,j] = max(S[i-1,j], S[i-1, prev_occ-1] + 1)
            int val_not_match = dp_prev[j]; // S[i-1,j]
            int val_match_before = dp_prev[prev_occ - 1] + 1;
            dp_curr[j] = (val_match_before > val_not_match) ? val_match_before : val_not_match;
        }
    }
}

void init(const std::string &X_input, const std::string &Y_input) {
    X_global = X_input;
    Y_global = Y_input;
}

int compute_lcs() {
    n = X_global.size();
    m = Y_global.size();
    if (n == 0 || m == 0) {
        return 0;
    }

    // Validate input characters
    for (char c : X_global) {
        if (char2idx(c) < 0) {
            std::cerr << "Error: X_global contains invalid character: " << c << std::endl;
            return 0;
        }
    }
    for (char c : Y_global) {
        if (char2idx(c) < 0) {
            std::cerr << "Error: Y_global contains invalid character: " << c << std::endl;
            return 0;
        }
    }

    // Build P on host
    build_P(Y_global);

    // Allocate DP rows and P on device (two rows of length m+1)
    cudaMalloc((void**)&d_dp_prev, (m+1)*sizeof(int));
    cudaMalloc((void**)&d_dp_curr, (m+1)*sizeof(int));

    // Initialize dp_prev to 0s (S[0,*] = 0)
    std::vector<int> zero_row(m+1, 0);
    cudaMemcpy(d_dp_prev, zero_row.data(), (m+1)*sizeof(int), cudaMemcpyHostToDevice);

    const int ALPHABET_SIZE = 52;
    // Copy P to device
    cudaMalloc((void**)&d_P, ALPHABET_SIZE*(m+1)*sizeof(int));
    cudaMemcpy(d_P, h_P.data(), ALPHABET_SIZE*(m+1)*sizeof(int), cudaMemcpyHostToDevice);

    // Copy X and Y to device
    char *d_X, *d_Y;
    cudaMalloc((void**)&d_X, n*sizeof(char));
    cudaMalloc((void**)&d_Y, m*sizeof(char));
    cudaMemcpy(d_X, X_global.data(), n*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y_global.data(), m*sizeof(char), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (int)((m + threadsPerBlock - 1) / threadsPerBlock);

    // Compute row by row
    for (int i = 1; i <= (int)n; i++) {
        lcs_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_dp_prev, d_dp_curr, d_P, d_X, d_Y, i, (int)m);
        cudaDeviceSynchronize();
        // Swap dp_prev and dp_curr
        int *temp = d_dp_prev;
        d_dp_prev = d_dp_curr;
        d_dp_curr = temp;
    }

    // After all rows, d_dp_prev contains S[n,*]
    int lcs_length = 0;
    cudaMemcpy(&lcs_length, d_dp_prev + m, sizeof(int), cudaMemcpyDeviceToHost);

    // Free temporary GPU memory
    cudaFree(d_X);
    cudaFree(d_Y);

    return lcs_length;
}

void free_memory() {
    if (d_dp_prev) { cudaFree(d_dp_prev); d_dp_prev = nullptr; }
    if (d_dp_curr) { cudaFree(d_dp_curr); d_dp_curr = nullptr; }
    if (d_P) { cudaFree(d_P); d_P = nullptr; }
}
