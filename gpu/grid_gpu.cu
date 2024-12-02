#include "../common/solver.hpp"
#include <cuda_runtime.h>

static std::string X_global;
static std::string Y_global;

void init(const std::string &X_input, const std::string &Y_input) {
    X_global = X_input;
    Y_global = Y_input;
}

__global__ void lcs_kernel() {
}

int compute_lcs() {
    return 0;
}

void free_memory() {
}
