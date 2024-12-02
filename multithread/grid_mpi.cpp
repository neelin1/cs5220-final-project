#include "../common/solver.hpp"
#include <mpi.h>
#include <vector>

static std::string X_global;
static std::string Y_global;

void init(const std::string &X_input, const std::string &Y_input) {
    X_global = X_input;
    Y_global = Y_input;
}

int compute_lcs() {
    return 0;
}

void free_memory() {
}
