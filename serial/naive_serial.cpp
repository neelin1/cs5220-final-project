#include "../common/solver.hpp"
#include <string>
#include <algorithm>

static std::string X_global;
static std::string Y_global;

void init(const std::string &X_input, const std::string &Y_input) {
    X_global = X_input;
    Y_global = Y_input;
}

static int naive_lcs(int m, int n) {
    if (m == 0 || n == 0) {
        return 0;
    } else if (X_global[m - 1] == Y_global[n - 1]) {
        return 1 + naive_lcs(m - 1, n - 1);
    } else {
        return std::max(naive_lcs(m - 1, n), naive_lcs(m, n - 1));
    }
}

int compute_lcs() {
    int m = X_global.size();
    int n = Y_global.size();
    return naive_lcs(m, n);
}

void free_memory() {
}
