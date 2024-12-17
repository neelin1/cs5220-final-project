#include "../common/solver.hpp"
#include <vector>
#include <algorithm>
#include <iostream>

static std::string X_global;
static std::string Y_global;
static std::vector<std::vector<int>> dp_table;

void init(const std::string &X_input, const std::string &Y_input) {
    X_global = X_input;
    Y_global = Y_input;
    int m = X_global.size();
    int n = Y_global.size();
    dp_table.resize(m + 1, std::vector<int>(n + 1, 0));
}

int compute_lcs() {
    int m = X_global.size();
    int n = Y_global.size();

    for (int d = 1; d <= m + n - 1; ++d) {
        int start = std::max(1, d - n + 1); 
        int end = std::min(d, m);

        for (int i = start; i <= end; ++i) {
            int j = d - i + 1;
            if (j < 1 || j > n) continue;

            if (X_global[i - 1] == Y_global[j - 1]) {
                dp_table[i][j] = dp_table[i - 1][j - 1] + 1;
            } else {
                dp_table[i][j] = std::max(dp_table[i - 1][j], dp_table[i][j - 1]);
            }
        }
    }

    return dp_table[m][n];
}


void free_memory() {
    dp_table.clear();
}
