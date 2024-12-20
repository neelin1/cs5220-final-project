#include "../common/solver.hpp"
#include <vector>
#include <algorithm>

static std::string X_global;
static std::string Y_global;
static std::string Z_global;
static std::vector<std::vector<std::vector<int>>> dp_table;

void init(const std::string &X_input, const std::string &Y_input, const std::string &Z_input) {
    X_global = X_input;
    Y_global = Y_input;
    Z_global = Z_input;

    int m = (int)X_global.size();
    int n = (int)Y_global.size();
    int o = (int)Z_global.size();

    // Resize the 3D DP table
    dp_table.resize(m+1);
    for (int i = 0; i <= m; ++i) {
        dp_table[i].resize(n+1);
        for (int j = 0; j <= n; ++j) {
            dp_table[i][j].resize(o+1, 0);
        }
    }
}

int compute_lcs() {
    int m = (int)X_global.size();
    int n = (int)Y_global.size();
    int o = (int)Z_global.size();

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            for (int k = 1; k <= o; k++) {
                if (X_global[i-1] == Y_global[j-1] && Y_global[j-1] == Z_global[k-1]) {
                    dp_table[i][j][k] = dp_table[i-1][j-1][k-1] + 1;
                } else {
                    dp_table[i][j][k] = std::max({dp_table[i-1][j][k], dp_table[i][j-1][k], dp_table[i][j][k-1]});
                }
            }
        }
    }

    return dp_table[m][n][o];
}

void free_memory() {
    // Clear the DP table
    dp_table.clear();
}
