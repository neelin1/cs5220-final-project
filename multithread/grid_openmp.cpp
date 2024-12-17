#include "../common/solver.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <omp.h>

static std::string X_global;
static std::string Y_global;
static std::vector<std::vector<int>> dp_table;
static std::vector<std::vector<int>> P;

inline int char2idx(char c) {
    if (c >= 'A' && c <= 'Z') {
        return c - 'A';
    } else if (c >= 'a' && c <= 'z') {
        return 26 + (c - 'a');
    } else {
        return -1;
    }
}

void build_P(const std::string &Y) {
    const int ALPHABET_SIZE = 52;
    size_t m = Y.size();

    P.assign(ALPHABET_SIZE, std::vector<int>(m + 1, 0));
    std::vector<int> last_occ(ALPHABET_SIZE, 0);

    for (int j = 1; j <= (int)m; j++) {
        int idx = char2idx(Y[j - 1]);
        if (idx >= 0) {
            last_occ[idx] = j;
        }

        for (int c = 0; c < ALPHABET_SIZE; c++) {
            P[c][j] = last_occ[c];
        }
    }
}

void init(const std::string &X_input, const std::string &Y_input) {
    X_global = X_input;
    Y_global = Y_input;

    for (char c : X_input) {
        if (char2idx(c) < 0) {
            std::cerr << "Error: X contains invalid character: " << c << std::endl;
            return;
        }
    }
    for (char c : Y_input) {
        if (char2idx(c) < 0) {
            std::cerr << "Error: Y contains invalid character: " << c << std::endl;
            return;
        }
    }

    build_P(Y_global);
    dp_table.assign(2, std::vector<int>(Y_global.size() + 1, 0));
}

int compute_lcs() {
    size_t n = X_global.size();
    size_t m = Y_global.size();

    if (n == 0 || m == 0) return 0;

    int curr = 1, prev = 0;

    for (size_t i = 1; i <= n; i++) {
        char x_c = X_global[i - 1];
        int x_idx = char2idx(x_c);

        // OpenMP parallel loop
        #pragma omp parallel for schedule(static) default(shared)
        for (size_t j = 1; j <= m; j++) {
            char y_c = Y_global[j - 1];              // Local variable
            int prev_occ = P[x_idx][j];             // Local variable
            int val_not_match, val_match_before;    // Local variables

            if (x_c == y_c) {
                // Match
                dp_table[curr][j] = dp_table[prev][j - 1] + 1;
            } else {
                // No match
                if (prev_occ == 0) {
                    dp_table[curr][j] = std::max(dp_table[prev][j], 0);
                } else {
                    val_not_match = dp_table[prev][j];
                    val_match_before = dp_table[prev][prev_occ - 1] + 1;
                    dp_table[curr][j] = std::max(val_match_before, val_not_match);
                }
            }
        }

        // Swap current and previous DP rows
        std::swap(curr, prev);
    }

    return dp_table[prev][m];
}


void free_memory() {
    dp_table.clear();
    dp_table.shrink_to_fit();
    P.clear();
    P.shrink_to_fit();
}
