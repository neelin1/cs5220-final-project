#include "../common/solver.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <cmath>

#define BLOCK_SIZE 32

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

void process_block(int block_i, int block_j, int m, int n) {
    int start_i = block_i * BLOCK_SIZE;
    int start_j = block_j * BLOCK_SIZE;
    int end_i = std::min((block_i + 1) * BLOCK_SIZE, m + 1);
    int end_j = std::min((block_j + 1) * BLOCK_SIZE, n + 1);

    for (int i = start_i; i < end_i; ++i) {
        for (int j = start_j; j < end_j; ++j) {
            // skipping at border initialization cells
            if (i == 0 || j == 0) {
                dp_table[i][j] = 0;
                continue;
            }
            
            if (X_global[i - 1] == Y_global[j - 1]) {
                dp_table[i][j] = dp_table[i - 1][j - 1] + 1;
            } else {
                dp_table[i][j] = std::max(dp_table[i - 1][j], dp_table[i][j - 1]);
            }
        }
    }
}

int compute_lcs() {
    int m = X_global.size();
    int n = Y_global.size();

    int num_blocks_i = (m + BLOCK_SIZE) / BLOCK_SIZE;
    int num_blocks_j = (n + BLOCK_SIZE) / BLOCK_SIZE;
    
    for (int block_d = 0; block_d < num_blocks_i + num_blocks_j - 1; ++block_d) {
        int block_start = std::max(0, block_d - num_blocks_j + 1);
        int block_end = std::min(block_d, num_blocks_i - 1);

        #pragma omp parallel for schedule(dynamic)
        for (int block_i = block_start; block_i <= block_end; ++block_i) {
            int block_j = block_d - block_i;
            
            if (block_j < 0 || block_j >= num_blocks_j) continue;

            process_block(block_i, block_j, m, n);
        }
    }

    return dp_table[m][n];
}

void free_memory() {
    dp_table.clear();
    dp_table.shrink_to_fit();
}