#include "../common/solver.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <omp.h>

static std::string X_global;
static std::string Y_global;
static std::vector<int> dp_prev;
static std::vector<int> dp_curr;
static std::vector<int> P; 
static size_t n, m; 

inline int char2idx(char c) {
    if (c >= 'A' && c <= 'Z') {
        return c - 'A';
    } else if (c >= 'a' && c <= 'z') {
        return 26 + (c - 'a');
    } else {
        return -1; // invalid character
    }
}

static void build_P(const std::string &Y) {
    m = Y.size();
    const int ALPHABET_SIZE = 52;
    P.resize(ALPHABET_SIZE * (m + 1), 0);

    std::vector<int> last_occ(ALPHABET_SIZE, 0);
    
    #pragma omp parallel
    {
        std::vector<int> private_last_occ(ALPHABET_SIZE, 0);
        
        #pragma omp for schedule(static)
        for (int j = 1; j <= (int)m; j++) {
            int idx = char2idx(Y[j-1]);
            private_last_occ[idx] = j;
            
            for (int c = 0; c < ALPHABET_SIZE; c++) {
                P[c*(m+1) + j] = private_last_occ[c];
            }
        }
    }
}

void init(const std::string &X_input, const std::string &Y_input) {
    X_global = X_input;
    Y_global = Y_input;
    n = X_global.size();
    m = Y_global.size();
}

int compute_lcs() {
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

    build_P(Y_global);

    dp_prev.resize(m + 1, 0);
    dp_curr.resize(m + 1, 0);

    for (int i = 1; i <= (int)n; i++) {
        char x_c = X_global[i-1];
        int x_idx = char2idx(x_c);

        #pragma omp parallel for schedule(dynamic, 256)
        for (int j = 1; j <= (int)m; j++) {
            char y_c = Y_global[j-1];

            if (x_c == y_c) {
                dp_curr[j] = dp_prev[j-1] + 1;
            } else {
                int prev_occ = P[x_idx*(m+1) + j];

                if (prev_occ == 0) {
                    int val = dp_prev[j];
                    dp_curr[j] = (val > 0) ? val : 0;
                } else {
                    int val_not_match = dp_prev[j];
                    int val_match_before = dp_prev[prev_occ - 1] + 1;
                    dp_curr[j] = std::max(val_match_before, val_not_match);
                }
            }
        }

        dp_prev.swap(dp_curr);
    }

    return dp_prev[m];
}

void free_memory() {
    std::vector<int>().swap(dp_prev);
    std::vector<int>().swap(dp_curr);
    std::vector<int>().swap(P);
}