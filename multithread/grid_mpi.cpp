#include <mpi.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>

static std::string X_global;
static std::string Y_global;
static std::vector<std::vector<int>> P;
static std::vector<int> dp_prev; 
static std::vector<int> dp_curr;

static int rank_world;
static int size_world;

inline int char2idx(char c) {
    if (c >= 'A' && c <= 'Z') {
        return c - 'A';
    } else if (c >= 'a' && c <= 'z') {
        return 26 + (c - 'a');
    } else {
        return -1;
    }
}

static void build_P(const std::string &Y) {
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
    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        MPI_Init(NULL, NULL);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
    MPI_Comm_size(MPI_COMM_WORLD, &size_world);

    if (rank_world == 0) {
        X_global = X_input;
        Y_global = Y_input;
    }

    int n, m;
    if (rank_world == 0) {
        n = (int)X_global.size();
        m = (int)Y_global.size();
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    X_global.resize(n);
    Y_global.resize(m);
    MPI_Bcast(&X_global[0], n, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Y_global[0], m, MPI_CHAR, 0, MPI_COMM_WORLD);

    for (char c : X_global) {
        if (char2idx(c) < 0) {
            if (rank_world == 0) {
                std::cerr << "Error: X contains invalid character: " << c << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    for (char c : Y_global) {
        if (char2idx(c) < 0) {
            if (rank_world == 0) {
                std::cerr << "Error: Y contains invalid character: " << c << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    build_P(Y_global);

    dp_prev.assign(m + 1, 0);
    dp_curr.assign(m + 1, 0);
}

int compute_lcs() {
    int n = (int)X_global.size();
    int m = (int)Y_global.size();

    if (n == 0 || m == 0) return 0;

    int base = m / size_world;  
    int remainder = m % size_world;
    int cols_for_this_rank = base + (rank_world < remainder ? 1 : 0);

    int start_col = 1;
    for (int r = 0; r < rank_world; r++) {
        start_col += base + (r < remainder ? 1 : 0);
    }
    int end_col = start_col + cols_for_this_rank - 1;

    for (int i = 1; i <= n; i++) {
        char x_c = X_global[i - 1];
        int x_idx = char2idx(x_c);

        for (int j = start_col; j <= end_col; j++) {
            char y_c = Y_global[j - 1];
            if (x_c == y_c) {
                dp_curr[j] = dp_prev[j - 1] + 1;
            } else {
                int prev_occ = P[x_idx][j];
                int val_not_match = dp_prev[j];
                int val_match_before = (prev_occ > 0) ? (dp_prev[prev_occ - 1] + 1) : 0;
                dp_curr[j] = std::max(val_match_before, val_not_match);
            }
        }

        MPI_Allgather(MPI_IN_PLACE, cols_for_this_rank, MPI_INT,
                      &dp_curr[1], cols_for_this_rank, MPI_INT, MPI_COMM_WORLD);

        dp_prev.swap(dp_curr);
    }

    int lcs_len = dp_prev[m];
    return lcs_len;
}

void free_memory() {
    dp_prev.clear();
    dp_curr.clear();
    P.clear();
}

