#include <mpi.h>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

static std::string X_global;
static std::string Y_global;
static std::vector<std::vector<int>> P;
static std::vector<int> dp_local;
static std::vector<int> dp_prev;

static int rank, size; // MPI rank and size
static int chunk_start, chunk_end; // Chunk boundaries for this process
static int local_chunk_size;

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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    X_global = X_input;
    Y_global = Y_input;

    if (rank == 0) {
        build_P(Y_global);
    }

    // Broadcast the P table to all processes
    const int ALPHABET_SIZE = 52;
    size_t m = Y_global.size();
    if (rank != 0) {
        P.assign(ALPHABET_SIZE, std::vector<int>(m + 1, 0));
    }

    for (int c = 0; c < ALPHABET_SIZE; c++) {
        MPI_Bcast(P[c].data(), m + 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Divide Y into chunks
    size_t n = X_global.size();
    size_t global_chunk_size = (Y_global.size() + size - 1) / size; // Ceiling division
    chunk_start = rank * global_chunk_size;
    chunk_end = std::min(static_cast<size_t>((rank + 1) * global_chunk_size), Y_global.size());
    local_chunk_size = chunk_end - chunk_start;

    dp_local.resize(local_chunk_size + 1, 0);
    dp_prev.resize(local_chunk_size + 1, 0);
}

int compute_lcs() {
    size_t n = X_global.size();
    size_t m = Y_global.size();

    if (n == 0 || m == 0) return 0;

    for (size_t i = 1; i <= n; i++) {
        char x_c = X_global[i - 1];
        int x_idx = char2idx(x_c);

        // Communicate boundary values with neighbors
        int left_boundary = (chunk_start > 0) ? dp_prev[0] : 0;
        int right_boundary = (chunk_end < m) ? dp_prev[local_chunk_size] : 0;
        if (rank > 0) {
            MPI_Sendrecv(&dp_prev[1], 1, MPI_INT, rank - 1, 0,
                         &left_boundary, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(&dp_prev[local_chunk_size], 1, MPI_INT, rank + 1, 0,
                         &right_boundary, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Compute current row locally
        for (size_t j = 0; j < local_chunk_size; j++) {
            int global_j = chunk_start + j + 1;
            if (x_c == Y_global[global_j - 1]) {
                dp_local[j + 1] = dp_prev[j] + 1;
            } else {
                int prev_occ = P[x_idx][global_j];
                int val_not_match = dp_prev[j + 1];
                int val_match_before = (prev_occ > 0) ? dp_prev[prev_occ - chunk_start] + 1 : 0;
                dp_local[j + 1] = std::max(val_not_match, val_match_before);
            }
        }

        // Update dp_prev for the next iteration
        dp_prev = dp_local;
    }

    // Gather results from all processes
    int local_result = dp_local[local_chunk_size];
    int global_result = 0;
    MPI_Reduce(&local_result, &global_result, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    return global_result;
}

void free_memory() {
    dp_local.clear();
    dp_local.shrink_to_fit();
    dp_prev.clear();
    dp_prev.shrink_to_fit();
    P.clear();
    P.shrink_to_fit();
}
