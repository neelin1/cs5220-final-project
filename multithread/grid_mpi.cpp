#include "../common/solver.hpp"
#include <mpi.h>
#include <vector>
#include <string>
#include <algorithm>

static std::string X_global;
static std::string Y_global;

static int *prev_row = nullptr;
static int *curr_row = nullptr;

void init(const std::string &X_input, const std::string &Y_input) {
    X_global = X_input;
    Y_global = Y_input;
}

void initialize_grid(int m) {
    prev_row = new int[m + 1]();
    curr_row = new int[m + 1]();
}

void compute_row(const std::string &X, const std::string &Y, int row_idx, int m, int rank, int num_procs) {
    int chunk_size = (m + num_procs - 1) / num_procs;
    int start_col = rank * chunk_size + 1;
    int end_col = std::min(start_col + chunk_size - 1, m);

    for (int j = start_col; j <= end_col; j++) {
        if (X[row_idx - 1] == Y[j - 1]) {
            curr_row[j] = prev_row[j - 1] + 1;
        } else {
            curr_row[j] = std::max(prev_row[j], curr_row[j - 1]);
        }
    }

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, curr_row, chunk_size, MPI_INT, MPI_COMM_WORLD);
}

int compute_lcs() {
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = X_global.size();
    int m = Y_global.size();

    if (n == 0 || m == 0) return 0;

    initialize_grid(m);

    for (int i = 1; i <= n; i++) {
        compute_row(X_global, Y_global, i, m, rank, num_procs);

        std::swap(prev_row, curr_row);
        std::fill(curr_row, curr_row + m + 1, 0);
    }

    int lcs_length = prev_row[m];

    delete[] prev_row;
    delete[] curr_row;

    return lcs_length;
}

void free_memory() {
    MPI_Finalize();
}
