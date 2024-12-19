#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <cassert>

#ifdef MPI
#include <mpi.h>
#endif

static std::string X_global;
static std::string Y_global;
static std::vector<std::vector<int>> dp_table;

void init(const std::string &X_input, const std::string &Y_input) {
    X_global = X_input;
    Y_global = Y_input;
    dp_table.clear();
}

#ifdef MPI

int compute_lcs() {
    int world_size = 1, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int m = (int)X_global.size();
    int n = (int)Y_global.size();

    // If only one process, just do a serial computation
    if (world_size == 1) {
        // Serial DP:
        dp_table.assign(m+1, std::vector<int>(n+1, 0));
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (X_global[i-1] == Y_global[j-1]) {
                    dp_table[i][j] = dp_table[i-1][j-1] + 1;
                } else {
                    dp_table[i][j] = std::max(dp_table[i-1][j], dp_table[i][j-1]);
                }
            }
        }
        return dp_table[m][n];
    }

    // For multiple processes:
    // Broadcast lengths
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast Y to all
    if (rank != 0) {
        Y_global.resize(n);
    }
    MPI_Bcast(&Y_global[0], n, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Determine row distribution
    int base_rows = m / world_size;
    int remainder = m % world_size;

    int local_rows = base_rows + ((rank < remainder) ? 1 : 0);

    // Compute start_row for each rank
    int start_row = 1; 
    for (int r = 0; r < rank; r++) {
        start_row += base_rows + ((r < remainder) ? 1 : 0);
    }
    int end_row = start_row + local_rows - 1;

    // Scatter X
    // First, gather counts and displacements
    std::vector<int> sendcounts(world_size), displs(world_size);
    {
        int disp = 0;
        for (int r = 0; r < world_size; r++) {
            int rows_for_r = base_rows + ((r < remainder) ? 1 : 0);
            sendcounts[r] = rows_for_r;
            displs[r] = disp;
            disp += rows_for_r;
        }
    }

    std::string X_local;
    X_local.resize(local_rows);
    MPI_Scatterv((rank == 0 ? &X_global[0] : nullptr), &sendcounts[0], &displs[0], MPI_CHAR,
                 &X_local[0], local_rows, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Allocate local dp_table
    dp_table.assign(local_rows+1, std::vector<int>(n+1, 0));

    // Buffers for halo exchange
    // upper_row represents dp for the row above start_row (owned by rank-1)
    std::vector<int> upper_row(n+1, 0);

    bool have_upper = (rank > 0);
    bool have_lower = (rank < world_size - 1);

    // Diagonal computation
    for (int d = 1; d <= m + n - 1; d++) {
        int global_i_start = std::max(1, d - n + 1);
        int global_i_end   = std::min(d, m);

        int local_i_start = std::max(global_i_start, start_row);
        int local_i_end   = std::min(global_i_end, end_row);

        // Compute dp for this diagonal
        for (int i = local_i_start; i <= local_i_end; i++) {
            int i_local = i - start_row + 1;
            int j = d - i + 1;
            if (j < 1 || j > n) continue;

            int top  = (i_local == 1) ? (have_upper ? upper_row[j] : 0) 
                                      : dp_table[i_local - 1][j];
            int left = dp_table[i_local][j-1];
            int diag = (i_local == 1) ? (have_upper ? upper_row[j-1] : 0)
                                      : dp_table[i_local - 1][j-1];

            if (X_local[i_local - 1] == Y_global[j - 1]) {
                dp_table[i_local][j] = diag + 1;
            } else {
                dp_table[i_local][j] = std::max(top, left);
            }
        }

        // Exchange halo rows
        // After computing diagonal d, send our bottom row to rank+1 (if any),
        // and receive the top row from rank-1 (if any).
        if (have_lower) {
            // Send bottom row (end_row global => local i_local = local_rows)
            MPI_Send(&dp_table[local_rows][0], n+1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
        }
        if (have_upper) {
            MPI_Recv(&upper_row[0], n+1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    int owner_of_last_row = 0;
    {
        int cumulative = 0;
        for (int r = 0; r < world_size; r++) {
            int rows_for_r = base_rows + ((r < remainder) ? 1 : 0);
            cumulative += rows_for_r;
            if (cumulative >= m) {
                owner_of_last_row = r;
                break;
            }
        }
    }

    int lcs_length_local = 0;
    if (rank == owner_of_last_row) {
        int i_local = m - start_row + 1;
        lcs_length_local = dp_table[i_local][n];
    }

    int lcs_length = 0;
    MPI_Bcast(&lcs_length_local, 1, MPI_INT, owner_of_last_row, MPI_COMM_WORLD);
    lcs_length = lcs_length_local;

    return lcs_length;
}

#else // If MPI is not defined, fallback to serial

int compute_lcs() {
    int m = (int)X_global.size();
    int n = (int)Y_global.size();

    dp_table.assign(m+1, std::vector<int>(n+1, 0));
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (X_global[i-1] == Y_global[j-1]) {
                dp_table[i][j] = dp_table[i-1][j-1] + 1;
            } else {
                dp_table[i][j] = std::max(dp_table[i-1][j], dp_table[i][j-1]);
            }
        }
    }
    return dp_table[m][n];
}

#endif // MPI

void free_memory() {
    dp_table.clear();
}

