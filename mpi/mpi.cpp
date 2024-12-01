#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

int nx, ny;

int local_nx;
int rank, num_procs;
int t = 0;

double *h, *u, *v;
double *dh, *du, *dv;
double *dh1, *du1, *dv1;
double *dh2, *du2, *dv2;

double H, g, dx, dy, dt;

#undef h
#undef u
#define h(i, j) h[(i + 1) * (ny + 1) + (j)]
#define u(i, j) u[(i + 1) * ny + (j)]

void init(double *h0, double *u0, double *v0, double length_, double width_,
          int nx_, int ny_, double H_, double g_, double dt_,
          int rank_, int num_procs_)
{
    rank = rank_;
    num_procs = num_procs_;

    nx = nx_;
    ny = ny_;

    H = H_;
    g = g_;
    dx = length_ / nx;
    dy = width_ / ny;
    dt = dt_;

    local_nx = nx / num_procs;
    int remainder = nx % num_procs;
    if (rank < remainder)
        local_nx++;

    h = (double *)calloc((local_nx + 2) * (ny + 1), sizeof(double));
    u = (double *)calloc((local_nx + 2) * ny, sizeof(double));
    v = (double *)calloc((local_nx + 2) * (ny + 1), sizeof(double));

    dh = (double *)calloc(local_nx * ny, sizeof(double));
    du = (double *)calloc(local_nx * ny, sizeof(double));
    dv = (double *)calloc(local_nx * ny, sizeof(double));

    dh1 = (double *)calloc(local_nx * ny, sizeof(double));
    du1 = (double *)calloc(local_nx * ny, sizeof(double));
    dv1 = (double *)calloc(local_nx * ny, sizeof(double));

    dh2 = (double *)calloc(local_nx * ny, sizeof(double));
    du2 = (double *)calloc(local_nx * ny, sizeof(double));
    dv2 = (double *)calloc(local_nx * ny, sizeof(double));

    int *sendcounts_h = NULL, *displs_h = NULL;
    int *sendcounts_u = NULL, *displs_u = NULL;
    int *sendcounts_v = NULL, *displs_v = NULL;

    if (rank == 0) {
        sendcounts_h = (int *)malloc(num_procs * sizeof(int));
        displs_h = (int *)malloc(num_procs * sizeof(int));
        sendcounts_u = (int *)malloc(num_procs * sizeof(int));
        displs_u = (int *)malloc(num_procs * sizeof(int));
        sendcounts_v = (int *)malloc(num_procs * sizeof(int));
        displs_v = (int *)malloc(num_procs * sizeof(int));

        int offset_h = 0, offset_u = 0, offset_v = 0;
        for (int i = 0; i < num_procs; i++) {
            int proc_nx = nx / num_procs;
            if (i < remainder)
                proc_nx++;

            sendcounts_h[i] = proc_nx * (ny + 1);
            displs_h[i] = offset_h;
            offset_h += proc_nx * (ny + 1);

            sendcounts_u[i] = proc_nx * ny;
            displs_u[i] = offset_u;
            offset_u += proc_nx * ny;

            sendcounts_v[i] = proc_nx * (ny + 1);
            displs_v[i] = offset_v;
            offset_v += proc_nx * (ny + 1);
        }
    }

    MPI_Scatterv(h0, sendcounts_h, displs_h, MPI_DOUBLE,
                 h + (ny + 1), local_nx * (ny + 1), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(u0, sendcounts_u, displs_u, MPI_DOUBLE,
                 u + ny, local_nx * ny, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(v0, sendcounts_v, displs_v, MPI_DOUBLE,
                 v + (ny + 1), local_nx * (ny + 1), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(sendcounts_h);
        free(displs_h);
        free(sendcounts_u);
        free(displs_u);
        free(sendcounts_v);
        free(displs_v);
    }
}

void compute_ghost_vertical()
{
    for (int i = 0; i < local_nx; i++)
        h(i, ny) = h(i, 0);
}

void compute_boundaries_vertical()
{
    for (int i = 0; i < local_nx; i++)
        v(i, 0) = v(i, ny);
}

void exchange_ghost_cells()
{
    int left = (rank - 1 + num_procs) % num_procs;
    int right = (rank + 1) % num_procs;

    MPI_Sendrecv(h + (local_nx * (ny + 1)), (ny + 1), MPI_DOUBLE, right, 0,
                 h, (ny + 1), MPI_DOUBLE, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(h + (ny + 1), (ny + 1), MPI_DOUBLE, left, 1,
                 h + ((local_nx + 1) * (ny + 1)), (ny + 1), MPI_DOUBLE, right, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(u + (local_nx * ny), ny, MPI_DOUBLE, right, 2,
                 u, ny, MPI_DOUBLE, left, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(u + ny, ny, MPI_DOUBLE, left, 3,
                 u + ((local_nx + 1) * ny), ny, MPI_DOUBLE, right, 3,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void compute_dh()
{
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            double du_dx = (u(i + 1, j) - u(i, j)) / dx;
            double dv_dy = (v(i, j + 1) - v(i, j)) / dy;
            dh(i, j) = -H * (du_dx + dv_dy);
        }
    }
}

void compute_du()
{
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            double dh_dx = (h(i + 1, j) - h(i, j)) / dx;
            du(i, j) = -g * dh_dx;
        }
    }
}

void compute_dv()
{
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            double dh_dy = (h(i, j + 1) - h(i, j)) / dy;
            dv(i, j) = -g * dh_dy;
        }
    }
}

void multistep(double a1, double a2, double a3_dt)
{
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            h(i, j) += (a1 * dh(i, j) + a2 * dh1(i, j) + a3_dt * dh2(i, j));
            u(i + 1, j) += (a1 * du(i, j) + a2 * du1(i, j) + a3_dt * du2(i, j));
            v(i, j + 1) += (a1 * dv(i, j) + a2 * dv1(i, j) + a3_dt * dv2(i, j));
        }
    }
}

void swap_buffers()
{
    double *tmp;

    tmp = dh2;
    dh2 = dh1;
    dh1 = dh;
    dh = tmp;

    tmp = du2;
    du2 = du1;
    du1 = du;
    du = tmp;

    tmp = dv2;
    dv2 = dv1;
    dv1 = dv;
    dv = tmp;
}

void step()
{
    compute_ghost_vertical();
    compute_boundaries_vertical();

    exchange_ghost_cells();

    compute_dh();
    compute_du();
    compute_dv();

    double a1, a2, a3;
    if (t == 0)
    {
        a1 = 1.0;
    }
    else if (t == 1)
    {
        a1 = 3.0 / 2.0;
        a2 = -1.0 / 2.0;
    }
    else
    {
        a1 = 23.0 / 12.0;
        a2 = -16.0 / 12.0;
        a3 = 5.0 / 12.0;
    }

    double a3_dt = dt * a3;

    multistep(a1, a2, a3_dt);

    swap_buffers();

    t++;
}

void transfer(double *h_recv)
{
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        recvcounts = (int *)malloc(num_procs * sizeof(int));
        displs = (int *)malloc(num_procs * sizeof(int));

        int offset = 0;
        int remainder = nx % num_procs;
        for (int i = 0; i < num_procs; i++) {
            int proc_nx = nx / num_procs;
            if (i < remainder)
                proc_nx++;

            recvcounts[i] = proc_nx * (ny + 1);
            displs[i] = offset;
            offset += proc_nx * (ny + 1);
        }
    }

    MPI_Gatherv(h + (ny + 1), local_nx * (ny + 1), MPI_DOUBLE,
                h_recv, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }
}

void free_memory()
{
    free(h);
    free(u);
    free(v);

    free(dh);
    free(du);
    free(dv);

    free(dh1);
    free(du1);
    free(dv1);

    free(dh2);
    free(du2);
    free(dv2);
}
