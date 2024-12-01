#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>  
#include <stdlib.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

int nx, ny;
double *d_h, *d_u, *d_v;
double *d_dh, *d_du, *d_dv, *d_dh1, *d_du1, *d_dv1, *d_dh2, *d_du2, *d_dv2;
double H, g, dx, dy, dt;
int t = 0;

#define CUDA_CHECK(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

void init(double *h0, double *u0, double *v0, double length_, double width_, int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_)
{
    nx = nx_;
    ny = ny_;

    H = H_;
    g = g_;
    dx = length_ / nx;
    dy = width_ / ny;
    dt = dt_;

    size_t size_h = (nx + 1) * (ny + 1) * sizeof(double);
    size_t size_u = (nx + 1) * ny * sizeof(double);
    size_t size_v = nx * (ny + 1) * sizeof(double);
    size_t size_deriv = nx * ny * sizeof(double);

    CUDA_CHECK(cudaMalloc((void **)&d_h, size_h));
    CUDA_CHECK(cudaMalloc((void **)&d_u, size_u));
    CUDA_CHECK(cudaMalloc((void **)&d_v, size_v));

    CUDA_CHECK(cudaMalloc((void **)&d_dh, size_deriv));
    CUDA_CHECK(cudaMalloc((void **)&d_du, size_deriv));
    CUDA_CHECK(cudaMalloc((void **)&d_dv, size_deriv));

    CUDA_CHECK(cudaMalloc((void **)&d_dh1, size_deriv));
    CUDA_CHECK(cudaMalloc((void **)&d_du1, size_deriv));
    CUDA_CHECK(cudaMalloc((void **)&d_dv1, size_deriv));

    CUDA_CHECK(cudaMalloc((void **)&d_dh2, size_deriv));
    CUDA_CHECK(cudaMalloc((void **)&d_du2, size_deriv));
    CUDA_CHECK(cudaMalloc((void **)&d_dv2, size_deriv));

    CUDA_CHECK(cudaMemcpy(d_h, h0, size_h, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u, u0, size_u, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, v0, size_v, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_dh, 0, size_deriv));
    CUDA_CHECK(cudaMemset(d_du, 0, size_deriv));
    CUDA_CHECK(cudaMemset(d_dv, 0, size_deriv));

    CUDA_CHECK(cudaMemset(d_dh1, 0, size_deriv));
    CUDA_CHECK(cudaMemset(d_du1, 0, size_deriv));
    CUDA_CHECK(cudaMemset(d_dv1, 0, size_deriv));

    CUDA_CHECK(cudaMemset(d_dh2, 0, size_deriv));
    CUDA_CHECK(cudaMemset(d_du2, 0, size_deriv));
    CUDA_CHECK(cudaMemset(d_dv2, 0, size_deriv));
}

__global__ void compute_ghost_horizontal(double *h, int nx, int ny)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < ny)
    {
        h[nx * (ny + 1) + j] = h[j];
    }
}

__global__ void compute_ghost_vertical(double *h, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx)
    {
        h[i * (ny + 1) + ny] = h[i * (ny + 1)];
    }
}

__global__ void compute_derivatives(double *u, double *v, double *h,
                                    double *dh, double *du, double *dv,
                                    int nx, int ny, double dx, double dy, double H, double g)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny;
    if (idx < total)
    {
        int i = idx / ny;
        int j = idx % ny;

        double du_dx = (u[(i + 1) * ny + j] - u[i * ny + j]) / dx;
        double dv_dy = (v[i * (ny + 1) + j + 1] - v[i * (ny + 1) + j]) / dy;
        dh[idx] = -H * (du_dx + dv_dy);

        double dh_dx = (h[(i + 1) * (ny + 1) + j] - h[i * (ny + 1) + j]) / dx;
        du[idx] = -g * dh_dx;

        double dh_dy = (h[i * (ny + 1) + j + 1] - h[i * (ny + 1) + j]) / dy;
        dv[idx] = -g * dh_dy;
    }
}

__global__ void multistep(double *h, double *u, double *v,
                          double *dh, double *du, double *dv,
                          double *dh1, double *du1, double *dv1,
                          double *dh2, double *du2, double *dv2,
                          int nx, int ny, double a1, double a2, double a3_dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny;
    if (idx < total)
    {
        int i = idx / ny;
        int j = idx % ny;

        h[i * (ny + 1) + j] += (a1 * dh[idx] + a2 * dh1[idx] + a3_dt * dh2[idx]);

        if (i + 1 < nx + 1)
        {
            int u_idx = (i + 1) * ny + j;
            u[u_idx] += (a1 * du[idx] + a2 * du1[idx] + a3_dt * du2[idx]);
        }

        if (j + 1 < ny + 1)
        {
            int v_idx = i * (ny + 1) + j + 1;
            v[v_idx] += (a1 * dv[idx] + a2 * dv1[idx] + a3_dt * dv2[idx]);
        }
    }
}

__global__ void compute_boundaries_horizontal(double *u, int nx, int ny)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < ny)
    {
        u[j] = u[nx * ny + j];
    }
}

__global__ void compute_boundaries_vertical(double *v, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx)
    {
        v[i * (ny + 1)] = v[i * (ny + 1) + ny];
    }
}

void swap_buffers()
{
    double *tmp;

    tmp = d_dh2;
    d_dh2 = d_dh1;
    d_dh1 = d_dh;
    d_dh = tmp;

    tmp = d_du2;
    d_du2 = d_du1;
    d_du1 = d_du;
    d_du = tmp;

    tmp = d_dv2;
    d_dv2 = d_dv1;
    d_dv1 = d_dv;
    d_dv = tmp;
}

void step()
{
    int blockSize = 256;
    int gridSize_h = (ny + blockSize - 1) / blockSize;
    int gridSize_v = (nx + blockSize - 1) / blockSize;
    int total_cells = nx * ny;
    int gridSize_total = (total_cells + blockSize - 1) / blockSize;

    compute_ghost_horizontal<<<gridSize_h, blockSize>>>(d_h, nx, ny);
    compute_ghost_vertical<<<gridSize_v, blockSize>>>(d_h, nx, ny);

    compute_derivatives<<<gridSize_total, blockSize>>>(d_u, d_v, d_h,
                                                       d_dh, d_du, d_dv,
                                                       nx, ny, dx, dy, H, g);

    CUDA_CHECK(cudaGetLastError());

    double a1 = 0.0, a2 = 0.0, a3_dt = 0.0;
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
        a3_dt = (5.0 / 12.0) * dt;
    }

    multistep<<<gridSize_total, blockSize>>>(
        d_h, d_u, d_v,
        d_dh, d_du, d_dv,
        d_dh1, d_du1, d_dv1,
        d_dh2, d_du2, d_dv2,
        nx, ny, a1, a2, a3_dt);

    compute_boundaries_horizontal<<<gridSize_h, blockSize>>>(d_u, nx, ny);
    compute_boundaries_vertical<<<gridSize_v, blockSize>>>(d_v, nx, ny);

    swap_buffers();

    t++;
}

void transfer(double *h_host)
{
    size_t size_h = (nx + 1) * (ny + 1) * sizeof(double);
    CUDA_CHECK(cudaMemcpy(h_host, d_h, size_h, cudaMemcpyDeviceToHost));
}

void free_memory()
{
    CUDA_CHECK(cudaFree(d_h));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_v));

    CUDA_CHECK(cudaFree(d_dh));
    CUDA_CHECK(cudaFree(d_du));
    CUDA_CHECK(cudaFree(d_dv));

    CUDA_CHECK(cudaFree(d_dh1));
    CUDA_CHECK(cudaFree(d_du1));
    CUDA_CHECK(cudaFree(d_dv1));

    CUDA_CHECK(cudaFree(d_dh2));
    CUDA_CHECK(cudaFree(d_du2));
    CUDA_CHECK(cudaFree(d_dv2));
}