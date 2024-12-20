#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -C cpu
#SBATCH --time=00:30:00
#SBATCH --qos=debug
#SBATCH --account=m4776

module load PrgEnv-gnu

# Run perf to profile your MPI program
srun perf record -o perf_wavefront_mpi.data ./build/wavefront_mpi
