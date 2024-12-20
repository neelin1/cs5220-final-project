#!/bin/bash
#SBATCH -N 1
#SBATCH -n 100
#SBATCH -C cpu
#SBATCH --time=00:30:00
#SBATCH --qos=shared
#SBATCH --account=m4776

echo "Starting wavefront_mpi with 1 rank..."
srun -n 1 --cpus-per-task=1 ./build/wavefront_mpi --input ./tests/test100000.txt
echo "Finished wavefront_mpi with 1 rank."

echo "Starting wavefront_mpi with 10 ranks..."
srun -n 1 --cpus-per-task=10 ./build/wavefront_mpi --input ./tests/test100000.txt
echo "Finished wavefront_mpi with 10 ranks."

echo "Starting wavefront_mpi with 31 ranks..."
srun -n 1 --cpus-per-task=31 ./build/wavefront_mpi --input ./tests/test100000.txt
echo "Finished wavefront_mpi with 31 ranks."

echo "Starting wavefront_mpi with 100 ranks..."
srun -n 1 --cpus-per-task=100 ./build/wavefront_mpi --input ./tests/test100000.txt
echo "Finished wavefront_mpi with 100 ranks."

echo "Starting grid_mpi with 1 rank..."
srun -n 1 --cpus-per-task=1 ./build/grid_mpi --input ./tests/test100000.txt
echo "Finished grid_mpi with 1 rank."

echo "Starting grid_mpi with 10 ranks..."
srun -n 1 --cpus-per-task=10 ./build/grid_mpi --input ./tests/test100000.txt
echo "Finished grid_mpi with 10 ranks."

echo "Starting grid_mpi with 31 ranks..."
srun -n 1 --cpus-per-task=31 ./build/grid_mpi --input ./tests/test100000.txt
echo "Finished grid_mpi with 31 ranks."

echo "Starting grid_mpi with 100 ranks..."
srun -n 1 --cpus-per-task=100 ./build/grid_mpi --input ./tests/test100000.txt
echo "Finished grid_mpi with 100 ranks."

echo "Starting grid_mpi with 10 ranks (small input)..."
srun -n 1 --cpus-per-task=10 ./build/grid_mpi --input ./tests/test10000.txt
echo "Finished grid_mpi with 10 ranks (small input)."

echo "Starting grid_mpi with 20 ranks..."
srun -n 1 --cpus-per-task=20 ./build/grid_mpi --input ./tests/test20000.txt
echo "Finished grid_mpi with 20 ranks."

echo "Starting grid_mpi with 30 ranks..."
srun -n 1 --cpus-per-task=30 ./build/grid_mpi --input ./tests/test30000.txt
echo "Finished grid_mpi with 30 ranks."

echo "Starting grid_mpi with 40 ranks..."
srun -n 1 --cpus-per-task=40 ./build/grid_mpi --input ./tests/test40000.txt
echo "Finished grid_mpi with 40 ranks."

echo "Starting grid_mpi with 50 ranks..."
srun -n 1 --cpus-per-task=50 ./build/grid_mpi --input ./tests/test50000.txt
echo "Finished grid_mpi with 50 ranks."

echo "Starting wavefront_mpi with 10 ranks (small input)..."
srun -n 1 --cpus-per-task=10 ./build/wavefront_mpi --input ./tests/test10000.txt
echo "Finished wavefront_mpi with 10 ranks (small input)."

echo "Starting wavefront_mpi with 20 ranks..."
srun -n 1 --cpus-per-task=20 ./build/wavefront_mpi --input ./tests/test20000.txt
echo "Finished wavefront_mpi with 20 ranks."

echo "Starting wavefront_mpi with 30 ranks..."
srun -n 1 --cpus-per-task=30 ./build/wavefront_mpi --input ./tests/test30000.txt
echo "Finished wavefront_mpi with 30 ranks."

echo "Starting wavefront_mpi with 40 ranks..."
srun -n 1 --cpus-per-task=40 ./build/wavefront_mpi --input ./tests/test40000.txt
echo "Finished wavefront_mpi with 40 ranks."

echo "Starting wavefront_mpi with 50 ranks..."
srun -n 1 --cpus-per-task=50 ./build/wavefront_mpi --input ./tests/test50000.txt
echo "Finished wavefront_mpi with 50 ranks."

echo "All tasks completed."
