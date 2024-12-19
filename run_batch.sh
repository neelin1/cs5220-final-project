#!/bin/bash
#SBATCH --account=m4776
#SBATCH -C gpu
#SBATCH --qos=debug
#SBATCH --time=00:10:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-task=1

# Load necessary modules (adjust according to your system)
# module load cuda
# module load gcc

# Path to the executable and input file
EXECUTABLE=./build/grid_gpu
INPUT_FILE=tests/test100000.txt


srun --ntasks-per-node=1 dcgmi profile --pause
srun ncu -o report --target-processes all $EXECUTABLE --input $INPUT_FILE
srun --ntasks-per-node=1 dcgmi profile --resume