#!/bin/bash
#SBATCH -A m4776
#SBATCH -C cpu
#SBATCH -c 1
#SBATCH --qos=debug
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -n 1

srun ./build/basic_serial