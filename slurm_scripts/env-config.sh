#!/bin/bash
#SBATCH --job-name=env_setup
#SBATCH --output=%x.o%j
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_test
#SBATCH --mem=50G

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Create env
conda env create --file environment.yml --force