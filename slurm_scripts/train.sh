#!/bin/bash
#SBATCH --job-name=met_naming_train
#SBATCH --output=%x.o%j
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate mlns

# Train model
python -m src.training.train $@