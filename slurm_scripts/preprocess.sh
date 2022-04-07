#!/bin/bash
#SBATCH --job-name=preprocess_code2
#SBATCH --output=%x.o%j
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --partition=cpu_short

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate mlns

# Train model
python -m src.preprocessing $@