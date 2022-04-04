#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=%x.o%j
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0

# Activate anaconda environment
source activate mlns

python -m src.training.train --gnn gcn --filename output_ogb \
 --batch_size 64 --num_workers 0 \
 --model_path $WORKDIR/mlns/models/gcn_model_preprocess.pt