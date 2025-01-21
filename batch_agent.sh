#!/bin/bash
#SBATCH -A herbrich-student
#SBATCH --partition magic
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=out/out_%j.log

#SBATCH --job-name=na_mst_cpu
#SBATCH --time=30:00:00

# I think the best way is to copy this file per experiment...

source ~/.bashrc # need this to allow activating a conda env
conda activate na_mst

wandb agent $1 –count 2
