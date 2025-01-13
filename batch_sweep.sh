#!/bin/bash
#SBATCH -A herbrich-student
#SBATCH --partition magic
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=out/out_%j.log

#SBATCH --job-name=na_mst_sweep
#SBATCH --time=12:00:00

# I think the best way is to copy this file per experiment...

source ~/.bashrc # need this to allow activating a conda env
conda activate na_mst

python src/sweep.py
