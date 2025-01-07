#!/bin/bash
#SBATCH -A herbrich-student
#SBATCH --partition cauldron
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=3GB
#SBATCH --job-name=na_mst_gpu
#SBATCH --output=out/gpu_%j.log
#SBATCH --time=24:00:00

# I think the best way is to copy this file per experiment...

source ~/.bashrc # need this to allow activating a conda env
conda activate na_mst
python src/train.py --config=configs/rew_on_ret.yaml