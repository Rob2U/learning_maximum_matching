#!/bin/bash
#SBATCH -A herbrich-student
#SBATCH --partition magic
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=out/out_%j.log

#SBATCH --job-name=na_mst
#SBATCH --time=30:00:00


# I think the best way is to copy this file per experiment...
echo "Starting job"
source ~/.bashrc # need this to allow activating a conda env
conda activate na_mst_gpu

# start many agents on the same node
for i in {1..4}
do
    echo "Starting agent $i"
    wandb agent --count 4 $1 &
done

wait
