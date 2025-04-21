#!/bin/bash -l
#SBATCH -A herbrich-student
#SBATCH --partition magic
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=out/out_%j.log
#SBATCH --constraint=ARCH:X86

#SBATCH --job-name=na_mst
#SBATCH --time=30:00:00


# I think the best way is to copy this file per experiment...
echo "Starting job"

# Remove the na_mst bin directory from PATH
export PATH=$(echo "$PATH" | sed 's|/hpi/fs00/scratch/bp-herbrich23/philipp.kolbe/na_mst/bin:||')
# Prepend the miniconda3 directories
export PATH="/hpi/fs00/home/philipp.kolbe/miniconda3/condabin:/hpi/fs00/home/philipp.kolbe/miniconda3/bin:$PATH"
unset DISPLAY

echo "PATH: $PATH"
which conda
conda info --envs
conda info -a
env | grep CONDA

# source ~/.bashrc # need this to allow activating a conda env
export PATH="/hpi/fs00/home/philipp.kolbe/miniconda3/bin:$PATH"
unset DISPLAY
source /hpi/fs00/home/philipp.kolbe/miniconda3/etc/profile.d/conda.sh
conda activate na_mst

# start many agents on the same node
for i in {1..4}
do
    echo "Starting agent $i"
    wandb agent --count 4 $1 &
done

wait
