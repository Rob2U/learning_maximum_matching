#!/bin/bash
# wandb: Run sweep agent with: wandb agent ...
$(wandb sweep ./configs/sweep.yaml) >temp_output.txt 2>&1

SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' temp_output.txt)

rm temp_output.txt

echo "SWEEP_ID: $SWEEP_ID"

# for loop
for i in {1..5}
do
    sbatch batch_agent.sh "$SWEEP_ID"
done