#!/bin/bash
# wandb: Creating sweep with ID: ....
SWEEP_ID=$(wandb sweep ./configs/sweep.yaml | grep "Creating sweep with ID:" | awk '{print $NF}')

echo "SWEEP_ID: $SWEEP_ID"

# for loop
for i in {1..5}
do
    sbatch batch_agent.sh "$SWEEP_ID"
done