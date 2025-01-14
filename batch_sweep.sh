#!/bin/bash
SWEEP_ID=$(wandb sweep ./configs/sweep.yaml | grep "Created sweep with ID:" | awk '{print $NF}')

echo "SWEEP_ID: $SWEEP_ID"

# for loop
for i in {1..5}
    sbatch batch_agent.sh "$SWEEP_ID"