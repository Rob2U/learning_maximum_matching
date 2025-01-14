#!/bin/bash
SWEEP_ID = wandb sweep ./configs/sweep.yaml

# for loop
for i in {1..5}
    sbatch batch_agent.sh "$SWEEP_ID"