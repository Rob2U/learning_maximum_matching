import math
import subprocess

import wandb

# Usage:
# - adjust hyperparameters to sweep over in sweep_config
# - python src/sweep.py
# - check that all agent subprocesses were started with `squeue -u <username>`

sweep_config = {
    "program": "src/train.py",
    "name": "better_rewards_mlp_beta0.5",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "best_ep_reward"},
    "parameters": {
        "config_path": {
            "values": ["configs/config.yaml"],
        },
        "beta": {
            "values": [0.5],
        },
        "learning_rate": {
            "distribution": "log_uniform",
            "min": math.log(1e-5),
            "max": math.log(1e-3),
        },
        "gamma": {
            "distribution": "log_uniform",
            "min": math.log(0.9),
            "max": math.log(0.999),
        },
        "ent_coef": {
            "distribution": "log_uniform",
            "min": math.log(1e-3),
            "max": math.log(0.2),
        },
        "clip_range": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.5,
        },
    },
}

NUM_AGENTS = 5


def start_agent(sweep_id: str) -> None:
    # Start the agent
    subprocess.run(
        [
            "sbatch",
            "batch_agent.sh",
            sweep_id,
        ]
    )


def sweep() -> None:
    sweep_id = wandb.sweep(
        sweep_config,
        entity="robert-weeke2-uni-potsdam",
        project="constrainedIS",
    )

    print("Sweep ID: ", sweep_id)

    # for i in range(NUM_AGENTS):
    #     start_agent(sweep_id)


if __name__ == "__main__":
    sweep()
