import wandb
import subprocess
import sys
from pathlib import Path
from args import GlobalArgs

# This Python script:
#  1) Creates a WandB sweep configuration that grid-searches over learning_rate and exploration_rate
#  2) Defines a "train" function that calls train.py with the hyperparameters from wandb.config
#  3) Launches the sweep agent locally to run the experiments

# (Make sure you have `pip install wandb` and are logged in with `wandb login`)

SWEEP_CONFIG = {
    "name": "learning_and_exploration_sweep",
    "method": "bayes",  # or "random", "bayes"
    "metric": {
        "name": "reward",  # The metric name you log in train.py (e.g., wandb.log({"reward": ...}))
        "goal": "maximize",
    },
    "parameters": {
        "learning_rate": {"values": [1e-3, 1e-4, 1e-5]},
        "exploration_rate": {"values": [0.1, 0.2, 0.3]},
    },
}


def train() -> None:
    # 1) Initialize wandb run
    wandb.init()
    # 2) Extract hyperparameters from wandb.config
    learning_rate = wandb.config.learning_rate
    exploration_rate = wandb.config.exploration_rate

    # 3) Copy your base config and override relevant fields
    base_args = GlobalArgs.load_yaml("configs/config.yaml").to_dict()
    base_args["learning_rate"] = learning_rate
    base_args["exploration_rate"] = exploration_rate

    # 4) Store the new config as a temporary file
    tmp_config_path = Path("configs/tmp_sweep_config.yaml")
    with tmp_config_path.open("w+") as f:
        GlobalArgs(**base_args).dump_yaml(f)

    # 5) Call train.py with this config
    subprocess.run(
        [sys.executable, "src/train.py", "--config", str(tmp_config_path)], check=True
    )

    # 6) Clean up
    tmp_config_path.unlink()


def main() -> None:
    # Create the sweep on WandB
    sweep_id = wandb.sweep(SWEEP_CONFIG, project="my_mst_project")
    # Launch the sweep agent: this will repeatedly call `train()` above
    wandb.agent(sweep_id, function=train)


if __name__ == "__main__":
    main()
