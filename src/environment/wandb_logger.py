from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


class WandbLoggingCallback(BaseCallback):
    """See https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#custom-callback"""

    def __init__(self, wandb_run: Any):
        super().__init__()
        self.wandb_run = wandb_run

    def _on_step(
        self,
    ) -> bool:  # NOTE(rob2u): necessary because abstract method in BaseCallback
        infos = (
            self.locals["infos"]
            if isinstance(self.locals["infos"], list)
            else [self.locals["infos"]]
        )
        for info in infos:
            for env_info in info:
                if "episode" in env_info.keys():
                    ep_reward = env_info["episode"]["r"]
                    ep_length = env_info["episode"]["l"]

                    self.wandb_run.log(
                        {
                            "ep_reward": ep_reward,
                            "ep_len": ep_length,
                            "ep_reward_avg": ep_reward / ep_length,
                        },
                        step=self.num_timesteps,
                    )

        return True
