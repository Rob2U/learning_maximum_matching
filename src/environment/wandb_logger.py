from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


class WandbLoggingCallback(BaseCallback):
    """See https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#custom-callback"""

    def __init__(self, wandb_run: Any):
        super().__init__()
        self.wandb_run = wandb_run
        self.best_reward_avg = -float("inf")

    def _on_step(
        self,
    ) -> bool:  # NOTE(rob2u): necessary because abstract method in BaseCallback
        infos = (
            self.locals["infos"]
            if isinstance(self.locals["infos"], list)
            else [self.locals["infos"]]
        )
        for info in infos:
            if type(info) is list:
                for env_info in info:
                    self._on_episode(env_info)
            else:
                self._on_episode(info)

        return True

    def _on_episode(self, env_info: dict[str, Any]) -> None:
        if "episode" in env_info.keys():
            ep_reward = env_info["episode"]["r"]
            ep_length = env_info["episode"]["l"]
            t = env_info["episode"]["t"]
            ep_reward_avg = ep_reward / ep_length

            if ep_reward_avg > self.best_reward_avg:
                self.best_reward_avg = ep_reward_avg

            program = env_info["terminal_observation"]

            self.wandb_run.log(
                {
                    "ep_reward": ep_reward,
                    "ep_len": ep_length,
                    "ep_reward_avg": ep_reward / ep_length,
                    "best_reward_avg": self.best_reward_avg,
                    # "program": program,
                    "t?": t,
                },
                step=self.num_timesteps,
            )
