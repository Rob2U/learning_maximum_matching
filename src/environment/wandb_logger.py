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
        """This method will be called after every model step."""
        infos = (
            self.locals["infos"]
            if isinstance(self.locals["infos"], list)
            else [self.locals["infos"]]
        )
        for info in infos:
            if type(info) is list:
                for env_info in info:
                    self._on_environment_step(env_info)
            else:
                self._on_environment_step(info)

        return True

    def _on_environment_step(self, env_info: dict[str, Any]) -> None:
        """This method will be called after every environment step.
        Args:
            - env_info: dict containing metrics accumulated from each VMs step + possibly episode information (only on episode end).
        """
        if "episode" in env_info.keys():
            self._on_episode_end(env_info)

    def _on_episode_end(self, env_info: dict[str, Any]) -> None:
        """This method will be called after every episode."""
        ep_reward = env_info["episode"]["r"]
        ep_length = env_info["episode"]["l"]
        t = env_info["episode"]["t"]
        ep_reward_avg = ep_reward / ep_length

        if ep_reward_avg > self.best_reward_avg:
            self.best_reward_avg = ep_reward_avg

        example_program = env_info["terminal_observation"]

        averages = {
            key + "_avg": sum(values) / len(values)
            for key, values in env_info.items()
            if key not in ["episode", "terminal_observation", "TimeLimit.truncated"]
        }

        self.wandb_run.log(
            {
                "ep_reward": ep_reward,
                "ep_len": ep_length,
                "ep_reward_avg": ep_reward / ep_length,
                "best_reward_avg": self.best_reward_avg,
                # "program": program,
                "t?": t,
                **averages,
            },
            step=self.num_timesteps,
        )
