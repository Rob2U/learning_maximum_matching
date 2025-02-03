from typing import Any, Dict, List, Type

from stable_baselines3.common.callbacks import BaseCallback

from environment.environment import Transpiler
from environment.vm_state import AbstractCommand
from models.policy_nets import CustomActorCriticPolicy


class WandbLoggingCallback(BaseCallback):
    """See https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#custom-callback"""

    def __init__(self, wandb_run: Any, global_args: dict[str, Any]) -> None:
        super().__init__()
        self.wandb_run = wandb_run
        self.global_args = global_args
        self.best_reward_avg = -float("inf")
        self.best_ep_reward = -float("inf")
        self.best_end_reward = -float("inf")
        self.best_code = []
        self.episode_counter = 0
        self.episode_entropies: List[float] = []

    def _on_step(
        self,
    ) -> bool:  # NOTE(rob2u): necessary because abstract method in BaseCallback
        """This method will be called after every environment step."""
        infos = (
            self.locals["infos"]
            if isinstance(self.locals["infos"], list)
            else [self.locals["infos"]]
        )
        if isinstance(self.model.policy, CustomActorCriticPolicy):
            self.episode_entropies.append(self.model.policy.entropy)

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
            self.episode_counter += 1

    def _on_episode_end(self, env_info: dict[str, Any]) -> None:
        """This method will be called after every episode."""
        ep_reward = env_info["episode"]["r"]
        ep_length = env_info["episode"]["l"]
        t = env_info["episode"]["t"]
        ep_reward_avg = ep_reward / ep_length

        if ep_reward_avg > self.best_reward_avg:
            self.best_reward_avg = ep_reward_avg

        example_program = self.get_code_from_state(env_info["terminal_observation"])

        averages = {
            key + "_avg": sum(values) / len(values)
            for key, values in env_info.items()
            if key not in ["episode", "terminal_observation", "TimeLimit.truncated"]
        }

        # NOTE(rob2u): 'step_reward_avg' is the reward for this step and the ep is over -> end_reward
        end_reward = averages["step_reward_avg"]
        del averages["step_reward_avg"]
        if end_reward > self.best_end_reward:
            self.best_end_reward = end_reward
            
        ep_reward = env_info["episode"]["r"]
        if ep_reward > self.best_ep_reward:
            self.best_ep_reward = ep_reward
            self.best_code = "[" + ", ".join([str(c()) for c in example_program]) + "]"

        avg_entropy = (
            sum(self.episode_entropies) / len(self.episode_entropies)
            if self.episode_entropies
            else 0
        )

        self.wandb_run.log(
            {
                "ep_reward": ep_reward,
                "ep_len": ep_length,
                "ep_reward_avg": ep_reward / ep_length,
                "best_reward_avg": self.best_reward_avg,
                "best_end_reward": self.best_end_reward,
                "best_code": self.best_code,
                "best_ep_reward": self.best_ep_reward,
                "end_reward": end_reward,
                "t?": t,
                "avg_entropy": avg_entropy,
                "episode": self.episode_counter,
                **averages,
            },
            step=self.num_timesteps,
        )

    def get_code_from_state(self, state: List[int]) -> List[Type[AbstractCommand]]:
        code_size = self.global_args["max_code_length"]
        code = state[-code_size:]
        return Transpiler.intToCommand([int(a) for a in code])  # type: ignore
