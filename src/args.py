from dataclasses import dataclass, field
from typing import List

from simple_parsing.helpers import Serializable  # type: ignore


@dataclass
class GlobalArgs(Serializable):
    iterations: int = 100_000
    max_code_length: int = 32
    reset_for_every_run: bool = False
    action_masking: bool = False
    only_reward_on_ret: bool = True

    # ENVIRONMENT -- GRAPH Generation
    num_vms_per_env: int = 10
    min_n: int = 3
    min_m: int = 3
    max_n: int = 3
    max_m: int = 3

    # REWARDS:
    punish_cap: int = 24

    # MODEL:
    policy_net: List[int] = field(default_factory=lambda: [256, 256, 256])
