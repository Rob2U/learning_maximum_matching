from dataclasses import dataclass, field
from typing import List

import torch
from simple_parsing.helpers import Serializable  # type: ignore


@dataclass
class GlobalArgs(Serializable):
    iterations: int = 100_000
    max_code_length: int = 32
    reset_for_every_run: bool = False
    action_masking: bool = False
    only_reward_on_ret: bool = True
    seed: int = 42
    vectorize_environment: bool = True
    num_envs: int = 16  # NOTE(rob2u): automatically set to 1 if not vectorized
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ENVIRONMENT -- GRAPH Generation
    num_vms_per_env: int = 100
    min_n: int = 3
    min_m: int = 3
    max_n: int = 3
    max_m: int = 3

    # REWARDS:
    punish_cap: int = 24

    # MODEL:
    # Transformer Features Extractor
    fe_d_model: int = 64
    fe_nhead: int = 4
    fe_num_blocks: int = 4

    # General policy network
    feature_dim: int = 64
    layer_dim_pi: int = 64
    layer_dim_vf: int = 64

    gamma: float = 0.99
    batch_size: int = 64
    learning_rate: float = 3e-4

    # rewards
    beta: float = 2.0
