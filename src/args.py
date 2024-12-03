from dataclasses import dataclass

from simple_parsing.helpers import Serializable  # type: ignore


@dataclass
class GlobalArgs(Serializable):
    iterations: int = 100_000
    max_code_length: int = 32
    reset_for_every_run: bool = False

    # ENVIRONMENT -- GRAPH Generation
    min_n: int = 3
    min_m: int = 3
    max_n: int = 3
    max_m: int = 3

    # REWARDS:
    punish_cap: int = 24
