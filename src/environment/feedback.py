from typing import Any

from .structure_elements import Graph


def reward_naive(graph: Graph, result: Any) -> float:
    """Naive feedback function for the MSTCodeEnvironment"""
    # We want to minimize the number of instructions
    return 0.0
