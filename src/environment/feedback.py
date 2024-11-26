from typing import Any, List, Type

from .commands import AbstractCommand
from .structure_elements import Graph


def reward_naive(
    graph: Graph, program: List[Type[AbstractCommand]], result: Any
) -> float:
    """Naive feedback function for the MSTCodeEnvironment. Placeholder"""
    # We want to minimize the number of instructions
    max_weight = max(graph.edges, key=lambda x: x.weight).weight
    return max_weight - abs(max_weight - float(result))
