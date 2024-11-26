from typing import Any, List

from .structure_elements import Graph
from .commands import AbstractCommand

def reward_naive(graph: Graph, program: List[AbstractCommand], result: Any) -> float:
    """Naive feedback function for the MSTCodeEnvironment"""
    # We want to minimize the number of instructions
    return result if result > 0.0 else 0.0
