from typing import Any, List, Type, Set

from .commands import AbstractCommand
from .structure_elements import Graph, Edge
from .algorithms import UnionFind, compute_mst

def reward(graph: Graph, program: List[Type[AbstractCommand]], result: Set[Edge], infinite: bool) -> float:
    rewards = [
        reward_finite_runtime,
        reward_valid_spanning_tree_length,
        reward_no_cycles,
        reward_covered_nodes,
        reward_minimality,
        reward_efficiency,
        reward_distance_to_MST,
    ]
    
    # TODO(mehdi): implement a mechanism to weight the rewards given a config. Make sure that all rewards are on the same scale so the weights are valid.

    return sum(reward(graph, program, result, infinite) for reward in rewards)

def reward_finite_runtime(graph: Graph, program: List[Type[AbstractCommand]], result: Set[Edge], infinite: bool) -> float:
    return -100.0 if infinite else 0.0

# Checks that the length of the returned spanning tree is n - 1
def reward_valid_spanning_tree_length(graph: Graph, program: List[Type[AbstractCommand]], result: Set[Edge], infinite: bool) -> float:
    squared_dist = (len(result) - len(graph.nodes)) ** 2
    return squared_dist

# Checks that the returned edge set is a spanning tree (i.e. connected, no cycles and it spans all nodes)
def reward_connected(graph: Graph, program: List[Type[AbstractCommand]], result: Set[Edge], infinite: bool) -> float:
    # TODO(mehdi): Implement this. i.e. number of unconnected graphs, distance between unconnected graphs, sizes of unconnected graphs?
    return 0.0

def reward_no_cycles(graph: Graph, program: List[Type[AbstractCommand]], result: Set[Edge], infinite: bool) -> float:
    # TODO(mehdi): Check if the graph has cycles?
    # TODO(mehdi): can we do this more finegraned? i.e. number of cycles, distance between cycles, size of cycles?
    return 0.0
    
def reward_covered_nodes(graph: Graph, program: List[Type[AbstractCommand]], result: Set[Edge], infinite: bool) -> float:
    count = len(set([edge.u for edge in result] + [edge.v for edge in result]))
    return count / len(graph.nodes)

def reward_minimality(graph: Graph, program: List[Type[AbstractCommand]], result: Set[Edge], infinite: bool) -> float:
    max_weight = max(graph.edges, key=lambda x: x.weight).weight
    return sum(max_weight - e.weight for e in result)

def reward_distance_to_MST(graph: Graph, program: List[Type[AbstractCommand]], result: Set[Edge], infinite: bool) -> float:
    mst = compute_mst(graph)
    # count differences between the two sets
    diff = mst.symmetric_difference(result)
    return len(diff)

def reward_efficiency(graph: Graph, program: List[Type[AbstractCommand]], result: Set[Edge], infinite: bool) -> float:
    # TODO(mehdi): Implement this properly i.e. number of instructions that were actually executed. Probably have to count this in the VM and pass the VM into this.
    return -len(program)

def reward_naive(
    graph: Graph, program: List[Type[AbstractCommand]], result: Set[Edge], infinite: bool
) -> float:
    return max(r.weight for r in result) / max(e.weight for e in graph.edges)
