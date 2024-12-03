from typing import Set

from .algorithms import compute_mst
from .code_state import CodeState
from .structure_elements import Edge
from .vm_state import VMState


# def reward_multiple_graphs():  # TODO
# pass


def reward(result: Set[Edge], vm_state: VMState, code_state: CodeState) -> float:
    rewards = [
        # reward_finite_runtime, # bad values (skew the reward)
        reward_valid_spanning_tree_length,
        # reward_no_cycles, # not implemented
        reward_covered_nodes,
        # reward_minimality, # bad values (skew the reward)
        reward_efficiency,
        reward_distance_to_MST,
    ]

    # TODO(mehdi): implement a mechanism to weight the rewards given a config. Make sure that all rewards are on the same scale so the weights are valid.

    return sum(
        reward(result=result, vm_state=vm_state, code_state=code_state)  # type: ignore
        for reward in rewards
    )


def reward_finite_runtime(
    result: Set[Edge], vm_state: VMState, code_state: CodeState
) -> float:
    return -100.0 if code_state.timeout else 0.0


# Checks that the length of the returned spanning tree is n - 1
def reward_valid_spanning_tree_length(
    result: Set[Edge], vm_state: VMState, code_state: CodeState
) -> float:
    squared_dist = (len(result) - len(vm_state.input.nodes)) ** 2
    return squared_dist


# Checks that the returned edge set is a spanning tree (i.e. connected, no cycles and it spans all nodes)
def reward_connected(
    result: Set[Edge], vm_state: VMState, code_state: CodeState
) -> float:
    # TODO(mehdi): Implement this. i.e. number of unconnected graphs, distance between unconnected graphs, sizes of unconnected graphs?
    return 0.0


def reward_no_cycles(
    result: Set[Edge], vm_state: VMState, code_state: CodeState
) -> float:
    # TODO(mehdi): Check if the graph has cycles?
    # TODO(mehdi): can we do this more finegraned? i.e. number of cycles, distance between cycles, size of cycles?
    return 0.0


def reward_covered_nodes(
    result: Set[Edge], vm_state: VMState, code_state: CodeState, factor: float = 100
) -> float:
    count = len(set([edge.u for edge in result] + [edge.v for edge in result]))
    return (count / len(vm_state.input.nodes)) * factor


def reward_minimality(
    result: Set[Edge], vm_state: VMState, code_state: CodeState
) -> float:
    max_weight = max(vm_state.input.edges, key=lambda x: x.weight).weight
    return sum(max_weight - e.weight for e in result)


def reward_distance_to_MST(
    result: Set[Edge], vm_state: VMState, code_state: CodeState
) -> float:
    mst = compute_mst(vm_state.input)
    # count differences between the two sets
    diff = mst.symmetric_difference(result)
    return -len(diff)


def reward_efficiency(
    result: Set[Edge], vm_state: VMState, code_state: CodeState
) -> float:
    # TODO(mehdi): Implement this properly i.e. number of instructions that were actually executed. Probably have to count this in the VM and pass the VM into this.
    return -len(code_state.code)


def reward_naive(result: Set[Edge], vm_state: VMState, code_state: CodeState) -> float:
    """Naive reward function that only checks if the Set contains the minimal edges"""
    if len(result) == 0:
        return 0.0
    minimal_edge = min(result, key=lambda x: x.weight)
    return (1.0 if minimal_edge in result else 0.0) + 0.5 * (
        1 - len(result) / len(vm_state.input.edges)
    )
