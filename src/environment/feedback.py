from typing import Callable, List, Set

from .algorithms import compute_mst
from .structure_elements import Edge
from .vm_state import VMState

# def reward_multiple_graphs():  # TODO
# pass


def reward(result: Set[Edge], vm_state: VMState) -> float:
    rewards: List[Callable[[Set[Edge], VMState], float]] = [
        # reward_finite_runtime, # bad values (skew the reward)
        # reward_valid_spanning_tree_length,
        # reward_no_cycles, # not implemented
        # reward_covered_nodes,
        # reward_minimality, # bad values (skew the reward)
        # reward_efficiency,
        # reward_distance_to_MST,
        reward_correct_edges,
        # reward_connected,
        punish_mst_weight_too_large,
        # punish_code_length,
        # f1_score_mst,
    ]

    # TODO(mehdi): implement a mechanism to weight the rewards given a config. Make sure that all rewards are on the same scale so the weights are valid.

    return sum(reward(result, vm_state) for reward in rewards)


def f1_score_mst(result: Set[Edge], vm_state: VMState) -> float:
    mst = compute_mst(vm_state.input)
    true_positives = len(result.intersection(mst))
    false_positives = len(result.difference(mst))
    false_negatives = len(mst.difference(result))

    recall: float
    precision: float
    if (true_positives + false_negatives) != 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 1.0

    if (true_positives + false_positives) != 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 1.0

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def punish_mst_weight_too_large(result: Set[Edge], vm_state: VMState) -> float:
    """Punish the algorithm for returning a spanning tree with a weight that is too large. Range: [-1, 0]

    Args:
        result (Set[Edge]): _description_
        vm_state (VMState): _description_
        code_state (CodeState): _description_

    Returns:
        float: _description_
    """
    mst_weight = sum(edge.weight for edge in result)
    actual_mst_weight = sum(edge.weight for edge in compute_mst(vm_state.input))

    return (
        -abs(mst_weight - actual_mst_weight)
        / sum([edge.weight for edge in vm_state.input.edges])
        if mst_weight > actual_mst_weight
        else 0.0
    )


def reward_correct_edges(result: Set[Edge], vm_state: VMState) -> float:
    """Reward the algorithm for returning the correct edges in the MST. Range: [0, 1]

    Args:
        result (Set[Edge]): _description_
        vm_state (VMState): _description_
        code_state (CodeState): _description_

    Returns:
        float: _description_
    """
    mst = compute_mst(vm_state.input)
    correct_edges = len(result.intersection(mst))
    return correct_edges / len(mst)


def punish_code_length(
    result: Set[Edge], vm_state: VMState, punish_cap: int = 24
) -> float:
    code_length = len(vm_state.code)
    return (
        -(code_length - punish_cap) / (32 - punish_cap)  # HACK HACK HACK
        if code_length > punish_cap
        else 0.0
    )


def reward_finite_runtime(result: Set[Edge], vm_state: VMState) -> float:
    return -100.0 if vm_state.timeout else 0.0


# Checks that the length of the returned spanning tree is n - 1
def reward_valid_spanning_tree_length(result: Set[Edge], vm_state: VMState) -> float:
    squared_dist = (len(result) - len(vm_state.input.nodes)) ** 2
    return squared_dist


# Checks that the returned edge set is a spanning tree (i.e. connected, no cycles and it spans all nodes)
def reward_connected(result: Set[Edge], vm_state: VMState) -> float:
    # TODO(mehdi): Implement this. i.e. number of unconnected graphs, distance between unconnected graphs, sizes of unconnected graphs?

    # HACK HACK HACK
    # THIS CODE IS BAD AND SHOULD BE REPLACED BUT OK FOR SIZE 3 GRAPHS
    connected_nodes = set()
    connected_nodes.add(vm_state.input.nodes[0])

    for edge in result:
        for edge in result:
            if edge.u in connected_nodes or edge.v in connected_nodes:
                connected_nodes.add(edge.u)
                connected_nodes.add(edge.v)

    return len(connected_nodes) / len(vm_state.input.nodes)


def reward_no_cycles(result: Set[Edge], vm_state: VMState) -> float:
    # TODO(mehdi): Check if the graph has cycles?
    # TODO(mehdi): can we do this more finegraned? i.e. number of cycles, distance between cycles, size of cycles?
    return 0.0


def reward_covered_nodes(
    result: Set[Edge], vm_state: VMState, factor: float = 100
) -> float:
    count = len(set([edge.u for edge in result] + [edge.v for edge in result]))
    return (count / len(vm_state.input.nodes)) * factor


def reward_minimality(result: Set[Edge], vm_state: VMState) -> float:
    max_weight = max(vm_state.input.edges, key=lambda x: x.weight).weight
    return sum(max_weight - e.weight for e in result)


def reward_distance_to_MST(result: Set[Edge], vm_state: VMState) -> float:
    mst = compute_mst(vm_state.input)
    # count differences between the two sets
    diff = mst.symmetric_difference(result)

    return -len(diff)


def reward_efficiency(result: Set[Edge], vm_state: VMState) -> float:
    # TODO(mehdi): Implement this properly i.e. number of instructions that were actually executed. Probably have to count this in the VM and pass the VM into this.
    return -len(vm_state.code)


def reward_naive(result: Set[Edge], vm_state: VMState) -> float:
    """Naive reward function that only checks if the Set contains the minimal edges"""
    if len(result) == 0:
        return 0.0
    minimal_edge = min(result, key=lambda x: x.weight)
    return (1.0 if minimal_edge in result else 0.0) + 0.5 * (
        1 - len(result) / len(vm_state.input.edges)
    )
