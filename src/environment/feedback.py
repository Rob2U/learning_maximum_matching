from typing import Any, Dict, List, Protocol, Set, Tuple, runtime_checkable

import wandb

from .algorithms import compute_mst
from .structure_elements import Edge
from .vm_state import VMState

# def reward_multiple_graphs():  # TODO
# pass


@runtime_checkable
class RewardFunction(Protocol):

    def __call__(
        self, result: Set[Edge], vm_state: VMState, **kwargs: Any
    ) -> Tuple[float, Dict[str, Any]]:
        pass


def reward(
    result: Set[Edge], vm_state: VMState, reward_fn: Dict[str, float], **kwargs: Any
) -> Tuple[float, Dict[str, Any]]:
    reward = 0.0
    metric_dict: Dict[str, Any] = {}

    for fn, weight in reward_fn.items():
        assert fn in globals(), f"Reward function {fn} not found in feedback.py"

        partial_reward, partial_metric_dict = eval(fn + "(result, vm_state, **kwargs)")
        reward += weight * partial_reward
        metric_dict.update(partial_metric_dict)

    # Also Add the reward:
    metric_dict["step_reward"] = reward
    
    return reward, metric_dict


def f_score_mst(
    result: Set[Edge], vm_state: VMState, beta: float = 2.0, **kwargs: Any
) -> Tuple[float, Dict[str, Any]]:
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
        return 0.0, {"precision": precision, "recall": recall, "f_beta": 0.0}

    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

    return f_beta, {"precision": precision, "recall": recall, "f_beta": f_beta}


def punish_mst_weight_too_large(
    result: Set[Edge], vm_state: VMState, **kwargs: Any
) -> Tuple[float, Dict[str, Any]]:
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

    punish_score = (
        -abs(mst_weight - actual_mst_weight)
        / sum([edge.weight for edge in vm_state.input.edges])
        if mst_weight > actual_mst_weight
        else 0.0
    )

    return punish_score, {
        "mst_weight": mst_weight,
        "actual_mst_weight": actual_mst_weight,
        "mst_weight_punish_score": punish_score,
    }


def reward_correct_edges(
    result: Set[Edge], vm_state: VMState, **kwargs: dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
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

    reward = correct_edges / len(mst)

    return reward, {"correct_edges": correct_edges, "proportion_correct_edges": reward}


def punish_code_length(
    result: Set[Edge], vm_state: VMState, punish_cap: int = 24, **kwargs: Any
) -> Tuple[float, Dict[str, Any]]:
    code_length = len(vm_state.code)
    punish_score = (
        -(code_length - punish_cap) / (32 - punish_cap)  # HACK HACK HACK
        if code_length > punish_cap
        else 0.0
    )

    return punish_score, {
        "code_length": code_length,
        "code_length_punish_score": punish_score,
    }


def reward_finite_runtime(
    result: Set[Edge], vm_state: VMState, **kwargs: Any
) -> Tuple[float, Dict[str, Any]]:
    return -100.0 if vm_state.timeout else 0.0, {}


# Checks that the length of the returned spanning tree is n - 1
def reward_valid_spanning_tree_length(
    result: Set[Edge], vm_state: VMState, **kwargs: Any
) -> Tuple[float, Dict[str, Any]]:
    squared_dist = (len(result) - len(vm_state.input.nodes)) ** 2
    return squared_dist, {}


# Checks that the returned edge set is a spanning tree (i.e. connected, no cycles and it spans all nodes)
def reward_connected(
    result: Set[Edge], vm_state: VMState, **kwargs: Any
) -> Tuple[float, Dict[str, Any]]:
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

    return len(connected_nodes) / len(vm_state.input.nodes), {}


def reward_no_cycles(
    result: Set[Edge], vm_state: VMState, **kwargs: Any
) -> Tuple[float, Dict[str, Any]]:
    # TODO(mehdi): Check if the graph has cycles?
    # TODO(mehdi): can we do this more finegraned? i.e. number of cycles, distance between cycles, size of cycles?
    return 0.0, {}


def reward_covered_nodes(
    result: Set[Edge], vm_state: VMState, factor: float = 100, **kwargs: Any
) -> Tuple[float, Dict[str, Any]]:
    count = len(set([edge.u for edge in result] + [edge.v for edge in result]))
    return (count / len(vm_state.input.nodes)) * factor, {}


def reward_minimality(
    result: Set[Edge], vm_state: VMState, **kwargs: Any
) -> Tuple[float, Dict[str, Any]]:
    max_weight = max(vm_state.input.edges, key=lambda x: x.weight).weight
    return sum(max_weight - e.weight for e in result), {}


def reward_distance_to_MST(
    result: Set[Edge], vm_state: VMState, **kwargs: Any
) -> Tuple[float, Dict[str, Any]]:
    mst = compute_mst(vm_state.input)
    # count differences between the two sets
    diff = mst.symmetric_difference(result)

    return -len(diff), {}


def reward_efficiency(
    result: Set[Edge], vm_state: VMState, **kwargs: Any
) -> Tuple[float, Dict[str, Any]]:
    # TODO(mehdi): Implement this properly i.e. number of instructions that were actually executed. Probably have to count this in the VM and pass the VM into this.
    return -len(vm_state.code), {}


def reward_naive(
    result: Set[Edge], vm_state: VMState, **kwargs: Any
) -> Tuple[float, Dict[str, Any]]:
    """Naive reward function that only checks if the Set contains the minimal edges"""
    if len(result) == 0:
        return 0.0, {}
    minimal_edge = min(result, key=lambda x: x.weight)
    return (1.0 if minimal_edge in result else 0.0) + 0.5 * (
        1 - len(result) / len(vm_state.input.edges)
    ), {}
