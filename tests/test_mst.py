import pytest

from environment.algorithms import compute_mst
from environment.commands import (
    ADD_EDGE_TO_SET,
    COMPUTE_MST,
    IF_EDGE_SET_CAPACITY_REMAINING,
    IF_EDGE_STACK_REMAINING,
    IF_EDGE_WEIGHT_LT,
    JUMP,
    POP_EDGE,
    POP_MARK,
    PUSH_LEGAL_EDGES,
    PUSH_MARK,
    RESET_EDGE_REGISTER,
    RET,
    WRITE_EDGE_REGISTER,
)
from environment.environment import VirtualMachine
from environment.generation import generate_graph


def test_compute_mst() -> None:
    # Generate a test graph
    n = 8
    m = 20
    test_graph = generate_graph(n, m, seed=42)
    mst = compute_mst(test_graph)

    # Check if the result is the correct MST
    assert len(mst) == n - 1, f"Expected MST of size {n - 1}, but got {len(mst)}"
    expected_mst = {
        test_graph.edges[0],
        test_graph.edges[2],
        test_graph.edges[3],
        test_graph.edges[8],
        test_graph.edges[9],
        test_graph.edges[12],
        test_graph.edges[13],
    }
    assert expected_mst == mst, f"Expected MST {expected_mst}, but got {mst}"


@pytest.mark.parametrize("n_nodes, m_edges", [(10, 45), (20, 150), (30, 417)])
def test_COMPUTE_MST_instruction(n_nodes: int, m_edges: int) -> None:
    # Generate a test graph
    test_graph = generate_graph(n_nodes, m_edges, seed=42)

    # Define the Prim's algorithm instructions
    code = [
        COMPUTE_MST,
        RET,
    ]

    # Create a virtual machine and run the code
    vm = VirtualMachine(code, test_graph, verbose=False)
    result, vm_state, code_state = vm.run()
    infinite = code_state.timeout

    # Check if the result is not infinite
    assert not infinite, "Max instructions reached."

    # Check if the result is a valid MST
    mst_weight = sum(edge.weight for edge in result)
    expected_mst_weight = sum(edge.weight for edge in compute_mst(test_graph))
    assert (
        mst_weight == expected_mst_weight
    ), f"Expected MST weight {expected_mst_weight}, but got {mst_weight}"


@pytest.mark.parametrize("n_nodes, m_edges", [(10, 45), (20, 150), (30, 417)])
def test_simple_prims_algorithm(n_nodes: int, m_edges: int) -> None:
    # Generate a test graph
    test_graph = generate_graph(n_nodes, m_edges)

    # Define the Prim's algorithm instructions
    code = [
        PUSH_MARK,  # LOOP START
        PUSH_LEGAL_EDGES,  # push stack of edges that are allowed to be added
        RESET_EDGE_REGISTER,
        PUSH_MARK,  # INNER LOOP START
        IF_EDGE_WEIGHT_LT,  # if top of edge stack is greater than edge register
        WRITE_EDGE_REGISTER,  # update edge register to edge on top of stack
        POP_EDGE,  # pop edge from edge stack
        IF_EDGE_STACK_REMAINING,  # INNER LOOP END CONDITION: if edge stack is empty
        JUMP,  # to INNER LOOP START
        ADD_EDGE_TO_SET,  # final command before inner loop ends: add the edge from our edge register to the set
        POP_MARK,  # INNER LOOP END
        IF_EDGE_SET_CAPACITY_REMAINING,  # LOOP END CONDITION: if n - 1 edges have been marked
        JUMP,  # to LOOP START
        POP_MARK,  # LOOP END
        RET,
    ]

    # Create a virtual machine and run the code
    vm = VirtualMachine(code, test_graph, verbose=False)
    result, vm_state, code_state = vm.run()
    infinite = code_state.timeout

    # Check if the result is not infinite
    assert not infinite, "Max instructions reached."

    # Check if the result is a valid MST
    mst_weight = sum(edge.weight for edge in result)
    expected_mst_weight = sum(edge.weight for edge in compute_mst(test_graph))
    assert (
        mst_weight == expected_mst_weight
    ), f"Expected MST weight {expected_mst_weight}, but got {mst_weight}"
