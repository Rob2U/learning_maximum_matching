import pytest

from environment.commands import (
    ADD_EDGE_TO_SET,
    IF_EDGE_WEIGHT_LT,
    POP_EDGE,
    PUSH_LEGAL_EDGES,
    RET,
    WRITE_EDGE_REGISTER,
)
from environment.generation import generate_graph
from environment.vm_state import VMState


def test_simplest_algo_has_valid_masks() -> None:
    code = [
        # pushes all three edges to stack
        PUSH_LEGAL_EDGES,
        WRITE_EDGE_REGISTER,
        POP_EDGE,
        IF_EDGE_WEIGHT_LT,  # compares top of stack with second top of stack
        WRITE_EDGE_REGISTER,
        ADD_EDGE_TO_SET,
        # has added one of the two smaller edges to set
        # adds the two remaining edges to stack
        PUSH_LEGAL_EDGES,
        WRITE_EDGE_REGISTER,
        POP_EDGE,
        IF_EDGE_WEIGHT_LT,  # compares top of stack with second top of stack
        WRITE_EDGE_REGISTER,
        ADD_EDGE_TO_SET,
        # has added second of the two smaller edges to set
        RET,
    ]

    graph = generate_graph(3, 3)
    for i in range(len(code)):
        state = VMState(graph, code[:i])
        command = code[i]()
        assert command.is_applicable(
            state
        ), f"Command {command} is not applicable at step {i}"
        command.execute(state)
