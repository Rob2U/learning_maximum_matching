from typing import List

from .commands import (
    ADD_TO_OUT,
    ADD_TO_SET,
    IF_EDGE_WEIGHT_GT,
    IF_IN_SET,
    IF_IS_NOT_FIRST_EDGE,
    IF_IS_NOT_FIRST_NODE,
    JUMP,
    NEXT_EDGE,
    NEXT_NODE,
    NOP,
    POP_MARK,
    POP_NODE,
    PUSH_CLONE_NODE,
    PUSH_MARK,
    PUSH_START_NODE,
    RESET_EDGE_WEIGHT,
    RET,
    TO_NEIGHBOR,
    WRITE_EDGE_WEIGHT,
    AbstractCommand,
)
from .generation import generate_graph
from .structure_elements import Graph
from .vm_state import State


class VirtualMachine:
    """Simple stack-based virtual machine for graph traversal


    See instruction_set.md for more info.
    """

    def __init__(self, code: List[AbstractCommand], input: Graph):
        self.code = code
        self.state = State(input)

    def run(self) -> int:
        while self.state.pc < len(self.code):
            op = self.code[self.state.pc]()
            op.execute(self.state)
            print(op)

            self.state.pc += 1
            if self.state.early_ret:
                break

        return int(self.state.ret_register)


COMMAND_REGISTRY = [
    NOP,
    RET,
    PUSH_MARK,
    JUMP,
    POP_MARK,
    PUSH_START_NODE,
    PUSH_CLONE_NODE,
    POP_NODE,
    ADD_TO_SET,
    IF_IN_SET,
    NEXT_NODE,
    NEXT_EDGE,
    TO_NEIGHBOR,
    IF_IS_NOT_FIRST_EDGE,
    IF_IS_NOT_FIRST_NODE,
    WRITE_EDGE_WEIGHT,
    RESET_EDGE_WEIGHT,
    ADD_TO_OUT,
    IF_EDGE_WEIGHT_GT,
]


class Transpiler:
    """Translates a list of integers to a list of AbstractCommands and back"""

    def commandToInt(self, code: List[AbstractCommand]) -> List[int]:
        return [COMMAND_REGISTRY.index(op) for op in code]

    def intToCommand(self, code: List[int]) -> List[AbstractCommand]:
        return [COMMAND_REGISTRY[op] for op in code]


if __name__ == "__main__":
    # Lets write PRIM algorithm in our instruction set
    test_graph = generate_graph(50, 200)

    code = [
        PUSH_START_NODE,
        ADD_TO_SET,
        PUSH_MARK,  # LOOP START
        NEXT_NODE,
        IF_IS_NOT_FIRST_NODE,  # LOOP END CONDITION
            JUMP,
        POP_MARK,  # LOOP END
        RET,
    ]
    
    vm = VirtualMachine(code, test_graph)
    print(vm.run())
