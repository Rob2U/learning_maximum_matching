from typing import List, Set

from instruction_set_rl.structure_elements import NodeEdgePointer, Graph


class State:
    """Current state of the virtual machine

    Attributes:
    pc: Program counter
    mark_stack: Stack of 'marks' for jumping (back)
    stack: Stack of NodeEdgePointers for graph traversal
    set: Set of nodes

    ret_register: Return register?
    value_register: Value register?
    early_ret: Flag for early return?    

    """
    input: Graph
    pc: int
    mark_stack: List[int]
    stack: List[NodeEdgePointer]
    set: Set[int]

    ret_register: int
    value_register: int
    early_ret: bool
    
    def __init__(self, input: Graph):
        self.input = input
        self.reset()
    
    def reset(self) -> None:
        self.pc: int = 0
        self.mark_stack: List[int] = []
        self.stack: List[NodeEdgePointer] = []
        self.set: Set[int] = set()

        self.ret_register: int = -1
        self.value_register: int = -1
        self.early_ret: bool = False
    
