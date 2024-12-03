from typing import List, Set

from .structure_elements import Edge, Graph, NodeEdgePointer


class VMState:
    """Current state of the virtual machine

    Attributes:
    pc: Program counter
    mark_stack: Stack of 'marks' for jumping (back)
    stack: Stack of NodeEdgePointers for graph traversal
    set: Set of nodes

    ret_register: Return register
    value_register: Value register
    early_ret: Flag for early return
    """

    input: Graph
    pc: int

    # state needed for simplest version
    edge_set: Set[Edge]
    edge_stack: List[Edge]
    edge_register: Edge | None

    # not needed for simplest version
    mark_stack: List[int]
    stack: List[NodeEdgePointer]
    set: Set[int]
    heap: List[NodeEdgePointer]
    ret_register: int
    value_register: int

    early_ret: bool
    execution_counter: int

    timeout: bool

    def __init__(self, input: Graph):
        self.input = input
        self.reset()

    def reset(self) -> None:
        self.pc = 0
        self.mark_stack = []
        self.stack = []
        self.set = set()
        self.heap = []
        self.edge_set = set()
        self.edge_stack = []

        self.ret_register = -1
        self.value_register = -1
        self.edge_register = None
        self.early_ret = False
