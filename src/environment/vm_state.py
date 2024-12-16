from abc import ABC, abstractmethod
from typing import List, Set

from .structure_elements import Edge, Graph, NodeEdgePointer


# unfortunately we need to define this here because of circular imports
class AbstractCommand(ABC):
    @abstractmethod
    def execute(self, state: "VMState") -> None:
        pass

    @abstractmethod
    def is_applicable(self, state: "VMState") -> bool:
        """Used for action masking: does executing the command have any effect?"""
        pass

    @abstractmethod
    def is_comparison(self) -> bool:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class VMState:
    """Current state of the virtual machine including the code state.

    Attributes:
    pc: Program counter
    mark_stack: Stack of 'marks' for jumping (back)
    stack: Stack of NodeEdgePointers for graph traversal
    set: Set of nodes

    ret_register: Return register
    value_register: Value register
    early_ret: Flag for early return

    code: List of commands
    finished: Flag for finished execution
    runtime_steps: Number of steps executed
    timeout: Flag for timeout execution
    truncated: Flag for truncated execution
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

    code: List[type[AbstractCommand]] = []
    runtime_steps: int = 0
    finished: bool = False
    timeout: bool = False
    truncated: bool = False

    def __init__(self, input: Graph, code: List[type[AbstractCommand]] = []) -> None:
        self.reset()

        self.input = input
        self.code = code

    def reset(self) -> None:
        self.pc = 0
        self.mark_stack = []
        self.stack = []
        self.set = set()
        self.heap = []
        self.edge_set = set()
        self.edge_stack = []

        self.code = []
        self.runtime_steps = 0
        self.finished = False
        self.timeout = False
        self.truncated = False

        self.ret_register = -1
        self.value_register = -1
        self.edge_register = None
        self.early_ret = False
