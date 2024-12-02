import heapq
from abc import ABC, abstractmethod

from .algorithms import UnionFind, compute_mst
from .structure_elements import NodeEdgePointer
from .vm_state import State

############### ABSTRACT COMMANDS ####################


class AbstractCommand(ABC):
    @abstractmethod
    def execute(self, state: State) -> None:
        pass

    @abstractmethod
    def is_applicable(self, state: State) -> bool:
        pass

    @abstractmethod  # NOTE(rob2u): not sure if necessary
    def is_comparison(self) -> bool:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class ConditionalCommand(AbstractCommand):
    """Abstract class for commands that are conditionals."""

    def is_applicable(self, state: State) -> bool:
        return True

    def is_comparison(self) -> bool:
        return True

    def execute(self, state: State) -> None:
        if not self.condition(state):
            state.pc += 1

    @abstractmethod
    def condition(self, state: State) -> bool:
        """Returns True if the next command should be executed."""
        pass


############### GENERAL COMMANDS ####################


class NOP(AbstractCommand):
    def execute(self, state: State) -> None:
        pass

    def is_applicable(self, state: State) -> bool:
        return True

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "NOP"


class RET(AbstractCommand):
    def execute(self, state: State) -> None:
        state.early_ret = True

    def is_applicable(self, state: State) -> bool:
        return True

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "RET"


############### MARKS + JUMPS FOR LOOPS ####################
class PUSH_MARK(AbstractCommand):
    """Adds a code marker at the position of the current pc. Using JUMP we can loop back to this position later."""

    def execute(self, state: State) -> None:
        state.mark_stack.append(state.pc - 1)

    def is_applicable(self, state: State) -> bool:
        return True

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "PUSH_MARK"


class POP_MARK(AbstractCommand):
    def execute(self, state: State) -> None:
        if state.mark_stack:
            state.mark_stack.pop()

    def is_applicable(self, state: State) -> bool:
        return True

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "POP_MARK"


class JUMP(AbstractCommand):
    def execute(self, state: State) -> None:
        if state.mark_stack:
            state.pc = state.mark_stack.pop()

    def is_applicable(self, state: State) -> bool:
        return True

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "JUMP"


################# EDGE REGISTER COMMANDS ####################


class WRITE_EDGE_REGISTER(AbstractCommand):
    def execute(self, state: State) -> None:
        state.edge_register = state.edge_stack[-1]

    def is_applicable(self, state: State) -> bool:
        return len(state.stack) > 0

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "WRITE_EDGE_REGISTER"


class RESET_EDGE_REGISTER(AbstractCommand):
    def execute(self, state: State) -> None:
        state.edge_register = None

    def is_applicable(self, state: State) -> bool:
        return True

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "RESET_EDGE_REGISTER"


################### EDGE SET COMMANDS ####################


class ADD_EDGE_TO_SET(AbstractCommand):
    def execute(self, state: State) -> None:
        if state.edge_register:
            state.edge_set.add(state.edge_register)

    def is_applicable(self, state: State) -> bool:
        return state.edge_register is not None

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "ADD_EDGE_TO_SET"


class REMOVE_EDGE_FROM_SET(AbstractCommand):
    def execute(self, state: State) -> None:
        if state.edge_register:
            state.edge_set.remove(state.edge_register)

    def is_applicable(self, state: State) -> bool:
        return state.edge_register is not None

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "REMOVE_EDGE_FROM_SET"


################### EDGE STACK COMMANDS ####################


class PUSH_EDGE(AbstractCommand):
    def execute(self, state: State) -> None:
        if state.edge_register:
            state.edge_stack.append(state.edge_register)

    def is_applicable(self, state: State) -> bool:
        return True  # if edge_register is None, we do nothing.

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "PUSH_EDGE"


class POP_EDGE(AbstractCommand):
    def execute(self, state: State) -> None:
        if len(state.edge_stack) > 0:
            state.edge_stack.pop()

    def is_applicable(self, state: State) -> bool:
        # if the edge_stack is already empty we do nothing. Therefore, always applicable.
        return True

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "POP_EDGE"


class IF_EDGE_STACK_REMAINING(ConditionalCommand):
    """Checks if there are any edges left on the edge_stack. If so, the next command is executed."""

    def condition(self, state: State) -> bool:
        return len(state.edge_stack) > 0

    def __str__(self) -> str:
        return "IF_EDGE_STACK_REMAINING"


class IF_EDGE_WEIGHT_LT(ConditionalCommand):
    """Compares the weight of the edge on the stack with the value register. If the weight is less than the value register, the next command is executed."""

    def condition(self, state: State) -> bool:
        return not state.edge_register or (
            len(state.edge_stack) > 0
            and state.edge_stack[-1].weight < state.edge_register.weight
        )

    def __str__(self) -> str:
        return "IF_EDGE_WEIGHT_LT"


################### VALUE AND RETURN REGISTER COMMANDS ####################


class WRITE_EDGE_WEIGHT(AbstractCommand):
    def execute(self, state: State) -> None:
        if state.stack:
            state.value_register = state.stack[-1].edge.weight

    def is_applicable(self, state: State) -> bool:
        return len(state.stack) > 0

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "WRITE_EDGE"


class RESET_EDGE_WEIGHT(AbstractCommand):
    def execute(self, state: State) -> None:
        state.value_register = -1

    def is_applicable(self, state: State) -> bool:
        return True

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "RESET_EDGE_WEIGHT"


class ADD_TO_OUT(AbstractCommand):
    """Adds the value register to the return register. The return register is probably not needed anymore if the edge_set is used as the return value."""

    def execute(self, state: State) -> None:
        if state.value_register != -1:
            state.ret_register += state.value_register

    def is_applicable(self, state: State) -> bool:
        return bool(state.value_register != -1)

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "ADD_TO_OUT"


################### NODE_EDGE STACK COMMANDS ####################
# Currently not needed but could be useful for future extensions


class PUSH_START_NODE(AbstractCommand):
    def execute(self, state: State) -> None:
        first_node = state.input.first_node()
        state.stack.append(
            NodeEdgePointer(first_node, state.input.first_edge(first_node))
        )

    def is_applicable(self, state: State) -> bool:
        return True

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "PUSH_START_NODE"


class PUSH_CLONE_NODE(AbstractCommand):
    def execute(self, state: State) -> None:
        if state.stack:
            clone_node = state.stack[-1].node
            state.stack.append(
                NodeEdgePointer(clone_node, state.input.first_edge(clone_node))
            )

    def is_applicable(self, state: State) -> bool:
        # NOTE(rob2u): we replace PUSH_START_NODE by using this but putting the start_node on the stack if empty
        return len(state.stack) > 0

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "PUSH_CLONE_NODE"


class POP_NODE(AbstractCommand):
    def execute(self, state: State) -> None:
        if state.stack:
            state.stack.pop()

    def is_applicable(self, state: State) -> bool:
        return len(state.stack) > 0

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "POP_NODE"


class NEXT_NODE(AbstractCommand):
    def execute(self, state: State) -> None:
        if state.stack:
            last_node = state.stack[-1].node
            next_node = state.input.next_node(last_node)
            state.stack[-1].node = next_node
            state.stack[-1].edge = state.input.first_edge(next_node)

    def is_applicable(self, state: State) -> bool:
        return len(state.stack) > 0

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "NEXT_NODE"


class NEXT_EDGE(AbstractCommand):
    def execute(self, state: State) -> None:
        if state.stack:
            state.stack[-1].edge = state.input.next_edge(
                state.stack[-1].node, state.stack[-1].edge
            )

    def is_applicable(self, state: State) -> bool:
        return len(state.stack) > 0

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "NEXT_EDGE"


class TO_NEIGHBOR(AbstractCommand):
    def execute(self, state: State) -> None:
        if state.stack:
            last_node = state.stack[-1].node
            last_edge = state.stack[-1].edge
            state.stack[-1].node = (
                last_edge.v if last_edge.u == last_node else last_edge.u
            )
            state.stack[-1].edge = last_edge

    def is_applicable(self, state: State) -> bool:
        return len(state.stack) > 0

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "TO_NEIGHBOR"


class IF_IS_NOT_FIRST_NODE(AbstractCommand):
    def execute(self, state: State) -> None:
        if state.stack:
            last_node = state.stack[-1].node
            if last_node == state.input.first_node():
                state.pc += 1

    def is_applicable(self, state: State) -> bool:
        return len(state.stack) > 0

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "IF_IS_NOT_FIRST_NODE"


class IF_IS_NOT_FIRST_EDGE(AbstractCommand):
    def execute(self, state: State) -> None:
        if state.stack:
            last_node = state.stack[-1].node
            last_edge = state.stack[-1].edge
            if last_edge == state.input.first_edge(last_node):
                state.pc += 1

    def is_applicable(self, state: State) -> bool:
        return len(state.stack) > 0

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "IF_IS_NOT_FIRST_EDGE"


################## NODE EDGE HEAP COMMANDS ####################


class PUSH_HEAP(AbstractCommand):
    def execute(self, state: State) -> None:
        if state.stack:
            heapq.heappush(state.heap, state.stack[-1])

    def is_applicable(self, state: State) -> bool:
        return len(state.stack) > 0

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "PUSH_HEAP"


class POP_HEAP(AbstractCommand):
    def execute(self, state: State) -> None:
        if len(state.heap) > 0:
            # TODO(philippkolbe): we have to decide where we want to pop the node to
            state.value_register = heapq.heappop(state.heap).node

    def is_applicable(self, state: State) -> bool:
        return len(state.heap) == 0

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "POP_HEAP"


class IF_HEAP_EMPTY(ConditionalCommand):
    def condition(self, state: State) -> bool:
        return len(state.heap) == 0

    def __str__(self) -> str:
        return "IF_HEAP_EMPTY"


################ CHEAT COMMANDS FOR MST ################


class IF_EDGE_SET_CAPACITY_REMAINING(ConditionalCommand):
    """This is also kind of cheating as it computes the end condition for MST directly."""

    def condition(self, state: State) -> bool:
        return len(state.edge_set) < len(state.input.nodes) - 1

    def __str__(self) -> str:
        return "IF_EDGE_SET_CAPACITY_REMAINING"


class PUSH_LEGAL_EDGES(AbstractCommand):
    """Note: This is a cheat command as it solves a part of the problem directly.
    Pushes all edges to the edge_stack that would be valid to add to a MST.
    We view the edges in the edge_set as the current MST.
    Initially (i.e. edge_set is empty), all edges are legal.
    Otherwise, all edges that are not yet part of the MST and do not create a cycle are legal.
    """

    def execute(self, state: State) -> None:
        if len(state.edge_set) == 0:
            # initially all edges are legal
            state.edge_stack += state.input.edges
        else:
            # its quite inefficient to always rebuild the union find when this command is called. Instead: union find could be updated every time edge_set is updated.
            uf = UnionFind(len(state.input.nodes))
            for e in state.edge_set:
                uf.union(e.u, e.v)

            for edge in state.input.edges:
                # legal edges are  all edges that are not already in the edge set and do not create a cycle (checked with union find data structure)
                if edge not in state.edge_set and not uf.connected(edge.u, edge.v):
                    state.edge_stack.append(edge)

    def is_applicable(self, state: State) -> bool:
        return True

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "PUSH_LEGAL_EDGES"


class COMPUTE_MST(AbstractCommand):
    """Note: This is a cheat command which returns the command directly. Can be used for testing."""

    def execute(self, state: State) -> None:
        state.edge_set = compute_mst(state.input)

    def is_applicable(self, state: State) -> bool:
        return True

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "COMPUTE_MST"
