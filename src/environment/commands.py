import heapq
from abc import ABC, abstractmethod

from .algorithms import UnionFind, compute_mst
from .structure_elements import NodeEdgePointer
from .vm_state import State


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


class PUSH_MARK(AbstractCommand):
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


class PUSH_LEGAL_EDGES(AbstractCommand):
    """Pushes all edges to the edge_stack that would be valid to add to a MST. 
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


class PUSH_EDGE(AbstractCommand):
    def execute(self, state: State) -> None:
        if state.edge_register:
            state.edge_stack.append(state.edge_register)

    def is_applicable(self, state: State) -> bool:
        return True # if edge_register is None, we do nothing.

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


class IF_EDGE_STACK_REMAINING(AbstractCommand):
    def execute(self, state: State) -> None:
        if len(state.edge_stack) == 0:
            state.pc += 1

    def is_applicable(self, state: State) -> bool:
        return True

    def is_comparison(self) -> bool:
        return True

    def __str__(self) -> str:
        return "IF_EDGE_STACK_REMAINING"


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


class IF_EDGE_SET_CAPACITY_REMAINING(AbstractCommand):
    def execute(self, state: State) -> None:
        if not (len(state.edge_set) < len(state.input.nodes) - 1):
            state.pc += 1

    def is_applicable(self, state: State) -> bool:
        return True

    def is_comparison(self) -> bool:
        return True

    def __str__(self) -> str:
        return "IF_EDGE_SET_CAPACITY_REMAINING"


class ADD_TO_SET(AbstractCommand):
    def execute(self, state: State) -> None:
        if state.stack:
            state.set.add(state.stack[-1].node)

    def is_applicable(self, state: State) -> bool:
        return len(state.stack) > 0

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "ADD_TO_SET"


class IF_IN_SET(AbstractCommand):  # NOTE: could be unnecessary
    def execute(self, state: State) -> None:
        if state.stack:
            if state.stack[-1].node not in state.set:
                state.pc += 1

    def is_applicable(self, state: State) -> bool:
        return len(state.stack) > 0

    def is_comparison(self) -> bool:
        return True

    def __str__(self) -> str:
        return "IF_IN_SET"


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


class IF_HEAP_EMPTY(AbstractCommand):
    def execute(self, state: State) -> None:
        if len(state.heap) > 0:
            state.pc += 1

    def is_applicable(self, state: State) -> bool:
        return True

    def is_comparison(self) -> bool:
        return True

    def __str__(self) -> str:
        return "POP_HEAP"


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
    def execute(self, state: State) -> None:
        if state.value_register != -1:
            state.ret_register += state.value_register

    def is_applicable(self, state: State) -> bool:
        return bool(state.value_register != -1)

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "ADD_TO_OUT"


class IF_EDGE_WEIGHT_LT(AbstractCommand):
    def execute(self, state: State) -> None:
        if len(state.edge_stack) > 0 and state.edge_register and state.edge_register.weight >= state.edge_stack[-1].weight:
            state.pc += 1

    def is_applicable(self, state: State) -> bool:
        return state.value_register != -1 and len(state.stack) > 0

    def is_comparison(self) -> bool:
        return True

    def __str__(self) -> str:
        return "IF_EDGE_WEIGHT_GT"


# This is a cheat command. can be used for testing.
class COMPUTE_MST(AbstractCommand):
    def execute(self, state: State) -> None:
        state.edge_set = compute_mst(state.input)
    
    def is_applicable(self, state: State) -> bool:
        return True

    def is_comparison(self) -> bool:
        return False
    
    def __str__(self) -> str:
        return "COMPUTE_MST"


# class IF_EQ(AbstractCommand):
