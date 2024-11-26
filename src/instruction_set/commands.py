from abc import ABC, abstractmethod

from .structure_elements import NodeEdgePointer
from .vm_state import State

import heapq

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


class IF_EDGE_WEIGHT_GT(AbstractCommand):
    def execute(self, state: State) -> None:
        if state.value_register >= state.stack[-1].edge.weight:
            state.pc += 1

    def is_applicable(self, state: State) -> bool:
        return state.value_register != -1 and len(state.stack) > 0

    def is_comparison(self) -> bool:
        return True

    def __str__(self) -> str:
        return "IF_EDGE_WEIGHT_GT"


# class IF_EQ(AbstractCommand):
