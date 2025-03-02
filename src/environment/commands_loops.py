import heapq
from abc import abstractmethod
from typing import List, Type

from .algorithms import UnionFind, compute_mst
from .masking_utils import (
    are_any_of_last_n_commands_different_to_all,
    does_any_command_exist,
    does_command_exist,
    is_last_command_different_to,
)
from .structure_elements import NodeEdgePointer
from .vm_state import AbstractCommand, VMState

############### ABSTRACT COMMANDS ####################


class ConditionalCommand(AbstractCommand):
    """Abstract class for commands that are conditionals."""

    def is_applicable(self, state: VMState) -> bool:
        # 2 if commands in a row make sense (its an implies operator) but 3 in a row is just weird.
        return are_any_of_last_n_commands_different_to_all(
            state.code, CONDITIONAL_COMMANDS, 3
        )

    def is_comparison(self) -> bool:
        return True

    def execute(self, state: VMState) -> None:
        if not self.condition(state):
            state.pc += 1

    @abstractmethod
    def condition(self, state: VMState) -> bool:
        """Returns True if the next command should be executed."""
        pass


############### GENERAL COMMANDS ####################


class NOP(AbstractCommand):
    def execute(self, state: VMState) -> None:
        pass

    def is_applicable(self, state: VMState) -> bool:
        # in our current instruction set there is no case where NOP will have any effect
        return False

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "NOP"


class RET(AbstractCommand):
    def execute(self, state: VMState) -> None:
        state.early_ret = True

    def is_applicable(self, state: VMState) -> bool:
        # Is only applicable if the position before the RET is not an IF command
        # find the last command that is not a NOP
        # if state.code and issubclass(ConditionalCommand, state.code[-1]):
        #     return False
        if state.code and isinstance(state.code[-1](), IF_EDGE_WEIGHT_LT):
            return False

        return True

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "RET"


############### MARKS + JUMPS FOR LOOPS ####################


# NOTE: !!!!!!!!!!!!!!!
# Different to original PUSH_MARK, only allowed once per program
class PUSH_MARK(AbstractCommand):
    """Adds a code marker at the position of the current pc. Using JUMP we can loop back to this position later."""

    def execute(self, state: VMState) -> None:
        state.mark_stack.append(state.pc - 1)

    def is_applicable(self, state: VMState) -> bool:
        # only allow PUSH_MARK once
        return not does_command_exist(state.code, PUSH_MARK)

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "PUSH_MARK"


# NOTE: !!!!!!!!!!!!!!!
# Designed to only be used once per program
class IF_EDGE_STACK_REMAINING_JUMP_ELSE_POP_MARK(AbstractCommand):
    def execute(self, state):
        if len(state.mark_stack) > 0:
            if len(state.edge_stack) > 0:
                print("jumping back to mark")
                state.pc = (
                    state.mark_stack.pop()
                )  # we jump before the PUSH_MARK which is therefore added to the mark_stack again
            else:
                # if there are no edges left we pop the mark_stack
                state.mark_stack.pop()

    def is_comparison(self):
        return False

    def is_applicable(self, state: VMState) -> bool:
        # Check that at least one:
        # - PUSH_LEGAL_EDGES is in the LOOP and not behind an IF (PUSH_LEGAL_EDGES not after IF)
        # - POP_AND_WRITE_EDGE_REGISTER is in the code and not behind an IF (only allow WRITE_EDGE_REGISTERS after IF)
        # - ADD_EDGE_TO_SET is in the code and not behind an IF (ADD_EDGE_TO_SET not after IF)
        # - move RESET_EDGE_REGISTER into ADD_EDGE_TO_SET
        # this ensures that the state will always change -> no endless loop possible

        if not does_command_exist(state.code, PUSH_MARK) or len(state.mark_stack) == 0:
            return False

        loop_code = state.code[state.mark_stack[-1] + 1 :]

        push_legal_edges_in_loop = does_any_command_exist(loop_code, [PUSH_LEGAL_EDGES])
        add_edge_to_set_in_loop = does_any_command_exist(
            loop_code, [ADD_EDGE_TO_SET_AND_RESET_REGISTER]
        )

        # Check if there is a POP_AND_WRITE_EDGE_REGISTER in the loop_code without an IF before
        write_edge_register_in_loop = loop_code[0] == POP_AND_WRITE_EDGE_REGISTER
        if not write_edge_register_in_loop:
            for i, command in enumerate(loop_code):
                if command == POP_AND_WRITE_EDGE_REGISTER and not isinstance(
                    loop_code[i - 1](), ConditionalCommand
                ):
                    write_edge_register_in_loop = True
                    break

        return (
            push_legal_edges_in_loop
            and add_edge_to_set_in_loop
            and write_edge_register_in_loop
        )

    def __str__(self) -> str:
        return "IF_EDGE_STACK_REMAINING_JUMP_ELSE_POP_MARK"


################# EDGE REGISTER COMMANDS ####################


# NOTE: !!!!!!!!!!!!!!!
# Different to original WRITE_EDGE_REGISTER, also POPs the edge_stack
class POP_AND_WRITE_EDGE_REGISTER(AbstractCommand):
    def execute(self, state: VMState) -> None:
        if state.edge_stack:
            state.edge_register = state.edge_stack.pop()

    def is_applicable(self, state: VMState) -> bool:
        # TODO(philipp): this is valid if we will push edges later and jump back to this command
        # TODO(philipp): the only way where it would be valid is if the last command was wrapped in a conditional but then the last action was invalid...
        return does_any_command_exist(
            state.code,
            PUSH_EDGE_COMMANDS,
        ) and is_last_command_different_to(state.code, POP_AND_WRITE_EDGE_REGISTER)

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "POP_AND_WRITE_EDGE_REGISTER"


################### EDGE SET COMMANDS ####################


# NOTE: !!!!!!!!!!!!!!!
# Different to original ADD_EDGE_TO_SET, also resets the edge register
# also not allowed directly after IF
class ADD_EDGE_TO_SET_AND_RESET_REGISTER(AbstractCommand):
    def execute(self, state: VMState) -> None:
        if state.edge_register:
            state.edge_set.add(state.edge_register)
            state.edge_register = None

    def is_applicable(self, state: VMState) -> bool:
        # TODO(philipp): this only makes sense if there was a WRITE_EDGE_REGISTER after the last ADD_EDGE_TO_SET
        # TODO(philipp): the only way where it would be valid is if the last command was wrapped in a conditional but then the last action was invalid...
        return (
            is_last_command_different_to(state.code, ADD_EDGE_TO_SET_AND_RESET_REGISTER)
            and does_command_exist(state.code, POP_AND_WRITE_EDGE_REGISTER)
            and is_last_command_different_to(state.code, IF_EDGE_WEIGHT_LT)
        )

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "ADD_EDGE_TO_SET_AND_RESET_REGISTER"


################### EDGE STACK COMMANDS ####################


class IF_EDGE_WEIGHT_LT(ConditionalCommand):
    """Compares the weight of the edge on the top of the stack with the value register.
    If the weight of the edge on top of the stack is less than the value register, the next command is executed.
    """

    def condition(self, state: VMState) -> bool:
        print(state.edge_stack[-1].weight, state.edge_register.weight)
        return not state.edge_register or (
            len(state.edge_stack) > 0
            and state.edge_stack[-1].weight < state.edge_register.weight
        )

    def is_applicable(self, state: VMState) -> bool:
        return (
            # super().is_applicable(state)
            is_last_command_different_to(state.code, IF_EDGE_WEIGHT_LT)
            # and does_command_exist(state.code, WRITE_EDGE_REGISTER)
            # and does_any_command_exist(state.code, PUSH_EDGE_COMMANDS)
        )

    def __str__(self) -> str:
        return "IF_EDGE_WEIGHT_LT"


################ "CHEAT" COMMANDS FOR MST ################
# ordered in ascending order by how much they "cheat" (i.e. how much of the problem they solve on a non-atomic-instruction level)


# NOTE: !!!!!!!!!!!!!!!
# Different to original PUSH_LEGAL_EDGES, not allowed directly after IF
# Also resets the EDGE STACK
class PUSH_LEGAL_EDGES(AbstractCommand):
    """Note: This is a cheat command as it solves a part of the problem directly.
    Pushes all edges to the edge_stack that would be valid to add to a MST.
    We view the edges in the edge_set as the current MST.
    Initially (i.e. edge_set is empty), all edges are legal.
    Otherwise, all edges that are not yet part of the MST and do not create a cycle are legal.
    """

    def execute(self, state: VMState) -> None:
        state.edge_stack = []
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

    def is_applicable(self, state: VMState) -> bool:
        # only allow PUSH_LEGAL_EDGES if there is no IF directly before
        return is_last_command_different_to(
            state.code, PUSH_LEGAL_EDGES
        ) and is_last_command_different_to(state.code, IF_EDGE_WEIGHT_LT)

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "PUSH_LEGAL_EDGES"


PUSH_EDGE_COMMANDS: List[Type[AbstractCommand]] = [PUSH_LEGAL_EDGES]
CONDITIONAL_COMMANDS: List[Type[AbstractCommand]] = [
    IF_EDGE_WEIGHT_LT,
]

if __name__ == "__main__":
    from .generation import generate_ring

    # This works for all graphs with n nodes and n edges
    # Compare two edges and add the one with the smaller weight to the edge_set. Do until no Legal edges are left.
    our_program = [
        PUSH_LEGAL_EDGES,
        PUSH_MARK,
        POP_AND_WRITE_EDGE_REGISTER,
        IF_EDGE_WEIGHT_LT,
        POP_AND_WRITE_EDGE_REGISTER,
        ADD_EDGE_TO_SET_AND_RESET_REGISTER,
        PUSH_LEGAL_EDGES,
        IF_EDGE_STACK_REMAINING_JUMP_ELSE_POP_MARK,
        RET,
    ]
    # test that the commands are applicable
    input_graph = generate_ring(4, 42)
    state = VMState(input=input_graph, code=[])

    program_counter = 0
    while state.pc < len(our_program):
        program_counter = state.pc
        command = our_program[program_counter]
        print(command(), ": \t", command().is_applicable(state))
        command().execute(state)
        print(state.edge_stack)
        print(state.edge_register)
        print(state.edge_set)
        print(state.mark_stack)

        state.pc += 1
        state.code = our_program[: state.pc]
    print(state.edge_set)
