import random
from typing import Any, Dict, List, Set, Tuple, Type

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from .code_state import CodeState
from .commands import (
    ADD_EDGE_TO_SET,
    ADD_TO_OUT,
    IF_EDGE_SET_CAPACITY_REMAINING,
    IF_EDGE_STACK_REMAINING,
    IF_EDGE_WEIGHT_LT,
    IF_HEAP_EMPTY,
    IF_IS_NOT_FIRST_EDGE,
    IF_IS_NOT_FIRST_NODE,
    JUMP,
    NEXT_EDGE,
    NEXT_NODE,
    NOP,
    POP_EDGE,
    POP_HEAP,
    POP_MARK,
    POP_NODE,
    PUSH_CLONE_NODE,
    PUSH_HEAP,
    PUSH_LEGAL_EDGES,
    PUSH_MARK,
    PUSH_START_NODE,
    REMOVE_EDGE_FROM_SET,
    RESET_EDGE_REGISTER,
    RESET_EDGE_WEIGHT,
    RET,
    TO_NEIGHBOR,
    WRITE_EDGE_REGISTER,
    WRITE_EDGE_WEIGHT,
    AbstractCommand,
)
from .feedback import reward
from .generation import generate_graph
from .structure_elements import Edge, Graph
from .vm_state import VMState


class MSTCodeEnvironment(gym.Env[npt.ArrayLike, int]):  # TODO(rob2u)
    def __init__(self) -> None:
        super().__init__()
        self.max_code_length: int = 128
        self.vm: VirtualMachine

        # Attributes used by the gymnasium library
        self.action_space = gym.spaces.Discrete(  # type: ignore
            len(COMMAND_REGISTRY)
        )  # action_space contains all the possible instructions for the next line
        self.observation_space = gym.spaces.MultiDiscrete(
            [len(COMMAND_REGISTRY) + 1] * self.max_code_length
        )

        self.max_n = 10
        self.max_m = 30
        self.reset()

    def step(
        self, action: int
    ) -> Tuple[npt.ArrayLike, float, bool, bool, Dict[str, Any]]:
        """Execute the action and return the new state, reward, and whether the episode is terminated and truncated"""
        instruction = Transpiler.intToCommand([action])[0]
        truncated = not self.vm.append_instruction(instruction)
        result, vm_state, code_state = self.vm.run()
        _reward = reward(result, vm_state, code_state)

        # NOTE(rob2u): might be worth trying to parse the entire return state of the VM + code
        observation: npt.NDArray[int, 1] = np.array(Transpiler.commandToInt(code_state.code))  # type: ignore
        observation = np.concatenate(
            [observation, (self.max_code_length - observation.shape[0]) * [0]]
        )
        assert observation.shape[0] == self.max_code_length, "Bad observation shape!"

        return (
            observation,
            _reward,
            code_state.timeout or code_state.finished,
            truncated,
            {},
        )

    def reset(
        self,
        code: List[Type[AbstractCommand]] | None = None,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[npt.ArrayLike, Dict[str, Any]]:

        # NOTE(rob2u): we use standard library random in our Graph generation and here so we use np_random for adaptability
        random.seed(seed)

        n = random.randint(2, self.max_n)
        m = random.randint(n - 1, min(n * (n - 1) // 2, self.max_m))

        graph = generate_graph(n, m, seed=None)
        self.vm = VirtualMachine([], graph, max_code_len=self.max_code_length)

        if code is not None:
            self.vm.code_state.code = code

        # NOTE(rob2u): might be worth trying to parse the entire state of the VM (as above)
        return np.array([0] * self.max_code_length), {}

    def close(self) -> None:
        pass


class VirtualMachine:
    """Simple stack-based virtual machine for graph traversal


    See instruction_set.md for more info.
    """

    def __init__(
        self,
        code: List[Type[AbstractCommand]],
        input: Graph,
        max_code_len: int = 1000,
        verbose: bool = False,
    ):
        self.vm_state = VMState(input)
        self.code_state = CodeState(code)

        self.max_runtime = 100 + len(input.edges) * len(input.nodes) ** 2
        self.max_code_len = max_code_len
        self.verbose = verbose

    def run(self) -> Tuple[Set[Edge], VMState, CodeState]:
        while self.vm_state.pc < len(self.code_state.code):
            op = self.code_state.code[self.vm_state.pc]()  # type: ignore
            self.log(op)
            op.execute(self.vm_state)

            self.vm_state.pc += 1
            if self.vm_state.early_ret:
                self.code_state.finished = True
                break

            self.code_state.runtime_steps += 1
            if self.code_state.runtime_steps > self.max_runtime:
                break

        return (
            self.vm_state.edge_set,
            self.vm_state,
            self.code_state,
        )

    def append_instruction(self, instruction: Type[AbstractCommand]) -> bool:
        """Add an instruction to the code if it is not too long. If it is too long, return False"""
        if len(self.code_state.code) < self.max_code_len:
            self.code_state.code.append(instruction)
            return True
        else:
            return False

    def log(self, item: Any) -> None:
        if self.verbose:
            print(item)


COMMAND_REGISTRY: List[Type[AbstractCommand]] = [
    NOP,
    RET,
    PUSH_MARK,
    JUMP,
    POP_MARK,
    PUSH_START_NODE,
    PUSH_CLONE_NODE,
    POP_NODE,
    PUSH_HEAP,
    POP_HEAP,
    IF_HEAP_EMPTY,
    NEXT_NODE,
    NEXT_EDGE,
    TO_NEIGHBOR,
    IF_IS_NOT_FIRST_EDGE,
    IF_IS_NOT_FIRST_NODE,
    WRITE_EDGE_WEIGHT,
    RESET_EDGE_WEIGHT,
    ADD_TO_OUT,
    IF_EDGE_WEIGHT_LT,
    WRITE_EDGE_REGISTER,
    RESET_EDGE_REGISTER,
    POP_EDGE,
    IF_EDGE_STACK_REMAINING,
    PUSH_LEGAL_EDGES,
    ADD_EDGE_TO_SET,
    REMOVE_EDGE_FROM_SET,
    IF_EDGE_SET_CAPACITY_REMAINING,
]


class Transpiler:
    """Translates a list of integers to a list of AbstractCommands and back"""

    @staticmethod
    def commandToInt(code: List[Type[AbstractCommand]]) -> List[int]:
        return [COMMAND_REGISTRY.index(op) for op in code]

    @staticmethod
    def intToCommand(code: List[int]) -> List[Type[AbstractCommand]]:
        # assert all([i >= 1 for i in code]), "Bad command encoding encountered!"
        return [COMMAND_REGISTRY[op] for op in code]


if __name__ == "__main__":
    # Lets write PRIM algorithm in our instruction set
    test_graph = generate_graph(10, 45)
    print("##################")
    print("Graph:")
    print(test_graph)

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
        RET,  # end of program (indicate that we do not need to generate more code)
    ]

    print("##################")
    print("Code:")
    print("\n".join([f"{i + 1}: {op.__name__}" for i, op in enumerate(code)]))

    print("##################")
    print("Stack Trace:")

    vm = VirtualMachine(code, test_graph, verbose=True)

    print("##################")
    print("Graph:")
    print(test_graph)

    print("##################")
    result, vm_state, code_state = vm.run()
    print("##################")

    if not code_state.finished:
        print("Program did not finish orderly!")
    else:
        print("Result:")
        print(result)

        print("##################")
        print("Actual MST:")
        print(vm.vm_state.edge_set)
