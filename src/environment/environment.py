import random
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from .commands import (
    ADD_TO_OUT,
    ADD_TO_SET,
    IF_EDGE_WEIGHT_GT,
    IF_HEAP_EMPTY,
    IF_IN_SET,
    IF_IS_NOT_FIRST_EDGE,
    IF_IS_NOT_FIRST_NODE,
    JUMP,
    NEXT_EDGE,
    NEXT_NODE,
    NOP,
    POP_HEAP,
    POP_MARK,
    POP_NODE,
    PUSH_CLONE_NODE,
    PUSH_HEAP,
    PUSH_MARK,
    PUSH_START_NODE,
    RESET_EDGE_WEIGHT,
    RET,
    TO_NEIGHBOR,
    WRITE_EDGE_WEIGHT,
    AbstractCommand,
)
from .feedback import reward_naive
from .generation import generate_graph
from .structure_elements import Graph
from .vm_state import State


class MSTCodeEnvironment(gym.Env[npt.ArrayLike, int]):  # TODO(rob2u)
    def __init__(self) -> None:
        super().__init__()
        self.max_code_length: int = 128
        self.vm: VirtualMachine
        # Attributes used by the gymnasium library
        self.action_space = gym.spaces.Discrete(  # type: ignore
            len(COMMAND_REGISTRY)
        )  # action_space contains all the possible instructions for the next line
        # self.observation_space = gym.spaces.Sequence(gym.spaces.Discrete(len(COMMAND_REGISTRY)))  # type: ignore
        self.observation_space = gym.spaces.MultiDiscrete([len(COMMAND_REGISTRY) + 1] * self.max_code_length)
        # observation_space contains the current state of the VM (as we have full control over our VM)
        # self.metadata = {
        #     "render.modes": [],
        #     "torch": True,
        # }  # metadata contains the render modes
        # self.spec = None # spec contains the specification of the environment

        self.max_n = 50
        self.max_m = 100
        self.reset()

    def step(
        self, action: int
    ) -> Tuple[npt.ArrayLike, float, bool, bool, Dict[str, Any]]:
        """Execute the action and return the new state, reward, and whether the episode is terminated and truncated"""
        instruction = Transpiler.intToCommand([action + 1])[0]
        truncated = not self.vm.append_instruction(instruction)
        result, terminated = self.vm.run()
        reward = reward_naive(self.vm.state, self.vm.code, result)

        # NOTE(rob2u): might be worth trying to parse the entire return state of the VM + code
        observation = np.array(Transpiler.commandToInt(self.vm.code))
        observation = np.concatenate([observation, (self.max_code_length  - len(observation) )* [0]])
        assert observation.shape[0] == self.max_code_length, "Bad observation shape!"
        
        return observation, reward, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[npt.ArrayLike, Dict[str, Any]]:

        # NOTE(rob2u): we use standard library random in our Graph generation and here so we use np_random for adaptability
        if seed is not None:
            self._np_random = np.random.RandomState(seed)  # type: ignore
            random.seed(seed)

        n = random.randint(2, self.max_n)
        m = random.randint(n - 1, min(n * (n - 1) // 2, self.max_m))

        graph = generate_graph(n, m, seed=None)
        self.vm = VirtualMachine([], graph, truncated_after=self.max_code_length)

        # NOTE(rob2u): might be worth trying to parse the entire state of the VM (as above)
        return np.array([0] * self.max_code_length), {}

    def close(self) -> None:
        pass


class VirtualMachine:
    """Simple stack-based virtual machine for graph traversal


    See instruction_set.md for more info.
    """

    def __init__(self, code: List[AbstractCommand], input: Graph, truncated_after: int = 1000):
        self.code = code
        self.state = State(input)
        self.truncated_after = truncated_after  # Maximum number of instructions before truncating
        self.timeout = (len(input.edges) * len(input.nodes)**2)  # we let it run for a quite a while

    def run(self) -> Tuple[int, bool]:
        execution_counter = 0
        while self.state.pc < len(self.code):
            op = self.code[self.state.pc]()
            op.execute(self.state)

            self.state.pc += 1
            if self.state.early_ret:
                break

            execution_counter += 1
            if execution_counter > self.timeout:
                break

        return int(self.state.ret_register), False

    def append_instruction(self, instruction: AbstractCommand) -> bool:
        """Add an instruction to the code if it is not too long. If it is too long, return False"""
        if len(self.code) < self.truncated_after:
            self.code.append(instruction)
            return True
        else:
            return False


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
    IF_EDGE_WEIGHT_GT,
]


class Transpiler:
    """Translates a list of integers to a list of AbstractCommands and back"""

    @staticmethod
    def commandToInt(code: List[AbstractCommand]) -> List[int]:
        return [COMMAND_REGISTRY.index(op) + 1 for op in code]

    @staticmethod
    def intToCommand(code: List[int]) -> List[AbstractCommand]:
        assert all([i >= 1 for i in code]), "Bad command encoding encountered!"
        return [COMMAND_REGISTRY[op - 1] for op in code]


if __name__ == "__main__":
    # Lets write PRIM algorithm in our instruction set
    test_graph = generate_graph(10, 20)

    code = [
        PUSH_START_NODE,
        ADD_TO_SET,
        PUSH_MARK,  # LOOP START
        PUSH_HEAP,
        NEXT_NODE,
        IF_IS_NOT_FIRST_NODE,  # LOOP END CONDITION
        JUMP,
        POP_MARK,  # LOOP END
        RET,
    ]

    vm = VirtualMachine(code, test_graph)
    print(vm.run())
