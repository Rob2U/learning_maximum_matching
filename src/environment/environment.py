import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Type

import gymnasium as gym
import numpy as np
import numpy.typing as npt

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
    ConditionalCommand,
)
from .feedback import reward
from .generation import generate_graph, generate_ring
from .vm import VirtualMachine
from .vm_state import AbstractCommand


class MSTCodeEnvironment(gym.Env[npt.ArrayLike, int]):

    def __init__(
        self,
        observation_size: int = 32,
        max_code_length: int = 32,
        reset_for_every_run: bool = False,
        num_vms_per_env: int = 100,
        graph_type: str = "random",
        min_n: int = 3,
        max_n: int = 3,
        min_m: int = 3,
        max_m: int = 3,
        tree_i: int = 1,
        only_reward_on_ret: bool = True,
        action_masking: bool = True,
        add_vm_state_to_observations: bool = False,
        only_reward_of_first_vm: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the environment

        Args:
            max_code_length: The maximum length of the code. Defaults to 32.
            reset_for_every_run: Only keep the code / state for every single call to run. Reset VM(s) + Generate new graph for every run(). Defaults to False.
            num_vms_per_env: The number of virtual machines to run. This is somewhat equivalent to the number of graphs we evaluate per run(). Defaults to 1.
            min_n: The minimum number of nodes in the graph. Defaults to 3.
            max_n: The maximum number of nodes in the graph. Defaults to 3.
            min_m: The minimum number of edges in the graph. Defaults to 3.
            max_m: The maximum number of edges in the graph. Defaults to 3.
            only_reward_on_ret: Toggles if the reward should only be returned when the predicted ACTION was RET.
            action_masking: Toggles if we want to use strong action masking (if False we also use action masking but only for branches)
            add_vm_state_to_observations: Toggles if we want to add the VM state to the observation space
            only_reward_of_first_vm: Toggles if we want to return the reward for the first VM only or the mean reward of all VMs
        """

        super().__init__()
        self.max_code_length: int = max_code_length
        self.num_vms_per_env: int = num_vms_per_env
        self.vms: List[VirtualMachine] = []

        # Attributes used by the gymnasium library
        self.action_space = gym.spaces.Discrete(  # type: ignore
            len(COMMAND_REGISTRY)
        )  # action_space contains all the possible instructions for the next line
        self.observation_space = gym.spaces.MultiDiscrete(
            [len(COMMAND_REGISTRY)] * observation_size
        )  # we need to cap the size of the edge_set and edge_stack to the size of the COMMAND_REGISTRY so that each state dimension has the same size. otherwise we would need to do annoying things in the transformer.

        self.observation_size = observation_size
        self.graph_type = graph_type
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m
        self.tree_i = tree_i
        self.reset_for_every_run = reset_for_every_run
        self.only_reward_on_ret = only_reward_on_ret
        self.action_masking = action_masking
        self.add_vm_state_to_observations = add_vm_state_to_observations
        self.only_reward_of_first_vm = only_reward_of_first_vm

        self.init_args = kwargs
        self.init_args.update(
            {
                "max_code_length": self.max_code_length,
                "graph_type": self.graph_type,
                "min_n": self.min_n,
                "max_n": self.max_n,
                "min_m": self.min_m,
                "max_m": self.max_m,
                "tree_i": tree_i,
                "reset_for_every_run": self.reset_for_every_run,
                "only_reward_on_ret": self.only_reward_on_ret,
                "action_masking": self.action_masking,
                "add_vm_state_to_observations": self.add_vm_state_to_observations,
                "only_reward_of_first_vm": self.only_reward_of_first_vm,
            }
        )

        self.episode_counter: int = 0
        self.current_episode_rewards: List[float] = []

        assert (
            self.min_n <= self.max_n and self.min_m >= self.min_n - 1
        ), "Bad n / m values"

        self.reset()
        self.best_program: Tuple[float, List[Type[AbstractCommand]]] = (-1.0, [])

    def step(self, action: int) -> Tuple[npt.ArrayLike, float, bool, Dict[str, Any]]:  # type: ignore
        """Execute the action on all VMs and return the new state, reward, and whether the episode is done

        Args:
            action (int): An as int encoded AbstractCommand provided by the agent

        Returns:
            observation: The new state of the environment (the code), encoded as np.array of ints (with length self.max_code_length), prepended with additional information about the VM state if add_vm_state_to_observations is True
            reward: The reward for the action
            terminal: Whether the episode is done (True if the action is RET)
            info: Additional information about the environment (in our case an empty dictionary)
        """
        observations, rewards, terminals, truncateds = [], [], [], []

        metrics: defaultdict[str, List[Any]] = defaultdict(list)
        for i in range(self.num_vms_per_env):
            observation, _reward, terminal, truncated, vm_metrics = self.step_vm(
                action, i
            )

            assert (not terminal and action != RET) or (terminal and action == 1), (
                "Bad terminal value encountered: "
                + str(terminal)
                + " for "
                + Transpiler.intToCommand([action])[0].__name__
            )

            if self.only_reward_on_ret and not terminal:
                _reward = 0.0
            observations.append(observation)
            rewards.append(_reward)
            terminals.append(terminal)
            truncateds.append(truncated)
            for k, v in vm_metrics.items():
                metrics[k].append(v)

        self.current_episode_rewards.append(sum(rewards) / len(rewards))

        if any(terminals):
            self.episode_counter += 1

        if sum(rewards) / len(rewards) > self.best_program[0]:
            self.best_program = (sum(rewards) / len(rewards), self.vms[0].vm_state.code)

        if self.episode_counter % 10000 == 0 and any(terminals):
            logging.info(
                "Episode: "
                + str(self.episode_counter)
                + " Mean Reward: "
                + str(
                    sum(self.current_episode_rewards)
                    / len(self.current_episode_rewards)
                )
            )
            logging.info(
                "Program written: "
                + "[ "
                + ", ".join([str(op()) for op in self.vms[0].vm_state.code])
                + " ]"
            )
            logging.info("Reward in Last Step: " + str(sum(rewards) / len(rewards)))

        assert all(
            [val == terminals[0] for val in terminals]
        ), "Bad terminal values: " + str(terminals)

        assert all(
            [val == truncateds[0] for val in truncateds]
        ), "Bad truncated values: " + str(truncated)

        return (
            observations[0],
            rewards[0] if self.only_reward_of_first_vm else sum(rewards) / len(rewards),
            terminals[0],
            truncateds[0],
            metrics,
        )  # type: ignore

    def step_vm(
        self, action: int, vm_index: int
    ) -> Tuple[npt.ArrayLike, float, bool, bool, Dict[str, Any]]:
        """Execute the action on a single VM and return the new state, reward, and whether the episode is done

        Args:
            action: The action to execute

        Returns:
            observation: The new state of the environment (the code), encoded as np.array of ints (with length self.max_code_length)
            reward: The reward for the action
            terminal: Whether the episode is done (True if the action is RET)
            truncated: Whether the code was truncated (True if the code is too long > self.max_code_length)
        """
        assert 0 <= vm_index < self.num_vms_per_env, (
            "Bad VM index!: " + str(vm_index) + " for " + str(self.num_vms_per_env)
        )

        if self.reset_for_every_run:
            self.reset(code=self.vms[vm_index].vm_state.code)

        instruction = Transpiler.intToCommand([action])[0]
        truncated = not self.vms[vm_index].append_instruction(instruction)
        result, vm_state = self.vms[vm_index].run()
        
        reward_args = {**self.init_args, "ep_prev_reward": sum(self.current_episode_rewards) if self.current_episode_rewards else 0}
        _reward, metrics = reward(result, vm_state, **reward_args)

        # NOTE(rob2u): might be worth trying to parse the entire return state of the VM + code
        vm_observation = np.array(
            [
                0 if vm_state.edge_register == None else 1,
                min(
                    len(vm_state.edge_set), len(COMMAND_REGISTRY) - 1
                ),  # because our state space is capped by the size of the COMMAND_REGISTRY, we need to cap the edge_set size as well
                min(
                    len(vm_state.edge_stack), len(COMMAND_REGISTRY) - 1
                ),  # because our state space is capped by the size of the COMMAND_REGISTRY, we need to cap the edge_stack size as well
            ]
        )
        code_observation: npt.NDArray[int, 1] = np.array(Transpiler.commandToInt(vm_state.code))  # type: ignore
        filler_observation = (self.max_code_length - code_observation.shape[0]) * [0]
        code_observation = np.concatenate(
            [
                vm_observation if self.add_vm_state_to_observations else [],
                code_observation,
                filler_observation,
            ]
        )

        assert (
            code_observation.shape[0] == self.observation_size
        ), "Bad observation shape!"

        return (
            code_observation,
            _reward,
            vm_state.finished,
            truncated,
            metrics,
        )

    def action_masks(self) -> npt.ArrayLike:
        mask = np.zeros(len(COMMAND_REGISTRY), dtype=int)
        if self.action_masking:
            for Command in COMMAND_REGISTRY:
                mask[Transpiler.commandToInt([Command])[0]] = Command().is_applicable(
                    self.vms[0].vm_state
                )
        else:
            mask = mask + 1
            mask[Transpiler.commandToInt([RET])[0]] = (
                1
                if not (
                    len(self.vms[0].vm_state.code) > 0
                    and issubclass(self.vms[0].vm_state.code[-1], ConditionalCommand)
                )
                else 0
            )

        return mask

    def reset(
        self,
        code: List[Type[AbstractCommand]] | None = None,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[npt.ArrayLike, Dict[str, Any]]:
        """Reset the environment to a new state (can optionally provide a code to start with)

        Args:
            code: The code to start with. Defaults to None.

        Returns:
            observation: The initial state of the environment (the code), encoded as np.array of ints (with length self.max_code_length)
            info: Additional information about the environment (in our case an empty dictionary)
        """

        # NOTE(rob2u): we use standard library random in our Graph generation and here so we use np_random for adaptability

        self.vms = []
        self.current_episode_rewards = []

        for _ in range(self.num_vms_per_env):
            if self.graph_type == "random":
                n = random.randint(self.min_n, self.max_n)
                m = random.randint(self.min_m, min(n * (n - 1) // 2, self.max_m))
                graph = generate_graph(n, m, seed=None)
            elif self.graph_type == "ring":
                n = random.randint(self.min_n, self.max_n)
                graph = generate_ring(n, seed=None)
            # elif self.graph_type == "almost_tree":
                # n = random.randint(self.min_n, self.max_n)
                # i = random.randint(self.min_tree_i, self.max_tree_i)
                # graph = generate_almost_tree(n, i, seed=None)
            else:
                raise ValueError("Bad graph type! " + self.graph_type)

            self.vms.append(
                VirtualMachine(
                    code if code is not None else [],
                    graph,
                    max_code_len=self.max_code_length,
                )
            )

        # NOTE(rob2u): might be worth trying to parse the entire state of the VM (as above)
        if code:
            assert (
                len(code) <= self.max_code_length
            ), "Code too long! Make sure other state variables are not part of the code."

            vm_observation = np.zeros(3)
            code_filler = [0] * (self.max_code_length - len(code))
            observation = np.concatenate(
                [
                    vm_observation if self.add_vm_state_to_observations else [],
                    code,
                    code_filler,
                ]
            )
        else:
            observation = np.array([0] * self.observation_size)

        assert (
            observation.shape[0] == self.observation_size
        ), "Bad observation shape on reset!"

        return observation, {}

    def close(self) -> None:
        pass


COMMAND_REGISTRY: List[Type[AbstractCommand]] = [
    NOP,
    RET,  # only applicable if no if statement was is before
    # PUSH_MARK,
    # JUMP,
    # POP_MARK,
    # PUSH_START_NODE,
    # PUSH_CLONE_NODE,
    # POP_NODE,
    # PUSH_HEAP,
    # POP_HEAP,
    # IF_HEAP_EMPTY,
    # NEXT_NODE,
    # NEXT_EDGE,
    # TO_NEIGHBOR,
    # IF_IS_NOT_FIRST_EDGE,
    # IF_IS_NOT_FIRST_NODE,
    # WRITE_EDGE_WEIGHT,  # always applicable
    # RESET_EDGE_WEIGHT,  # always applicable
    # ADD_TO_OUT,
    IF_EDGE_WEIGHT_LT,  # not two in a row
    WRITE_EDGE_REGISTER,  # <-NOT SEEN      if PUSH_LEGAL_EDGES before and not two in a row
    RESET_EDGE_REGISTER,  # always applicable
    POP_EDGE,  # <- NOT SEEN       only if PUSH_LEGAL_EDGES before
    # IF_EDGE_STACK_REMAINING,
    PUSH_LEGAL_EDGES,  # always applicable
    ADD_EDGE_TO_SET,  # not two in a row
    # REMOVE_EDGE_FROM_SET,
    # IF_EDGE_SET_CAPACITY_REMAINING,
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
    result, vm_state = vm.run()
    print("##################")

    if not vm_state.finished:
        print("Program did not finish orderly!")
    else:
        print("Result:")
        print(result)

        print("##################")
        print("Actual MST:")
        print(vm.vm_state.edge_set)
