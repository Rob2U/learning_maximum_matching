import logging
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Tuple, Type

import gymnasium as gym
import numpy as np
import torch
from gymnasium import Env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from simple_parsing import parse
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import wandb
from args import GlobalArgs
from environment.commands import (
    ADD_EDGE_TO_SET,
    IF_EDGE_SET_CAPACITY_REMAINING,
    IF_EDGE_STACK_REMAINING,
    IF_EDGE_WEIGHT_LT,
    JUMP,
    POP_EDGE,
    POP_MARK,
    PUSH_LEGAL_EDGES,
    PUSH_MARK,
    RESET_EDGE_REGISTER,
    RET,
    WRITE_EDGE_REGISTER,
)
from environment.environment import COMMAND_REGISTRY, MSTCodeEnvironment, Transpiler
from environment.feedback import reward
from environment.vm_state import AbstractCommand
from environment.wandb_logger import WandbLoggingCallback
from models.policy_nets import CustomActorCriticPolicy
from models.transformer_fe import TransformerFeaturesExtractor

# from wandb.integration.sb3 import WandbCallback
# from simple_parsing import ArgumentParser

# Configure logging
time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def execute_program(
    env_args: Dict[Any, Any], program: List[Type[AbstractCommand]], n: int = 1000
) -> Tuple[List[float], defaultdict[str, List[Any]]]:
    rewards: List[float] = []
    metrics: defaultdict[str, List[Any]] = defaultdict(list)

    env_args["num_vms_per_env"] = 1
    test_environment = MSTCodeEnvironment(**env_args)  # type: ignore
    for _ in range(n):  # TODO(rob2u): could we simply switch to multiple vms directly?
        test_environment.reset(code=program)

        # run the code
        result, vm_state = test_environment.vms[0].run()
        observed_reward, observed_metric = reward(result, vm_state, **env_args)
        rewards.append(observed_reward)
        for key, value in observed_metric.items():
            metrics[key].append(value)

    test_environment.close()  # type: ignore

    return rewards, metrics


def infer_program(env_args: Dict[str, Any], model: Any) -> List[Type[AbstractCommand]]:
    new_env: Env[Any, Any] = gym.make("MSTCode-v0", **env_args)
    state, _ = new_env.reset()
    is_terminated = False
    is_truncated = False
    while not (is_terminated or is_truncated):
        action, _ = model.predict(observation=state, action_masks=get_action_masks(new_env), deterministic=True)  # type: ignore
        state, _, is_terminated, is_truncated, _ = new_env.step(action)  # type: ignore

        # logging.info(Transpiler.intToCommand([action])[0]())  # type: ignore

    new_env.close()  # type: ignore
    return Transpiler.intToCommand([int(a) for a in state])  # type: ignore


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed=seed)


if __name__ == "__main__":
    #  Load Config: We can specify config via: --config_path="path/to/config.yml"
    config_path = "./configs/config.yaml"
    global_args = asdict(
        parse(GlobalArgs, config_path=config_path)
    )  # if config_path is set, default values are loaded from there and overwritten by the command line arguments

    observation_size = global_args["max_code_length"]
    if global_args["add_vm_state_to_observations"]:
        observation_size += 3  # add 3 more dimensions for the VM state
    global_args["observation_size"] = observation_size

    # Setup WandB:
    wandb_run = wandb.init(
        entity="robert-weeke2-uni-potsdam",
        project="constrainedIS",
        config=global_args,  # type: ignore
    )

    gym.register("MSTCode-v0", entry_point=MSTCodeEnvironment)  # type: ignore

    def get_env(**kwargs: Any) -> MSTCodeEnvironment:
        return MSTCodeEnvironment(**kwargs)

    train_env: MSTCodeEnvironment | SubprocVecEnv
    if global_args["vectorize_environment"]:
        train_env = make_vec_env(get_env, env_kwargs=global_args, n_envs=global_args["num_envs"], vec_env_cls=SubprocVecEnv)  # type: ignore
    else:
        train_env = gym.make("MSTCode-v0", **global_args)  # type: ignore

    # check if masking is supported:
    assert is_masking_supported(
        train_env
    ), "Action masking not supported by Env but required!"

    feature_dim = global_args["feature_dim"]
    d_model = global_args["fe_d_model"]
    nhead = global_args["fe_nhead"]
    num_blocks = global_args["fe_num_blocks"]
    num_instructions = len(COMMAND_REGISTRY)

    layer_dim_pi = global_args["layer_dim_pi"]
    layer_dim_vf = global_args["layer_dim_vf"]

    policy_kwargs = dict(
        features_extractor_class=TransformerFeaturesExtractor,  # type: ignore
        features_extractor_kwargs=dict(
            features_dim=feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_blocks=num_blocks,
            num_instructions=num_instructions,
            observation_size=observation_size,
            device=global_args["device"],
        ),
        # MLP layers for actor and critic after the transformer:
        feature_dim=feature_dim,
        layer_dim_pi=layer_dim_pi,
        layer_dim_vf=layer_dim_vf,
    )

    model = MaskablePPO(
        CustomActorCriticPolicy,
        train_env,
        verbose=1,
        device=global_args["device"],
        policy_kwargs=policy_kwargs,
        gamma=global_args["gamma"],
        seed=global_args["seed"],
        batch_size=global_args["batch_size"],
        learning_rate=global_args["learning_rate"],
    )  # type: ignore

    logging.info("Policy network overview:")
    logging.info(model.policy)

    model.learn(
        total_timesteps=global_args["iterations"],
        callback=WandbLoggingCallback(wandb_run),
    )  # type: ignore
    model.save("ppo_mst_code")

    # check what the model has learned
    learned_program = infer_program(env_args=global_args, model=model)
    learned_program_results, learned_program_metrics = execute_program(
        global_args, learned_program
    )

    logging.info("Learned program: ")
    logging.info([str(c()) for c in learned_program])
    logging.info(
        f"Average reward: {sum(learned_program_results) / len(learned_program_results)}"
    )

    averages = {
        key + "_avg": sum(values) / len(values)
        for key, values in learned_program_metrics.items()
    }
    log_metrics = {"learned_program/" + key: value for key, value in averages.items()}
    log_metrics.update(
        {
            "learned_program/code": " ".join([str(c()) for c in learned_program]),
            "learned_program/reward_avg": sum(learned_program_results)
            / len(learned_program_results),
        }
    )
    wandb.log(log_metrics)

    # Best program during training:
    # get best program of all envs in vec_env
    # best_program = train_env.get_wrapper_attr("best_program")
    if global_args["vectorize_environment"]:
        best_program = train_env.get_attr("best_program")  # type: ignore
        best_program = max(best_program, key=lambda x: x[0])
    else:
        best_program = train_env.unwrapped.best_program  # type: ignore

    best_program_results, best_program_metrics = execute_program(
        env_args=global_args, program=best_program[1]
    )

    logging.info("Best program during train: ")
    logging.info([str(c()) for c in best_program[1]])
    logging.info(
        f"Average reward: {sum(best_program_results) / len(best_program_results)}"
    )

    averages = {
        key + "_avg": sum(values) / len(values)
        for key, values in best_program_metrics.items()
    }
    log_metrics = {"best_program/" + key: value for key, value in averages.items()}
    log_metrics.update(
        {
            "best_program/code": (
                "[ " + ", ".join([str(c()) for c in best_program[1]]) + " ]"
            ),
            "best_program/reward_avg": sum(best_program_results)
            / len(best_program_results),
        }
    )

    wandb.log(log_metrics)

    # the actual mst implementation
    # our_program = [
    #     PUSH_MARK,  # LOOP START
    #     PUSH_LEGAL_EDGES,  # push stack of edges that are allowed to be added
    #     RESET_EDGE_REGISTER,
    #     PUSH_MARK,  # INNER LOOP START
    #     IF_EDGE_WEIGHT_LT,  # if top of edge stack is greater than edge register
    #     WRITE_EDGE_REGISTER,  # update edge register to edge on top of stack
    #     POP_EDGE,  # pop edge from edge stack
    #     IF_EDGE_STACK_REMAINING,  # INNER LOOP END CONDITION: if edge stack is empty
    #     JUMP,  # to INNER LOOP START
    #     ADD_EDGE_TO_SET,  # final command before inner loop ends: add the edge from our edge register to the set
    #     POP_MARK,  # INNER LOOP END
    #     IF_EDGE_SET_CAPACITY_REMAINING,  # LOOP END CONDITION: if n - 1 edges have been marked
    #     JUMP,  # to LOOP START
    #     POP_MARK,  # LOOP END
    #     RET,  # end of program (indicate that we do not need to generate more code)
    # ]

    # very naive program that just adds the smallest to edges to the set (we have n=3 and m=3)
    our_program = [
        PUSH_LEGAL_EDGES,
        RESET_EDGE_REGISTER,
        WRITE_EDGE_REGISTER,
        POP_EDGE,
        IF_EDGE_WEIGHT_LT,
        WRITE_EDGE_REGISTER,
        POP_EDGE,
        IF_EDGE_WEIGHT_LT,
        WRITE_EDGE_REGISTER,
        POP_EDGE,
        ADD_EDGE_TO_SET,
        # continue for 2nd edge
        PUSH_LEGAL_EDGES,
        RESET_EDGE_REGISTER,
        WRITE_EDGE_REGISTER,
        POP_EDGE,
        IF_EDGE_WEIGHT_LT,
        WRITE_EDGE_REGISTER,
        POP_EDGE,
        ADD_EDGE_TO_SET,
        RET,
    ]

    our_program_results, our_program_metrics = execute_program(global_args, our_program)

    # TODO: average and log metrics

    logging.info("Our program:")
    logging.info([str(c()) for c in our_program])  # type: ignore
    logging.info(
        f"Average reward: {sum(our_program_results) / len(our_program_results)}"
    )

    averages = {
        key + "_avg": sum(values) / len(values)
        for key, values in our_program_metrics.items()
    }
    log_metrics = {"our_program/" + key: value for key, value in averages.items()}
    log_metrics.update(
        {
            "our_program/code": " ".join([str(c()) for c in our_program]),  # type: ignore
            "our_program/reward_avg": sum(our_program_results)
            / len(our_program_results),
        }
    )

    wandb.log(log_metrics)
