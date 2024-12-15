import logging
from datetime import datetime
from typing import Any, Dict, List, Type

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from simple_parsing import ArgumentParser
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

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

# from simple_parsing import ArgumentParser

# Configure logging
time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="train_{}.log".format(time_stamp),
    filemode="w+",
)
logger = logging.getLogger(__name__)


def execute_program(
    env_args: Dict[Any, Any], program: List[Type[AbstractCommand]], n: int = 1000
) -> List[float]:
    rewards: List[float] = []
    env_args["num_vms_per_env"] = 1
    test_environment = MSTCodeEnvironment(**env_args)  # type: ignore
    for _ in range(n):
        test_environment.reset(code=program)

        # run the code
        result, vm_state = test_environment.vms[0].run()
        rewards.append(reward(result, vm_state))

    test_environment.close()  # type: ignore

    return rewards


def infer_program(
    env_args: Dict[str, Any], model: Any, action_masking: bool
) -> List[Type[AbstractCommand]]:
    new_env: MSTCodeEnvironment = gym.make("MSTCode-v0", **env_args)  # type: ignore
    state, _ = new_env.reset()
    is_terminated = False
    is_truncated = False
    while not (is_terminated or is_truncated):
        if action_masking:
            action, _ = model.predict(observation=state, action_masks=get_action_masks(new_env), deterministic=True)  # type: ignore
        else:
            action, _ = model.predict(state, deterministic=True)  # type: ignore
        state, is_terminated, is_truncated, _ = new_env.step(action)  # type: ignore

        logging.info(Transpiler.intToCommand([action])[0]())  # type: ignore

    new_env.close()
    return Transpiler.intToCommand([int(a) for a in state])  # type: ignore


if __name__ == "__main__":
    # load configuration
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    call_args = parser.parse_args()
    config_path = call_args.config

    global_args = GlobalArgs.load_yaml(config_path).to_dict()
    logging.info(global_args)

    # Setup WandB:
    wandb.init(
        # entity="TODO",
        project="constrainedIS",
    )
    wandb.config = dict(global_args)  # type: ignore

    gym.register("MSTCode-v0", entry_point=MSTCodeEnvironment)  # type: ignore

    vec_env = make_vec_env("MSTCode-v0", env_kwargs=dict(global_args), n_envs=4)  # type: ignore

    # check if masking is supported:
    if is_masking_supported(vec_env):
        logging.info("Masking is supported")
    else:
        logging.info("Masking is not supported")
        if global_args["action_masking"]:
            logging.error(
                "Action masking is enabled but not supported by the environment"
            )
            exit(1)

    policy_kwargs = dict(
        net_arch=dict(pi=global_args["policy_net"], qf=global_args["policy_net"])
    )
    if global_args["action_masking"]:
        model = MaskablePPO("MlpPolicy", vec_env, verbose=1, device="cpu", policy_kwargs=policy_kwargs)  # type: ignore
    else:
        model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu", policy_kwargs=policy_kwargs)  # type: ignore

    model.learn(total_timesteps=global_args["iterations"])
    model.save("ppo_mst_code")

    # check what the model has learned
    learned_program = infer_program(
        env_args=global_args, model=model, action_masking=global_args["action_masking"]
    )
    learned_program_results = execute_program(global_args, learned_program)

    logging.info("Learned program: ")
    logging.info(learned_program_results)
    logging.info(
        f"Average reward: {sum(learned_program_results) / len(learned_program_results)}"
    )

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

    our_program_results = execute_program(global_args, our_program)

    logging.info("Our program:")
    logging.info(our_program_results)
    logging.info(
        f"Average reward: {sum(our_program_results) / len(our_program_results)}"
    )
