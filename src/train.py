import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from args import GlobalArgs
from environment.environment import MSTCodeEnvironment, Transpiler
from environment.feedback import reward
from environment.commands import (
    PUSH_MARK,
    PUSH_LEGAL_EDGES,
    RESET_EDGE_REGISTER,
    IF_EDGE_WEIGHT_LT,
    WRITE_EDGE_REGISTER,
    POP_EDGE,
    IF_EDGE_STACK_REMAINING,
    JUMP,
    ADD_EDGE_TO_SET,
    POP_MARK,
    IF_EDGE_SET_CAPACITY_REMAINING,
    RET,
)

# from simple_parsing import ArgumentParser


if __name__ == "__main__":
    # load configuration
    global_args = GlobalArgs.load_yaml("configs/config.yaml").to_dict()
    print(global_args)

    gym.register("MSTCode-v0", entry_point=MSTCodeEnvironment)  # type: ignore

    vec_env = make_vec_env("MSTCode-v0", env_kwargs=dict(global_args), n_envs=4)  # type: ignore
    # model = MaskablePPO("MlpPolicy", vec_env, verbose=1, device="cpu")  # type: ignore
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")  # type: ignore

    model.learn(total_timesteps=global_args["iterations"])

    model.save("ppo_mst_code")

    # check what the model has learned
    new_env: MSTCodeEnvironment = gym.make("MSTCode-v0")  # type: ignore
    state, _ = new_env.reset()
    is_terminated = False
    is_truncated = False
    curr_reward = 0.0
    while not (is_terminated or is_truncated):  # TODO(rob2u): check if this is correct
        # action, _ = model.predict(observation=state, action_masks=get_action_masks(new_env))  # type: ignore
        action, _ = model.predict(state, deterministic=True)  # type: ignore
        state, curr_reward, is_terminated, is_truncated, _ = new_env.step(action)  # type: ignore

        # print(Transpiler.intToCommand([action + 1])[0]())  # type: ignore

    # After finishing the training process, we evaluate the found program by running it n times
    n = 100
    program = Transpiler.intToCommand([int(a) for a in state])  # type: ignore
    rewards = []
    print(f"Running the program {n} times")
    print("Program:")
    print([str(a()) for a in program])

    test_environment = MSTCodeEnvironment(num_vms_per_env=1, **global_args)  # type: ignore
    for i in range(n):
        test_environment.reset(code=program)

        # run the code
        result, vm_state = test_environment.vms[0].run()
        rewards.append(reward(result, vm_state))

    print("Rewards: ")
    print(rewards)
    print(f"Average reward: {sum(rewards) / n}")
    test_environment.close()  # type: ignore
    # our Programs reward:
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

    test_environment = MSTCodeEnvironment(num_vms_per_env=1, **global_args)  # type: ignore
    for i in range(n):
        test_environment.reset(code=program)

        # run the code
        result, vm_state = test_environment.vms[0].run()
        rewards.append(reward(result, vm_state))

    print("Rewards: ")
    print(rewards)
    print(f"Average reward: {sum(rewards) / n}")
    test_environment.close()
    vec_env.close()  # type: ignore
    new_env.close()  # type: ignore
