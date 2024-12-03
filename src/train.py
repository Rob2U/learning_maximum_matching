import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env

from environment.environment import MSTCodeEnvironment, Transpiler
from environment.feedback import reward

if __name__ == "__main__":
    gym.register("MSTCode-v0", entry_point=MSTCodeEnvironment)  # type: ignore

    vec_env = make_vec_env("MSTCode-v0", n_envs=4)  # type: ignore
    model = MaskablePPO("MlpPolicy", vec_env, verbose=1, device="cpu")  # type: ignore

    model.learn(total_timesteps=100_000)

    model.save("ppo_mst_code")

    # check what the model has learned
    new_env: MSTCodeEnvironment = gym.make("MSTCode-v0")  # type: ignore
    state, _ = new_env.reset()
    is_terminated = False
    is_truncated = False
    curr_reward = 0.0
    while not (is_terminated or is_truncated):  # TODO(rob2u): check if this is correct
        action, _ = model.predict(observation=state, action_masks=get_action_masks(new_env))  # type: ignore
        state, curr_reward, is_terminated, is_truncated, _ = new_env.step(action)  # type: ignore

        # print(Transpiler.intToCommand([action + 1])[0]())  # type: ignore

    # After finishing the training process, we evaluate the found program by running it n times
    n = 100
    program = Transpiler.intToCommand([int(a) for a in state])  # type: ignore
    rewards = []
    print(f"Running the program {n} times")
    print("Program:")
    print([str(a()) for a in program])

    test_environment = MSTCodeEnvironment()
    for i in range(n):
        test_environment.reset(code=program)

        # run the code
        result, vm_state = test_environment.vm.run()
        rewards.append(reward(result, vm_state))

    print("Rewards: ")
    print(rewards)
    print(f"Average reward: {sum(rewards) / n}")

    vec_env.close()  # type: ignore
    new_env.close()  # type: ignore
