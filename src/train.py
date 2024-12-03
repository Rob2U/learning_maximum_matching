import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env

from environment.environment import MSTCodeEnvironment, Transpiler
from environment.feedback import reward

if __name__ == "__main__":
    gym.register("MSTCode-v0", entry_point=MSTCodeEnvironment)  # type: ignore

    vec_env = make_vec_env("MSTCode-v0", n_envs=4)  # type: ignore
    model = sb3.PPO("MlpPolicy", vec_env, verbose=1, device="cpu")  # type: ignore

    model.learn(total_timesteps=1_000_000)

    model.save("ppo_mst_code")

    # check what the model has learned
    new_env: MSTCodeEnvironment = gym.make("MSTCode-v0")  # type: ignore
    state, _ = new_env.reset()
    is_done = False
    is_truncated = False
    curr_reward = 0.0
    while not (is_done or is_truncated):
        action, _ = model.predict(observation=state)  # type: ignore
        state, curr_reward, is_done, is_truncated, _ = new_env.step(action)  # type: ignore

        # print(Transpiler.intToCommand([action + 1])[0]())  # type: ignore

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
        result, vm_state, code_state = test_environment.vm.run()
        rewards.append(reward(result, vm_state, code_state))

    print("Rewards: ")
    print(rewards)
    print(f"Average reward: {sum(rewards) / n}")

    vec_env.close()  # type: ignore
    new_env.close()  # type: ignore
