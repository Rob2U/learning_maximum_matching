import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env

from .environment.environment import MSTCodeEnvironment, Transpiler

if __name__ == "__main__":
    gym.register("MSTCode-v0", entry_point=MSTCodeEnvironment)

    vectorized_env = make_vec_env("MSTCode-v0", n_envs=10)
    model = sb3.PPO("MlpPolicy", vectorized_env, verbose=1, device="cpu")  # type: ignore

    model.learn(total_timesteps=10000)

    model.save("ppo_mst_code")

    # check what the model has learned
    new_env: MSTCodeEnvironment = gym.make("MSTCode-v0")
    state, _ = new_env.reset()
    is_done = False
    while not is_done:
        action, _ = model.predict(observation=state)
        state, _, is_done, _ = new_env.step(action)

        print(Transpiler.intToCommand([action]))
