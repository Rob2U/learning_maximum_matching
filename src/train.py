import gymnasium as gym
import logging
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env

from args import GlobalArgs
from environment.environment import MSTCodeEnvironment, Transpiler
from environment.feedback import reward

# from simple_parsing import ArgumentParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="train.log",
    filemode="w+",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # load configuration
    global_args = GlobalArgs.load_yaml("configs/config.yaml").to_dict()
    logging.info(global_args)

    gym.register("MSTCode-v0", entry_point=MSTCodeEnvironment)  # type: ignore

    vec_env = make_vec_env("MSTCode-v0", env_kwargs=dict(global_args), n_envs=4)  # type: ignore
    model = MaskablePPO("MlpPolicy", vec_env, verbose=1, device="cpu")  # type: ignore

    model.learn(total_timesteps=global_args["iterations"])

    model.save("ppo_mst_code")

    # check what the model has learned
    new_env: MSTCodeEnvironment = gym.make("MSTCode-v0")  # type: ignore
    state, _ = new_env.reset()
    is_terminated = False
    is_truncated = False
    curr_reward = 0.0
    logging.info("Final program:")
    while not (is_terminated or is_truncated):  # TODO(rob2u): check if this is correct
        action, _ = model.predict(observation=state, action_masks=get_action_masks(new_env))  # type: ignore
        state, curr_reward, is_terminated, is_truncated, _ = new_env.step(action)  # type: ignore

        logging.info(Transpiler.intToCommand([action])[0]())  # type: ignore

    # After finishing the training process, we evaluate the found program by running it n times
    n = 100
    program = Transpiler.intToCommand([int(a) for a in state])  # type: ignore
    rewards = []
    logging.info(f"Running the program {n} times")
    logging.info("Program:")
    logging.info([str(a()) for a in program])

    test_environment = MSTCodeEnvironment(num_vms_per_env=1, **global_args)  # type: ignore
    for i in range(n):
        test_environment.reset(code=program)

        # run the code
        result, vm_state = test_environment.vms[0].run()
        rewards.append(reward(result, vm_state))

    logging.info("Rewards: ")
    logging.info(rewards)
    logging.info(f"Average reward: {sum(rewards) / n}")

    vec_env.close()  # type: ignore
    new_env.close()  # type: ignore
