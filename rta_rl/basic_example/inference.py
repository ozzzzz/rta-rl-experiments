import time
from pathlib import Path

import numpy as np
from loguru import logger
from stable_baselines3 import PPO

from .custom_env import EnvParams, ProblemParams, SimpleEnv, Solution


def evaluate(model_path: Path, grid_size: int, num_episodes: int = 5, *, render: bool = True) -> None:
    """Run the trained model to evaluate its performance."""
    env = SimpleEnv(EnvParams(grid_size=grid_size))

    # Load the trained model
    model = PPO.load(model_path)

    # Run episodes
    episode_rewards = []

    for episode in range(num_episodes):
        obs, _info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0

        logger.info(f"\nEpisode {episode + 1}")
        if render:
            logger.info("Initial state:")
            env.render()

        while not terminated and not truncated:
            # Get action from the model
            action, _states = model.predict(obs, deterministic=True)

            # Perform action
            obs, reward, terminated, truncated, _info = env.step(action)

            total_reward += reward
            step_count += 1

            if render:
                logger.info(f"Step {step_count}, Action: {action}, Reward: {reward:.2f}")
                env.render()
                time.sleep(0.5)  # Pause for observation

        episode_rewards.append(total_reward)
        logger.info(f"Episode {episode + 1} finished. Steps: {step_count}, Reward: {total_reward:.2f}")

    # Output statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    logger.info("\nInference results:")
    logger.info(f"Average reward per episode: {avg_reward:.2f} ± {std_reward:.2f}")
    logger.info(f"Min/Max rewards: {min(episode_rewards):.2f} / {max(episode_rewards):.2f}")


def run(model_path: Path, grid_size: int, problem: ProblemParams) -> Solution:
    """Run the trained model to solve a specific problem."""
    env = SimpleEnv(EnvParams(grid_size=grid_size))
    env.set_problem(problem_params=problem)

    model = PPO.load(model_path)
    solution = Solution(steps=[])
    while True:
        observation = env.get_observation()
        action, info = model.predict(observation)
        obs, reward, terminated, truncated, _ = env.step(action)
        solution.steps.append(int(action))
        if terminated or truncated:
            break
    return solution
