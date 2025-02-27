from pathlib import Path

import numpy as np
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv

from .custom_env import EnvParams, SimpleEnv


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """Callback for saving the model when the best mean reward is achieved."""

    def __init__(self, models_dir: Path, log_dir: Path, check_freq: int = 1000, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = models_dir / "best_model"
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get x and y for plotting rewards
            x, y = ts2xy(load_results(str(self.log_dir)), "timesteps")
            if len(x) > 0:
                # Mean reward for the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    logger.info(f"Step {self.num_timesteps}")
                    logger.info(f"Current mean reward: {mean_reward:.2f}")

                # Save the best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        logger.info(f"Saving new best model to: {self.save_path}")
                    self.model.save(self.save_path / "ppo_simple_env")

        return True


def train(models_dir: Path, logs_dir: Path, grid_size: int, total_timesteps: int = 100_000) -> Path:
    """Function to train the PPO model on our environment."""
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Function to create the environment
    def make_env() -> Monitor:
        env = SimpleEnv(EnvParams(grid_size=grid_size))
        return Monitor(env, str(logs_dir))

    # Create vectorized environment
    env = DummyVecEnv([make_env])

    # Model name and path
    model_name = "ppo_simple_env"

    # Callback for saving the best model
    callback = SaveOnBestTrainingRewardCallback(models_dir=models_dir, log_dir=logs_dir, check_freq=1000)

    # Create and train the model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=str(logs_dir))

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the final model
    final_model_path = models_dir / f"{model_name}_final"
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Close the environment
    env.close()

    return final_model_path
