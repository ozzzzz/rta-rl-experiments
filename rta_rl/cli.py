from pathlib import Path

import numpy as np
import typer
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

from rta_rl import basic_example
from rta_rl.massing_generation.env_v1 import ConstructionEnv

app = typer.Typer(name="rta", pretty_exceptions_enable=False)

basic_example_app = typer.Typer(name="basic", add_completion=False)
app.add_typer(basic_example_app)


@basic_example_app.command("train")
def basic_example_train(
    grid_size: int,
    models_dir: Path = Path("data/models/basic_example"),
    logs_dir: Path = Path("data/models/basic_example/logs"),
    timesteps: int = 100_000,
) -> None:
    """Train the basic example model."""
    logger.info(f"Starting basic example training with {timesteps} timesteps")
    logger.info(f"Models will be saved to {models_dir}")
    logger.info(f"Logs will be saved to {logs_dir}")

    basic_example.train(
        models_dir=models_dir,
        logs_dir=logs_dir,
        grid_size=grid_size,
        total_timesteps=timesteps,
    )


@basic_example_app.command("evaluate")
def basic_example_evaluate(
    grid_size: int,
    model_path: Path = Path("data/models/basic_example/ppo_simple_env_final.zip"),
    num_episodes: int = 5,
) -> None:
    """Train the basic example model."""
    logger.info(f"Run stats for model {model_path}")
    basic_example.evaluate(model_path=model_path, grid_size=grid_size, num_episodes=num_episodes)


@basic_example_app.command("run")
def basic_example_run(
    grid_size: int,
    actor_x: int,
    actor_y: int,
    goal_x: int,
    goal_y: int,
    model_path: Path = Path("data/models/basic_example/ppo_simple_env_final.zip"),
) -> None:
    """Train the basic example model."""
    logger.info(f"Run stats for model {model_path}")
    problem_params = basic_example.ProblemParams(
        agent_pos=np.array([actor_x, actor_y]),
        goal_pos=np.array([goal_x, goal_y]),
    )
    solution = basic_example.run(model_path=model_path, grid_size=grid_size, problem=problem_params)
    logger.info(f"Solution: {solution.pretty_str()}")


@app.command("run")
def run2(
    area_width: float = 100.0,
    area_length: float = 100.0,
    zone_ratio: float = 0.5,
    height_limit: float = 30.0,
    max_buildings: int = 5,
    timesteps: int = 10000,
) -> None:
    """Run the RL training and testing."""
    logger.info("Initializing Construction Environment")
    logger.info(f"Area: {area_width}x{area_length}, Zone ratio: {zone_ratio}")
    logger.info(f"Height limit: {height_limit}, Max buildings: {max_buildings}")

    env = ConstructionEnv(
        area_shape=(area_width, area_length),
        construction_zone_ratio=zone_ratio,
        height_limit=height_limit,
        max_buildings=max_buildings,
    )

    logger.info(f"Training model for {timesteps} timesteps")
    model = PPO(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=timesteps)

    logger.info("Starting evaluation episode")
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()

    logger.success(f"Total reward: {total_reward}")
