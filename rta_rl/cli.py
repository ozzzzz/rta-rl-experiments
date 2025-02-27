import logging
from pathlib import Path
import typer
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from rta_rl.massing_generation.env_v1 import ConstructionEnv

from rta_rl.basic_example import train as basic_train

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = typer.Typer(name="rta", pretty_exceptions_enable=False)

basic = typer.Typer(name="basic", add_completion=False)
app.add_typer(basic)


@basic.command("train")
def basic_example_train(
    models_dir: Path = Path("data/models/basic_example"),
    logs_dir: Path = Path("data/models/basic_example/logs"),
    timesteps: int = 100_000,
):
    basic_train.train(
        models_dir=models_dir,
        logs_dir=logs_dir,
        total_timesteps=timesteps,
    )


@app.command("run")
def run2(
    area_width: float = 100.0,
    area_length: float = 100.0,
    zone_ratio: float = 0.5,
    height_limit: float = 30.0,
    max_buildings: int = 5,
    timesteps: int = 10000,
):
    """Run the RL training and testing."""
    env = ConstructionEnv(
        area_shape=(area_width, area_length),
        construction_zone_ratio=zone_ratio,
        height_limit=height_limit,
        max_buildings=max_buildings,
    )

    model = PPO(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=timesteps)

    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()

    print(f"Total reward: {total_reward}")
