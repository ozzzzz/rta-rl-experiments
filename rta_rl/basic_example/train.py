import os
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.callbacks import BaseCallback

from .custom_env import SimpleEnv


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Колбек для сохранения модели при достижении лучшей средней награды за эпизод
    """

    def __init__(
        self, models_dir: Path, log_dir: Path, check_freq: int = 1000, verbose=1
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(models_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Создание директории для сохранения, если её нет
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Получение x и y для построения графика наград
            x, y = ts2xy(load_results(str(self.log_dir)), "timesteps")
            if len(x) > 0:
                # Средняя награда за последние 100 эпизодов
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Шаг {self.num_timesteps}")
                    print(f"Текущая средняя награда: {mean_reward:.2f}")

                # Сохранение лучшей модели
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Сохранение новой лучшей модели: {self.save_path}")
                    self.model.save(os.path.join(self.save_path, "ppo_simple_env"))

        return True


def train(models_dir: Path, logs_dir: Path, total_timesteps: int = 100_000) -> Path:
    """Функция для обучения модели PPO на нашем окружении."""

    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Функция для создания окружения
    def make_env():
        env = SimpleEnv(grid_size=10)
        env = Monitor(env, str(logs_dir))
        return env

    # Создание векторизованного окружения
    env = DummyVecEnv([make_env])

    # Имя и путь модели
    model_name = "ppo_simple_env"

    # Колбек для сохранения лучшей модели
    callback = SaveOnBestTrainingRewardCallback(
        models_dir=models_dir, log_dir=logs_dir, check_freq=1000
    )

    # Создание и обучение модели
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=str(logs_dir))

    # Обучение модели
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Сохранение финальной модели
    final_model_path = models_dir / f"{model_name}_final"
    model.save(final_model_path)
    print(f"Финальная модель сохранена в {final_model_path}")

    # Закрытие окружения
    env.close()

    return final_model_path
