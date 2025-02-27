import os
import time
import numpy as np
from stable_baselines3 import PPO

from custom_env import SimpleEnv


def inference(model_path, num_episodes=5, render=True):
    """
    Запуск обученной модели для оценки её производительности.

    Args:
        model_path (str): Путь к сохраненной модели.
        num_episodes (int): Количество эпизодов для оценки.
        render (bool): Флаг для включения визуализации.
    """
    # Создание окружения
    env = SimpleEnv(grid_size=10)

    # Загрузка обученной модели
    model = PPO.load(model_path)

    # Запуск эпизодов
    episode_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        print(f"\nЭпизод {episode + 1}")
        if render:
            print("Начальное состояние:")
            env.render()

        while not done:
            # Получение действия от модели
            action, _states = model.predict(obs, deterministic=True)

            # Выполнение действия
            obs, reward, done, info = env.step(action)

            total_reward += reward
            step_count += 1

            if render:
                print(f"Шаг {step_count}, Действие: {action}, Награда: {reward:.2f}")
                env.render()
                time.sleep(0.5)  # Пауза для наблюдения

        episode_rewards.append(total_reward)
        print(
            f"Эпизод {episode + 1} завершен. Шагов: {step_count}, Награда: {total_reward:.2f}"
        )

    # Вывод статистики
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print("\nРезультаты инференса:")
    print(f"Среднее награда за эпизод: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Мин/Макс награды: {min(episode_rewards):.2f} / {max(episode_rewards):.2f}")


if __name__ == "__main__":
    # Путь к сохраненной лучшей модели
    best_model_path = os.path.join("models", "best_model", "ppo_simple_env")

    # Запуск инференса
    if os.path.exists(best_model_path + ".zip"):
        inference(best_model_path, num_episodes=5, render=True)
    else:
        # Если лучшей модели нет, используем финальную
        final_model_path = os.path.join("models", "ppo_simple_env_final")
        if os.path.exists(final_model_path + ".zip"):
            inference(final_model_path, num_episodes=5, render=True)
        else:
            print("Модель не найдена. Сначала запустите обучение (train.py)")
