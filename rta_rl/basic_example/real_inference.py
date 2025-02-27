from pathlib import Path
import time
import numpy as np
import json
import gymnasium as gym
from stable_baselines3 import PPO
from custom_env import SimpleEnv


class RealDataInference:
    """
    Класс для запуска обученной модели на реальных данных.
    Может принимать данные окружения извне и выполнять инференс модели.
    """

    def __init__(self, model_path: Path):
        # Загрузка модели
        self.model = PPO.load(model_path)
        print(f"Модель загружена из {model_path}")

        # Создаем базовое окружение только для получения информации о форматах данных
        self._reference_env = SimpleEnv()

        # Получаем размерность пространства наблюдений
        self.observation_shape = self._reference_env.observation_space.shape

        # Границы пространства наблюдений
        if isinstance(self._reference_env.observation_space, gym.spaces.Box):
            self.observation_low = self._reference_env.observation_space.low
            self.observation_high = self._reference_env.observation_space.high
        else:
            raise TypeError("Unsupported observation space type")

        # Размерность пространства действий
        if isinstance(self._reference_env.action_space, gym.spaces.Discrete):
            self.action_space_size = self._reference_env.action_space.n
        else:
            raise TypeError("Unsupported action space type")

        print(
            f"Готов к инференсу. Формат наблюдения: {self.observation_shape}, "
            f"Количество возможных действий: {self.action_space_size}"
        )

    def validate_observation(self, observation):
        """
        Проверяет, что наблюдение имеет корректный формат.

        Args:
            observation: Наблюдение для проверки

        Returns:
            bool: True если формат корректен, иначе False
        """
        # Проверка типа и размерности
        if not isinstance(observation, (list, np.ndarray)):
            print(
                f"Ошибка: наблюдение должно быть списком или массивом numpy, получено {type(observation)}"
            )
            return False

        # Преобразуем в numpy, если нужно
        if isinstance(observation, list):
            observation = np.array(observation, dtype=np.float32)

        # Проверка размерности
        if observation.shape != self.observation_shape:
            print(
                f"Ошибка: неверная размерность наблюдения. "
                f"Ожидается {self.observation_shape}, получено {observation.shape}"
            )
            return False

        # Проверка границ значений
        if np.any(observation < self.observation_low) or np.any(
            observation > self.observation_high
        ):
            print("Предупреждение: значения наблюдения выходят за допустимые пределы.")
            # Обрезаем значения по границам
            observation = np.clip(
                observation, self.observation_low, self.observation_high
            )

        return True

    def predict(self, observation):
        """
        Получение действия от модели на основе наблюдения.

        Args:
            observation: Наблюдение окружения (список или numpy массив)

        Returns:
            int: Предсказанное действие
            dict: Дополнительная информация о предсказании
        """
        # Проверка формата наблюдения
        if not self.validate_observation(observation):
            raise ValueError("Некорректный формат наблюдения")

        # Преобразуем в numpy, если нужно
        if isinstance(observation, list):
            observation = np.array(observation, dtype=np.float32)

        # Получаем действие от модели
        action, _states = self.model.predict(observation, deterministic=True)

        # Для модели PPO action - это одно значение, а не массив
        if isinstance(action, np.ndarray) and action.size == 1:
            action = action.item()

        # Возвращаем действие и дополнительную информацию
        info = {
            "action_meaning": self._get_action_meaning(action),
            "confidence": 1.0,  # В PPO нет прямой меры уверенности
        }

        return action, info

    def _get_action_meaning(self, action):
        """Получение текстового описания действия."""
        action_meanings = ["вверх", "вправо", "вниз", "влево"]
        if 0 <= action < len(action_meanings):
            return action_meanings[action]
        return "неизвестное действие"

    def run_episode_from_file(self, data_file):
        """
        Запуск модели на эпизоде с данными из файла.

        Args:
            data_file (str): Путь к JSON-файлу с данными эпизода
        """
        try:
            with open(data_file, encoding="utf-8") as f:
                episode_data = json.load(f)

            print(f"Загружены данные эпизода из {data_file}")
            self.run_episode(episode_data)

        except Exception as e:
            print(f"Ошибка при загрузке или обработке файла: {e}")

    def run_episode(self, episode_data):
        """
        Запуск модели на эпизоде с данными.

        Args:
            episode_data (list): Список наблюдений эпизода
        """
        if not isinstance(episode_data, list):
            print("Ошибка: данные эпизода должны быть списком наблюдений")
            return

        total_reward = 0
        print("\n=== Запуск эпизода ===")

        for step, observation in enumerate(episode_data):
            try:
                print(f"\nШаг {step+1}/{len(episode_data)}")
                print(f"Наблюдение: {observation}")

                # Предсказание действия
                action, info = self.predict(observation)

                print(f"Действие: {action} ({info['action_meaning']})")

                # В реальных данных награду обычно вычисляет внешняя система
                # Здесь для примера просто дадим +1 за каждый шаг
                reward = 1.0
                total_reward += reward

                print(f"Награда: {reward:.2f}, Общая награда: {total_reward:.2f}")

                # Можно добавить задержку для наблюдения
                time.sleep(0.5)

            except Exception as e:
                print(f"Ошибка на шаге {step+1}: {e}")
                break

        print("\n=== Эпизод завершен ===")
        print(f"Общая награда: {total_reward:.2f}")

    def interactive_mode(self):
        """
        Интерактивный режим, где пользователь вводит наблюдения,
        а модель возвращает действия.
        """
        print("\n=== Интерактивный режим ===")
        print("Вводите наблюдения в формате: x_агента, y_агента, x_цели, y_цели")
        print("Например: 2, 3, 8, 8")
        print("Для выхода введите 'выход' или 'exit'")

        while True:
            try:
                user_input = input("\nВведите наблюдение: ").strip().lower()

                if user_input in ["выход", "exit", "quit", "q"]:
                    print("Выход из интерактивного режима")
                    break

                # Разбор ввода пользователя
                values = user_input.replace(",", " ").split()

                if len(values) != 4:
                    print(
                        "Ошибка: нужно ввести 4 значения (x_агента, y_агента, x_цели, y_цели)"
                    )
                    continue

                # Преобразование в числа
                observation = [float(v) for v in values]

                # Предсказание действия
                action, info = self.predict(observation)

                print(f"Предсказанное действие: {action} ({info['action_meaning']})")

            except ValueError as e:
                print(f"Ошибка в формате ввода: {e}")
            except Exception as e:
                print(f"Ошибка: {e}")


def generate_sample_data(
    filename="sample_episode.json", num_observations=10, grid_size=10
):
    """
    Генерирует пример файла с данными эпизода для демонстрации.

    Args:
        filename (str): Имя файла для сохранения данных
        num_observations (int): Количество наблюдений в эпизоде
        grid_size (int): Размер сетки
    """
    # Создаем случайную начальную позицию и позицию цели
    agent_pos = np.random.randint(0, grid_size, size=2)
    goal_pos = np.random.randint(0, grid_size, size=2)

    # Убедимся, что начальная позиция и цель различаются
    while np.array_equal(agent_pos, goal_pos):
        goal_pos = np.random.randint(0, grid_size, size=2)

    # Создаем путь движения агента к цели
    observations = []

    for _ in range(num_observations):
        # Добавляем текущее наблюдение
        current_observation = np.concatenate([agent_pos, goal_pos]).tolist()
        observations.append(current_observation)

        # Двигаем агента ближе к цели (простая эвристика)
        for dim in range(2):
            if agent_pos[dim] < goal_pos[dim]:
                agent_pos[dim] += 1
                break
            elif agent_pos[dim] > goal_pos[dim]:
                agent_pos[dim] -= 1
                break

        # Если агент достиг цели, завершаем эпизод
        if np.array_equal(agent_pos, goal_pos):
            # Добавляем последнее наблюдение, где агент достиг цели
            last_observation = np.concatenate([agent_pos, goal_pos]).tolist()
            observations.append(last_observation)
            break

    # Сохраняем данные в файл
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(observations, f, indent=2)

    print(f"Сгенерирован пример данных эпизода в файле: {filename}")
    return filename


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Инференс модели RL на реальных данных"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Путь к обученной модели (опционально)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Путь к файлу с данными эпизода (опционально)",
    )
    parser.add_argument(
        "--generate", action="store_true", help="Сгенерировать пример данных эпизода"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Запустить в интерактивном режиме"
    )

    args = parser.parse_args()

    # Создаем инференс
    try:
        inference = RealDataInference(model_path=args.model)

        # Если нужно сгенерировать пример данных
        if args.generate:
            sample_file = generate_sample_data()

            # Если не указан другой файл, используем сгенерированный
            if args.file is None:
                args.file = sample_file

        # Если указан файл с данными
        if args.file:
            inference.run_episode_from_file(args.file)

        # Если запрошен интерактивный режим
        if args.interactive or (not args.file and not args.generate):
            inference.interactive_mode()

    except Exception as e:
        print(f"Ошибка: {e}")
