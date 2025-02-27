import typing
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SimpleEnv(gym.Env):
    """
    Простое окружение, где агент должен достичь цели.
    Представляет собой двумерную сетку размером grid_size x grid_size.
    """

    def __init__(self, grid_size=10):
        super().__init__()

        self.grid_size = grid_size

        # Пространство действий: 0 - вверх, 1 - вправо, 2 - вниз, 3 - влево
        self.action_space = spaces.Discrete(4)

        # Пространство наблюдений: текущая позиция (x, y) и позиция цели (goal_x, goal_y)
        self.observation_space = spaces.Box(
            low=0, high=grid_size - 1, shape=(4,), dtype=np.float32
        )

        # # Инициализация среды
        # self.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, typing.Any] | None = None,
    ):
        """Сбросить окружение в начальное состояние."""
        super().reset(seed=seed)
        # Случайное размещение агента
        self.agent_pos = np.random.randint(0, self.grid_size, size=2)

        # Случайное размещение цели (не совпадающее с позицией агента)
        self.goal_pos = self.agent_pos
        while np.array_equal(self.goal_pos, self.agent_pos):
            self.goal_pos = np.random.randint(0, self.grid_size, size=2)

        # Текущий шаг
        self.current_step = 0

        # Максимальное количество шагов в эпизоде
        self.max_steps = 2 * self.grid_size

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """Выполнить действие."""
        self.current_step += 1

        # Перемещение агента в соответствии с действием
        if action == 0:  # вверх
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 1:  # вправо
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # вниз
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # влево
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)

        # Проверка, достиг ли агент цели
        terminated = False
        truncated = False
        done = np.array_equal(self.agent_pos, self.goal_pos)

        # Награда: положительная за достижение цели, отрицательная за каждый шаг
        if done:
            reward = 10.0
            terminated = True
        else:
            # Евклидово расстояние до цели
            distance = np.linalg.norm(self.agent_pos - self.goal_pos)
            # Поощряем уменьшение расстояния до цели
            reward = -0.1 - distance / self.grid_size

        # Проверка на превышение максимального количества шагов
        if self.current_step >= self.max_steps and not done:
            truncated = True
            reward = -1.0

        info = {}
        # observation, reward, terminated, truncated, info = self.env.step(action)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_info(self):
        """Получить информацию о текущем состоянии окружения."""
        return {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "current_step": self.current_step,
        }

    def _get_observation(self):
        """Получить текущее наблюдение."""
        return np.concatenate([self.agent_pos, self.goal_pos]).astype(np.float32)

    def render(self, mode="human"):
        """Визуализация текущего состояния окружения."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid.fill(".")

        # Отметка позиции агента
        grid[self.agent_pos[1], self.agent_pos[0]] = "A"

        # Отметка позиции цели
        grid[self.goal_pos[1], self.goal_pos[0]] = "G"

        # Если агент и цель в одной позиции
        if np.array_equal(self.agent_pos, self.goal_pos):
            grid[self.agent_pos[1], self.agent_pos[0]] = "X"

        # Вывод сетки
        for row in reversed(grid):  # Разворачиваем, чтобы (0,0) было внизу слева
            print(" ".join(row))
        print("\n")
