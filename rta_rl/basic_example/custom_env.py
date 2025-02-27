import typing

import gymnasium as gym
import numpy as np
import pydantic
from gymnasium import spaces
from loguru import logger


class EnvParams(pydantic.BaseModel):
    """Parameters for the SimpleEnv environment."""

    grid_size: int


class ProblemParams(pydantic.BaseModel):
    """Parameters for the problem setup."""

    agent_pos: np.ndarray
    goal_pos: np.ndarray

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    @classmethod
    def randomize(cls, grid_size: int) -> "ProblemParams":
        """Create random problem parameters."""
        rng = np.random.default_rng()
        agent_pos = rng.integers(0, grid_size, size=2)
        goal_pos = rng.integers(0, grid_size, size=2)
        while np.array_equal(goal_pos, agent_pos):
            goal_pos = rng.integers(0, grid_size, size=2)
        return cls(agent_pos=agent_pos, goal_pos=goal_pos)


class Solution(pydantic.BaseModel):
    """Solution to the problem."""

    steps: list[int]

    def pretty_str(self) -> str:
        """String representation of the solution."""
        moves = {0: "up", 1: "right", 2: "down", 3: "left"}
        return f"Steps: {" ".join([moves[step] for step in self.steps])}"


class SimpleEnv(gym.Env):
    """Simple environment where the agent must reach a goal.

    Represents a 2D grid of size grid_size x grid_size.
    """

    def __init__(self, env_params: EnvParams) -> None:
        super().__init__()

        self.grid_size = env_params.grid_size

        # Action space: 0 - up, 1 - right, 2 - down, 3 - left
        self.action_space = spaces.Discrete(4)

        # Observation space: current position (x, y) and goal position (goal_x, goal_y)
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(4,), dtype=np.float32)

        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([0, 0])

    def set_problem(self, problem_params: ProblemParams) -> None:
        """Set the problem parameters."""
        self.reset()
        self.agent_pos = problem_params.agent_pos
        self.goal_pos = problem_params.goal_pos

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, typing.Any] | None = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, dict[str, typing.Any]]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)

        problem = ProblemParams.randomize(self.grid_size)
        self.agent_pos = problem.agent_pos
        self.goal_pos = problem.goal_pos

        # Random placement of the agent
        rng = np.random.default_rng(seed)
        self.agent_pos = rng.integers(0, self.grid_size, size=2)

        # Random placement of the goal (not coinciding with the agent's position)
        self.goal_pos = self.agent_pos
        while np.array_equal(self.goal_pos, self.agent_pos):
            self.goal_pos = rng.integers(0, self.grid_size, size=2)

        # Current step
        self.current_step = 0

        # Maximum number of steps in an episode
        self.max_steps = 2 * self.grid_size

        observation = self.get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, typing.Any]]:
        """Perform an action."""
        self.current_step += 1

        # Move the agent according to the action
        if action == 0:  # up
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 1:  # right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # down # noqa: PLR2004
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # left # noqa: PLR2004
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)

        # Check if the agent has reached the goal
        terminated = False
        truncated = False
        done = np.array_equal(self.agent_pos, self.goal_pos)

        # Reward: positive for reaching the goal, negative for each step
        if done:
            reward = 10.0
            terminated = True
        else:
            # Euclidean distance to the goal
            distance = np.linalg.norm(self.agent_pos - self.goal_pos)
            # Encourage reducing the distance to the goal
            reward = -0.1 - float(distance / self.grid_size)

        # Check for exceeding the maximum number of steps
        if self.current_step >= self.max_steps and not done:
            truncated = True
            reward = -1.0

        info = {}
        return self.get_observation(), reward, terminated, truncated, info

    def _get_info(self) -> dict[str, typing.Any]:
        """Get information about the current state of the environment."""
        return {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "current_step": self.current_step,
        }

    def get_observation(self) -> np.ndarray:
        """Get the current observation."""
        return np.concatenate([self.agent_pos, self.goal_pos]).astype(np.float32)

    def render(self, mode: str = "human") -> None:  # noqa: ARG002
        """Visualize the current state of the environment."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid.fill(".")

        # Mark the agent's position
        grid[self.agent_pos[1], self.agent_pos[0]] = "A"

        # Mark the goal's position
        grid[self.goal_pos[1], self.goal_pos[0]] = "G"

        # If the agent and the goal are in the same position
        if np.array_equal(self.agent_pos, self.goal_pos):
            grid[self.agent_pos[1], self.agent_pos[0]] = "X"

        # Output the grid
        for row in reversed(grid):  # Reverse to have (0,0) at the bottom left
            logger.info(" ".join(row))
        logger.info("\n")
