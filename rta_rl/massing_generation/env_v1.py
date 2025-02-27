import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any


class ConstructionEnv(gym.Env):
    """
    A simple RL environment for building construction within constraints.

    The agent places rectangular buildings within a construction area,
    while respecting zoning and height limitations.
    """

    def __init__(
        self,
        area_shape: tuple[float, float],
        construction_zone_ratio: float = 0.5,
        height_limit: float = 30.0,
        max_buildings: int = 5,
    ):
        """
        Initialize the construction environment.

        Args:
            area_shape: Tuple of (width, length) of the total area in meters
            construction_zone_ratio: Ratio of total area that can be built on (0-1)
            height_limit: Maximum height of buildings in meters
            max_buildings: Maximum number of buildings that can be placed
        """
        super().__init__()

        # Environment parameters
        self.area_shape = area_shape
        self.construction_zone_ratio = construction_zone_ratio
        self.height_limit = height_limit
        self.max_buildings = max_buildings

        # Calculate construction zone (simplified as centered rectangle)
        self.zone_width = area_shape[0] * np.sqrt(construction_zone_ratio)
        self.zone_length = area_shape[1] * np.sqrt(construction_zone_ratio)
        self.zone_x_min = (area_shape[0] - self.zone_width) / 2
        self.zone_y_min = (area_shape[1] - self.zone_length) / 2
        self.zone_x_max = self.zone_x_min + self.zone_width
        self.zone_y_max = self.zone_y_min + self.zone_length

        # State variables
        self.buildings = []
        self.current_step = 0

        # Action space: [x, y, width, length, height]
        # x, y: coordinates of the bottom-left corner of the building
        # width, length, height: dimensions of the building
        self.action_space = spaces.Box(
            low=np.array([0, 0, 1, 1, 1]),
            high=np.array(
                [
                    area_shape[0],
                    area_shape[1],
                    area_shape[0],
                    area_shape[1],
                    height_limit,
                ]
            ),
            dtype=np.float32,
        )

        # Observation space: current state of the construction area
        # We'll use a simplified representation: a list of buildings + remaining area info
        # Each building: [x, y, width, length, height]
        # Plus 3 more values: [total_building_area, remaining_zone_area, steps_remaining]
        max_building_dims = 5 * max_buildings  # 5 values per building * max_buildings
        self.observation_space = spaces.Box(
            low=0,
            high=np.max(area_shape + (height_limit,)),
            shape=(max_building_dims + 3,),
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        """Convert the current state into an observation vector."""
        # Initialize with zeros
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Fill in existing buildings
        for i, building in enumerate(self.buildings):
            if i < self.max_buildings:
                start_idx = i * 5
                obs[start_idx : start_idx + 5] = building

        # Calculate total building area
        total_building_area = sum(b[2] * b[3] for b in self.buildings)

        # Calculate remaining construction zone area
        zone_area = self.zone_width * self.zone_length
        remaining_zone_area = max(0, zone_area - total_building_area)

        # Add summary info
        obs[-3] = total_building_area
        obs[-2] = remaining_zone_area
        obs[-1] = self.max_buildings - len(self.buildings)

        return obs

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.buildings = []
        self.current_step = 0

        return self._get_observation(), {}

    def _is_valid_building(self, building) -> bool:
        """Check if the building placement is valid."""
        x, y, width, length, height = building

        # Check height limit
        if height > self.height_limit:
            return False

        # Check if building is completely within construction area
        if (
            x < 0
            or y < 0
            or x + width > self.area_shape[0]
            or y + length > self.area_shape[1]
        ):
            return False

        # Check for overlap with other buildings (simplified)
        for b in self.buildings:
            b_x, b_y, b_width, b_length, _ = b
            if (
                x < b_x + b_width
                and x + width > b_x
                and y < b_y + b_length
                and y + length > b_y
            ):
                return False

        return True

    def _calculate_reward(self, building, is_valid: bool) -> float:
        """
        Calculate the reward for placing a building.

        Args:
            building: The building parameters [x, y, width, length, height]
            is_valid: Whether the building placement is valid

        Returns:
            float: The calculated reward
        """
        if not is_valid:
            # Penalty for invalid building placement
            return -2.0

        # Calculate building area
        building_area = building[2] * building[3]

        # Check if building is inside the allowable construction zone
        x, y, width, length, _ = building
        in_zone = (
            x >= self.zone_x_min
            and y >= self.zone_y_min
            and x + width <= self.zone_x_max
            and y + length <= self.zone_y_max
        )

        reward = 0.0
        if in_zone:
            # Reward for placing a building inside the zone
            reward += 1.0

            # Additional reward for efficient use of space
            zone_area = self.zone_width * self.zone_length
            reward += 0.5 * (building_area / zone_area)
        else:
            # Penalty for placing outside the zone
            reward -= 1.0

        # Bonus reward for using all available buildings
        if len(self.buildings) >= self.max_buildings:
            reward += 2.0

        return reward

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment by placing a building.

        Args:
            action: [x, y, width, length, height] of the building to place

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # Place a building according to the action
        new_building = action.copy()

        # Initialize return values
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Check if the building is valid
        is_valid = self._is_valid_building(new_building)
        if is_valid:
            self.buildings.append(new_building)

            # Check if we've reached the max number of buildings
            if len(self.buildings) >= self.max_buildings:
                terminated = True
        else:
            # Don't add the building, but still count the step
            info["invalid_building"] = True

        # Calculate reward based on the building and its validity
        reward = self._calculate_reward(new_building, is_valid)

        # End episode if we've reached max steps (one per building)
        if self.current_step >= self.max_buildings:
            truncated = True

        # Get the new observation
        observation = self._get_observation()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Simple text rendering of the current state."""
        print(f"Construction Area: {self.area_shape}")
        print(
            f"Construction Zone: ({self.zone_x_min}, {self.zone_y_min}) to ({self.zone_x_max}, {self.zone_y_max})"
        )
        print(f"Buildings placed: {len(self.buildings)}/{self.max_buildings}")

        for i, building in enumerate(self.buildings):
            x, y, width, length, height = building
            print(
                f"Building {i+1}: Position ({x:.1f}, {y:.1f}), Size {width:.1f}x{length:.1f}x{height:.1f}m"
            )

        return None
