from __future__ import annotations

import numpy as np


class GridWorld:
    """Toy continuous-observation environment with vector salience and rewards."""

    def __init__(self, size: int = 5, seed: int | None = None) -> None:
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.position = np.array([0, 0], dtype=int)
        self.goal = np.array([size - 1, size - 1], dtype=int)
        self.hazard = np.array([size // 2, size // 2], dtype=int)
        self.action_effects = np.array(
            [
                [0, -1],
                [1, 0],
                [0, 1],
                [-1, 0],
            ],
            dtype=int,
        )

    @property
    def action_count(self) -> int:
        return len(self.action_effects)

    @property
    def observation_dim(self) -> int:
        return 8

    def observe(self) -> np.ndarray:
        pos = self.position / max(1, self.size - 1)
        goal_delta = (self.goal - self.position) / max(1, self.size - 1)
        hazard_delta = (self.hazard - self.position) / max(1, self.size - 1)
        noise = self.rng.normal(0.0, 0.015, 2)
        x = np.concatenate([pos, goal_delta, hazard_delta, noise])
        return np.clip((x + 1.0) / 2.0, 0.0, 1.0)

    def salience(self) -> np.ndarray:
        projected = self.action_effects @ (self.goal - self.position)
        salience = projected.astype(float)
        salience -= np.mean(salience)
        denom = np.max(np.abs(salience)) + 1e-9
        return 0.18 * salience / denom

    def urgency(self) -> float:
        distance_to_hazard = np.linalg.norm(self.position - self.hazard)
        return float(np.exp(-distance_to_hazard))

    def step(self, action_index: int) -> float:
        self.position = np.clip(
            self.position + self.action_effects[action_index],
            0,
            self.size - 1,
        )
        goal_reward = np.exp(-np.linalg.norm(self.position - self.goal))
        hazard_penalty = np.exp(-2.0 * np.linalg.norm(self.position - self.hazard))
        reward = float(goal_reward - hazard_penalty)
        if np.array_equal(self.position, self.goal):
            self.goal = self.rng.integers(0, self.size, size=2)
        return reward
