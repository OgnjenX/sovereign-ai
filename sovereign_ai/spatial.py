from __future__ import annotations

import numpy as np


class StripeCells:
    """Temporal trace bank useful as a spatial/motion substrate."""

    def __init__(self, dim: int, decay: float = 0.82) -> None:
        self.trace = np.zeros(dim, dtype=float)
        self.decay = decay

    def update(self, motion: np.ndarray) -> np.ndarray:
        self.trace = self.decay * self.trace + (1.0 - self.decay) * np.asarray(motion, dtype=float)
        return self.trace


class SOMLayer:
    """Small self-organizing map layer for grid/place-like vector codes."""

    def __init__(self, input_dim: int, units: int, learning_rate: float = 0.08, seed: int | None = None) -> None:
        self.learning_rate = learning_rate
        rng = np.random.default_rng(seed)
        self.weights = rng.random((units, input_dim))

    def activate(self, x: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(self.weights - x, axis=1)
        inv = 1.0 / (distances + 1e-9)
        return inv / (np.sum(inv) + 1e-9)

    def learn(self, x: np.ndarray) -> int:
        distances = np.linalg.norm(self.weights - x, axis=1)
        winner = int(np.argmin(distances))
        self.weights[winner] += self.learning_rate * (x - self.weights[winner])
        return winner


class SpatialModule:
    def __init__(self, motion_dim: int, grid_units: int = 9, place_units: int = 6, seed: int | None = None) -> None:
        self.stripes = StripeCells(motion_dim)
        self.grid = SOMLayer(motion_dim, grid_units, seed=seed)
        self.place = SOMLayer(grid_units, place_units, seed=None if seed is None else seed + 1)

    def process(self, motion: np.ndarray, learn: bool = True) -> np.ndarray:
        trace = self.stripes.update(motion)
        if learn:
            self.grid.learn(trace)
        grid_code = self.grid.activate(trace)
        if learn:
            self.place.learn(grid_code)
        return self.place.activate(grid_code)
