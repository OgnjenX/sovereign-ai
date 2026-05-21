"""Spatial coding primitives for motion traces and place representations."""

from __future__ import annotations

import numpy as np


class StripeCells:
    """Temporal trace bank useful as a spatial/motion substrate."""

    def __init__(self, dim: int, decay: float = 0.82) -> None:
        self.trace = np.zeros(dim, dtype=float)
        self.decay = decay

    def update(self, motion: np.ndarray) -> np.ndarray:
        """Update exponentially decayed motion trace."""

        self.trace = self.decay * self.trace + (1.0 - self.decay) * np.asarray(motion, dtype=float)
        return self.trace

    def reset(self) -> None:
        """Reset accumulated trace to zeros."""

        self.trace = np.zeros_like(self.trace)


class SOMLayer:
    """Small self-organizing map layer for grid/place-like vector codes."""

    def __init__(
        self,
        input_dim: int,
        units: int,
        learning_rate: float = 0.08,
        seed: int | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        rng = np.random.default_rng(seed)
        self.weights = rng.random((units, input_dim))

    def activate(self, x: np.ndarray) -> np.ndarray:
        """Return normalized inverse-distance activations over SOM units."""

        distances = np.linalg.norm(self.weights - x, axis=1)
        inv = 1.0 / (distances + 1e-9)
        return inv / (np.sum(inv) + 1e-9)

    def learn(self, x: np.ndarray) -> int:
        """Adapt winner unit toward input and return winner index."""

        distances = np.linalg.norm(self.weights - x, axis=1)
        winner = int(np.argmin(distances))
        self.weights[winner] += self.learning_rate * (x - self.weights[winner])
        return winner


class SpatialModule:
    """Pipeline combining stripe traces, grid coding, and place coding."""

    def __init__(
        self,
        motion_dim: int,
        grid_units: int = 9,
        place_units: int = 6,
        seed: int | None = None,
    ) -> None:
        self.stripes = StripeCells(motion_dim)
        self.grid = SOMLayer(motion_dim, grid_units, seed=seed)
        self.place = SOMLayer(grid_units, place_units, seed=None if seed is None else seed + 1)

    def process(self, motion: np.ndarray, learn: bool = True) -> np.ndarray:
        """Encode motion into place-code activation vector."""

        trace = self.stripes.update(motion)
        if learn:
            self.grid.learn(trace)
        grid_code = self.grid.activate(trace)
        if learn:
            self.place.learn(grid_code)
        return self.place.activate(grid_code)

    def reset(self) -> None:
        """Reset internal temporal trace state."""

        self.stripes.reset()
