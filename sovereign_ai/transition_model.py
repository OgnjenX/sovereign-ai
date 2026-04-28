from __future__ import annotations

import numpy as np


class LinearTransitionModel:
    """Online linear transition model over state-action vectors."""

    def __init__(
        self,
        state_dim: int,
        action_count: int,
        learning_rate: float = 0.05,
        uncertainty_decay: float = 0.98,
        seed: int | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_count = action_count
        self.learning_rate = learning_rate
        self.uncertainty_decay = uncertainty_decay
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.02, (state_dim + action_count, state_dim))
        self.uncertainty = np.ones(action_count, dtype=float)

    def predict(self, state: np.ndarray, action_distribution: np.ndarray) -> np.ndarray:
        features = self._features(state, action_distribution)
        delta = features @ self.weights
        return np.clip(state + delta, 0.0, 1.0)

    def learn(
        self,
        state: np.ndarray,
        action_distribution: np.ndarray,
        next_state: np.ndarray,
        learning_rate: float | None = None,
    ) -> None:
        action_distribution = np.asarray(action_distribution, dtype=float)
        action_distribution = action_distribution / (np.sum(action_distribution) + 1e-9)
        features = self._features(state, action_distribution)
        prediction = self.predict(state, action_distribution)
        error = next_state - prediction
        rate = self.learning_rate if learning_rate is None else learning_rate
        self.weights += rate * np.outer(features, error)
        error_norm = float(np.linalg.norm(error))
        self.uncertainty = (
            self.uncertainty_decay * self.uncertainty
            + (1.0 - self.uncertainty_decay) * action_distribution * error_norm
        )

    def _features(self, state: np.ndarray, action_distribution: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(state, dtype=float), np.asarray(action_distribution, dtype=float)])
