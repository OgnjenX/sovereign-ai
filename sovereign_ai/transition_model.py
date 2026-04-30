from __future__ import annotations

import numpy as np

from sovereign_ai.art_field import ARTField


class ARTTemporalField(ARTField):
    """ART sequence field over previous state, action, and current state."""

    def __init__(
        self,
        state_dim: int,
        action_count: int,
        max_categories: int = 32,
        learning_rate: float = 0.08,
        seed: int | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_count = action_count
        self.uncertainty = np.ones(action_count, dtype=float)
        self.next_state_expectations = np.empty((0, state_dim), dtype=float)
        self._last_state = np.zeros(state_dim, dtype=float)
        super().__init__(
            input_dim=state_dim + action_count + state_dim,
            max_categories=max_categories,
            vigilance=0.6,
            competition_temperature=0.25,
            learning_rate=learning_rate,
            seed=seed,
            debug=False,
            name="temporal",
        )

    def predict(self, state: np.ndarray, action_distribution: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        action_distribution = self._action_distribution(action_distribution)
        if len(self.categories) == 0:
            return np.clip(state, 0.0, 1.0)
        probe = self._transition_input(self._last_state, action_distribution, state)
        result = self.process(probe, category_bias=self._action_category_bias(action_distribution), learn=False)
        self._ensure_next_state_expectations(state)
        activation = result.category_activation
        expected = activation @ self.next_state_expectations[: len(activation)]
        if np.linalg.norm(expected) <= 1e-9:
            return np.clip(state, 0.0, 1.0)
        return np.clip(expected, 0.0, 1.0)

    def learn(
        self,
        state: np.ndarray,
        action_distribution: np.ndarray,
        next_state: np.ndarray,
        learning_rate: float | None = None,
    ) -> None:
        state = np.asarray(state, dtype=float)
        next_state = np.asarray(next_state, dtype=float)
        action_distribution = self._action_distribution(action_distribution)
        x = self._transition_input(state, action_distribution, next_state)
        before_count = len(self.categories)
        result = self.process(x, category_bias=self._action_category_bias(action_distribution), learn=False)
        self._ensure_next_state_expectations(next_state)
        rate = self.learning_rate if learning_rate is None else learning_rate
        self.learn_category(result.category_index, x, next_state, rate)
        prediction = self.predict(state, action_distribution)
        error_norm = float(np.linalg.norm(next_state - prediction))
        self.uncertainty = 0.98 * self.uncertainty + 0.02 * action_distribution * error_norm
        if before_count != len(self.categories):
            self.uncertainty += 0.01 * action_distribution
        self._last_state = next_state.copy()

    def _ensure_next_state_expectations(self, default_state: np.ndarray) -> None:
        if len(self.next_state_expectations) >= len(self.categories):
            return
        missing = len(self.categories) - len(self.next_state_expectations)
        additions = np.repeat(np.asarray(default_state, dtype=float).reshape(1, -1), missing, axis=0)
        if len(self.next_state_expectations) == 0:
            self.next_state_expectations = additions
        else:
            self.next_state_expectations = np.vstack([self.next_state_expectations, additions])

    def learn_category(
        self,
        category_index: int,
        transition_input: np.ndarray,
        next_state: np.ndarray,
        learning_rate: float,
    ) -> None:
        super().learn(category_index, transition_input, learning_rate)
        self.next_state_expectations[category_index] = np.clip(
            (1.0 - learning_rate) * self.next_state_expectations[category_index]
            + learning_rate * next_state,
            0.0,
            1.0,
        )

    def prediction_bias(self, state: np.ndarray, action_distribution: np.ndarray) -> np.ndarray:
        return self.predict(state, action_distribution) - np.asarray(state, dtype=float)

    def _transition_input(
        self,
        previous_state: np.ndarray,
        action_distribution: np.ndarray,
        current_state: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate(
            [
                np.clip(np.asarray(previous_state, dtype=float), 0.0, 1.0),
                self._action_distribution(action_distribution),
                np.clip(np.asarray(current_state, dtype=float), 0.0, 1.0),
            ]
        )

    def _action_distribution(self, action_distribution: np.ndarray) -> np.ndarray:
        fitted = np.zeros(self.action_count, dtype=float)
        action_distribution = np.asarray(action_distribution, dtype=float)
        fitted[: min(len(action_distribution), self.action_count)] = action_distribution[
            : min(len(action_distribution), self.action_count)
        ]
        if np.sum(fitted) <= 1e-9:
            fitted += 1.0 / self.action_count
        return fitted / (np.sum(fitted) + 1e-9)

    def _action_category_bias(self, action_distribution: np.ndarray) -> np.ndarray:
        if len(self.categories) == 0:
            return np.empty(0, dtype=float)
        action_distribution = self._action_distribution(action_distribution)
        action_slice = self.categories[:, self.state_dim : self.state_dim + self.action_count]
        bias = action_slice @ action_distribution
        if np.sum(bias) <= 1e-9:
            return np.zeros(len(self.categories), dtype=float)
        return bias / (np.sum(bias) + 1e-9)


LinearTransitionModel = ARTTemporalField
