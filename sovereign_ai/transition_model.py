"""Temporal transition ART field and linear compatibility wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sovereign_ai.art_field import ARTField


@dataclass(frozen=True)
class TemporalPrediction:
    """Prediction summary returned by the transition model."""

    perceptual_bias: np.ndarray
    action_bias: np.ndarray
    confidence: float
    mismatch: float
    reset: bool
    search_path: list[int]


class ARTTemporalField(ARTField):
    """ART sequence field with chunk categories and mismatch-driven reset."""

    def __init__(
        self,
        state_dim: int,
        action_count: int,
        perceptual_category_count: int | None = None,
        action_category_count: int | None = None,
        context_width: int = 12,
        max_categories: int = 32,
        learning_rate: float = 0.08,
        seed: int | None = None,
    ) -> None:
        """Initialize the temporal field and its transition buffers."""

        self.state_dim = state_dim
        self.action_count = action_count
        self.perceptual_category_count = (
            state_dim
            if perceptual_category_count is None
            else perceptual_category_count
        )
        self.action_category_count = (
            action_count if action_category_count is None else action_category_count
        )
        self.context_width = context_width
        self.input_width = (
            self.perceptual_category_count
            + self.action_category_count
            + self.perceptual_category_count
            + self.context_width
            + 1
        )
        self.uncertainty = np.ones(action_count, dtype=float)
        self.next_state_expectations = np.empty((0, state_dim), dtype=float)
        self.next_perceptual_bias = np.empty(
            (0, self.perceptual_category_count), dtype=float
        )
        self.next_action_bias = np.empty((0, self.action_category_count), dtype=float)
        self.sequence_counts = np.empty(0, dtype=float)
        self.mismatch_trace: list[float] = []
        self.last_prediction = TemporalPrediction(
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            0.0,
            1.0,
            False,
            [],
        )
        self._last_state = np.zeros(state_dim, dtype=float)
        super().__init__(
            input_dim=self.input_width,
            max_categories=max_categories,
            vigilance=0.6,
            competition_temperature=0.25,
            learning_rate=learning_rate,
            seed=seed,
            debug=False,
            name="temporal",
        )

    def predict(self, state: np.ndarray, action_distribution: np.ndarray) -> np.ndarray:
        """Predict the next state from the current state and action mix."""

        if len(self.categories) == 0:
            return np.clip(np.asarray(state, dtype=float), 0.0, 1.0)
        probe = self.sequence_input(
            previous_percept=self._state_to_category(
                self._last_state, self.perceptual_category_count
            ),
            action_category=self._fit(action_distribution, self.action_category_count),
            current_percept=self._state_to_category(
                state, self.perceptual_category_count
            ),
            context=np.empty(0),
        )
        result = self.process(
            probe,
            category_bias=self._action_category_bias(action_distribution),
            learn=False,
        )
        self._ensure_associations(np.asarray(state, dtype=float))
        activation = result.category_activation
        expected = activation @ self.next_state_expectations[: len(activation)]
        mismatch = 1.0 - max(result.resonance_trace) if result.resonance_trace else 1.0
        self.last_prediction = TemporalPrediction(
            activation @ self.next_perceptual_bias[: len(activation)],
            activation @ self.next_action_bias[: len(activation)],
            float(max(result.resonance_trace) if result.resonance_trace else 0.0),
            float(mismatch),
            not result.resonance,
            result.search_path,
        )
        self.mismatch_trace.append(float(mismatch))
        if np.linalg.norm(expected) <= 1e-9:
            return np.clip(np.asarray(state, dtype=float), 0.0, 1.0)
        return np.clip(expected, 0.0, 1.0)

    def learn(
        self,
        category_index: int | np.ndarray,
        x: np.ndarray,
        learning_rate: float | np.ndarray | None = None,
        previous_percept: np.ndarray | None = None,
        action_category: np.ndarray | None = None,
        current_percept: np.ndarray | None = None,
        context: np.ndarray | None = None,
    ) -> None:
        """Update transition statistics from either a category or raw state."""

        if isinstance(category_index, (int, np.integer)):
            rate = (
                learning_rate
                if isinstance(learning_rate, (int, float, np.floating))
                else None
            )
            super().learn(int(category_index), x, None if rate is None else float(rate))
            return

        state = np.asarray(category_index, dtype=float)
        action_distribution = np.asarray(x, dtype=float)
        next_state = (
            state if learning_rate is None else np.asarray(learning_rate, dtype=float)
        )
        self._learn_transition_core(
            state,
            action_distribution,
            next_state,
            previous_percept=previous_percept,
            action_category=action_category,
            current_percept=current_percept,
            context=context,
        )

    def _learn_transition_core(
        self,
        state: np.ndarray,
        action_distribution: np.ndarray,
        next_state: np.ndarray,
        *,
        previous_percept: np.ndarray | None = None,
        action_category: np.ndarray | None = None,
        current_percept: np.ndarray | None = None,
        context: np.ndarray | None = None,
    ) -> None:
        """Apply the core transition-learning update for a state/action pair."""

        previous = (
            self._state_to_category(state, self.perceptual_category_count)
            if previous_percept is None
            else previous_percept
        )
        action = self._fit(
            action_distribution if action_category is None else action_category,
            self.action_category_count,
        )
        current = (
            self._state_to_category(next_state, self.perceptual_category_count)
            if current_percept is None
            else current_percept
        )
        sequence = self.sequence_input(
            previous, action, current, np.empty(0) if context is None else context
        )
        before_count = len(self.categories)
        result = self.process(
            sequence, category_bias=self._action_category_bias(action), learn=False
        )
        self._ensure_associations(np.asarray(next_state, dtype=float))
        rate = self.learning_rate
        super().learn(result.category_index, sequence, rate)
        self.next_state_expectations[result.category_index] += rate * (
            np.asarray(next_state, dtype=float)
            - self.next_state_expectations[result.category_index]
        )
        self.next_perceptual_bias[result.category_index] += rate * (
            self._fit(current, self.perceptual_category_count)
            - self.next_perceptual_bias[result.category_index]
        )
        self.next_action_bias[result.category_index] += rate * (
            self._fit(action, self.action_category_count)
            - self.next_action_bias[result.category_index]
        )
        self.sequence_counts[result.category_index] += 1.0
        prediction = self.predict(state, action_distribution)
        error_norm = float(
            np.linalg.norm(np.asarray(next_state, dtype=float) - prediction)
        )
        action_distribution = self._action_distribution(action_distribution)
        self.uncertainty = (
            0.98 * self.uncertainty + 0.02 * action_distribution * error_norm
        )
        if before_count != len(self.categories):
            self.uncertainty += 0.01 * action_distribution
        self._last_state = np.asarray(next_state, dtype=float).copy()

    def predict_categories(
        self,
        previous_percept: np.ndarray,
        action_category: np.ndarray,
        current_percept: np.ndarray,
        context: np.ndarray,
    ) -> TemporalPrediction:
        """Predict category-level future bias without learning."""

        if len(self.categories) == 0:
            return self.last_prediction
        x = self.sequence_input(
            previous_percept, action_category, current_percept, context
        )
        result = self.process(
            x, category_bias=self._action_category_bias(action_category), learn=False
        )
        self._ensure_associations(np.zeros(self.state_dim, dtype=float))
        activation = result.category_activation
        confidence = float(
            max(result.resonance_trace) if result.resonance_trace else 0.0
        )
        prediction = TemporalPrediction(
            activation @ self.next_perceptual_bias[: len(activation)],
            activation @ self.next_action_bias[: len(activation)],
            confidence,
            1.0 - confidence,
            not result.resonance,
            result.search_path,
        )
        self.last_prediction = prediction
        self.mismatch_trace.append(prediction.mismatch)
        return prediction

    def sequence_input(
        self,
        previous_percept: np.ndarray,
        action_category: np.ndarray,
        current_percept: np.ndarray,
        context: np.ndarray,
        phase: float = 0.0,
    ) -> np.ndarray:
        """Pack perceptual, action, and contextual traces into one vector."""

        return np.concatenate(
            [
                self._fit(previous_percept, self.perceptual_category_count),
                self._fit(action_category, self.action_category_count),
                self._fit(current_percept, self.perceptual_category_count),
                self._fit(context, self.context_width),
                np.array([np.clip(phase, 0.0, 1.0)], dtype=float),
            ]
        )

    def prediction_bias(
        self, state: np.ndarray, action_distribution: np.ndarray
    ) -> np.ndarray:
        """Return the prediction residual for a state-action pair."""

        return self.predict(state, action_distribution) - np.asarray(state, dtype=float)

    def _add_category(self, x: np.ndarray) -> int:
        index = super()._add_category(x)
        self._ensure_associations(np.zeros(self.state_dim, dtype=float))
        return index

    def _ensure_associations(self, default_state: np.ndarray) -> None:
        count = len(self.categories)
        if len(self.next_state_expectations) < count:
            additions = np.repeat(
                np.asarray(default_state, dtype=float).reshape(1, -1),
                count - len(self.next_state_expectations),
                axis=0,
            )
            self.next_state_expectations = (
                additions
                if len(self.next_state_expectations) == 0
                else np.vstack([self.next_state_expectations, additions])
            )
        if self.next_perceptual_bias.shape != (count, self.perceptual_category_count):
            resized = np.zeros((count, self.perceptual_category_count), dtype=float)
            rows = min(count, self.next_perceptual_bias.shape[0])
            if rows:
                resized[:rows] = self.next_perceptual_bias[:rows]
            self.next_perceptual_bias = resized
        if self.next_action_bias.shape != (count, self.action_category_count):
            resized = np.zeros((count, self.action_category_count), dtype=float)
            rows = min(count, self.next_action_bias.shape[0])
            if rows:
                resized[:rows] = self.next_action_bias[:rows]
            self.next_action_bias = resized
        if len(self.sequence_counts) < count:
            self.sequence_counts = np.pad(
                self.sequence_counts, (0, count - len(self.sequence_counts))
            )

    def _action_distribution(self, action_distribution: np.ndarray) -> np.ndarray:
        return self._fit(action_distribution, self.action_count)

    def _action_category_bias(self, action_distribution: np.ndarray) -> np.ndarray:
        if len(self.categories) == 0:
            return np.empty(0, dtype=float)
        action = self._fit(action_distribution, self.action_category_count)
        start = self.perceptual_category_count
        action_slice = self.categories[:, start : start + self.action_category_count]
        bias = action_slice @ action
        if np.sum(bias) <= 1e-9:
            return np.zeros(len(self.categories), dtype=float)
        return bias / (np.sum(bias) + 1e-9)

    def _state_to_category(self, state: np.ndarray, width: int) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        fitted = np.zeros(width, dtype=float)
        fitted[: min(width, len(state))] = state[: min(width, len(state))]
        if np.min(fitted) < 0.0:
            fitted = fitted - np.min(fitted)
        if np.sum(fitted) <= 1e-9:
            return fitted
        return fitted / (np.sum(fitted) + 1e-9)

    def _fit(self, values: np.ndarray, width: int) -> np.ndarray:
        fitted = np.zeros(width, dtype=float)
        values = np.asarray(values, dtype=float)
        fitted[: min(width, len(values))] = values[: min(width, len(values))]
        if np.min(fitted) < 0.0:
            fitted = fitted - np.min(fitted)
        if np.sum(fitted) <= 1e-9:
            return fitted
        return fitted / (np.sum(fitted) + 1e-9)


class LinearTransitionModel(ARTTemporalField):
    """Compatibility wrapper preserving the old constructor name."""
