from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sovereign_ai.art_field import ARTField, ARTFieldState
from sovereign_ai.utils import softmax


@dataclass(frozen=True)
class ActionResult:
    action_index: int
    action_distribution: np.ndarray
    go: np.ndarray
    stop: np.ndarray
    drives: np.ndarray
    pathway: str


@dataclass(frozen=True)
class ActionState:
    result: ActionResult
    change: float


class ARTActionField(ARTField):
    """ART field whose categories are action schemas."""

    def __init__(
        self,
        max_categories: int,
        action_count: int,
        value_width: int = 6,
        vigilance: float = 0.58,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        self.perceptual_width = max_categories
        self.action_count = action_count
        self.value_width = value_width
        self.schema_width = max_categories + value_width + action_count * 3
        super().__init__(
            input_dim=self.schema_width,
            max_categories=max(action_count * 3, action_count),
            vigilance=vigilance,
            competition_temperature=0.24,
            learning_rate=0.08,
            seed=seed,
            debug=debug,
            name="action",
        )
        self.action_preferences = np.empty((0, action_count), dtype=float)
        self._seed_action_schemas()

    @property
    def category_action_weights(self) -> np.ndarray:
        return self.action_preferences

    def select(
        self,
        category_activation: np.ndarray,
        value: float,
        salience: np.ndarray,
        imagined_action_prior: np.ndarray | None = None,
        sequence_bias: np.ndarray | None = None,
        pathway: str = "planned",
    ) -> ActionResult:
        value_state = np.zeros(self.value_width, dtype=float)
        value_state[0 if value >= 0 else min(3, self.value_width - 1)] = min(1.0, abs(value))
        state = self._compute_action_state(
            category_activation,
            value_state,
            salience,
            np.zeros(self.action_count) if imagined_action_prior is None else imagined_action_prior,
            np.zeros(self.action_count) if sequence_bias is None else sequence_bias,
            np.asarray(np.ones(self.action_count) / self.action_count, dtype=float),
            pathway=pathway,
        )
        return state.result

    def compute_action_state(
        self,
        category_activation: np.ndarray,
        value_state: np.ndarray,
        salience: np.ndarray,
        imagined_action_prior: np.ndarray,
        sequence_bias: np.ndarray,
        reactive_distribution: np.ndarray,
        previous_distribution: np.ndarray | None = None,
        pathway: str = "coupled",
    ) -> ActionState:
        return self._compute_action_state(
            category_activation,
            value_state,
            salience,
            imagined_action_prior,
            sequence_bias,
            reactive_distribution,
            previous_distribution=previous_distribution,
            pathway=pathway,
        )

    def _compute_action_state(
        self,
        category_activation: np.ndarray,
        value_state: np.ndarray,
        salience: np.ndarray,
        imagined_action_prior: np.ndarray,
        sequence_bias: np.ndarray,
        reactive_distribution: np.ndarray,
        previous_distribution: np.ndarray | None = None,
        pathway: str = "coupled",
    ) -> ActionState:
        x = self._schema_input(category_activation, value_state, salience, imagined_action_prior, sequence_bias)
        category_bias = self._action_category_bias(reactive_distribution, imagined_action_prior, sequence_bias)
        value_activation = np.asarray(value_state, dtype=float)
        risk = value_activation[3] if len(value_activation) > 3 else 0.0
        reward = value_activation[0] if len(value_activation) > 0 else 0.0
        vigilance_modulation = float(np.clip(0.1 * risk - 0.04 * reward, -0.06, 0.14))
        previous_category = self._distribution_to_category_bias(previous_distribution)
        if len(previous_category):
            category_bias = category_bias + previous_category
        state = super().update_state(
            x,
            category_bias=category_bias,
            vigilance_modulation=vigilance_modulation,
            learn=False,
        )
        schema_activation = state.result.category_activation
        action_distribution = self._schema_to_action_distribution(schema_activation)
        action_index = int(np.argmax(action_distribution))
        go = action_distribution
        stop = softmax(1.0 - action_distribution, temperature=0.35)
        drives = action_distribution - stop
        previous = (
            np.ones(self.action_count, dtype=float) / self.action_count
            if previous_distribution is None
            else np.asarray(previous_distribution, dtype=float)
        )
        previous = previous / (np.sum(previous) + 1e-9)
        change = float(np.linalg.norm(action_distribution - previous) + state.change)
        if self.debug:
            print(
                "[action-dyn] category="
                f"{state.result.category_index} action={action_index} change={change:.4f} "
                f"vigilance={state.result.effective_vigilance:.3f} distribution={np.round(action_distribution, 3)}"
            )
        return ActionState(ActionResult(action_index, action_distribution, go, stop, drives, pathway), change)

    def update_state(
        self,
        x: np.ndarray,
        *,
        previous_activation: np.ndarray | None = None,
        top_down_bias: np.ndarray | None = None,
        category_bias: np.ndarray | None = None,
        vigilance_modulation: float = 0.0,
        learn: bool = False,
    ) -> ARTFieldState:
        return super().update_state(
            x,
            previous_activation=previous_activation,
            top_down_bias=top_down_bias,
            category_bias=category_bias,
            vigilance_modulation=vigilance_modulation,
            learn=learn,
        )

    def slot_action_bias(self, slots: np.ndarray, category_activation: np.ndarray) -> np.ndarray:
        if len(slots) == 0 or len(self.action_preferences) == 0:
            return np.zeros(self.action_count, dtype=float)
        category_bias = self._distribution_to_category_bias(self._schema_to_action_distribution(category_activation))
        schema_activation = self._resize_activation(category_bias)
        return self._schema_to_action_distribution(schema_activation)

    def learn_action(
        self,
        category_activation: np.ndarray,
        action_index: int,
        reward_prediction_error: float,
        learning_rate: float = 0.08,
    ) -> None:
        x = self._schema_input(
            category_activation,
            np.zeros(self.value_width, dtype=float),
            self._one_hot(action_index),
            self._one_hot(action_index),
            np.zeros(self.action_count, dtype=float),
        )
        result = self.process(x, category_bias=self._one_hot_schema(action_index), learn=False)
        self.learn(result.category_index, x, learning_rate=learning_rate)
        preference_target = self._one_hot(action_index)
        sign = 1.0 if reward_prediction_error >= 0.0 else -0.5
        self.action_preferences[result.category_index] = np.clip(
            self.action_preferences[result.category_index]
            + learning_rate * sign * abs(reward_prediction_error) * preference_target,
            0.0,
            1.0,
        )
        self.action_preferences[result.category_index] /= np.sum(self.action_preferences[result.category_index]) + 1e-9
        if self.debug:
            print(
                "[action-learning] schema="
                f"{result.category_index} action={action_index} prediction_error={reward_prediction_error:.3f}"
            )

    def _schema_input(
        self,
        category_activation: np.ndarray,
        value_state: np.ndarray,
        salience: np.ndarray,
        imagined_action_prior: np.ndarray,
        sequence_bias: np.ndarray,
    ) -> np.ndarray:
        x = np.zeros(self.schema_width, dtype=float)
        percept = np.asarray(category_activation, dtype=float)
        value = np.asarray(value_state, dtype=float)
        salience = np.asarray(salience, dtype=float)
        imagined = np.asarray(imagined_action_prior, dtype=float)
        sequence = np.asarray(sequence_bias, dtype=float)
        x[: min(len(percept), self.perceptual_width)] = percept[: self.perceptual_width]
        start = self.perceptual_width
        x[start : start + min(len(value), self.value_width)] = value[: self.value_width]
        start += self.value_width
        x[start : start + self.action_count] = self._fit_action_signal(salience)
        start += self.action_count
        x[start : start + self.action_count] = self._fit_action_signal(imagined)
        start += self.action_count
        x[start : start + self.action_count] = self._fit_action_signal(sequence)
        return np.clip(x, 0.0, 1.0)

    def _fit_action_signal(self, signal: np.ndarray) -> np.ndarray:
        fitted = np.zeros(self.action_count, dtype=float)
        fitted[: min(len(signal), self.action_count)] = signal[: min(len(signal), self.action_count)]
        if np.min(fitted) < 0.0:
            fitted = fitted - np.min(fitted)
        if np.max(fitted) > 1.0:
            fitted = fitted / (np.max(fitted) + 1e-9)
        return fitted

    def _seed_action_schemas(self) -> None:
        schemas = []
        preferences = []
        for action_index in range(self.action_count):
            base = self._schema_input(
                np.zeros(self.perceptual_width),
                np.zeros(self.value_width),
                self._one_hot(action_index),
                self._one_hot(action_index),
                self._one_hot(action_index),
            )
            schemas.append(base)
            preferences.append(self._one_hot(action_index))
        self.categories = np.asarray(schemas, dtype=float)
        self.action_preferences = np.asarray(preferences, dtype=float)

    def _add_category(self, x: np.ndarray) -> int:
        index = super()._add_category(x)
        action_slice_start = self.perceptual_width + self.value_width
        action_signal = np.zeros(self.action_count, dtype=float)
        for offset in range(3):
            start = action_slice_start + offset * self.action_count
            action_signal = np.maximum(action_signal, x[start : start + self.action_count])
        if np.sum(action_signal) <= 1e-9:
            action_signal += 1.0 / self.action_count
        action_signal = action_signal / (np.sum(action_signal) + 1e-9)
        if len(self.action_preferences) < len(self.categories):
            self.action_preferences = np.vstack([self.action_preferences, action_signal])
        return index

    def _action_category_bias(
        self,
        reactive_distribution: np.ndarray,
        imagined_action_prior: np.ndarray,
        sequence_bias: np.ndarray,
    ) -> np.ndarray:
        action_signal = self._fit_action_signal(reactive_distribution) + self._fit_action_signal(imagined_action_prior)
        action_signal += self._fit_action_signal(sequence_bias)
        if np.sum(action_signal) <= 1e-9:
            return np.zeros(len(self.categories), dtype=float)
        action_signal = action_signal / (np.sum(action_signal) + 1e-9)
        return self.action_preferences @ action_signal

    def _distribution_to_category_bias(self, distribution: np.ndarray | None) -> np.ndarray:
        if distribution is None or len(self.action_preferences) == 0:
            return np.zeros(len(self.categories), dtype=float)
        distribution = self._fit_action_signal(np.asarray(distribution, dtype=float))
        if np.sum(distribution) <= 1e-9:
            return np.zeros(len(self.categories), dtype=float)
        distribution = distribution / (np.sum(distribution) + 1e-9)
        return self.action_preferences @ distribution

    def _schema_to_action_distribution(self, schema_activation: np.ndarray) -> np.ndarray:
        activation = self._resize_activation(schema_activation)
        distribution = activation @ self.action_preferences[: len(activation)]
        if np.sum(distribution) <= 1e-9:
            distribution += 1.0 / self.action_count
        return distribution / (np.sum(distribution) + 1e-9)

    def _one_hot(self, index: int) -> np.ndarray:
        x = np.zeros(self.action_count, dtype=float)
        x[index % self.action_count] = 1.0
        return x

    def _one_hot_schema(self, action_index: int) -> np.ndarray:
        if len(self.categories) == 0:
            return np.empty(0, dtype=float)
        return self.action_preferences @ self._one_hot(action_index)


BasalGangliaActionSelection = ARTActionField
