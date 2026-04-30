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
    """ART field whose sensorimotor action schemas are learned from experience."""

    def __init__(
        self,
        max_categories: int,
        action_count: int,
        value_width: int = 12,
        goal_width: int | None = None,
        temporal_width: int = 8,
        vigilance: float = 0.58,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        self.perceptual_width = max_categories
        self.action_count = action_count
        self.value_width = value_width
        self.goal_width = max_categories if goal_width is None else goal_width
        self.temporal_width = temporal_width
        self.schema_width = (
            self.perceptual_width
            + self.goal_width
            + self.value_width
            + self.temporal_width
            + self.action_count
            + 1
        )
        super().__init__(
            input_dim=self.schema_width,
            max_categories=max(max_categories * 2, action_count * 4),
            vigilance=vigilance,
            competition_temperature=0.24,
            learning_rate=0.08,
            seed=seed,
            debug=debug,
            name="action",
        )
        self.action_associations = np.empty((0, action_count), dtype=float)
        self.outcome_associations = np.empty(0, dtype=float)
        self.exploration_cursor = 0

    @property
    def category_action_weights(self) -> np.ndarray:
        return self.action_associations

    @property
    def action_preferences(self) -> np.ndarray:
        return self.action_associations

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
        value_state[0] = np.clip((value + 1.0) * 0.5, 0.0, 1.0)
        temporal = sequence_bias if sequence_bias is not None else np.empty(0)
        context = self.schema_input(
            category_activation,
            goal_activation=np.empty(0),
            value_activation=value_state,
            temporal_activation=temporal,
            action_distribution=np.empty(0),
        )
        state = self.resonate_action(
            context,
            exploratory_signal=salience if imagined_action_prior is None else imagined_action_prior,
            pathway=pathway,
        )
        return state.result

    def resonate_action(
        self,
        context_input: np.ndarray,
        category_bias: np.ndarray | None = None,
        vigilance_modulation: float = 0.0,
        exploratory_signal: np.ndarray | None = None,
        previous_distribution: np.ndarray | None = None,
        pathway: str = "coupled",
    ) -> ActionState:
        previous = self._fit_action(previous_distribution)
        if len(self.categories) == 0:
            action = self._exploratory_action(exploratory_signal)
            schema = self._schema_with_action(context_input, self._one_hot(action))
            result = self.process(schema, learn=True)
            self._ensure_associations()
            self.action_associations[result.category_index] = self._one_hot(action)
        else:
            result = self.process(
                context_input,
                category_bias=category_bias,
                vigilance_modulation=vigilance_modulation,
                learn=False,
            )
        self._ensure_associations()
        distribution = self._schema_to_action_distribution(result.category_activation, exploratory_signal)
        action_index = int(np.argmax(distribution))
        go = distribution
        stop = softmax(1.0 - distribution, temperature=0.35)
        drives = distribution - stop
        change = float(np.linalg.norm(distribution - previous))
        if self.debug:
            print(
                "[action-dyn] category="
                f"{result.category_index} action={action_index} resonance={result.resonance} "
                f"search={result.search_path} vigilance={result.effective_vigilance:.3f}"
            )
        return ActionState(ActionResult(action_index, distribution, go, stop, drives, pathway), change)

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

    def schema_input(
        self,
        perceptual_activation: np.ndarray,
        goal_activation: np.ndarray,
        value_activation: np.ndarray,
        temporal_activation: np.ndarray,
        action_distribution: np.ndarray,
        outcome: float = 0.0,
    ) -> np.ndarray:
        x = np.zeros(self.schema_width, dtype=float)
        cursor = 0
        cursor = self._write_slice(x, cursor, perceptual_activation, self.perceptual_width)
        cursor = self._write_slice(x, cursor, goal_activation, self.goal_width)
        cursor = self._write_slice(x, cursor, value_activation, self.value_width)
        cursor = self._write_slice(x, cursor, temporal_activation, self.temporal_width)
        cursor = self._write_slice(x, cursor, action_distribution, self.action_count)
        x[cursor] = np.clip((outcome + 1.0) * 0.5, 0.0, 1.0)
        return x

    def learn_action(
        self,
        category_activation: np.ndarray,
        action_index: int,
        reward_prediction_error: float,
        learning_rate: float = 0.08,
        goal_activation: np.ndarray | None = None,
        value_activation: np.ndarray | None = None,
        temporal_activation: np.ndarray | None = None,
    ) -> None:
        schema = self.schema_input(
            category_activation,
            np.empty(0) if goal_activation is None else goal_activation,
            np.empty(0) if value_activation is None else value_activation,
            np.empty(0) if temporal_activation is None else temporal_activation,
            self._one_hot(action_index),
            outcome=reward_prediction_error,
        )
        result = self.process(schema, category_bias=self._action_category_bias(self._one_hot(action_index)), learn=True)
        self._ensure_associations()
        action_target = self._one_hot(action_index)
        outcome = float(np.clip(reward_prediction_error, -1.0, 1.0))
        self.action_associations[result.category_index] += learning_rate * (
            action_target - self.action_associations[result.category_index]
        )
        if outcome < 0.0:
            self.action_associations[result.category_index, action_index] *= max(0.0, 1.0 + outcome * learning_rate)
        self.action_associations[result.category_index] = self._fit_action(self.action_associations[result.category_index])
        self.outcome_associations[result.category_index] += learning_rate * (
            outcome - self.outcome_associations[result.category_index]
        )
        if self.debug:
            print(
                "[action-learning] schema="
                f"{result.category_index} action={action_index} prediction_error={reward_prediction_error:.3f}"
            )

    def slot_action_bias(self, slots: np.ndarray, category_activation: np.ndarray) -> np.ndarray:
        if len(self.action_associations) == 0:
            return np.zeros(self.action_count, dtype=float)
        schema_bias = self._resize_activation(category_activation)
        return self._schema_to_action_distribution(schema_bias)

    def _add_category(self, x: np.ndarray) -> int:
        index = super()._add_category(x)
        self._ensure_associations()
        action_start = self.perceptual_width + self.goal_width + self.value_width + self.temporal_width
        action = self._fit_action(x[action_start : action_start + self.action_count])
        if np.sum(action) <= 1e-9:
            action = self._one_hot(self.exploration_cursor)
        self.action_associations[index] = action
        return index

    def _ensure_associations(self) -> None:
        count = len(self.categories)
        if self.action_associations.shape != (count, self.action_count):
            resized = np.zeros((count, self.action_count), dtype=float)
            rows = min(count, self.action_associations.shape[0])
            if rows:
                resized[:rows] = self.action_associations[:rows]
            self.action_associations = resized
        if len(self.outcome_associations) < count:
            self.outcome_associations = np.pad(self.outcome_associations, (0, count - len(self.outcome_associations)))

    def _schema_with_action(self, context_input: np.ndarray, action_distribution: np.ndarray) -> np.ndarray:
        schema = np.asarray(context_input, dtype=float).copy()
        action_start = self.perceptual_width + self.goal_width + self.value_width + self.temporal_width
        schema[action_start : action_start + self.action_count] = self._fit_action(action_distribution)
        return schema

    def _schema_to_action_distribution(
        self,
        schema_activation: np.ndarray,
        exploratory_signal: np.ndarray | None = None,
    ) -> np.ndarray:
        self._ensure_associations()
        if len(self.action_associations) == 0:
            action = self._one_hot(self._exploratory_action(exploratory_signal))
            return action
        activation = self._resize_activation(schema_activation)
        distribution = activation @ self.action_associations[: len(activation)]
        if np.sum(distribution) <= 1e-9:
            distribution = self._one_hot(self._exploratory_action(exploratory_signal))
        return self._fit_action(distribution)

    def _action_category_bias(self, action_distribution: np.ndarray) -> np.ndarray:
        self._ensure_associations()
        if len(self.action_associations) == 0:
            return np.empty(0, dtype=float)
        action = self._fit_action(action_distribution)
        bias = self.action_associations @ action
        if np.sum(bias) <= 1e-9:
            return np.zeros(len(self.action_associations), dtype=float)
        return bias / (np.sum(bias) + 1e-9)

    def _exploratory_action(self, signal: np.ndarray | None = None) -> int:
        if signal is not None:
            fitted = self._fit_action(signal)
            if np.sum(fitted) > 1e-9:
                return int(np.argmax(fitted))
        action = self.exploration_cursor % self.action_count
        self.exploration_cursor += 1
        return action

    def _fit_action(self, signal: np.ndarray | None) -> np.ndarray:
        fitted = np.zeros(self.action_count, dtype=float)
        if signal is not None:
            signal = np.asarray(signal, dtype=float)
            fitted[: min(self.action_count, len(signal))] = signal[: min(self.action_count, len(signal))]
        if np.min(fitted) < 0.0:
            fitted = fitted - np.min(fitted)
        if np.sum(fitted) <= 1e-9:
            return fitted
        return fitted / (np.sum(fitted) + 1e-9)

    def _one_hot(self, index: int) -> np.ndarray:
        x = np.zeros(self.action_count, dtype=float)
        x[index % self.action_count] = 1.0
        return x

    def _write_slice(self, target: np.ndarray, cursor: int, values: np.ndarray, width: int) -> int:
        values = np.asarray(values, dtype=float)
        target[cursor : cursor + min(width, len(values))] = values[: min(width, len(values))]
        return cursor + width


class BasalGangliaActionSelection(ARTActionField):
    """Compatibility wrapper preserving the old constructor name."""

    pass
