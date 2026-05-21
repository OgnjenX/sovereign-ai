"""Value-association ART field and compatibility helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sovereign_ai.art_field import ARTField


@dataclass(frozen=True)
class ValueResult:
    """Scalar value prediction and decomposition of its components."""

    value: float
    reward_component: float
    novelty_component: float
    prediction_component: float
    prediction_error: float
    goal_component: float


@dataclass(frozen=True)
class ValueState:
    """Value field state after resonance and optional learning."""

    result: ValueResult
    activation: np.ndarray
    scalar: float
    change: float


class ARTValueField(ARTField):
    """ART field with learned value associations, not fixed value contexts."""

    def __init__(
        self,
        max_perceptual_categories: int = 16,
        context_count: int = 12,
        vigilance: float = 0.62,
        learning_rate: float = 0.08,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the value field and its learned association tables."""

        self.perceptual_width = max_perceptual_categories
        self.feature_width = max_perceptual_categories + 6
        super().__init__(
            input_dim=self.feature_width,
            max_categories=context_count,
            vigilance=vigilance,
            competition_temperature=0.22,
            learning_rate=learning_rate,
            seed=seed,
            debug=debug,
            name="value",
        )
        self.category_values = np.empty(0, dtype=float)
        self.category_vigilance = np.empty(0, dtype=float)
        self.attentional_preferences = np.empty(
            (0, max_perceptual_categories),
            dtype=float,
        )
        self.expected_value_weights = np.zeros(max_perceptual_categories, dtype=float)

    def evaluate(
        self,
        category_activation: np.ndarray,
        reward: float,
        novelty: float,
        context: float = 0.0,
        goal_alignment: float = 0.0,
        learn: bool = True,
    ) -> ValueResult:
        """Evaluate a category activation against reward and novelty."""

        return self.resonate_value(
            category_activation,
            reward=reward,
            novelty=novelty,
            context=context,
            goal_alignment=goal_alignment,
            action_distribution=np.empty(0, dtype=float),
            learn=learn,
        ).result

    def resonate_value(
        self,
        category_activation: np.ndarray,
        reward: float,
        novelty: float,
        context: float,
        goal_alignment: float,
        action_distribution: np.ndarray,
        previous_state: np.ndarray | None = None,
        category_bias: np.ndarray | None = None,
        vigilance_modulation: float = 0.0,
        learn: bool = False,
    ) -> ValueState:
        """Resonate the value field and optionally update learned weights."""

        expected_value = self._expected_value(category_activation)
        prediction_error = reward - expected_value
        x = self._value_input(
            category_activation,
            reward,
            novelty,
            context,
            goal_alignment,
            action_distribution,
            prediction_error,
        )
        state = super().update_state(
            x,
            previous_activation=previous_state,
            category_bias=category_bias,
            vigilance_modulation=vigilance_modulation,
            learn=learn,
        )
        self._ensure_associations()
        activation = state.result.category_activation
        scalar = (
            float(activation @ self.category_values[: len(activation)])
            if len(activation)
            else 0.0
        )
        if learn:
            self._learn_value_associations(
                activation, category_activation, reward, prediction_error, novelty
            )
            scalar = (
                float(activation @ self.category_values[: len(activation)])
                if len(activation)
                else 0.0
            )
            self._learn_expected_value(category_activation, prediction_error)
        result = ValueResult(
            scalar,
            reward,
            novelty,
            expected_value,
            prediction_error,
            goal_alignment,
        )
        if self.debug:
            print(
                "[value-dyn] category="
                f"{state.result.category_index} scalar={scalar:.3f} "
                f"change={state.change:.4f} vigilance={state.result.effective_vigilance:.3f} "
                f"search={state.result.search_path} "
                f"assoc={np.round(self.category_values[:len(activation)], 3)}"
            )
        return ValueState(result, activation, scalar, state.change)

    def evaluate_slots(
        self,
        category_activation: np.ndarray,
        slots: np.ndarray,
        reward: float,
        novelty: float,
        context: float = 0.0,
        goal_alignment: float = 0.0,
        learn: bool = False,
    ) -> ValueResult:
        """Evaluate a value state while incorporating slot-wise novelty."""

        slot_signal = (
            float(np.mean(np.linalg.norm(slots, axis=1))) if len(slots) else 0.0
        )
        return self.evaluate(
            category_activation,
            reward=reward,
            novelty=float(np.clip(novelty + slot_signal, 0.0, 1.0)),
            context=context,
            goal_alignment=goal_alignment,
            learn=learn,
        )

    def vigilance_signal(self, activation: np.ndarray | None = None) -> float:
        """Return a bounded vigilance adjustment based on activation."""

        self._ensure_associations()
        if activation is None:
            activation = (
                self.last_result.category_activation
                if self.last_result is not None
                else np.empty(0)
            )
        activation = self._fit_activation(activation, len(self.category_vigilance))
        if len(activation) == 0:
            return 0.0
        return float(np.clip(activation @ self.category_vigilance, -0.2, 0.2))

    def category_preference(
        self, size: int, activation: np.ndarray | None = None
    ) -> np.ndarray:
        """Return a normalized preference vector for perceptual categories."""

        self._ensure_associations()
        if len(self.attentional_preferences) == 0:
            return np.zeros(size, dtype=float)
        if activation is None:
            activation = (
                self.last_result.category_activation
                if self.last_result is not None
                else np.empty(0)
            )
        activation = self._fit_activation(activation, len(self.attentional_preferences))
        preference = activation @ self.attentional_preferences[: len(activation)]
        fitted = np.zeros(size, dtype=float)
        fitted[: min(size, len(preference))] = preference[: min(size, len(preference))]
        if np.sum(fitted) <= 1e-9:
            return fitted
        return fitted / (np.sum(fitted) + 1e-9)

    def _value_input(
        self,
        category_activation: np.ndarray,
        reward: float,
        novelty: float,
        context: float,
        goal_alignment: float,
        action_distribution: np.ndarray,
        prediction_error: float,
    ) -> np.ndarray:
        """Assemble the feature vector used for value resonance."""

        x = np.zeros(self.feature_width, dtype=float)
        category_activation = np.asarray(category_activation, dtype=float)
        width = min(len(category_activation), self.perceptual_width)
        x[:width] = category_activation[:width]
        action_confidence = (
            float(np.linalg.norm(action_distribution, ord=2))
            if len(action_distribution)
            else 0.0
        )
        x[self.perceptual_width :] = np.array(
            [
                np.clip((reward + 1.0) * 0.5, 0.0, 1.0),
                np.clip(novelty, 0.0, 1.0),
                np.clip((context + 1.0) * 0.5, 0.0, 1.0),
                np.clip((goal_alignment + 1.0) * 0.5, 0.0, 1.0),
                np.clip(action_confidence, 0.0, 1.0),
                np.clip((prediction_error + 1.0) * 0.5, 0.0, 1.0),
            ],
            dtype=float,
        )
        return x

    def _add_category(self, x: np.ndarray) -> int:
        """Create a new value category and keep its tables in sync."""

        index = super()._add_category(x)
        self._ensure_associations()
        return index

    def _ensure_associations(self) -> None:
        """Resize learned value tables to match the current category count."""

        count = len(self.categories)
        if len(self.category_values) < count:
            self.category_values = np.pad(
                self.category_values, (0, count - len(self.category_values))
            )
        if len(self.category_vigilance) < count:
            self.category_vigilance = np.pad(
                self.category_vigilance, (0, count - len(self.category_vigilance))
            )
        if self.attentional_preferences.shape != (count, self.perceptual_width):
            resized = np.zeros((count, self.perceptual_width), dtype=float)
            rows = min(count, self.attentional_preferences.shape[0])
            cols = min(
                self.perceptual_width,
                (
                    self.attentional_preferences.shape[1]
                    if self.attentional_preferences.ndim == 2
                    else 0
                ),
            )
            if rows and cols:
                resized[:rows, :cols] = self.attentional_preferences[:rows, :cols]
            self.attentional_preferences = resized

    def _learn_value_associations(
        self,
        value_activation: np.ndarray,
        perceptual_activation: np.ndarray,
        reward: float,
        prediction_error: float,
        novelty: float,
    ) -> None:
        """Update category value, vigilance, and attentional preferences."""

        self._ensure_associations()
        value_activation = self._fit_activation(
            value_activation, len(self.category_values)
        )
        perceptual = self._fit_activation(perceptual_activation, self.perceptual_width)
        outcome = float(np.clip(reward + 0.25 * prediction_error, -1.0, 1.0))
        self.category_values += (
            self.learning_rate * value_activation * (outcome - self.category_values)
        )
        vigilance_target = float(np.clip(max(abs(prediction_error), novelty), 0.0, 1.0))
        self.category_vigilance += (
            self.learning_rate
            * value_activation
            * (vigilance_target - self.category_vigilance)
        )
        self.attentional_preferences += (
            self.learning_rate
            * value_activation[:, None]
            * (perceptual[None, :] - self.attentional_preferences)
        )

    def _expected_value(self, category_activation: np.ndarray) -> float:
        """Estimate expected value from learned category weights."""

        category_activation = np.asarray(category_activation, dtype=float)
        if len(self.expected_value_weights) < len(category_activation):
            self.expected_value_weights = np.pad(
                self.expected_value_weights,
                (0, len(category_activation) - len(self.expected_value_weights)),
            )
        return float(
            category_activation
            @ self.expected_value_weights[: len(category_activation)]
        )

    def _learn_expected_value(
        self, category_activation: np.ndarray, prediction_error: float
    ) -> None:
        """Update the expected-value weights using prediction error."""

        category_activation = np.asarray(category_activation, dtype=float)
        if len(self.expected_value_weights) < len(category_activation):
            self.expected_value_weights = np.pad(
                self.expected_value_weights,
                (0, len(category_activation) - len(self.expected_value_weights)),
            )
        self.expected_value_weights[: len(category_activation)] += (
            self.learning_rate * prediction_error * category_activation
        )

    def _fit_activation(self, activation: np.ndarray, size: int) -> np.ndarray:
        """Pad or trim an activation vector and normalize if non-zero."""

        fitted = np.zeros(size, dtype=float)
        activation = np.asarray(activation, dtype=float)
        fitted[: min(size, len(activation))] = activation[: min(size, len(activation))]
        if np.sum(fitted) <= 1e-9:
            return fitted
        return fitted / (np.sum(fitted) + 1e-9)


def compute_value_state(
    field: ARTValueField,
    category_activation: np.ndarray,
    reward: float,
    novelty: float,
    context: float,
    goal_alignment: float,
    action_distribution: np.ndarray,
    previous_state: np.ndarray | None = None,
    learn: bool = False,
) -> ValueState:
    """Convenience wrapper returning a value field state."""

    return field.resonate_value(
        category_activation,
        reward=reward,
        novelty=novelty,
        context=context,
        goal_alignment=goal_alignment,
        action_distribution=action_distribution,
        previous_state=previous_state,
        learn=learn,
    )


class ValueSystem(ARTValueField):
    """Compatibility wrapper preserving the old no-argument constructor."""
