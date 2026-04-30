from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sovereign_ai.art_field import ARTField


@dataclass(frozen=True)
class ValueResult:
    value: float
    reward_component: float
    novelty_component: float
    prediction_component: float
    prediction_error: float
    goal_component: float


@dataclass(frozen=True)
class ValueState:
    result: ValueResult
    activation: np.ndarray
    scalar: float
    change: float


def compute_value_state(
    field: "ARTValueField",
    category_activation: np.ndarray,
    reward: float,
    novelty: float,
    context: float,
    goal_alignment: float,
    action_distribution: np.ndarray,
    previous_state: np.ndarray | None = None,
    learn: bool = False,
) -> ValueState:
    x = field._value_input(category_activation, reward, novelty, context, goal_alignment, action_distribution)
    expected_value = field._expected_value(category_activation)
    prediction_error = reward - expected_value
    vigilance_modulation = float(np.clip(0.08 * novelty + 0.06 * abs(prediction_error), -0.08, 0.18))
    state = ARTField.update_state(
        field,
        x,
        previous_activation=previous_state,
        vigilance_modulation=vigilance_modulation,
        learn=learn,
    )
    if learn:
        field._learn_expected_value(category_activation, prediction_error)

    activation = state.result.category_activation
    scalar = float(activation @ field.context_values[: len(activation)])
    reward_component = float(activation[0] if len(activation) > 0 else 0.0) * reward
    novelty_component = float(activation[1] if len(activation) > 1 else 0.0) * novelty
    prediction_component = float(activation[2] if len(activation) > 2 else 0.0) * expected_value
    goal_component = float(activation[4] if len(activation) > 4 else 0.0) * goal_alignment
    result = ValueResult(
        scalar + reward_component + novelty_component + prediction_component + goal_component,
        reward_component,
        novelty_component,
        prediction_component,
        prediction_error,
        goal_component,
    )
    if field.debug:
        print(
            "[value-dyn] category="
            f"{state.result.category_index} scalar={result.value:.3f} change={state.change:.4f} "
            f"vigilance={state.result.effective_vigilance:.3f} activation={np.round(activation, 3)}"
        )
    return ValueState(result, activation, result.value, state.change)


class ARTValueField(ARTField):
    """ART value field whose categories are value contexts."""

    def __init__(
        self,
        max_perceptual_categories: int = 16,
        context_count: int = 6,
        vigilance: float = 0.62,
        learning_rate: float = 0.08,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        self.perceptual_width = max_perceptual_categories
        self.feature_width = max_perceptual_categories + 5
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
        self.context_values = np.array([0.85, 0.45, 0.25, -0.55, 0.65, -0.25], dtype=float)[:context_count]
        self.expected_value_weights = np.zeros(max_perceptual_categories, dtype=float)
        self.categories = self._initial_value_contexts(context_count)

    def evaluate(
        self,
        category_activation: np.ndarray,
        reward: float,
        novelty: float,
        context: float = 0.0,
        goal_alignment: float = 0.0,
        learn: bool = True,
    ) -> ValueResult:
        state = compute_value_state(
            self,
            category_activation,
            reward=reward,
            novelty=novelty,
            context=context,
            goal_alignment=goal_alignment,
            action_distribution=np.empty(0, dtype=float),
            learn=learn,
        )
        return state.result


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
        slot_signal = float(np.mean(np.linalg.norm(slots, axis=1))) if len(slots) else 0.0
        return self.evaluate(
            category_activation,
            reward=reward,
            novelty=float(np.clip(novelty + slot_signal, 0.0, 1.0)),
            context=context,
            goal_alignment=goal_alignment,
            learn=learn,
        )

    def vigilance_signal(self, activation: np.ndarray) -> float:
        activation = np.asarray(activation, dtype=float)
        if len(activation) == 0:
            return 0.0
        risk = activation[3] if len(activation) > 3 else 0.0
        novelty = activation[1] if len(activation) > 1 else 0.0
        reward = activation[0] if len(activation) > 0 else 0.0
        return float(np.clip(0.12 * risk + 0.08 * novelty - 0.05 * reward, -0.08, 0.18))

    def category_preference(self, size: int) -> np.ndarray:
        if len(self.expected_value_weights) < size:
            self.expected_value_weights = np.pad(self.expected_value_weights, (0, size - len(self.expected_value_weights)))
        preference = np.maximum(self.expected_value_weights[:size], 0.0)
        if np.sum(preference) <= 1e-9:
            return np.zeros(size, dtype=float)
        return preference / (np.sum(preference) + 1e-9)

    def _value_input(
        self,
        category_activation: np.ndarray,
        reward: float,
        novelty: float,
        context: float,
        goal_alignment: float,
        action_distribution: np.ndarray,
    ) -> np.ndarray:
        x = np.zeros(self.feature_width, dtype=float)
        category_activation = np.asarray(category_activation, dtype=float)
        width = min(len(category_activation), self.perceptual_width)
        x[:width] = category_activation[:width]
        action_confidence = float(np.linalg.norm(action_distribution, ord=2)) if len(action_distribution) else 0.0
        tail = np.array(
            [
                np.clip((reward + 1.0) * 0.5, 0.0, 1.0),
                np.clip(novelty, 0.0, 1.0),
                np.clip((context + 1.0) * 0.5, 0.0, 1.0),
                np.clip((goal_alignment + 1.0) * 0.5, 0.0, 1.0),
                np.clip(action_confidence, 0.0, 1.0),
            ],
            dtype=float,
        )
        x[self.perceptual_width :] = tail
        return x

    def _expected_value(self, category_activation: np.ndarray) -> float:
        category_activation = np.asarray(category_activation, dtype=float)
        if len(self.expected_value_weights) < len(category_activation):
            self.expected_value_weights = np.pad(
                self.expected_value_weights,
                (0, len(category_activation) - len(self.expected_value_weights)),
            )
        return float(category_activation @ self.expected_value_weights[: len(category_activation)])

    def _learn_expected_value(self, category_activation: np.ndarray, prediction_error: float) -> None:
        category_activation = np.asarray(category_activation, dtype=float)
        self.expected_value_weights[: len(category_activation)] += self.learning_rate * prediction_error * category_activation

    def _initial_value_contexts(self, count: int) -> np.ndarray:
        contexts = np.zeros((count, self.feature_width), dtype=float)
        for index in range(count):
            contexts[index, self.perceptual_width :] = 0.45
        if count > 0:
            contexts[0, self.perceptual_width] = 1.0
        if count > 1:
            contexts[1, self.perceptual_width + 1] = 1.0
        if count > 2:
            contexts[2, self.perceptual_width + 2] = 0.85
        if count > 3:
            contexts[3, self.perceptual_width] = 0.0
            contexts[3, self.perceptual_width + 4] = 0.75
        if count > 4:
            contexts[4, self.perceptual_width + 3] = 1.0
        if count > 5:
            contexts[5, self.perceptual_width + 1] = 0.7
            contexts[5, self.perceptual_width + 3] = 0.2
        return contexts


ValueSystem = ARTValueField
