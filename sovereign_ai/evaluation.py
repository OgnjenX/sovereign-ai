from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValueResult:
    value: float
    reward_component: float
    novelty_component: float
    prediction_component: float
    prediction_error: float


class ValueSystem:
    """Combines reward and novelty into a scalar state value."""

    def __init__(
        self,
        reward_weight: float = 0.75,
        novelty_weight: float = 0.25,
        prediction_weight: float = 0.35,
        context_weight: float = 0.1,
        learning_rate: float = 0.08,
        debug: bool = False,
    ) -> None:
        self.reward_weight = reward_weight
        self.novelty_weight = novelty_weight
        self.prediction_weight = prediction_weight
        self.context_weight = context_weight
        self.learning_rate = learning_rate
        self.debug = debug
        self.expected_value_weights = np.empty(0, dtype=float)

    def evaluate(
        self,
        category_activation: np.ndarray,
        x: np.ndarray,
        reward: float,
        novelty: float,
        context: float = 0.0,
        learn: bool = True,
    ) -> ValueResult:
        if len(self.expected_value_weights) != len(category_activation):
            resized = np.zeros(len(category_activation), dtype=float)
            resized[: min(len(resized), len(self.expected_value_weights))] = self.expected_value_weights[
                : min(len(resized), len(self.expected_value_weights))
            ]
            self.expected_value_weights = resized

        distributed_gain = float(np.linalg.norm(category_activation, ord=2))
        expected_value = float(category_activation @ self.expected_value_weights)
        prediction_error = reward - expected_value
        if learn:
            self.expected_value_weights += self.learning_rate * prediction_error * category_activation
        reward_component = self.reward_weight * reward
        novelty_component = self.novelty_weight * novelty * distributed_gain
        prediction_component = self.prediction_weight * expected_value
        context_component = self.context_weight * context
        value = reward_component + novelty_component + prediction_component + context_component
        if self.debug:
            print(
                "[value] reward="
                f"{reward:.3f} expected={expected_value:.3f} prediction_error={prediction_error:.3f} "
                f"novelty={novelty:.3f} context={context:.3f} value={value:.3f}"
            )
        return ValueResult(
            value,
            reward_component,
            novelty_component,
            prediction_component,
            prediction_error,
        )
