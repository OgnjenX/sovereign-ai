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
    goal_component: float


@dataclass(frozen=True)
class ValueState:
    result: ValueResult
    activation: np.ndarray
    scalar: float
    change: float


class ValueSystem:
    """Combines reward and novelty into a scalar state value."""

    def __init__(
        self,
        reward_weight: float = 0.75,
        novelty_weight: float = 0.25,
        prediction_weight: float = 0.35,
        context_weight: float = 0.1,
        goal_weight: float = 0.4,
        learning_rate: float = 0.08,
        debug: bool = False,
    ) -> None:
        self.reward_weight = reward_weight
        self.novelty_weight = novelty_weight
        self.prediction_weight = prediction_weight
        self.context_weight = context_weight
        self.goal_weight = goal_weight
        self.learning_rate = learning_rate
        self.debug = debug
        self.expected_value_weights = np.empty(0, dtype=float)

    def evaluate(
        self,
        category_activation: np.ndarray,
        reward: float,
        novelty: float,
        context: float = 0.0,
        goal_alignment: float = 0.0,
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
        goal_component = self.goal_weight * goal_alignment
        value = reward_component + novelty_component + prediction_component + context_component + goal_component
        if self.debug:
            print(
                "[value] reward="
                f"{reward:.3f} expected={expected_value:.3f} prediction_error={prediction_error:.3f} "
                f"novelty={novelty:.3f} context={context:.3f} goal={goal_alignment:.3f} value={value:.3f}"
            )
        return ValueResult(
            value,
            reward_component,
            novelty_component,
            prediction_component,
            prediction_error,
            goal_component,
        )

    def update_state(
        self,
        category_activation: np.ndarray,
        reward: float,
        novelty: float,
        context: float,
        goal_alignment: float,
        action_distribution: np.ndarray,
        previous_state: np.ndarray | None = None,
        learn: bool = False,
    ) -> ValueState:
        result = self.evaluate(
            category_activation,
            reward,
            novelty,
            context,
            goal_alignment,
            learn=learn,
        )
        action_confidence = float(np.linalg.norm(action_distribution, ord=2))
        raw = np.array(
            [
                max(0.0, result.reward_component),
                max(0.0, result.novelty_component),
                max(0.0, result.prediction_component),
                max(0.0, self.context_weight * context),
                max(0.0, result.goal_component),
                action_confidence,
            ],
            dtype=float,
        )
        excitation = raw / (np.sum(raw) + 1e-9)
        inhibition = 1.0 - excitation
        previous = np.zeros_like(excitation) if previous_state is None else np.asarray(previous_state, dtype=float)
        if len(previous) != len(excitation):
            resized = np.zeros_like(excitation)
            resized[: min(len(previous), len(resized))] = previous[: min(len(previous), len(resized))]
            previous = resized
        if np.sum(previous) <= 1e-9:
            previous += 1.0 / len(previous)
        previous = previous / (np.sum(previous) + 1e-9)
        updated = np.clip(previous + 0.45 * (excitation * (1.0 - previous) - inhibition * previous), 0.0, 1.0)
        updated = updated / (np.sum(updated) + 1e-9)
        scalar = float(updated @ np.array([1.0, 0.45, 0.7, 0.25, 0.65, 0.35]))
        change = float(np.linalg.norm(updated - previous))
        if self.debug:
            print(
                "[value-dyn] scalar="
                f"{scalar:.3f} change={change:.4f} activation={np.round(updated, 3)}"
            )
        return ValueState(result, updated, scalar, change)

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
        result = self.evaluate(
            category_activation,
            reward=reward,
            novelty=novelty,
            context=context,
            goal_alignment=goal_alignment,
            learn=learn,
        )
        if len(slots) == 0:
            return result

        slot_norms = np.linalg.norm(slots, axis=1)
        slot_gain = float(np.mean(slot_norms))
        slot_value = self.novelty_weight * novelty * slot_gain + self.goal_weight * goal_alignment
        return ValueResult(
            result.value + slot_value,
            result.reward_component,
            result.novelty_component + self.novelty_weight * novelty * slot_gain,
            result.prediction_component,
            result.prediction_error,
            result.goal_component + self.goal_weight * goal_alignment,
        )
