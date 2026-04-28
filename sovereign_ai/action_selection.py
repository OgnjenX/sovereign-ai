from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sovereign_ai.utils import softmax


@dataclass(frozen=True)
class ActionResult:
    action_index: int
    action_distribution: np.ndarray
    go: np.ndarray
    stop: np.ndarray
    drives: np.ndarray
    pathway: str


class BasalGangliaActionSelection:
    """GO/STOP competitive action selection over distributed category activations."""

    def __init__(
        self,
        max_categories: int,
        action_count: int,
        temperature: float = 0.35,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        self.action_count = action_count
        self.temperature = temperature
        self.debug = debug
        rng = np.random.default_rng(seed)
        self.category_action_weights = rng.normal(0.0, 0.08, (max_categories, action_count))
        self.value_affordance = rng.normal(0.15, 0.05, action_count)

    def select(
        self,
        category_activation: np.ndarray,
        value: float,
        salience: np.ndarray,
        imagined_action_prior: np.ndarray | None = None,
        sequence_bias: np.ndarray | None = None,
        pathway: str = "planned",
    ) -> ActionResult:
        category_activation = np.asarray(category_activation, dtype=float)
        salience = np.asarray(salience, dtype=float)
        active_weights = self.category_action_weights[: len(category_activation)]
        imagined = np.zeros(self.action_count) if imagined_action_prior is None else imagined_action_prior
        sequence = np.zeros(self.action_count) if sequence_bias is None else sequence_bias
        drives = (
            category_activation @ active_weights
            + value * self.value_affordance
            + salience
            + imagined
            + sequence
        )
        go = softmax(drives, self.temperature)
        stop = softmax(-drives, self.temperature)
        gated = go * (1.0 - stop)
        action_distribution = gated / (np.sum(gated) + 1e-9)
        action_index = int(np.argmax(action_distribution))
        if self.debug:
            print(
                "[action] pathway="
                f"{pathway} action={action_index} drives={np.round(drives, 3)} "
                f"go={np.round(go, 3)} stop={np.round(stop, 3)}"
            )
        return ActionResult(action_index, action_distribution, go, stop, drives, pathway)

    def learn_action(
        self,
        category_activation: np.ndarray,
        action_index: int,
        reward_prediction_error: float,
        learning_rate: float = 0.08,
    ) -> None:
        active_weights = self.category_action_weights[: len(category_activation)]
        target = np.zeros(self.action_count)
        target[action_index] = reward_prediction_error
        active_weights += learning_rate * np.outer(category_activation, target)
        if self.debug:
            print(
                "[action-learning] action="
                f"{action_index} prediction_error={reward_prediction_error:.3f}"
            )
