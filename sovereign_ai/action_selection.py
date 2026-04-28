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


@dataclass(frozen=True)
class ActionState:
    result: ActionResult
    change: float


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

    def update_state(
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
        category_activation = np.asarray(category_activation, dtype=float)
        active_weights = self.category_action_weights[: len(category_activation)]
        learned_affordance = category_activation @ active_weights
        value_gain = float(value_state @ np.array([0.8, 0.35, 0.5, 0.2, 0.3]))
        components = np.vstack(
            [
                softmax(learned_affordance, self.temperature),
                softmax(value_gain * self.value_affordance, self.temperature),
                softmax(salience, self.temperature),
                softmax(imagined_action_prior, self.temperature),
                softmax(sequence_bias, self.temperature),
                reactive_distribution,
            ]
        )
        excitation = np.mean(components, axis=0)
        inhibition = softmax(-excitation, self.temperature)
        previous = (
            np.ones(self.action_count, dtype=float) / self.action_count
            if previous_distribution is None
            else np.asarray(previous_distribution, dtype=float)
        )
        previous = previous / (np.sum(previous) + 1e-9)
        updated = np.clip(previous + 0.5 * (excitation * (1.0 - previous) - inhibition * previous), 0.0, 1.0)
        action_distribution = updated / (np.sum(updated) + 1e-9)
        action_index = int(np.argmax(action_distribution))
        drives = excitation - inhibition
        go = softmax(excitation, self.temperature)
        stop = softmax(inhibition, self.temperature)
        change = float(np.linalg.norm(action_distribution - previous))
        if self.debug:
            print(
                "[action-dyn] action="
                f"{action_index} change={change:.4f} excitation={np.round(excitation, 3)} "
                f"inhibition={np.round(inhibition, 3)}"
            )
        return ActionState(
            ActionResult(action_index, action_distribution, go, stop, drives, pathway),
            change,
        )

    def learn_action(
        self,
        category_activation: np.ndarray,
        action_index: int,
        reward_prediction_error: float,
        learning_rate: float = 0.08,
    ) -> None:
        target = np.zeros(self.action_count)
        target[action_index] = reward_prediction_error
        self.category_action_weights[: len(category_activation)] += learning_rate * np.outer(
            category_activation,
            target,
        )
        if self.debug:
            print(
                "[action-learning] action="
                f"{action_index} prediction_error={reward_prediction_error:.3f}"
            )
