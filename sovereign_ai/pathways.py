from __future__ import annotations

import numpy as np

from sovereign_ai.action_selection import ActionResult, BasalGangliaActionSelection
from sovereign_ai.utils import softmax


class ReactivePathway:
    """Fast direct input-to-action competition."""

    def __init__(
        self,
        input_dim: int,
        action_count: int,
        temperature: float = 0.3,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        self.temperature = temperature
        self.debug = debug
        rng = np.random.default_rng(seed)
        self.input_action_weights = rng.normal(0.0, 0.12, (input_dim, action_count))

    def select(self, x: np.ndarray, salience: np.ndarray) -> ActionResult:
        drives = x @ self.input_action_weights + salience
        go = softmax(drives, self.temperature)
        stop = softmax(-drives, self.temperature)
        gated = go * (1.0 - stop)
        action_distribution = gated / (np.sum(gated) + 1e-9)
        action_index = int(np.argmax(action_distribution))
        if self.debug:
            print(
                "[action] pathway=reactive "
                f"action={action_index} drives={np.round(drives, 3)}"
            )
        return ActionResult(action_index, action_distribution, go, stop, drives, "reactive")


class PlannedPathway:
    """Slower perception-evaluation-action pathway."""

    def __init__(self, action_selector: BasalGangliaActionSelection) -> None:
        self.action_selector = action_selector

    def select(
        self,
        category_activation: np.ndarray,
        value: float,
        salience: np.ndarray,
        imagined_action_prior: np.ndarray | None = None,
        sequence_bias: np.ndarray | None = None,
    ) -> ActionResult:
        return self.action_selector.select(
            category_activation,
            value,
            salience,
            imagined_action_prior,
            sequence_bias,
            "planned",
        )


class PathwayGate:
    def __init__(self, urgency_threshold: float = 0.68, debug: bool = False) -> None:
        self.urgency_threshold = urgency_threshold
        self.debug = debug

    def weights(self, urgency: float) -> tuple[float, float]:
        reactive_weight = float(1.0 / (1.0 + np.exp(-12.0 * (urgency - self.urgency_threshold))))
        planned_weight = 1.0 - reactive_weight
        if self.debug:
            print(
                f"[gate] urgency={urgency:.3f} "
                f"reactive_weight={reactive_weight:.3f} planned_weight={planned_weight:.3f}"
            )
        return reactive_weight, planned_weight
