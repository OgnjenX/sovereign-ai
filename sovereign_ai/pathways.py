from __future__ import annotations

import numpy as np

from sovereign_ai.action_selection import ARTActionField, ActionResult


class ReactivePathway:
    """Compatibility adapter for older callers.

    Direct input-to-action scoring has been removed. New code should use
    ARTActionField inside the coupled architecture.
    """

    def __init__(
        self,
        input_dim: int,
        action_count: int,
        temperature: float = 0.3,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        self.action_field = ARTActionField(input_dim, action_count, seed=seed, debug=debug)

    def select(self, x: np.ndarray, salience: np.ndarray) -> ActionResult:
        category_activation = np.asarray(x, dtype=float)
        if np.sum(category_activation) <= 1e-9:
            category_activation = np.ones_like(category_activation) / max(1, len(category_activation))
        else:
            category_activation = category_activation / (np.sum(category_activation) + 1e-9)
        return self.action_field.select(
            category_activation,
            value=0.0,
            salience=salience,
            imagined_action_prior=np.zeros(self.action_field.action_count),
            sequence_bias=np.zeros(self.action_field.action_count),
            pathway="art-compat",
        )


class PlannedPathway:
    """Compatibility adapter around the ART action field."""

    def __init__(self, action_selector: ARTActionField) -> None:
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
            "art-planned",
        )


class PathwayGate:
    """Compatibility shim: pathway mixing is no longer part of the architecture."""

    def __init__(self, urgency_threshold: float = 0.68, debug: bool = False) -> None:
        self.urgency_threshold = urgency_threshold
        self.debug = debug

    def weights(self, urgency: float = 0.0) -> tuple[float, float]:
        return 0.0, 1.0
