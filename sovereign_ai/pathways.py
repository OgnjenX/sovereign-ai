"""Compatibility pathway adapters for action selection."""

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
        **opts: int | float | bool | None,
    ) -> None:
        temperature = opts.get("temperature")
        self.temperature = 0.3 if temperature is None else float(temperature)
        seed = opts.get("seed")
        self.action_field = ARTActionField(
            input_dim,
            action_count,
            seed=None if seed is None else int(seed),
            debug=bool(opts.get("debug", False)),
        )

    def select(self, x: np.ndarray, salience: np.ndarray) -> ActionResult:
        """Select action from normalized category activation and salience."""

        category_activation = np.asarray(x, dtype=float)
        if np.sum(category_activation) <= 1e-9:
            category_activation = np.ones_like(category_activation) / max(
                1, len(category_activation)
            )
        else:
            category_activation = category_activation / (
                np.sum(category_activation) + 1e-9
            )
        return self.action_field.select(
            category_activation,
            value=0.0,
            salience=salience,
            imagined_action_prior=np.zeros(self.action_field.action_count),
            sequence_bias=np.zeros(self.action_field.action_count),
            pathway="art-compat",
        )

    def action_count(self) -> int:
        """Expose number of supported actions."""

        return self.action_field.action_count


class PlannedPathway:
    """Compatibility adapter around the ART action field."""

    def __init__(self, action_selector: ARTActionField) -> None:
        self.action_selector = action_selector

    def select(
        self,
        category_activation: np.ndarray,
        **opts: np.ndarray | float | None,
    ) -> ActionResult:
        """Delegate planned action selection to wrapped action field."""

        value = float(opts.get("value", 0.0) or 0.0)
        salience = np.asarray(
            opts.get("salience", np.zeros(self.action_selector.action_count)),
            dtype=float,
        )
        imagined_action_prior = opts.get("imagined_action_prior")
        if not isinstance(imagined_action_prior, np.ndarray):
            imagined_action_prior = None
        sequence_bias = opts.get("sequence_bias")
        if not isinstance(sequence_bias, np.ndarray):
            sequence_bias = None
        return self.action_selector.select(
            category_activation,
            value,
            salience,
            imagined_action_prior,
            sequence_bias,
            "art-planned",
        )

    def action_count(self) -> int:
        """Expose number of actions from wrapped selector."""

        return self.action_selector.action_count


class PathwayGate:
    """Compatibility shim: pathway mixing is no longer part of the architecture."""

    def __init__(self, urgency_threshold: float = 0.68, debug: bool = False) -> None:
        self.urgency_threshold = urgency_threshold
        self.debug = debug

    def weights(self) -> tuple[float, float]:
        """Return fixed pathway mixture weights for compatibility mode."""

        return 0.0, 1.0

    def pathway(self, urgency: float) -> str:
        """Return active pathway label for compatibility API consumers."""

        _ = urgency
        return "planned"
