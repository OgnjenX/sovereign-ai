"""Learning gate and prototype update helpers for ART fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LearningDecision:
    """Decision output for whether learning updates are allowed."""

    allowed: bool
    reason: str


class GatedLearning:
    """Prototype update gated by resonance, reward, or novelty."""

    def __init__(
        self,
        learning_rate: float = 0.18,
        intersection_rate: float = 0.75,
        **gate_options: float | bool,
    ) -> None:
        self.learning_rate = learning_rate
        self.intersection_rate = intersection_rate
        self.reward_threshold = float(gate_options.get("reward_threshold", 0.1))
        self.novelty_threshold = float(gate_options.get("novelty_threshold", 0.35))
        self.debug = bool(gate_options.get("debug", False))

    def gate(self, resonance: bool, reward: float, novelty: float) -> LearningDecision:
        """Return learning decision from resonance, reward, and novelty signals."""

        signals = np.array(
            [
                float(resonance),
                max(0.0, reward - self.reward_threshold),
                max(0.0, novelty - self.novelty_threshold),
            ]
        )
        allowed = bool(np.max(signals) > 0.0)
        reason = ["resonance", "reward", "novelty"][int(np.argmax(signals))]
        if not allowed:
            reason = "closed"
        return LearningDecision(allowed, reason)

    def update(self, prototypes: np.ndarray, category_index: int, x: np.ndarray) -> None:
        """Apply hybrid intersection-average update to one prototype."""

        before = prototypes[category_index].copy()
        intersection = np.minimum(prototypes[category_index], x)
        average = prototypes[category_index] + self.learning_rate * (
            x - prototypes[category_index]
        )
        prototypes[category_index] = (
            self.intersection_rate * intersection + (1.0 - self.intersection_rate) * average
        )
        if self.debug:
            delta = float(np.linalg.norm(prototypes[category_index] - before))
            print(
                f"[learning] category={category_index} delta={delta:.4f} "
                f"mode=art-hybrid-intersection"
            )
