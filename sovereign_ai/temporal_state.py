"""Temporal rollout container for near/far future imagined states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sovereign_ai.utils import normalize


@dataclass
class TemporalState:
    """Present and predicted future feature vectors."""

    present: np.ndarray
    future_1: np.ndarray
    future_2: np.ndarray

    @classmethod
    def from_present(cls, present: np.ndarray) -> "TemporalState":
        """Create temporal state with normalized present and empty futures."""

        present = normalize(np.asarray(present, dtype=float))
        return cls(
            present=present,
            future_1=np.zeros_like(present),
            future_2=np.zeros_like(present),
        )

    def unfold(self, transition_model, action_distribution: np.ndarray) -> None:
        """Predict one- and two-step futures using transition dynamics."""

        self.future_1 = normalize(transition_model.predict(self.present, action_distribution))
        self.future_2 = normalize(transition_model.predict(self.future_1, action_distribution))

    def imagined_input(self, near_weight: float = 0.7, far_weight: float = 0.3) -> np.ndarray:
        """Blend near/far futures into one normalized imagined input."""

        combined = near_weight * self.future_1 + far_weight * self.future_2
        return np.clip(normalize(combined), 0.0, 1.0)
