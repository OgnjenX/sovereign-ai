from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sovereign_ai.utils import normalize


@dataclass
class TemporalState:
    present: np.ndarray
    future_1: np.ndarray
    future_2: np.ndarray

    @classmethod
    def from_present(cls, present: np.ndarray) -> "TemporalState":
        present = normalize(np.asarray(present, dtype=float))
        return cls(
            present=present,
            future_1=np.zeros_like(present),
            future_2=np.zeros_like(present),
        )

    def unfold(self, transition_model, action_distribution: np.ndarray) -> None:
        self.future_1 = normalize(transition_model.predict(self.present, action_distribution))
        self.future_2 = normalize(transition_model.predict(self.future_1, action_distribution))

    def imagined_input(self, near_weight: float = 0.7, far_weight: float = 0.3) -> np.ndarray:
        combined = near_weight * self.future_1 + far_weight * self.future_2
        return np.clip(normalize(combined), 0.0, 1.0)
