from __future__ import annotations

import numpy as np


class Memory:
    """Vector STM/LTM plus temporal traces for short action sequences."""

    def __init__(self, input_dim: int, sequence_length: int = 5, decay: float = 0.82) -> None:
        self.stm = np.zeros(input_dim, dtype=float)
        self.trace = np.zeros(input_dim, dtype=float)
        self.ltm = np.empty((0, input_dim), dtype=float)
        self.decay = decay
        self.sequence_length = sequence_length
        self.category_trace: list[int] = []
        self.action_trace: list[int] = []

    def update_stm(self, activation_pattern: np.ndarray) -> None:
        pattern = np.asarray(activation_pattern, dtype=float)
        self.stm = pattern.copy()
        self.trace = self.decay * self.trace + (1.0 - self.decay) * pattern

    def bind_ltm(self, learned_weights: np.ndarray) -> None:
        self.ltm = np.asarray(learned_weights, dtype=float)

    def record_transition(self, category_index: int, action_index: int) -> None:
        self.category_trace.append(category_index)
        self.action_trace.append(action_index)
        del self.category_trace[:-self.sequence_length]
        del self.action_trace[:-self.sequence_length]

    def sequence_context(self) -> float:
        if not self.action_trace:
            return 0.0
        recency = np.linspace(0.3, 1.0, len(self.action_trace))
        return float(np.mean(recency))
