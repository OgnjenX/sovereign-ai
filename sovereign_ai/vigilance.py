"""Adaptive vigilance modulation utilities for coupled ART fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VigilanceTrace:
    """Recorded update from one vigilance learning step."""

    field_name: str
    adjustment: float
    mismatch: float
    reward: float
    update_norm: float


class VigilanceController:
    """Learns field-specific vigilance adjustments from mismatch and outcome."""

    def __init__(
        self,
        field_names: list[str],
        feature_dim: int = 4,
        learning_rate: float = 0.05,
        debug: bool = False,
    ) -> None:
        """Initialize field-specific vigilance weights."""

        self.field_names = list(field_names)
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.debug = debug
        self.weights = {
            name: np.zeros(feature_dim, dtype=float) for name in self.field_names
        }
        self.last_traces: list[VigilanceTrace] = []

    def modulation(
        self,
        field_name: str,
        field_activation: np.ndarray,
        mismatch: float = 0.0,
        reward: float = 0.0,
    ) -> float:
        """Compute a bounded vigilance modulation for one field."""

        features = self._features(field_activation, mismatch, reward)
        return float(
            np.clip(
                features
                @ self.weights.setdefault(field_name, np.zeros(self.feature_dim)),
                -0.2,
                0.2,
            )
        )

    def learn(
        self,
        field_name: str,
        field_activation: np.ndarray,
        mismatch: float,
        reward: float,
    ) -> VigilanceTrace:
        """Learn a new vigilance adjustment from mismatch and reward."""

        features = self._features(field_activation, mismatch, reward)
        target = np.clip(0.15 * mismatch - 0.08 * max(reward, 0.0), -0.2, 0.2)
        current = self.modulation(field_name, field_activation, mismatch, reward)
        error = target - current
        update = self.learning_rate * error * features
        self.weights.setdefault(field_name, np.zeros(self.feature_dim))
        self.weights[field_name] += update
        trace = VigilanceTrace(
            field_name,
            self.modulation(field_name, field_activation, mismatch, reward),
            mismatch,
            reward,
            float(np.linalg.norm(update)),
        )
        self.last_traces.append(trace)
        if self.debug:
            print(
                "[vigilance] field="
                f"{field_name} adjustment={trace.adjustment:.4f} mismatch={mismatch:.3f} "
                f"reward={reward:.3f} update={trace.update_norm:.4f}"
            )
        return trace

    def _features(
        self, activation: np.ndarray, mismatch: float, reward: float
    ) -> np.ndarray:
        """Build the feature vector used by the vigilance controller."""

        activation = np.asarray(activation, dtype=float)
        confidence = float(np.max(activation)) if len(activation) else 0.0
        dispersion = (
            float(1.0 - np.linalg.norm(activation, ord=2)) if len(activation) else 1.0
        )
        return np.array(
            [
                1.0,
                np.clip(mismatch, 0.0, 1.0),
                np.clip(reward, -1.0, 1.0),
                np.clip(confidence - dispersion, -1.0, 1.0),
            ],
            dtype=float,
        )
