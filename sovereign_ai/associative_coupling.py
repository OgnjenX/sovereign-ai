from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sovereign_ai.art_field import ARTField


@dataclass(frozen=True)
class ProjectionTrace:
    name: str
    source_size: int
    target_size: int
    update_norm: float


class AssociativeProjection:
    """Learned category-to-category projection between ART fields."""

    def __init__(
        self,
        source_field: ARTField,
        target_field: ARTField,
        name: str,
        learning_rate: float = 0.08,
        debug: bool = False,
    ) -> None:
        self.source_field = source_field
        self.target_field = target_field
        self.name = name
        self.learning_rate = learning_rate
        self.debug = debug
        self.weights = np.empty((0, 0), dtype=float)
        self.last_trace: ProjectionTrace | None = None

    def project(self, source_activation: np.ndarray | None = None) -> np.ndarray:
        self._ensure_shape()
        if self.weights.size == 0:
            return np.zeros(len(self.target_field.categories), dtype=float)
        source = self._source_activation(source_activation)
        target = source @ self.weights
        if np.sum(target) <= 1e-9:
            return np.zeros(self.weights.shape[1], dtype=float)
        return target / (np.sum(target) + 1e-9)

    def top_down(self, source_activation: np.ndarray | None = None) -> np.ndarray:
        bias = self.project(source_activation)
        if len(bias) == 0:
            return np.zeros(self.target_field.input_dim, dtype=float)
        return self.target_field.reconstruct(bias)

    def learn(
        self,
        source_activation: np.ndarray | None = None,
        target_activation: np.ndarray | None = None,
        rate: float | None = None,
    ) -> ProjectionTrace:
        self._ensure_shape()
        source = self._source_activation(source_activation)
        target = self._target_activation(target_activation)
        if len(source) == 0 or len(target) == 0:
            self.last_trace = ProjectionTrace(self.name, len(source), len(target), 0.0)
            return self.last_trace
        update_rate = self.learning_rate if rate is None else rate
        target_prediction = self.project(source)
        error = target - target_prediction
        update = update_rate * np.outer(source, error)
        self.weights += update
        self.weights = np.clip(self.weights, 0.0, None)
        row_sums = np.sum(self.weights, axis=1, keepdims=True)
        active_rows = row_sums[:, 0] > 1e-9
        self.weights[active_rows] = self.weights[active_rows] / row_sums[active_rows]
        self.last_trace = ProjectionTrace(self.name, len(source), len(target), float(np.linalg.norm(update)))
        if self.debug:
            print(
                "[projection] name="
                f"{self.name} source={len(source)} target={len(target)} update={self.last_trace.update_norm:.4f}"
            )
        return self.last_trace

    def _ensure_shape(self) -> None:
        source_size = len(self.source_field.categories)
        target_size = len(self.target_field.categories)
        if self.weights.shape == (source_size, target_size):
            return
        resized = np.zeros((source_size, target_size), dtype=float)
        rows = min(source_size, self.weights.shape[0])
        cols = min(target_size, self.weights.shape[1])
        if rows and cols:
            resized[:rows, :cols] = self.weights[:rows, :cols]
        self.weights = resized

    def _source_activation(self, activation: np.ndarray | None) -> np.ndarray:
        if activation is None:
            if self.source_field.last_result is None:
                return np.zeros(len(self.source_field.categories), dtype=float)
            activation = self.source_field.last_result.category_activation
        return self._fit_activation(activation, len(self.source_field.categories))

    def _target_activation(self, activation: np.ndarray | None) -> np.ndarray:
        if activation is None:
            if self.target_field.last_result is None:
                return np.zeros(len(self.target_field.categories), dtype=float)
            activation = self.target_field.last_result.category_activation
        return self._fit_activation(activation, len(self.target_field.categories))

    def _fit_activation(self, activation: np.ndarray, size: int) -> np.ndarray:
        fitted = np.zeros(size, dtype=float)
        activation = np.asarray(activation, dtype=float)
        fitted[: min(size, len(activation))] = activation[: min(size, len(activation))]
        if np.sum(fitted) <= 1e-9:
            return fitted
        return fitted / (np.sum(fitted) + 1e-9)
