from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sovereign_ai.art_field import ARTField, ARTFieldResult, ARTFieldState


@dataclass(frozen=True)
class PerceptionResult(ARTFieldResult):
    """Perception-specialized result preserving ART field contract."""


@dataclass(frozen=True)
class PerceptionState:
    result: PerceptionResult
    effective_input: np.ndarray
    change: float


class ARTPerceptualField(ARTField):
    """Perceptual ART field with recurrent top-down and expectation coupling."""

    def __init__(
        self,
        input_dim: int,
        max_categories: int = 16,
        vigilance: float = 0.72,
        competition_temperature: float = 0.12,
        resonance_iterations: int = 6,
        feedback_rate: float = 0.45,
        convergence_tolerance: float = 1e-3,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            max_categories=max_categories,
            vigilance=vigilance,
            competition_temperature=competition_temperature,
            resonance_iterations=resonance_iterations,
            feedback_rate=feedback_rate,
            convergence_tolerance=convergence_tolerance,
            seed=seed,
            debug=debug,
            name="perception",
        )

    def process(
        self,
        x: np.ndarray,
        *,
        category_bias: np.ndarray | None = None,
        top_down_bias: np.ndarray | None = None,
        vigilance_modulation: float = 0.0,
        learn: bool = False,
    ) -> PerceptionResult:
        return self._to_perception_result(
            super().process(
                x,
                category_bias=category_bias,
                top_down_bias=top_down_bias,
                vigilance_modulation=vigilance_modulation,
                learn=learn,
            )
        )

    def update_state(
        self,
        x: np.ndarray,
        *,
        previous_activation: np.ndarray | None = None,
        top_down_bias: np.ndarray | None = None,
        category_bias: np.ndarray | None = None,
        vigilance_modulation: float = 0.0,
        learn: bool = False,
    ) -> ARTFieldState:
        return super().update_state(
            x,
            previous_activation=previous_activation,
            top_down_bias=top_down_bias,
            category_bias=category_bias,
            vigilance_modulation=vigilance_modulation,
            learn=learn,
        )

    def update_state_with_imagination(
        self,
        x: np.ndarray,
        *,
        previous_activation: np.ndarray | None = None,
        top_down_bias: np.ndarray | None = None,
        imagined_input: np.ndarray | None = None,
        imagined_category_bias: np.ndarray | None = None,
        real_input_weight: float = 0.82,
        vigilance_modulation: float = 0.0,
        learn: bool = False,
    ) -> PerceptionState:
        x = np.asarray(x, dtype=float)
        if imagined_input is not None:
            imagined = np.asarray(imagined_input, dtype=float)
            effective_input = np.clip(real_input_weight * x + (1.0 - real_input_weight) * imagined, 0.0, 1.0)
        else:
            effective_input = x

        base_state = self.update_state(
            effective_input,
            previous_activation=previous_activation,
            top_down_bias=top_down_bias,
            category_bias=imagined_category_bias,
            vigilance_modulation=vigilance_modulation,
            learn=learn,
        )
        return PerceptionState(self._to_perception_result(base_state.result), base_state.effective_input, base_state.change)

    def compose_activation(
        self,
        category_activation: np.ndarray,
        slot_count: int = 2,
    ) -> tuple[np.ndarray, np.ndarray]:
        activation = self._resize_activation(category_activation)
        if len(activation) == 0:
            return activation, np.empty((0, self.input_dim), dtype=float)
        slot_count = min(slot_count, len(activation))
        component_indices = np.argsort(activation)[-slot_count:]
        component_weights = activation[component_indices]
        component_weights = component_weights / (np.sum(component_weights) + 1e-9)
        composed = np.zeros_like(activation)
        composed[component_indices] = component_weights
        slots = self.prototypes[component_indices] * component_weights[:, None]
        if self.debug:
            print(
                "[composition] components="
                f"{component_indices.tolist()} weights={np.round(component_weights, 3)}"
            )
        return composed, slots

    def _to_perception_result(self, result: ARTFieldResult) -> PerceptionResult:
        return PerceptionResult(
            result.category_index,
            result.category_activation,
            result.similarities,
            result.resonance,
            result.novelty,
            result.search_path,
            result.resonance_trace,
            result.top_down_match,
            result.effective_vigilance,
        )


class ARTPerception(ARTPerceptualField):
    """Compatibility wrapper preserving the historical perception class name."""

    pass
