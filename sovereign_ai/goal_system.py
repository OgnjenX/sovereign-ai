"""Goal-state ART field and compatibility wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sovereign_ai.art_field import ARTField
from sovereign_ai.utils import normalize


@dataclass(frozen=True)
class GoalState:
    """Snapshot of the current goal representation and alignment metrics."""

    active_goal: np.ndarray
    goal_activation: np.ndarray
    alignment: float
    resonance: bool = True
    search_path: list[int] | None = None


@dataclass(frozen=True)
class GoalUpdateContext:
    """Optional context values used when updating goal activations."""

    future_alignment: float = 0.0
    vigilance_modulation: float = 0.0


class ARTGoalField(ARTField):
    """ART goal field whose categories are persistent goal states."""

    def __init__(
        self,
        input_dim: int,
        goal_count: int = 3,
        update_rate: float = 0.04,
        **field_options: int | bool | None,
    ) -> None:
        seed = field_options.get("seed")
        debug = bool(field_options.get("debug", False))
        super().__init__(
            input_dim=input_dim,
            max_categories=goal_count,
            vigilance=0.6,
            competition_temperature=0.3,
            learning_rate=update_rate,
            seed=seed,
            debug=debug,
            name="goal",
        )
        self.update_rate = update_rate
        rng = np.random.default_rng(seed)
        self.categories = rng.random((goal_count, input_dim))
        self.goal_activation = np.ones(goal_count, dtype=float) / goal_count

    def update(
        self,
        state: np.ndarray,
        reward: float,
        novelty: float,
        context: GoalUpdateContext | None = None,
    ) -> GoalState:
        """Update goal activations and return the resulting goal state."""

        update_context = context or GoalUpdateContext()
        category_bias = np.zeros(len(self.categories), dtype=float)
        if reward > 0.0 or update_context.future_alignment > 0.0:
            category_bias += self.goal_activation
        field_state = super().update_state(
            state,
            previous_activation=self.goal_activation,
            category_bias=category_bias,
            vigilance_modulation=update_context.vigilance_modulation,
            learn=reward > 0.0 or novelty > 0.35,
        )
        self.goal_activation = field_state.result.category_activation
        active_goal = self.goal_activation @ self.categories
        alignment = float(normalize(state) @ normalize(active_goal))
        if self.debug:
            print(
                "[goal-dyn] category="
                f"{field_state.result.category_index} alignment={alignment:.3f} "
                f"activation={np.round(self.goal_activation, 3)}"
            )
        return GoalState(
            active_goal,
            self.goal_activation.copy(),
            alignment,
            field_state.result.resonance,
            field_state.result.search_path,
        )

    def state(self, current_state: np.ndarray) -> GoalState:
        """Compute goal state for the current input without learning."""

        result = self.process(current_state, category_bias=self.goal_activation, learn=False)
        active_goal = result.category_activation @ self.categories
        alignment = float(normalize(current_state) @ normalize(active_goal))
        return GoalState(
            active_goal,
            result.category_activation,
            alignment,
            result.resonance,
            result.search_path,
        )


class GoalSystem(ARTGoalField):
    """Compatibility wrapper preserving the old goal-system constructor."""
