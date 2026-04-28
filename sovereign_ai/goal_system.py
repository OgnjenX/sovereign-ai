from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sovereign_ai.utils import cosine_similarity, normalize, softmax


@dataclass(frozen=True)
class GoalState:
    active_goal: np.ndarray
    goal_activation: np.ndarray
    alignment: float


class GoalSystem:
    """Persistent distributed goals with slow competitive dynamics."""

    def __init__(
        self,
        input_dim: int,
        goal_count: int = 3,
        update_rate: float = 0.04,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        self.update_rate = update_rate
        self.debug = debug
        rng = np.random.default_rng(seed)
        self.goals = rng.random((goal_count, input_dim))
        self.goal_activation = np.ones(goal_count, dtype=float) / goal_count

    def update(
        self,
        state: np.ndarray,
        reward: float,
        novelty: float,
        future_alignment: float = 0.0,
    ) -> GoalState:
        state = np.asarray(state, dtype=float)
        similarities = cosine_similarity(state, self.goals)
        reward_pull = max(0.0, reward) + 0.25 * novelty + 0.15 * max(0.0, future_alignment)
        excitation = softmax(similarities + reward_pull, temperature=0.35)
        inhibition = 1.0 - excitation
        self.goal_activation = np.clip(
            self.goal_activation
            + self.update_rate * (excitation * (1.0 - self.goal_activation) - inhibition * self.goal_activation),
            0.0,
            1.0,
        )
        self.goal_activation = self.goal_activation / (np.sum(self.goal_activation) + 1e-9)
        winner = int(np.argmax(self.goal_activation))
        self.goals[winner] = normalize(
            (1.0 - self.update_rate) * self.goals[winner] + self.update_rate * state
        )
        active_goal = self.goal_activation @ self.goals
        alignment = float(normalize(state) @ normalize(active_goal))
        if self.debug:
            print(
                "[goal] alignment="
                f"{alignment:.3f} activation={np.round(self.goal_activation, 3)}"
            )
        return GoalState(active_goal, self.goal_activation.copy(), alignment)

    def state(self, current_state: np.ndarray) -> GoalState:
        active_goal = self.goal_activation @ self.goals
        alignment = float(normalize(current_state) @ normalize(active_goal))
        return GoalState(active_goal, self.goal_activation.copy(), alignment)
