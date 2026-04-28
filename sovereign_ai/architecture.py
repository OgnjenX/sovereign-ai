from __future__ import annotations

import itertools

import numpy as np

from sovereign_ai.action_selection import BasalGangliaActionSelection
from sovereign_ai.environment import GridWorld
from sovereign_ai.evaluation import ValueSystem
from sovereign_ai.imagination import ImaginationLoop
from sovereign_ai.learning import GatedLearning
from sovereign_ai.memory import Memory
from sovereign_ai.pathways import PathwayGate, PlannedPathway, ReactivePathway
from sovereign_ai.perception import ARTPerception, PerceptionResult
from sovereign_ai.spatial import SpatialModule
from sovereign_ai.utils import compact_vector


class CognitiveArchitecture:
    def __init__(
        self,
        input_dim: int,
        action_count: int,
        max_categories: int = 16,
        vigilance: float = 0.95,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        self.debug = debug
        self.memory = Memory(input_dim)
        self.perception = ARTPerception(
            input_dim,
            max_categories,
            vigilance=vigilance,
            seed=seed,
            debug=debug,
        )
        self.value_system = ValueSystem(debug=debug)
        self.action_selector = BasalGangliaActionSelection(
            max_categories,
            action_count,
            seed=None if seed is None else seed + 1,
            debug=debug,
        )
        self.learning = GatedLearning(debug=debug)
        self.reactive = ReactivePathway(
            input_dim,
            action_count,
            seed=None if seed is None else seed + 2,
            debug=debug,
        )
        self.planned = PlannedPathway(self.action_selector)
        self.pathway_gate = PathwayGate(debug=debug)
        self.imagination = ImaginationLoop(
            self.perception,
            self.value_system,
            self.action_selector.category_action_weights,
            seed=None if seed is None else seed + 3,
            debug=debug,
        )
        self.previous_reward = 0.0
        self.previous_prediction_error = 0.0
        self.spatial = SpatialModule(motion_dim=2, seed=None if seed is None else seed + 4)
        self.spatial_context = np.zeros(6, dtype=float)

    def observe_environment(self, env: GridWorld) -> np.ndarray:
        return env.observe()

    def evaluate(self, perception: PerceptionResult, x: np.ndarray, reward: float) -> float:
        result = self.value_system.evaluate(
            perception.category_activation,
            x,
            reward=reward,
            novelty=perception.novelty,
            context=self.memory.sequence_context(),
        )
        self.previous_prediction_error = result.prediction_error
        return result.value

    def select_action(
        self,
        x: np.ndarray,
        perception: PerceptionResult,
        value: float,
        salience: np.ndarray,
        urgency: float,
    ) -> int:
        reactive_weight, planned_weight = self.pathway_gate.weights(urgency)
        imagined_prior = self.imagination.action_prior(count=4, keep=2)
        sequence_bias = self._sequence_bias(len(salience))
        reactive_result = self.reactive.select(x, salience)
        planned_result = self.planned.select(
            perception.category_activation,
            value,
            salience,
            imagined_prior,
            sequence_bias,
        )
        distribution = (
            reactive_weight * reactive_result.action_distribution
            + planned_weight * planned_result.action_distribution
        )
        action = int(np.argmax(distribution))
        if self.debug:
            print(
                "[action-mix] action="
                f"{action} reactive_weight={reactive_weight:.3f} planned_weight={planned_weight:.3f} "
                f"distribution={np.round(distribution, 3)}"
            )
        return action

    def _sequence_bias(self, action_count: int) -> np.ndarray:
        bias = np.zeros(action_count, dtype=float)
        if self.memory.action_trace:
            recent = self.memory.action_trace[-1]
            bias[recent] -= 0.08
            bias[(recent + 1) % action_count] += 0.04
        if len(self.spatial_context):
            spatial_bias = self.spatial_context[:action_count]
            bias += 0.03 * (spatial_bias - np.mean(spatial_bias))
        return bias

    def execute(self, env: GridWorld, action: int) -> float:
        return env.step(action)

    def learning_condition(self, perception: PerceptionResult, reward: float) -> bool:
        decision = self.learning.gate(perception.resonance, reward, perception.novelty)
        if self.debug:
            print(f"[learning-gate] allowed={decision.allowed} reason={decision.reason}")
        return decision.allowed

    def update_weights(self, perception: PerceptionResult, x: np.ndarray) -> None:
        self.learning.update(self.perception.prototypes, perception.category_index, x)
        self.memory.bind_ltm(self.perception.prototypes)

    def step(self, env: GridWorld, step_index: int) -> None:
        x = self.observe_environment(env)
        self.memory.update_stm(x)
        perception = self.perception.process(x)
        value = self.evaluate(perception, x, self.previous_reward)
        action = self.select_action(x, perception, value, env.salience(), env.urgency())
        previous_position = env.position.copy()
        reward = self.execute(env, action)
        motion = (env.position - previous_position).astype(float) / max(1, env.size - 1)
        self.spatial_context = self.spatial.process(motion)
        self.memory.record_transition(perception.category_index, action)
        self.action_selector.learn_action(
            perception.category_activation,
            action,
            reward_prediction_error=reward - value,
        )
        if self.learning_condition(perception, reward):
            self.update_weights(perception, x)

        if self.debug:
            print(
                "[loop] step="
                f"{step_index} pos={env.position.tolist()} x={compact_vector(x)} "
                f"action={action} reward={reward:.3f}\n"
            )
        self.previous_reward = reward

    def run(self, env: GridWorld, steps: int | None = 32) -> None:
        iterator = itertools.count() if steps is None else range(steps)
        for step_index in iterator:
            self.step(env, int(step_index))
