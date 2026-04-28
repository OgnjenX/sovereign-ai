from __future__ import annotations

import itertools

import numpy as np

from sovereign_ai.action_selection import ActionResult, BasalGangliaActionSelection
from sovereign_ai.environment import GridWorld
from sovereign_ai.evaluation import ValueSystem
from sovereign_ai.imagination import ImaginationLoop
from sovereign_ai.learning import GatedLearning
from sovereign_ai.memory import Memory
from sovereign_ai.pathways import PathwayGate, PlannedPathway, ReactivePathway
from sovereign_ai.perception import ARTPerception, PerceptionResult
from sovereign_ai.goal_system import GoalSystem
from sovereign_ai.spatial import SpatialModule
from sovereign_ai.temporal_state import TemporalState
from sovereign_ai.transition_model import LinearTransitionModel
from sovereign_ai.utils import compact_vector, normalize


class CognitiveArchitecture:
    """Dynamical control loop.

    real input + imagination -> perception field <-> value field <-> action field
                         ^             |              |              |
                         +-------------+--------------+--------------+
    Learning happens after the intra-step fields converge.
    """

    def __init__(
        self,
        input_dim: int,
        action_count: int,
        max_categories: int = 16,
        vigilance: float = 0.95,
        seed: int | None = None,
        debug: bool = False,
        convergence_iterations: int = 7,
        convergence_tolerance: float = 0.025,
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
        self.goal_system = GoalSystem(
            input_dim,
            seed=None if seed is None else seed + 5,
            debug=debug,
        )
        self.transition_model = LinearTransitionModel(
            input_dim,
            action_count,
            seed=None if seed is None else seed + 6,
        )
        self.convergence_iterations = convergence_iterations
        self.convergence_tolerance = convergence_tolerance
        self.input_dim = input_dim
        self.action_count = action_count

    def observe_environment(self, env: GridWorld) -> np.ndarray:
        return env.observe()

    def evaluate(self, perception: PerceptionResult, reward: float) -> float:
        result = self.value_system.evaluate(
            perception.category_activation,
            reward=reward,
            novelty=perception.novelty,
            context=self.memory.sequence_context(),
            goal_alignment=self.goal_system.state(self.memory.stm).alignment,
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

    def converge_states(
        self,
        x: np.ndarray,
        reward: float,
        salience: np.ndarray,
        urgency: float,
    ) -> tuple[PerceptionResult, float, ActionResult, np.ndarray, float]:
        seeded_perception = self.perception.process(x)
        temporal_state = TemporalState.from_present(x)
        perception_state = self.perception.update_state(
            x,
            previous_activation=seeded_perception.category_activation,
        )
        category_activation = perception_state.result.category_activation
        action_distribution = np.ones(self.action_count, dtype=float) / self.action_count
        value_activation = np.ones(6, dtype=float) / 6.0
        top_down_bias = np.zeros(self.input_dim, dtype=float)
        effective_input = x.copy()
        value_scalar = 0.0
        future_alignment = 0.0
        future_value = 0.0
        future_value_weight = 0.35
        action_result: ActionResult | None = None
        self._adaptive_imagination_strength(
            perception_state.result.novelty,
            action_distribution,
            self.goal_system.state(effective_input).alignment,
        )
        temporal_state.unfold(self.transition_model, action_distribution)
        temporal_state.present.copy()
        previous_total_change: float | None = None

        for iteration in range(self.convergence_iterations):
            temporal_imagined = temporal_state.imagined_input()
            imagined_input, imagined_category_bias, imagined_action_prior = self.imagination.coupled_priors(
                count=4,
                keep=2,
            )
            imagined_input = np.clip(0.65 * temporal_imagined + 0.35 * imagined_input, 0.0, 1.0)
            transition_uncertainty = float(action_distribution @ self.transition_model.uncertainty)
            goal_probe = self.goal_system.state(effective_input)
            alpha = self._adaptive_imagination_strength(
                perception_state.result.novelty,
                action_distribution,
                goal_probe.alignment,
            )
            effective_input = np.clip((1.0 - alpha) * x + alpha * imagined_input, 0.0, 1.0)
            goal_state = self.goal_system.state(effective_input)
            perception_state = self.perception.update_state(
                effective_input,
                previous_activation=category_activation,
                top_down_bias=top_down_bias + 0.12 * goal_state.active_goal,
                imagined_category_bias=imagined_category_bias,
                real_input_weight=1.0,
            )
            category_activation, compositional_slots = self.perception.compose_activation(
                perception_state.result.category_activation
            )
            effective_input = perception_state.effective_input
            previous_present = temporal_state.present.copy()
            temporal_state.present = normalize(effective_input)
            present_change = float(np.linalg.norm(temporal_state.present - previous_present))
            goal_state = self.goal_system.state(effective_input)
            future_value, future_alignment = self._evaluate_future_state(
                temporal_state.future_1,
                category_activation,
            )
            value_state = self.value_system.update_state(
                category_activation,
                reward=reward,
                novelty=perception_state.result.novelty,
                context=self.memory.sequence_context(),
                goal_alignment=float(np.clip(goal_state.alignment + 0.25 * future_alignment, -1.0, 1.0)),
                action_distribution=action_distribution,
                previous_state=value_activation,
                learn=False,
            )
            value_activation = value_state.activation
            action_value_activation = value_activation.copy()
            action_value_activation[4] = max(0.0, action_value_activation[4] + future_value_weight * future_value)
            action_value_activation = action_value_activation / (np.sum(action_value_activation) + 1e-9)
            value_scalar = value_state.scalar + future_value_weight * future_value
            reactive_weight, planned_weight = self.pathway_gate.weights(urgency)
            reactive_result = self.reactive.select(effective_input, salience)
            sequence_bias = self._sequence_bias(len(salience))
            action_state = self.action_selector.update_state(
                category_activation,
                action_value_activation,
                salience,
                imagined_action_prior
                + self.action_selector.slot_action_bias(compositional_slots, category_activation),
                sequence_bias,
                reactive_result.action_distribution,
                previous_distribution=action_distribution,
            )
            action_distribution = (
                reactive_weight * reactive_result.action_distribution
                + planned_weight * action_state.result.action_distribution
            )
            action_distribution = action_distribution / (np.sum(action_distribution) + 1e-9)
            action_result = ActionResult(
                int(np.argmax(action_distribution)),
                action_distribution,
                action_state.result.go,
                action_state.result.stop,
                action_state.result.drives,
                "coupled",
            )
            temporal_state.unfold(self.transition_model, action_distribution)
            top_down_bias = self._derive_top_down_bias(
                action_value_activation,
                action_distribution,
                imagined_input,
                goal_state.active_goal,
                temporal_state.future_1 - temporal_state.present,
            )
            total_change = perception_state.change + value_state.change + action_state.change + present_change
            oscillating = previous_total_change is not None and total_change > previous_total_change * 1.25
            previous_total_change = total_change
            if self.debug:
                print(
                    "[convergence] iter="
                    f"{iteration} total_change={total_change:.4f} "
                    f"p_change={perception_state.change:.4f} "
                    f"v_change={value_state.change:.4f} a_change={action_state.change:.4f} "
                    f"present_change={present_change:.4f} unfold=post_action "
                    f"top_down_norm={np.linalg.norm(top_down_bias):.3f} "
                    f"goal_alignment={goal_state.alignment:.3f} "
                    f"future_alignment={future_alignment:.3f} future_value={future_value:.3f} "
                    f"future_contrib={future_value_weight * future_value:.3f} "
                    f"alpha={alpha:.2f} transition_uncertainty={transition_uncertainty:.3f} "
                    f"oscillation={oscillating} "
                    f"slot_norms={np.round(np.linalg.norm(compositional_slots, axis=1), 3).tolist()}"
                )
            if total_change < self.convergence_tolerance:
                break

        if action_result is None:
            action_result = self.action_selector.select(category_activation, value_scalar, salience, pathway="coupled")
        final_slots = self.perception.compose_activation(category_activation)[1]
        result = self.value_system.evaluate_slots(
            category_activation,
            final_slots,
            reward=reward,
            novelty=perception_state.result.novelty,
            context=self.memory.sequence_context(),
            goal_alignment=float(np.clip(self.goal_system.state(effective_input).alignment, -1.0, 1.0)),
            learn=True,
        )
        self.previous_prediction_error = result.prediction_error
        return perception_state.result, result.value + future_value_weight * future_value, action_result, effective_input, future_alignment

    def _adaptive_imagination_strength(
        self,
        novelty: float,
        action_distribution: np.ndarray,
        goal_alignment: float,
    ) -> float:
        action_distribution = np.asarray(action_distribution, dtype=float)
        action_distribution = action_distribution / (np.sum(action_distribution) + 1e-9)
        transition_uncertainty = float(action_distribution @ self.transition_model.uncertainty)
        signal = 1.15 * novelty + 0.75 * transition_uncertainty + 0.55 * goal_alignment - 1.0
        adaptive = 0.6 * (1.0 / (1.0 + np.exp(-signal)))
        return float(np.clip(adaptive, 0.0, 0.6))

    def _evaluate_future_state(
        self,
        future_state: np.ndarray,
        previous_activation: np.ndarray,
    ) -> tuple[float, float]:
        if np.linalg.norm(future_state) <= 1e-9:
            return 0.0, 0.0

        future_perception = self.perception.update_state(
            future_state,
            previous_activation=previous_activation,
            real_input_weight=1.0,
        )
        future_activation, future_slots = self.perception.compose_activation(
            future_perception.result.category_activation
        )
        future_goal = self.goal_system.state(future_state)
        future_result = self.value_system.evaluate_slots(
            future_activation,
            future_slots,
            reward=0.0,
            novelty=future_perception.result.novelty,
            context=self.memory.sequence_context(),
            goal_alignment=future_goal.alignment,
            learn=False,
        )
        return future_result.value, future_goal.alignment

    def _derive_top_down_bias(
        self,
        value_activation: np.ndarray,
        action_distribution: np.ndarray,
        imagined_input: np.ndarray,
        goal_vector: np.ndarray,
        future_bias: np.ndarray,
    ) -> np.ndarray:
        category_count = len(self.perception.prototypes)
        if category_count == 0:
            return np.zeros(self.input_dim, dtype=float)
        value_weights = self.value_system.expected_value_weights[:category_count]
        if len(value_weights) < category_count:
            value_weights = np.pad(value_weights, (0, category_count - len(value_weights)))
        value_attention = value_weights @ self.perception.prototypes
        action_attention = action_distribution @ self.reactive.input_action_weights.T
        imagination_attention = imagined_input - self.memory.stm
        raw = (
            0.18 * value_activation[2] * value_attention
            + 0.12 * action_attention
            + 0.10 * imagination_attention
            + 0.12 * goal_vector
            + 0.18 * future_bias
        )
        return np.clip(raw, -0.15, 0.15)

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
        perception, value, action_state, effective_input, future_alignment = self.converge_states(
            x,
            self.previous_reward,
            env.salience(),
            env.urgency(),
        )
        action = action_state.action_index
        previous_position = env.position.copy()
        reward = self.execute(env, action)
        next_x = self.observe_environment(env)
        self.transition_model.learn(x, action_state.action_distribution, next_x)
        self.goal_system.update(effective_input, reward, perception.novelty, future_alignment=future_alignment)
        motion = (env.position - previous_position).astype(float) / max(1, env.size - 1)
        self.spatial_context = self.spatial.process(motion)
        self.memory.record_transition(perception.category_index, action)
        self.action_selector.learn_action(
            perception.category_activation,
            action,
            reward_prediction_error=reward - value,
        )
        if self.learning_condition(perception, reward):
            self.update_weights(perception, effective_input)

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
