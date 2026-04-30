from __future__ import annotations

import itertools

import numpy as np

from sovereign_ai.action_selection import ARTActionField, ActionResult
from sovereign_ai.associative_coupling import AssociativeProjection
from sovereign_ai.environment import GridWorld
from sovereign_ai.evaluation import ARTValueField
from sovereign_ai.imagination import ARTExpectationField
from sovereign_ai.learning import GatedLearning
from sovereign_ai.memory import Memory
from sovereign_ai.perception import ARTPerceptualField, PerceptionResult
from sovereign_ai.goal_system import ARTGoalField
from sovereign_ai.spatial import SpatialModule
from sovereign_ai.temporal_state import TemporalState
from sovereign_ai.transition_model import ARTTemporalField
from sovereign_ai.utils import compact_vector, normalize, softmax
from sovereign_ai.vigilance import VigilanceController


class CognitiveArchitecture:
    """Network of interacting ART resonant fields.

    Architecture diagram:

        PerceptualField <-> ValueField <-> ActionField
              ^               ^              ^
              |               |              |
        GoalField ------> top-down bias ------+
              ^                              |
              |                              |
        TemporalField ---- prediction ----> PerceptualField
              ^
              |
        ExpectationField -- hypothesis --> PerceptualField/ActionField

    Coupling changes category activation, match scores, or vigilance. Field
    states are never overwritten by another field.
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
        self.perception: ARTPerceptualField = ARTPerceptualField(
            input_dim,
            max_categories,
            vigilance=vigilance,
            seed=seed,
            debug=debug,
        )
        self.value_system: ARTValueField = ARTValueField(
            max_categories,
            seed=None if seed is None else seed + 1,
            debug=debug,
        )
        self.action_selector: ARTActionField = ARTActionField(
            max_categories,
            action_count,
            seed=None if seed is None else seed + 2,
            debug=debug,
        )
        self.learning = GatedLearning(debug=debug)
        self.imagination: ARTExpectationField = ARTExpectationField(
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
        self.goal_system = ARTGoalField(
            input_dim,
            seed=None if seed is None else seed + 5,
            debug=debug,
        )
        self.transition_model = ARTTemporalField(
            input_dim,
            action_count,
            perceptual_category_count=max_categories,
            seed=None if seed is None else seed + 6,
        )
        self.vigilance_controller = VigilanceController(
            ["perception", "value", "goal", "action", "temporal", "expectation"],
            debug=debug,
        )
        self.projections = {
            "perception_value": AssociativeProjection(self.perception, self.value_system, "perception->value", debug=debug),
            "value_perception": AssociativeProjection(self.value_system, self.perception, "value->perception", debug=debug),
            "goal_perception": AssociativeProjection(self.goal_system, self.perception, "goal->perception", debug=debug),
            "goal_action": AssociativeProjection(self.goal_system, self.action_selector, "goal->action", debug=debug),
            "temporal_perception": AssociativeProjection(self.transition_model, self.perception, "temporal->perception", debug=debug),
            "expectation_action": AssociativeProjection(self.imagination, self.action_selector, "expectation->action", debug=debug),
            "perception_temporal": AssociativeProjection(self.perception, self.transition_model, "perception->temporal", debug=debug),
            "action_temporal": AssociativeProjection(self.action_selector, self.transition_model, "action->temporal", debug=debug),
        }
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
    ) -> int:
        value_state = self.value_system.update_state(
            perception.category_activation,
            reward=value,
            novelty=perception.novelty,
            context=self.memory.sequence_context(),
            goal_alignment=self.goal_system.state(x).alignment,
            action_distribution=np.full(self.action_count, 1.0 / self.action_count, dtype=float),
            learn=False,
        )
        action_context = self.action_selector.schema_input(
            perception.category_activation,
            self.goal_system.goal_activation,
            value_state.activation,
            np.empty(0),
            np.empty(0),
        )
        return self.action_selector.resonate_action(action_context, exploratory_signal=salience, pathway="art").result.action_index

    def converge_states(
        self,
        x: np.ndarray,
        reward: float,
        salience: np.ndarray,
        urgency: float,
    ) -> tuple[PerceptionResult, float, ActionResult, np.ndarray, float]:
        seeded_perception = self.perception.process(x)
        temporal_state = TemporalState.from_present(x)
        perception_state = self.perception.update_state_with_imagination(
            x,
            previous_activation=seeded_perception.category_activation,
        )
        category_activation = perception_state.result.category_activation
        action_distribution = np.ones(self.action_count, dtype=float) / self.action_count
        value_activation = np.empty(0, dtype=float)
        effective_input = x.copy()
        value_scalar = 0.0
        future_alignment = 0.0
        action_result: ActionResult | None = None
        temporal_state.unfold(self.transition_model, action_distribution)
        previous_total_change: float | None = None

        for iteration in range(self.convergence_iterations):
            temporal_imagined = temporal_state.imagined_input()
            transition_uncertainty = float(action_distribution @ self.transition_model.uncertainty)
            goal_state = self.goal_system.state(effective_input)
            rollout = self.imagination.prospective_rollout(
                category_activation,
                goal_state.goal_activation,
                value_activation,
                self.transition_model,
                self.action_selector,
            )
            value_bias = self.projections["perception_value"].project(category_activation)
            perception_bias = self._projection_bias(
                self.projections["value_perception"].project(value_activation),
                self.projections["goal_perception"].project(goal_state.goal_activation),
                self.projections["temporal_perception"].project(self._field_activation(self.transition_model)),
                self._fit(rollout.value_category_bias, len(self.perception.categories)),
            )
            top_down_bias = self._projection_expectation(
                self.projections["goal_perception"].top_down(goal_state.goal_activation),
                self.projections["value_perception"].top_down(value_activation),
                self.projections["temporal_perception"].top_down(self._field_activation(self.transition_model)),
                rollout.perceptual_bias,
                temporal_imagined,
            )
            perception_mismatch = 1.0 - max(perception_state.result.resonance_trace or [0.0])
            perception_vigilance = self.vigilance_controller.modulation(
                "perception",
                category_activation,
                mismatch=perception_mismatch,
                reward=reward,
            )
            perception_state = self.perception.update_state_with_imagination(
                x,
                previous_activation=category_activation,
                top_down_bias=top_down_bias,
                imagined_input=rollout.perceptual_bias,
                imagined_category_bias=perception_bias,
                real_input_weight=1.0,
                vigilance_modulation=perception_vigilance,
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
            value_mismatch = 1.0 - max(self.value_system.last_result.resonance_trace) if self.value_system.last_result else 0.0
            value_state = self.value_system.update_state(
                category_activation,
                reward=reward,
                novelty=perception_state.result.novelty,
                context=self.memory.sequence_context(),
                goal_alignment=float(np.clip(max(goal_state.alignment, future_alignment), -1.0, 1.0)),
                action_distribution=action_distribution,
                previous_state=value_activation,
                category_bias=value_bias,
                vigilance_modulation=self.vigilance_controller.modulation(
                    "value",
                    value_activation,
                    mismatch=value_mismatch,
                    reward=reward,
                ),
                learn=False,
            )
            value_activation = value_state.activation
            value_scalar = value_state.scalar
            previous_action_distribution = action_distribution.copy()
            action_context = self.action_selector.schema_input(
                category_activation,
                goal_state.goal_activation,
                value_activation,
                self._field_activation(self.transition_model),
                np.empty(0),
            )
            action_category_bias = self._projection_bias(
                self.projections["goal_action"].project(goal_state.goal_activation),
                self.projections["expectation_action"].project(self._field_activation(self.imagination)),
                self._fit(rollout.action_category_bias, len(self.action_selector.categories)),
            )
            action_state = self.action_selector.resonate_action(
                action_context,
                category_bias=action_category_bias,
                vigilance_modulation=self.vigilance_controller.modulation(
                    "action",
                    self._field_activation(self.action_selector),
                    mismatch=0.0 if self.action_selector.last_result is None else 1.0 - max(self.action_selector.last_result.resonance_trace),
                    reward=reward,
                ),
                exploratory_signal=salience,
                previous_distribution=previous_action_distribution,
            )
            action_distribution = action_state.result.action_distribution
            action_result = action_state.result
            temporal_state.unfold(self.transition_model, action_distribution)
            previous = previous_action_distribution / (np.sum(previous_action_distribution) + 1e-9)
            action_change = float(np.linalg.norm(action_distribution - previous) + action_state.change)
            total_change = perception_state.change + value_state.change + action_change + present_change
            oscillating = previous_total_change is not None and total_change > previous_total_change * 1.25
            previous_total_change = total_change
            if self.debug:
                print(
                    "[convergence] iter="
                    f"{iteration} total_change={total_change:.4f} "
                    f"p_change={perception_state.change:.4f} "
                    f"v_change={value_state.change:.4f} a_change={action_change:.4f} "
                    f"present_change={present_change:.4f} unfold=post_action "
                    f"top_down_norm={np.linalg.norm(top_down_bias):.3f} "
                    f"goal_alignment={goal_state.alignment:.3f} "
                    f"future_alignment={future_alignment:.3f} future_value={future_value:.3f} "
                    f"perception_vigilance={perception_vigilance:.3f} transition_uncertainty={transition_uncertainty:.3f} "
                    f"oscillation={oscillating} "
                    f"slot_norms={np.round(np.linalg.norm(compositional_slots, axis=1), 3).tolist()}"
                )
            if total_change < self.convergence_tolerance:
                break

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
        final_action_result: ActionResult = action_result if action_result is not None else self.action_selector.select(
            category_activation,
            value_scalar,
            salience,
            pathway="coupled",
        )
        effective_input = np.asarray(effective_input, dtype=float)
        return perception_state.result, result.value, final_action_result, effective_input, future_alignment

    def _evaluate_future_state(
        self,
        future_state: np.ndarray,
        previous_activation: np.ndarray,
    ) -> tuple[float, float]:
        if np.linalg.norm(future_state) <= 1e-9:
            return 0.0, 0.0

        future_perception = self.perception.update_state_with_imagination(
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

    def _projection_expectation(self, *signals: np.ndarray) -> np.ndarray:
        expectation = np.zeros(self.input_dim, dtype=float)
        for signal in signals:
            signal = np.asarray(signal, dtype=float)
            if len(signal) != self.input_dim:
                continue
            expectation += signal
        if np.max(expectation) > 1e-9:
            expectation = expectation / np.max(expectation)
        return np.clip(expectation, 0.0, 1.0)

    def _projection_bias(self, *signals: np.ndarray) -> np.ndarray:
        size = max((len(np.asarray(signal, dtype=float)) for signal in signals), default=0)
        bias = np.zeros(size, dtype=float)
        for signal in signals:
            signal = np.asarray(signal, dtype=float)
            bias[: len(signal)] += signal
        if np.sum(bias) <= 1e-9:
            return bias
        return bias / (np.sum(bias) + 1e-9)

    def _field_activation(self, field) -> np.ndarray:
        if getattr(field, "last_result", None) is None:
            return np.zeros(len(field.categories), dtype=float)
        return field.last_result.category_activation

    def _fit(self, values: np.ndarray, size: int) -> np.ndarray:
        fitted = np.zeros(size, dtype=float)
        values = np.asarray(values, dtype=float)
        fitted[: min(size, len(values))] = values[: min(size, len(values))]
        if np.sum(fitted) <= 1e-9:
            return fitted
        return fitted / (np.sum(fitted) + 1e-9)

    def _urgency_signal(self, urgency: float, salience: np.ndarray) -> np.ndarray:
        salience = np.asarray(salience, dtype=float)
        signal = np.zeros(self.action_count, dtype=float)
        signal[: min(self.action_count, len(salience))] = salience[: min(self.action_count, len(salience))]
        if urgency > 0.0 and np.sum(signal) <= 1e-9:
            signal += urgency
        if np.sum(signal) <= 1e-9:
            signal += 1.0 / self.action_count
        return signal / (np.sum(signal) + 1e-9)

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
        goal_update = self.goal_system.update(
            effective_input,
            reward,
            perception.novelty,
            future_alignment=future_alignment,
            vigilance_modulation=self.vigilance_controller.modulation(
                "goal",
                self._field_activation(self.goal_system),
                mismatch=0.0 if self.goal_system.last_result is None else 1.0 - max(self.goal_system.last_result.resonance_trace),
                reward=reward,
            ),
        )
        value_activation = self._field_activation(self.value_system)
        action_activation = self._field_activation(self.action_selector)
        self.transition_model.learn(
            x,
            action_state.action_distribution,
            next_x,
            previous_percept=perception.category_activation,
            action_category=action_activation,
            current_percept=self.perception.process(next_x).category_activation,
            context=value_activation,
        )
        motion = (env.position - previous_position).astype(float) / max(1, env.size - 1)
        self.spatial_context = self.spatial.process(motion)
        self.memory.record_transition(perception.category_index, action)
        self.action_selector.learn_action(
            perception.category_activation,
            action,
            reward_prediction_error=reward - value,
            goal_activation=goal_update.goal_activation,
            value_activation=value_activation,
            temporal_activation=self._field_activation(self.transition_model),
        )
        self.imagination.learn_expectation(effective_input)
        self._learn_projections()
        self._learn_vigilance(reward)
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

    def _learn_projections(self) -> None:
        for projection in self.projections.values():
            projection.learn()

    def _learn_vigilance(self, reward: float) -> None:
        fields = [
            ("perception", self.perception),
            ("value", self.value_system),
            ("goal", self.goal_system),
            ("action", self.action_selector),
            ("temporal", self.transition_model),
            ("expectation", self.imagination),
        ]
        for name, field in fields:
            activation = self._field_activation(field)
            mismatch = 1.0
            if field.last_result is not None and field.last_result.resonance_trace:
                mismatch = 1.0 - max(field.last_result.resonance_trace)
            self.vigilance_controller.learn(name, activation, mismatch, reward)
