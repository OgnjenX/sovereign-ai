from __future__ import annotations

import itertools

import numpy as np

from sovereign_ai.action_selection import ARTActionField, ActionResult
from sovereign_ai.environment import GridWorld
from sovereign_ai.evaluation import ARTValueField, compute_value_state
from sovereign_ai.imagination import ARTExpectationField
from sovereign_ai.learning import GatedLearning
from sovereign_ai.memory import Memory
from sovereign_ai.perception import ARTPerceptualField, PerceptionResult
from sovereign_ai.goal_system import ARTGoalField
from sovereign_ai.spatial import SpatialModule
from sovereign_ai.temporal_state import TemporalState
from sovereign_ai.transition_model import ARTTemporalField
from sovereign_ai.utils import compact_vector, normalize, softmax


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
    ) -> int:
        imagined_prior = self.imagination.action_prior(count=4, keep=2)
        sequence_bias = self._sequence_bias(len(salience))
        value_state = compute_value_state(
            self.value_system,
            perception.category_activation,
            reward=value,
            novelty=perception.novelty,
            context=self.memory.sequence_context(),
            goal_alignment=self.goal_system.state(x).alignment,
            action_distribution=np.full(self.action_count, 1.0 / self.action_count, dtype=float),
            learn=False,
        )
        action_result = self.action_selector.select(
            perception.category_activation,
            value_state.scalar,
            salience,
            imagined_prior,
            sequence_bias,
            pathway="art",
        )
        return action_result.action_index

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
        value_activation = np.ones(6, dtype=float) / 6.0
        np.zeros(self.input_dim, dtype=float)
        effective_input = x.copy()
        value_scalar = 0.0
        future_alignment = 0.0
        action_result: ActionResult | None = None
        temporal_state.unfold(self.transition_model, action_distribution)
        previous_total_change: float | None = None

        for iteration in range(self.convergence_iterations):
            temporal_imagined = temporal_state.imagined_input()
            imagined_input, imagined_category_bias, imagined_action_prior = self.imagination.coupled_priors(
                count=4,
                keep=2,
            )
            transition_uncertainty = float(action_distribution @ self.transition_model.uncertainty)
            goal_state = self.goal_system.state(effective_input)
            value_vigilance = self.value_system.vigilance_signal(value_activation)
            category_preference = self.value_system.category_preference(len(category_activation))
            top_down_bias = self._top_down_expectation(
                goal_state.active_goal,
                temporal_imagined,
                imagined_input,
            )
            perception_state = self.perception.update_state_with_imagination(
                x,
                previous_activation=category_activation,
                top_down_bias=top_down_bias,
                imagined_input=imagined_input,
                imagined_category_bias=self._category_bias_union(imagined_category_bias, category_preference),
                real_input_weight=1.0,
                vigilance_modulation=value_vigilance,
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
            value_state = compute_value_state(
                self.value_system,
                category_activation,
                reward=reward,
                novelty=perception_state.result.novelty,
                context=self.memory.sequence_context(),
                goal_alignment=float(np.clip(max(goal_state.alignment, future_alignment), -1.0, 1.0)),
                action_distribution=action_distribution,
                previous_state=value_activation,
                learn=False,
            )
            value_activation = value_state.activation
            value_scalar = value_state.scalar
            sequence_bias = self._sequence_bias(len(salience))
            previous_action_distribution = action_distribution.copy()
            slot_bias = np.zeros(self.action_count, dtype=float)
            if len(compositional_slots) and len(self.action_selector.action_preferences):
                category_bias = self.action_selector._distribution_to_category_bias(
                    self.action_selector._schema_to_action_distribution(category_activation)
                )
                schema_activation = self.action_selector._resize_activation(category_bias)
                slot_bias = self.action_selector._schema_to_action_distribution(schema_activation)
            action_prior = self._action_bias_union(imagined_action_prior, slot_bias)
            action_schema = self.action_selector._schema_input(
                category_activation,
                value_activation,
                salience,
                action_prior,
                sequence_bias,
            )
            action_category_bias = self.action_selector._action_category_bias(
                previous_action_distribution,
                action_prior,
                sequence_bias,
            )
            previous_category = self.action_selector._distribution_to_category_bias(previous_action_distribution)
            if len(previous_category):
                action_category_bias = action_category_bias + previous_category
            action_vigilance = float(
                np.clip(
                    0.1 * (value_activation[3] if len(value_activation) > 3 else 0.0)
                    - 0.04 * (value_activation[0] if len(value_activation) > 0 else 0.0)
                    + 0.02 * self._urgency_signal(urgency, salience),
                    -0.06,
                    0.14,
                )
            )
            action_state = self.action_selector.update_state(
                action_schema,
                category_bias=action_category_bias,
                vigilance_modulation=action_vigilance,
                learn=False,
            )
            action_distribution = self.action_selector._schema_to_action_distribution(action_state.result.category_activation)
            action_result = ActionResult(
                int(np.argmax(action_distribution)),
                action_distribution,
                action_distribution,
                softmax(1.0 - action_distribution, temperature=0.35),
                action_distribution - softmax(1.0 - action_distribution, temperature=0.35),
                "coupled",
            )
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
                    f"value_vigilance={value_vigilance:.3f} transition_uncertainty={transition_uncertainty:.3f} "
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

    def _top_down_expectation(self, *signals: np.ndarray) -> np.ndarray:
        usable = []
        for signal in signals:
            signal = np.asarray(signal, dtype=float)
            if len(signal) == self.input_dim and np.linalg.norm(signal) > 1e-9:
                usable.append(np.clip(signal, 0.0, 1.0))
        if not usable:
            return np.zeros(self.input_dim, dtype=float)
        return np.maximum.reduce(usable)

    def _category_bias_union(self, *signals: np.ndarray) -> np.ndarray:
        size = len(self.perception.prototypes)
        if size == 0:
            return np.empty(0, dtype=float)
        bias = np.zeros(size, dtype=float)
        for signal in signals:
            signal = np.asarray(signal, dtype=float)
            bias[: min(size, len(signal))] = np.maximum(bias[: min(size, len(signal))], signal[: min(size, len(signal))])
        if np.sum(bias) <= 1e-9:
            return bias
        return bias / (np.sum(bias) + 1e-9)

    def _action_bias_union(self, *signals: np.ndarray) -> np.ndarray:
        bias = np.zeros(self.action_count, dtype=float)
        for signal in signals:
            signal = np.asarray(signal, dtype=float)
            fitted = np.zeros(self.action_count, dtype=float)
            fitted[: min(self.action_count, len(signal))] = signal[: min(self.action_count, len(signal))]
            bias = np.maximum(bias, fitted)
        if np.sum(bias) <= 1e-9:
            return bias
        return bias / (np.sum(bias) + 1e-9)

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
