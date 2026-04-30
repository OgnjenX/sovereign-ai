from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sovereign_ai.action_selection import ARTActionField
from sovereign_ai.art_field import ARTField
from sovereign_ai.evaluation import ARTValueField
from sovereign_ai.perception import ARTPerceptualField
from sovereign_ai.transition_model import ARTTemporalField


@dataclass(frozen=True)
class ImaginedCandidate:
    category_index: int
    activation: np.ndarray
    reconstruction: np.ndarray
    value: float
    action_prior: np.ndarray


@dataclass(frozen=True)
class ImaginationRollout:
    perceptual_bias: np.ndarray
    action_category_bias: np.ndarray
    value_category_bias: np.ndarray
    trace: list[dict[str, object]]


class ARTExpectationField(ARTField):
    """Prospective ART field for short horizon resonant imagination."""

    def __init__(
        self,
        perception: ARTPerceptualField,
        value_system: ARTValueField,
        action_preferences: np.ndarray,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(
            input_dim=perception.input_dim,
            max_categories=perception.max_categories,
            vigilance=0.56,
            competition_temperature=0.28,
            learning_rate=0.06,
            seed=seed,
            debug=debug,
            name="expectation",
        )
        self.perception = perception
        self.value_system = value_system
        self.action_preferences = action_preferences
        self.last_rollout = ImaginationRollout(
            np.zeros(perception.input_dim, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            [],
        )

    def prospective_rollout(
        self,
        current_percept: np.ndarray,
        goal_activation: np.ndarray,
        value_activation: np.ndarray,
        temporal_field: ARTTemporalField,
        action_field: ARTActionField,
        horizon: int = 3,
    ) -> ImaginationRollout:
        self._sync_with_perception()
        perceptual_bias = np.zeros(self.input_dim, dtype=float)
        action_bias = np.zeros(len(action_field.categories), dtype=float)
        value_bias = np.zeros(len(self.value_system.categories), dtype=float)
        trace: list[dict[str, object]] = []
        percept_activation = self._fit(current_percept, len(self.perception.categories))
        action_activation = np.zeros(max(1, len(action_field.categories)), dtype=float)

        for step in range(horizon):
            (
                step_perceptual_bias,
                step_action_bias,
                step_value_bias,
                step_trace,
                percept_activation,
                action_activation,
                accepted,
            ) = self._rollout_step(
                percept_activation,
                action_activation,
                goal_activation,
                value_activation,
                temporal_field,
                action_field,
                step,
            )
            perceptual_bias += step_perceptual_bias
            action_bias = self._accumulate(action_bias, step_action_bias, len(action_bias))
            value_bias = self._accumulate(value_bias, step_value_bias, len(value_bias))
            trace.append(step_trace)
            if not accepted:
                break

        if np.sum(perceptual_bias) > 1e-9:
            perceptual_bias = perceptual_bias / np.max(perceptual_bias)
        if np.sum(action_bias) > 1e-9:
            action_bias = action_bias / (np.sum(action_bias) + 1e-9)
        if np.sum(value_bias) > 1e-9:
            value_bias = value_bias / (np.sum(value_bias) + 1e-9)
        self.last_rollout = ImaginationRollout(perceptual_bias, action_bias, value_bias, trace)
        if self.debug:
            print(f"[expectation-rollout] trace={trace}")
        return self.last_rollout

    def _rollout_step(
        self,
        percept_activation: np.ndarray,
        action_activation: np.ndarray,
        goal_activation: np.ndarray,
        value_activation: np.ndarray,
        temporal_field: ARTTemporalField,
        action_field: ARTActionField,
        step: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], np.ndarray, np.ndarray, bool]:
        temporal_prediction = temporal_field.predict_categories(
            percept_activation,
            action_activation,
            percept_activation,
            value_activation,
        )
        candidate_activation = temporal_prediction.perceptual_bias
        if np.sum(candidate_activation) <= 1e-9 and len(self.perception.categories):
            candidate_activation = self._next_activation(percept_activation, len(self.perception.categories))
        candidate_state = self.perception.reconstruct(candidate_activation)
        expectation = self.process(candidate_state, category_bias=candidate_activation, learn=False)
        perceptual_test = self.perception.update_state_with_imagination(
            np.zeros(self.input_dim, dtype=float),
            imagined_input=expectation.top_down_match,
            imagined_category_bias=expectation.category_activation,
            real_input_weight=0.0,
        )
        accepted = expectation.resonance and perceptual_test.result.resonance and not temporal_prediction.reset
        action_context = action_field.schema_input(
            perceptual_test.result.category_activation,
            goal_activation,
            value_activation,
            temporal_prediction.action_bias,
            np.empty(0),
        )
        action_state = action_field.resonate_action(
            action_context,
            category_bias=self._fit(temporal_prediction.action_bias, len(action_field.categories)),
            exploratory_signal=temporal_prediction.action_bias,
            pathway="imagined",
        )
        value_state = self.value_system.resonate_value(
            perceptual_test.result.category_activation,
            reward=0.0,
            novelty=perceptual_test.result.novelty,
            context=temporal_prediction.confidence,
            goal_alignment=float(np.max(goal_activation)) if len(goal_activation) else 0.0,
            action_distribution=action_state.result.action_distribution,
            previous_state=value_activation,
            learn=False,
        )
        perceptual_bias = np.zeros(self.input_dim, dtype=float)
        action_bias = np.zeros(len(action_field.categories), dtype=float)
        value_bias = np.zeros(len(self.value_system.categories), dtype=float)
        if accepted and action_state.result.action_distribution.size:
            perceptual_bias += perceptual_test.result.top_down_match
            action_bias = self._accumulate(action_bias, action_state.result.action_distribution, len(action_bias))
            value_bias = self._accumulate(value_bias, value_state.activation, len(value_bias))
        trace = {
            "step": step,
            "expectation_resonance": expectation.resonance,
            "perception_resonance": perceptual_test.result.resonance,
            "action": action_state.result.action_index,
            "temporal_mismatch": temporal_prediction.mismatch,
            "accepted": accepted,
            "search": expectation.search_path,
        }
        return (
            perceptual_bias,
            action_bias,
            value_bias,
            trace,
            perceptual_test.result.category_activation,
            self._fit(action_state.result.action_distribution, len(action_field.categories)),
            accepted,
        )

    def sample_candidates(
        self,
        count: int = 4,
        keep: int = 2,
    ) -> list[ImaginedCandidate]:
        self._sync_with_perception()
        if len(self.categories) == 0:
            return []
        candidates: list[ImaginedCandidate] = []
        for index in range(min(count, len(self.categories))):
            activation = np.zeros(len(self.categories), dtype=float)
            activation[index] = 1.0
            expectation = self.process(self.reconstruct(activation), category_bias=activation, learn=False)
            perceptual_test = self.perception.update_state_with_imagination(
                np.zeros(self.input_dim, dtype=float),
                imagined_input=expectation.top_down_match,
                imagined_category_bias=expectation.category_activation,
                real_input_weight=0.0,
            )
            value = self.value_system.evaluate(
                perceptual_test.result.category_activation,
                reward=0.0,
                novelty=perceptual_test.result.novelty,
                learn=False,
            ).value
            candidates.append(
                ImaginedCandidate(
                    expectation.category_index,
                    expectation.category_activation,
                    expectation.top_down_match,
                    value,
                    self._action_prior(expectation.category_activation),
                )
            )
        return sorted(candidates, key=lambda item: item.value, reverse=True)[:keep]

    def action_prior(self, count: int = 5, keep: int = 2) -> np.ndarray:
        if len(self.last_rollout.action_category_bias):
            return self.last_rollout.action_category_bias
        candidates = self.sample_candidates(count=count, keep=keep)
        action_size = self._action_size()
        if not candidates:
            return np.zeros(action_size, dtype=float)
        prior = np.zeros(action_size, dtype=float)
        for candidate in candidates:
            prior += candidate.action_prior
        if np.sum(prior) <= 1e-9:
            return prior
        return prior / (np.sum(prior) + 1e-9)

    def coupled_priors(self, count: int = 5, keep: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.last_rollout.trace:
            return (
                self.last_rollout.perceptual_bias,
                np.zeros(len(self.perception.categories), dtype=float),
                self.last_rollout.action_category_bias,
            )
        candidates = self.sample_candidates(count=count, keep=keep)
        action_size = self._action_size()
        if not candidates:
            return (
                np.zeros(self.input_dim, dtype=float),
                np.zeros(len(self.perception.prototypes), dtype=float),
                np.zeros(action_size, dtype=float),
            )
        imagined_input = np.zeros(self.input_dim, dtype=float)
        category_bias = np.zeros(len(self.perception.prototypes), dtype=float)
        action_prior = np.zeros(action_size, dtype=float)
        for candidate in candidates:
            imagined_input += candidate.reconstruction
            category_bias[: min(len(category_bias), len(candidate.activation))] += candidate.activation[
                : min(len(category_bias), len(candidate.activation))
            ]
            action_prior += candidate.action_prior
        if np.sum(action_prior) > 1e-9:
            action_prior = action_prior / (np.sum(action_prior) + 1e-9)
        return imagined_input, category_bias, action_prior

    def learn_expectation(self, perceptual_input: np.ndarray) -> None:
        self.process(perceptual_input, learn=True)

    def _sync_with_perception(self) -> None:
        if len(self.perception.prototypes) == 0:
            return
        if len(self.categories) == 0:
            self.categories = self.perception.prototypes.copy()
            return
        if len(self.categories) < len(self.perception.prototypes):
            missing = self.perception.prototypes[len(self.categories) :]
            self.categories = np.vstack([self.categories, missing])

    def _action_prior(self, expectation_activation: np.ndarray) -> np.ndarray:
        action_size = self._action_size()
        if action_size == 0:
            return np.empty(0, dtype=float)
        if len(self.action_preferences) == 0:
            return np.zeros(action_size, dtype=float)
        activation = self._fit(expectation_activation, len(self.action_preferences))
        prior = activation @ self.action_preferences[: len(activation)]
        if np.sum(prior) <= 1e-9:
            return np.zeros(action_size, dtype=float)
        return prior / (np.sum(prior) + 1e-9)

    def _next_activation(self, activation: np.ndarray, size: int) -> np.ndarray:
        fitted = self._fit(activation, size)
        if np.sum(fitted) <= 1e-9:
            fitted[0] = 1.0
            return fitted
        return np.roll(fitted, 1)

    def _accumulate(self, target: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
        if size == 0:
            return target
        fitted = self._fit(values, size)
        return target + fitted

    def _fit(self, values: np.ndarray, size: int) -> np.ndarray:
        fitted = np.zeros(size, dtype=float)
        values = np.asarray(values, dtype=float)
        fitted[: min(size, len(values))] = values[: min(size, len(values))]
        if np.sum(fitted) <= 1e-9:
            return fitted
        return fitted / (np.sum(fitted) + 1e-9)

    def _action_size(self) -> int:
        if len(self.action_preferences) == 0:
            return 0
        return self.action_preferences.shape[1]


class ImaginationLoop(ARTExpectationField):
    """Compatibility wrapper preserving the old constructor name."""

    pass
