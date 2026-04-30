from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sovereign_ai.art_field import ARTField
from sovereign_ai.evaluation import ARTValueField
from sovereign_ai.perception import ARTPerceptualField


@dataclass(frozen=True)
class ImaginedCandidate:
    category_index: int
    activation: np.ndarray
    reconstruction: np.ndarray
    value: float
    action_prior: np.ndarray


class ARTExpectationField(ARTField):
    """Top-down ART field for imagined expectations and hypothesis testing."""

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

    def sample_candidates(
        self,
        count: int = 4,
        keep: int = 2,
    ) -> list[ImaginedCandidate]:
        self._sync_with_perception()
        if len(self.categories) == 0:
            return []

        candidates: list[ImaginedCandidate] = []
        base_activation = np.ones(len(self.categories), dtype=float) / len(self.categories)
        for index in range(min(count, len(self.categories))):
            category_bias = np.roll(base_activation, index)
            hypothesis = self.reconstruct(category_bias)
            expectation = self.process(hypothesis, category_bias=category_bias, learn=False)
            reconstruction = expectation.top_down_match
            perceptual_test = self.perception.update_state_with_imagination(
                reconstruction,
                imagined_input=reconstruction,
                imagined_category_bias=expectation.category_activation,
                real_input_weight=0.0,
            )
            value = self.value_system.evaluate(
                perceptual_test.result.category_activation,
                reward=0.0,
                novelty=perceptual_test.result.novelty,
                learn=False,
            ).value
            action_prior = self._action_prior(expectation.category_activation, value)
            candidates.append(
                ImaginedCandidate(
                    expectation.category_index,
                    expectation.category_activation,
                    reconstruction,
                    value,
                    action_prior,
                )
            )

        ranked = sorted(candidates, key=lambda item: item.value, reverse=True)[:keep]
        if self.debug and ranked:
            trace = [(c.category_index, round(c.value, 3)) for c in ranked]
            print(f"[expectation] kept={trace}")
        return ranked

    def action_prior(self, count: int = 5, keep: int = 2) -> np.ndarray:
        candidates = self.sample_candidates(count=count, keep=keep)
        action_size = self._action_size()
        if not candidates:
            return np.zeros(action_size, dtype=float)
        activation = self._candidate_activation(candidates)
        prior = np.zeros(action_size, dtype=float)
        for weight, candidate in zip(activation, candidates):
            prior += weight * candidate.action_prior
        return prior / (np.sum(prior) + 1e-9)

    def coupled_priors(self, count: int = 5, keep: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        candidates = self.sample_candidates(count=count, keep=keep)
        action_size = self._action_size()
        if not candidates:
            return (
                np.zeros(self.input_dim, dtype=float),
                np.zeros(len(self.perception.prototypes), dtype=float),
                np.zeros(action_size, dtype=float),
            )
        activation = self._candidate_activation(candidates)
        imagined_input = np.zeros(self.input_dim, dtype=float)
        category_bias = np.zeros(len(self.perception.prototypes), dtype=float)
        action_prior = np.zeros(action_size, dtype=float)
        for weight, candidate in zip(activation, candidates):
            imagined_input += weight * candidate.reconstruction
            category_bias[: min(len(category_bias), len(candidate.activation))] += (
                weight * candidate.activation[: min(len(category_bias), len(candidate.activation))]
            )
            action_prior += weight * candidate.action_prior
        if self.debug:
            print(
                "[expectation-coupled] "
                f"input_norm={np.linalg.norm(imagined_input):.3f} "
                f"category_bias={np.round(category_bias, 3)} action_prior={np.round(action_prior, 3)}"
            )
        return imagined_input, category_bias, action_prior / (np.sum(action_prior) + 1e-9)

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

    def _action_prior(self, expectation_activation: np.ndarray, value: float) -> np.ndarray:
        action_size = self._action_size()
        if action_size == 0:
            return np.empty(0, dtype=float)
        if len(self.action_preferences) == 0:
            return np.ones(action_size, dtype=float) / action_size
        activation = np.zeros(len(self.action_preferences), dtype=float)
        activation[: min(len(activation), len(expectation_activation))] = expectation_activation[
            : min(len(activation), len(expectation_activation))
        ]
        prior = activation @ self.action_preferences[: len(activation)]
        if value > 0.0:
            prior = np.maximum(prior, 0.0)
        if np.sum(prior) <= 1e-9:
            prior += 1.0 / action_size
        return prior / (np.sum(prior) + 1e-9)

    def _candidate_activation(self, candidates: list[ImaginedCandidate]) -> np.ndarray:
        scores = np.asarray([candidate.value for candidate in candidates], dtype=float)
        scores = scores - np.min(scores)
        if np.sum(scores) <= 1e-9:
            return np.ones(len(candidates), dtype=float) / len(candidates)
        return scores / (np.sum(scores) + 1e-9)

    def _action_size(self) -> int:
        if len(self.action_preferences) == 0:
            return 0
        return self.action_preferences.shape[1]


ImaginationLoop = ARTExpectationField
