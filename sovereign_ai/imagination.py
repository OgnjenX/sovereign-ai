from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sovereign_ai.evaluation import ValueSystem
from sovereign_ai.perception import ARTPerception
from sovereign_ai.utils import cosine_similarity, softmax


@dataclass(frozen=True)
class ImaginedCandidate:
    category_index: int
    activation: np.ndarray
    reconstruction: np.ndarray
    value: float
    action_prior: np.ndarray


class ImaginationLoop:
    """Top-down category-to-input reconstruction with partial activations."""

    def __init__(
        self,
        perception: ARTPerception,
        value_system: ValueSystem,
        category_action_weights: np.ndarray,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        self.perception = perception
        self.value_system = value_system
        self.category_action_weights = category_action_weights
        self.rng = np.random.default_rng(seed)
        self.debug = debug

    def sample_candidates(
        self,
        count: int = 4,
        keep: int = 2,
        partial_temperature: float = 0.4,
    ) -> list[ImaginedCandidate]:
        category_count = len(self.perception.prototypes)
        if category_count == 0:
            return []

        base_scores = self.rng.normal(0.0, 1.0, category_count)
        candidates: list[ImaginedCandidate] = []
        for _ in range(count):
            center = int(self.rng.integers(0, category_count))
            scores = base_scores.copy()
            scores[center] += 1.5
            activation = softmax(scores, partial_temperature)
            x_hat = self.perception.reconstruct(activation)
            similarities = cosine_similarity(x_hat, self.perception.prototypes)
            novelty = float(max(0.0, 1.0 - np.max(similarities)))
            value = self.value_system.evaluate(
                activation,
                reward=0.0,
                novelty=novelty,
                learn=False,
            ).value
            active_weights = self.category_action_weights[: len(activation)]
            action_prior = softmax(activation @ active_weights + value, temperature=0.4)
            candidates.append(ImaginedCandidate(center, activation, x_hat, value, action_prior))

        ranked = sorted(candidates, key=lambda item: item.value, reverse=True)[:keep]
        if self.debug and ranked:
            trace = [(c.category_index, round(c.value, 3)) for c in ranked]
            print(f"[imagination] kept={trace}")
        return ranked

    def action_prior(self, count: int = 5, keep: int = 2) -> np.ndarray:
        candidates = self.sample_candidates(count=count, keep=keep)
        if not candidates:
            return np.zeros(self.category_action_weights.shape[1], dtype=float)

        weights = softmax(np.asarray([candidate.value for candidate in candidates]), temperature=0.35)
        prior = np.zeros_like(candidates[0].action_prior)
        for weight, candidate in zip(weights, candidates):
            prior += weight * candidate.action_prior
        if self.debug:
            print(f"[imagination-action] prior={np.round(prior, 3)}")
        return prior

    def coupled_priors(self, count: int = 5, keep: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        candidates = self.sample_candidates(count=count, keep=keep)
        action_size = self.category_action_weights.shape[1]
        category_count = len(self.perception.prototypes)
        if not candidates:
            return (
                np.zeros(self.perception.input_dim, dtype=float),
                np.zeros(category_count, dtype=float),
                np.zeros(action_size, dtype=float),
            )

        weights = softmax(np.asarray([candidate.value for candidate in candidates]), temperature=0.35)
        imagined_input = np.zeros(self.perception.input_dim, dtype=float)
        category_bias = np.zeros(category_count, dtype=float)
        action_prior = np.zeros(action_size, dtype=float)
        for weight, candidate in zip(weights, candidates):
            imagined_input += weight * candidate.reconstruction
            category_bias += weight * candidate.activation
            action_prior += weight * candidate.action_prior
        if self.debug:
            print(
                "[imagination-coupled] "
                f"input_norm={np.linalg.norm(imagined_input):.3f} "
                f"category_bias={np.round(category_bias, 3)} action_prior={np.round(action_prior, 3)}"
            )
        return imagined_input, category_bias, action_prior
