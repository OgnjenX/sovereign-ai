from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sovereign_ai.utils import cosine_similarity, softmax


@dataclass(frozen=True)
class PerceptionResult:
    category_index: int
    category_activation: np.ndarray
    similarities: np.ndarray
    resonance: bool
    novelty: float
    search_path: list[int]
    resonance_trace: list[float]
    top_down_match: np.ndarray


class ARTPerception:
    """ART-like competitive category system with vigilance and reset search."""

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
        self.input_dim = input_dim
        self.max_categories = max_categories
        self.vigilance = vigilance
        self.competition_temperature = competition_temperature
        self.resonance_iterations = resonance_iterations
        self.feedback_rate = feedback_rate
        self.convergence_tolerance = convergence_tolerance
        self.debug = debug
        self.rng = np.random.default_rng(seed)
        self.prototypes = np.empty((0, input_dim), dtype=float)

    def process(self, x: np.ndarray) -> PerceptionResult:
        x = np.asarray(x, dtype=float)
        if self.prototypes.size == 0:
            self.prototypes = x.reshape(1, -1).copy()
            activation = np.array([1.0])
            return PerceptionResult(0, activation, np.array([1.0]), True, 1.0, [0], [1.0], x.copy())

        similarities = cosine_similarity(x, self.prototypes)
        available = np.ones(len(similarities), dtype=bool)
        search_path: list[int] = []

        while np.any(available):
            masked = np.where(available, similarities, -np.inf)
            winner = int(np.argmax(masked))
            search_path.append(winner)
            resonance, resonance_trace, top_down_match = self._resonate(x, winner)
            if resonance:
                activation = softmax(similarities, self.competition_temperature)
                novelty = float(max(0.0, 1.0 - np.max(similarities)))
                self._log(winner, similarities, resonance, novelty, search_path, resonance_trace)
                return PerceptionResult(
                    winner,
                    activation,
                    similarities,
                    resonance,
                    novelty,
                    search_path,
                    resonance_trace,
                    top_down_match,
                )
            available[winner] = False

        if len(self.prototypes) < self.max_categories:
            self.prototypes = np.vstack([self.prototypes, x])
            winner = len(self.prototypes) - 1
            similarities = cosine_similarity(x, self.prototypes)
            activation = softmax(similarities, self.competition_temperature)
            novelty = 1.0
            resonance = True
            resonance_trace = [1.0]
            top_down_match = x.copy()
            search_path.append(winner)
        else:
            winner = int(np.argmax(similarities))
            activation = softmax(similarities, self.competition_temperature)
            novelty = float(max(0.0, 1.0 - np.max(similarities)))
            resonance, resonance_trace, top_down_match = self._resonate(x, winner)

        self._log(winner, similarities, resonance, novelty, search_path, resonance_trace)
        return PerceptionResult(
            winner,
            activation,
            similarities,
            resonance,
            novelty,
            search_path,
            resonance_trace,
            top_down_match,
        )

    def _resonate(self, x: np.ndarray, category_index: int) -> tuple[bool, list[float], np.ndarray]:
        prototype = self.prototypes[category_index]
        bottom_up = x.copy()
        trace: list[float] = []
        previous_match = 0.0

        for _ in range(self.resonance_iterations):
            top_down = np.minimum(bottom_up, prototype)
            refined = (1.0 - self.feedback_rate) * bottom_up + self.feedback_rate * top_down
            match = float(np.sum(top_down) / (np.sum(x) + 1e-9))
            trace.append(match)
            if match >= self.vigilance and abs(match - previous_match) <= self.convergence_tolerance:
                return True, trace, top_down
            bottom_up = refined
            previous_match = match

        return trace[-1] >= self.vigilance, trace, np.minimum(bottom_up, prototype)

    def reconstruct(self, category_activation: np.ndarray) -> np.ndarray:
        activation = np.asarray(category_activation, dtype=float)
        activation = activation / (np.sum(activation) + 1e-9)
        return activation @ self.prototypes

    def generate_from_category(self, category_index: int, noise_scale: float = 0.03) -> np.ndarray:
        prototype = self.prototypes[category_index]
        noise = self.rng.normal(0.0, noise_scale, self.input_dim)
        return np.clip(prototype + noise, 0.0, 1.0)

    def _log(
        self,
        winner: int,
        similarities: np.ndarray,
        resonance: bool,
        novelty: float,
        search_path: list[int],
        resonance_trace: list[float],
    ) -> None:
        if self.debug:
            sims = np.array2string(similarities, precision=2, suppress_small=True)
            trace = np.array2string(np.asarray(resonance_trace), precision=3, suppress_small=True)
            print(
                "[perception] winner="
                f"{winner} resonance={resonance} novelty={novelty:.3f} "
                f"search={search_path} similarities={sims} resonance_trace={trace}"
            )
