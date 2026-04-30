from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sovereign_ai.utils import cosine_similarity, normalize, softmax


@dataclass(frozen=True)
class ARTFieldResult:
    category_index: int
    category_activation: np.ndarray
    similarities: np.ndarray
    resonance: bool
    novelty: float
    search_path: list[int]
    resonance_trace: list[float]
    top_down_match: np.ndarray
    effective_vigilance: float


@dataclass(frozen=True)
class ARTFieldState:
    result: ARTFieldResult
    effective_input: np.ndarray
    change: float


class ARTField:
    """Reusable ART-style resonant field.

    Architecture:
        bottom-up signal + inter-field bias -> category competition
        winning category prototype -> top-down expectation
        match >= vigilance -> resonance and optional prototype learning
        match < vigilance -> reset/search over alternate categories

    Coupling is expressed as match/activation/vigilance modulation. No caller may
    overwrite the resonant state of another field directly.
    """

    def __init__(
        self,
        input_dim: int,
        max_categories: int = 16,
        vigilance: float = 0.72,
        competition_temperature: float = 0.12,
        resonance_iterations: int = 6,
        feedback_rate: float = 0.45,
        convergence_tolerance: float = 1e-3,
        learning_rate: float = 0.08,
        seed: int | None = None,
        debug: bool = False,
        name: str = "art-field",
    ) -> None:
        self.input_dim = input_dim
        self.max_categories = max_categories
        self.vigilance = vigilance
        self.competition_temperature = competition_temperature
        self.resonance_iterations = resonance_iterations
        self.feedback_rate = feedback_rate
        self.convergence_tolerance = convergence_tolerance
        self.learning_rate = learning_rate
        self.debug = debug
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.categories = np.empty((0, input_dim), dtype=float)
        self.last_result: ARTFieldResult | None = None

    @property
    def prototypes(self) -> np.ndarray:
        return self.categories

    @prototypes.setter
    def prototypes(self, value: np.ndarray) -> None:
        self.categories = np.asarray(value, dtype=float)

    def process(
        self,
        x: np.ndarray,
        *,
        category_bias: np.ndarray | None = None,
        top_down_bias: np.ndarray | None = None,
        vigilance_modulation: float = 0.0,
        learn: bool = False,
    ) -> ARTFieldResult:
        x = self._prepare_input(x, top_down_bias)
        effective_vigilance = float(np.clip(self.vigilance + vigilance_modulation, 0.0, 1.0))
        if self.categories.size == 0:
            self._add_category(x)
            activation = np.array([1.0])
            result = ARTFieldResult(0, activation, np.array([1.0]), True, 1.0, [0], [1.0], x.copy(), effective_vigilance)
            self.last_result = result
            return result

        similarities = self.match(x)
        category_drive = self._category_drive(similarities, category_bias)
        available = np.ones(len(similarities), dtype=bool)
        search_path: list[int] = []

        while np.any(available):
            masked = np.where(available, category_drive, -np.inf)
            winner = int(np.argmax(masked))
            search_path.append(winner)
            resonance, resonance_trace, top_down_match = self.resonance(x, winner, effective_vigilance)
            if resonance:
                activation = softmax(category_drive, self.competition_temperature)
                novelty = float(max(0.0, 1.0 - np.max(similarities)))
                if learn:
                    self.learn(winner, x)
                self._log(winner, similarities, resonance, novelty, search_path, resonance_trace)
                result = ARTFieldResult(
                    winner,
                    activation,
                    similarities,
                    resonance,
                    novelty,
                    search_path,
                    resonance_trace,
                    top_down_match,
                    effective_vigilance,
                )
                self.last_result = result
                return result
            available[winner] = False

        if len(self.categories) < self.max_categories:
            winner = self._add_category(x)
            similarities = self.match(x)
            category_drive = self._category_drive(similarities, category_bias)
            activation = softmax(category_drive, self.competition_temperature)
            novelty = 1.0
            resonance_trace = [1.0]
            top_down_match = x.copy()
            search_path.append(winner)
            resonance = True
        else:
            winner = int(np.argmax(category_drive))
            activation = softmax(category_drive, self.competition_temperature)
            novelty = float(max(0.0, 1.0 - np.max(similarities)))
            resonance, resonance_trace, top_down_match = self.resonance(x, winner, effective_vigilance)

        self._log(winner, similarities, resonance, novelty, search_path, resonance_trace)
        result = ARTFieldResult(
            winner,
            activation,
            similarities,
            resonance,
            novelty,
            search_path,
            resonance_trace,
            top_down_match,
            effective_vigilance,
        )
        self.last_result = result
        return result

    def update_state(
        self,
        x: np.ndarray,
        *,
        previous_activation: np.ndarray | None = None,
        top_down_bias: np.ndarray | None = None,
        category_bias: np.ndarray | None = None,
        vigilance_modulation: float = 0.0,
        learn: bool = False,
    ) -> ARTFieldState:
        previous = self._resize_activation(previous_activation)
        result = self.process(
            x,
            category_bias=category_bias,
            top_down_bias=top_down_bias,
            vigilance_modulation=vigilance_modulation,
            learn=learn,
        )
        if len(previous) != len(result.category_activation):
            previous = self._resize_activation(previous)
        change = float(np.linalg.norm(result.category_activation - previous))
        return ARTFieldState(result, result.top_down_match.copy(), change)

    def match(self, x: np.ndarray) -> np.ndarray:
        return cosine_similarity(x, self.categories)

    def resonance(
        self,
        x: np.ndarray,
        category_index: int,
        vigilance: float | None = None,
    ) -> tuple[bool, list[float], np.ndarray]:
        threshold = self.vigilance if vigilance is None else vigilance
        prototype = self.categories[category_index]
        bottom_up = np.asarray(x, dtype=float).copy()
        trace: list[float] = []
        previous_match = 0.0

        for _ in range(self.resonance_iterations):
            top_down = np.minimum(bottom_up, prototype)
            refined = (1.0 - self.feedback_rate) * bottom_up + self.feedback_rate * top_down
            match = float(np.sum(top_down) / (np.sum(np.maximum(x, 0.0)) + 1e-9))
            trace.append(match)
            if match >= threshold and abs(match - previous_match) <= self.convergence_tolerance:
                return True, trace, top_down
            bottom_up = refined
            previous_match = match

        return trace[-1] >= threshold, trace, np.minimum(bottom_up, prototype)

    def reset(self, available: np.ndarray, category_index: int) -> np.ndarray:
        updated = available.copy()
        updated[category_index] = False
        return updated

    def learn(self, category_index: int, x: np.ndarray, learning_rate: float | None = None) -> None:
        rate = self.learning_rate if learning_rate is None else learning_rate
        x = np.asarray(x, dtype=float)
        prototype = self.categories[category_index]
        art_intersection = np.minimum(prototype, x)
        self.categories[category_index] = normalize((1.0 - rate) * prototype + rate * art_intersection)

    def reconstruct(self, category_activation: np.ndarray) -> np.ndarray:
        if len(self.categories) == 0:
            return np.zeros(self.input_dim, dtype=float)
        activation = self._resize_activation(category_activation)
        return activation @ self.categories

    def generate_from_category(self, category_index: int, noise_scale: float = 0.03) -> np.ndarray:
        prototype = self.categories[category_index]
        noise = self.rng.normal(0.0, noise_scale, self.input_dim)
        return np.clip(prototype + noise, 0.0, 1.0)

    def _prepare_input(self, x: np.ndarray, top_down_bias: np.ndarray | None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(x) != self.input_dim:
            resized = np.zeros(self.input_dim, dtype=float)
            resized[: min(len(x), self.input_dim)] = x[: min(len(x), self.input_dim)]
            x = resized
        if top_down_bias is not None:
            bias = np.asarray(top_down_bias, dtype=float)
            if len(bias) != self.input_dim:
                resized = np.zeros(self.input_dim, dtype=float)
                resized[: min(len(bias), self.input_dim)] = bias[: min(len(bias), self.input_dim)]
                bias = resized
            x = np.clip(x + bias, 0.0, 1.0)
        return x

    def _category_drive(
        self,
        similarities: np.ndarray,
        category_bias: np.ndarray | None,
    ) -> np.ndarray:
        drive = similarities.copy()
        if category_bias is not None:
            drive += self._resize_activation(category_bias)
        return drive

    def _add_category(self, x: np.ndarray) -> int:
        if self.categories.size == 0:
            self.categories = np.asarray(x, dtype=float).reshape(1, -1).copy()
        else:
            self.categories = np.vstack([self.categories, np.asarray(x, dtype=float)])
        return len(self.categories) - 1

    def _resize_activation(self, activation: np.ndarray | None) -> np.ndarray:
        size = len(self.categories)
        if size == 0:
            return np.empty(0, dtype=float)
        resized = np.zeros(size, dtype=float)
        if activation is not None:
            activation = np.asarray(activation, dtype=float)
            resized[: min(size, len(activation))] = activation[: min(size, len(activation))]
        if np.sum(resized) <= 1e-9:
            resized += 1.0 / size
        return resized / (np.sum(resized) + 1e-9)

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
                f"[{self.name}] winner={winner} resonance={resonance} novelty={novelty:.3f} "
                f"search={search_path} similarities={sims} resonance_trace={trace}"
            )
