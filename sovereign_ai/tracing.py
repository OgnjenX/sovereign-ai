"""Structured tracing primitives and recorder for ART dynamics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FieldMetrics:
    """Per-iteration match and vigilance metrics for a field event."""

    resonance: bool
    vigilance: float
    match: float
    search_path: list[int]
    novelty: float
    change: float


@dataclass(frozen=True)
class FieldTrace:
    """Recorded iteration for one field during convergence."""

    field_name: str
    step: int
    iteration: int
    category_index: int
    metrics: FieldMetrics

    @property
    def resonance(self) -> bool:
        """Expose resonance state for backward-compatible access."""

        return self.metrics.resonance

    @property
    def vigilance(self) -> float:
        """Expose effective vigilance used during category search."""

        return self.metrics.vigilance

    @property
    def match(self) -> float:
        """Expose best match score for backward-compatible access."""

        return self.metrics.match

    @property
    def search_path(self) -> list[int]:
        """Expose evaluated category path for backward-compatible access."""

        return self.metrics.search_path

    @property
    def novelty(self) -> float:
        """Expose novelty score for backward-compatible access."""

        return self.metrics.novelty

    @property
    def change(self) -> float:
        """Expose activation delta for backward-compatible access."""

        return self.metrics.change


@dataclass(frozen=True)
class ProjectionTrace:
    """Recorded update event for an associative projection."""

    name: str
    step: int
    update_norm: float
    source_size: int
    target_size: int


@dataclass(frozen=True)
class BehaviorTrace:
    """Recorded behavior-level outcome for a simulation step."""

    step: int
    action: int
    reward: float
    value: float
    goal_alignment: float
    temporal_mismatch: float
    imagination_accepted: bool


class TraceRecorder:
    """Machine-readable trace collector for ART field dynamics."""

    def __init__(self) -> None:
        self.field_traces: list[FieldTrace] = []
        self.projection_traces: list[ProjectionTrace] = []
        self.behavior_traces: list[BehaviorTrace] = []
        self.category_counts: dict[str, int] = {}

    def record_field(
        self,
        trace: FieldTrace,
        category_count: int | None = None,
    ) -> None:
        """Record a single field convergence event."""

        normalized = FieldTrace(
            field_name=trace.field_name,
            step=int(trace.step),
            iteration=int(trace.iteration),
            category_index=int(trace.category_index),
            metrics=FieldMetrics(
                resonance=bool(trace.resonance),
                vigilance=float(trace.vigilance),
                match=float(trace.match),
                search_path=[int(item) for item in trace.search_path],
                novelty=float(trace.novelty),
                change=float(trace.change),
            ),
        )
        self.field_traces.append(normalized)
        if category_count is not None:
            self.category_counts[normalized.field_name] = int(category_count)

    def record_projection(self, trace: ProjectionTrace) -> None:
        """Record a projection learning event."""

        self.projection_traces.append(
            ProjectionTrace(
                name=trace.name,
                step=int(trace.step),
                update_norm=float(trace.update_norm),
                source_size=int(trace.source_size),
                target_size=int(trace.target_size),
            )
        )

    def record_behavior(self, trace: BehaviorTrace) -> None:
        """Record one behavior-level step trace."""

        self.behavior_traces.append(
            BehaviorTrace(
                step=int(trace.step),
                action=int(trace.action),
                reward=float(trace.reward),
                value=float(trace.value),
                goal_alignment=float(trace.goal_alignment),
                temporal_mismatch=float(trace.temporal_mismatch),
                imagination_accepted=bool(trace.imagination_accepted),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize all traces and summary metrics to a dictionary."""

        return {
            "fields": [_field_trace_to_dict(item) for item in self.field_traces],
            "projections": [asdict(item) for item in self.projection_traces],
            "behavior": [asdict(item) for item in self.behavior_traces],
            "summary": self.summary(),
        }

    def to_json(self, path: str | Path) -> None:
        """Write trace payload to a JSON file."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def summary(self) -> dict[str, Any]:
        """Compute aggregate trace statistics across fields and behavior."""

        field_names = sorted({trace.field_name for trace in self.field_traces})
        field_summary: dict[str, dict[str, float | int]] = {}
        for name in field_names:
            traces = [trace for trace in self.field_traces if trace.field_name == name]
            field_summary[name] = {
                "categories": self.category_counts.get(name, 0),
                "events": len(traces),
                "resonance_rate": _mean([1.0 if trace.resonance else 0.0 for trace in traces]),
                "average_search_length": _mean([len(trace.search_path) for trace in traces]),
                "average_vigilance": _mean([trace.vigilance for trace in traces]),
                "average_match": _mean([trace.match for trace in traces]),
            }

        projection_summary: dict[str, dict[str, float | int]] = {}
        for name in sorted({trace.name for trace in self.projection_traces}):
            traces = [trace for trace in self.projection_traces if trace.name == name]
            projection_summary[name] = {
                "events": len(traces),
                "average_update_norm": _mean([trace.update_norm for trace in traces]),
                "total_update_norm": float(np.sum([trace.update_norm for trace in traces])),
            }

        actions = [trace.action for trace in self.behavior_traces]
        action_distribution = {
            str(action): actions.count(action) for action in sorted(set(actions))
        }
        rewards = [trace.reward for trace in self.behavior_traces]
        temporal = [trace.temporal_mismatch for trace in self.behavior_traces]
        imagination = [1.0 if trace.imagination_accepted else 0.0 for trace in self.behavior_traces]
        return {
            "steps": len(self.behavior_traces),
            "fields": field_summary,
            "projections": projection_summary,
            "action_distribution": action_distribution,
            "reward_mean": _mean(rewards),
            "reward_trend": _trend(rewards),
            "temporal_mismatch_mean": _mean(temporal),
            "temporal_mismatch_trend": _trend(temporal),
            "imagination_acceptance_rate": _mean(imagination),
        }


def _mean(values: list[float]) -> float:
    """Return mean value or zero for an empty list."""

    if not values:
        return 0.0
    return float(np.mean(values))


def _trend(values: list[float]) -> float:
    """Estimate trend as second-half mean minus first-half mean."""

    if len(values) < 2:
        return 0.0
    split = max(1, len(values) // 2)
    return _mean(values[split:]) - _mean(values[:split])


def _field_trace_to_dict(trace: FieldTrace) -> dict[str, Any]:
    """Serialize field trace using legacy flat keys for compatibility."""

    return {
        "field_name": trace.field_name,
        "step": trace.step,
        "iteration": trace.iteration,
        "category_index": trace.category_index,
        "resonance": trace.resonance,
        "vigilance": trace.vigilance,
        "match": trace.match,
        "search_path": trace.search_path,
        "novelty": trace.novelty,
        "change": trace.change,
    }
