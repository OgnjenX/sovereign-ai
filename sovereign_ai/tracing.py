from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FieldTrace:
    field_name: str
    step: int
    iteration: int
    category_index: int
    resonance: bool
    vigilance: float
    match: float
    search_path: list[int]
    novelty: float
    change: float


@dataclass(frozen=True)
class ProjectionTrace:
    name: str
    step: int
    update_norm: float
    source_size: int
    target_size: int


@dataclass(frozen=True)
class BehaviorTrace:
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
        field_name: str,
        step: int,
        iteration: int,
        category_index: int,
        resonance: bool,
        vigilance: float,
        match: float,
        search_path: list[int],
        novelty: float,
        change: float,
        category_count: int | None = None,
    ) -> None:
        self.field_traces.append(
            FieldTrace(
                field_name,
                int(step),
                int(iteration),
                int(category_index),
                bool(resonance),
                float(vigilance),
                float(match),
                [int(item) for item in search_path],
                float(novelty),
                float(change),
            )
        )
        if category_count is not None:
            self.category_counts[field_name] = int(category_count)

    def record_projection(
        self,
        name: str,
        step: int,
        update_norm: float,
        source_size: int,
        target_size: int,
    ) -> None:
        self.projection_traces.append(
            ProjectionTrace(name, int(step), float(update_norm), int(source_size), int(target_size))
        )

    def record_behavior(
        self,
        step: int,
        action: int,
        reward: float,
        value: float,
        goal_alignment: float,
        temporal_mismatch: float,
        imagination_accepted: bool,
    ) -> None:
        self.behavior_traces.append(
            BehaviorTrace(
                int(step),
                int(action),
                float(reward),
                float(value),
                float(goal_alignment),
                float(temporal_mismatch),
                bool(imagination_accepted),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": [asdict(item) for item in self.field_traces],
            "projections": [asdict(item) for item in self.projection_traces],
            "behavior": [asdict(item) for item in self.behavior_traces],
            "summary": self.summary(),
        }

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def summary(self) -> dict[str, Any]:
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
    if not values:
        return 0.0
    return float(np.mean(values))


def _trend(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    split = max(1, len(values) // 2)
    return _mean(values[split:]) - _mean(values[:split])
