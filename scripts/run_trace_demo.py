from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sovereign_ai.architecture import CognitiveArchitecture
from sovereign_ai.environment import GridWorld
from sovereign_ai.tracing import TraceRecorder


def main() -> None:
    recorder = TraceRecorder()
    env = GridWorld(size=5, seed=7)
    agent = CognitiveArchitecture(
        input_dim=env.observation_dim,
        action_count=env.action_count,
        max_categories=12,
        seed=3,
        debug=False,
        trace_recorder=recorder,
    )
    for step in range(16):
        agent.step(env, step)

    output_path = PROJECT_ROOT / "artifacts" / "trace_demo.json"
    recorder.to_json(output_path)
    print(json.dumps(recorder.summary(), indent=2))
    print(f"trace_json={output_path}")


if __name__ == "__main__":
    main()
