# Sovereign AI

A small research prototype for a SOVEREIGN-like cognitive architecture in Python.

This is not an implementation of Stephen Grossberg's SOVEREIGN model. It is a compact, runnable system that borrows several of the important design pressures: pattern-based processing, distributed category representations, recurrent field dynamics, vigilance-regulated reset/search, gated learning, learned inter-field associations, and prospective imagination.

The goal is to make the architecture easy to inspect, modify, and argue with.

## What It Does

The demo runs an agent in a tiny vector-observed grid world. On each environment step, the agent first runs an intra-step convergence loop across ART fields:

```text
PerceptualField <-- learned projections --> ValueField
      ^                                      |
      |                                      v
GoalField ---- learned projections ----> ActionField
      ^                                      ^
      |                                      |
TemporalField -- sequence prediction --> ExpectationField

learning gates open only after the fields settle
```

Inside that loop, perception, value, action, expectation, goals, and predicted future states repeatedly bias each other through learned category-to-category projections until the state change is small or the iteration budget is exhausted. The selected action is the result of a resonant learned sensorimotor action-schema category, not a one-pass scoring pipeline.

The printed trace is intentionally verbose so the internal dynamics are visible while experimenting.

## Architecture

The modules live in `sovereign_ai/` and are meant to be independently testable.

- `art_field.py`: shared ART field base class with categories, bottom-up input, top-down expectation, match, vigilance, reset/search, resonance, and prototype learning.
- `associative_coupling.py`: learned category-to-category projections between ART fields.
- `vigilance.py`: learned per-field vigilance controller driven by mismatch and reward outcome.
- `perception.py`: `ARTPerceptualField`, a perceptual ART field with top-down expectation and imagined category coupling.
- `learning.py`: gated prototype learning with an ART-style component-wise intersection hybrid.
- `memory.py`: STM, decaying traces, LTM binding, and short category/action sequence memory.
- `evaluation.py`: `ARTValueField`, whose value categories and scalar/vigilance/attention associations are learned from reward, novelty, prediction error, goal alignment, action outcome, and perceptual activation.
- `action_selection.py`: `ARTActionField`, whose sensorimotor action schemas and category-to-action associations are learned from state/action/outcome experience.
- `pathways.py`: compatibility adapters around ART action selection; direct reactive/planned vector scoring has been removed from the architecture.
- `imagination.py`: `ARTExpectationField`, which performs short prospective rollouts using temporal proposals, no-input perceptual resonance, and action-schema resonance.
- `goal_system.py`: `ARTGoalField`, whose categories are persistent goal states that bias perception and action through top-down signals.
- `temporal_state.py`: present/future state buffer used for recurrent internal unfolding.
- `transition_model.py`: `ARTTemporalField`, whose sequence chunk categories learn ordered percept/action/context transitions, predicted next perceptual/action category biases, confidence, and mismatch/reset traces.
- `spatial.py`: optional stripe/SOM-style spatial substrate, now lightly integrated into the action bias.
- `environment.py`: simple toy grid world used by the demo.
- `architecture.py`: recurrent convergence loop that wires the fields together before execution and learning.

## Run It

Install dependencies:

```bash
pip install -e .
```

Run the demo:

```bash
python3 main.py
```

You should see logs like:

```text
[perception] winner=0 resonance=True novelty=0.005 search=[0] ...
[value-dyn] category=1 scalar=0.002 change=0.0771 vigilance=0.620 search=[...] assoc=[...]
[expectation-rollout] trace=[{'step': 0, 'expectation_resonance': False, ...}]
[goal-dyn] category=0 alignment=0.915 activation=[0.911 0.04  0.049]
[projection] name=goal->action source=3 target=12 update=0.0268
[vigilance] field=expectation adjustment=0.0664 mismatch=0.646 reward=0.015 update=0.0022
[action-dyn] category=6 action=1 resonance=True search=[6] vigilance=0.595
[convergence] iter=3 total_change=0.0062 p_change=0.0043 v_change=0.0017 a_change=0.0002 ...
[learning] category=0 delta=0.1408 mode=art-hybrid-intersection
```

## Why This Shape

The code avoids symbolic task rules inside the agent. The environment has discrete actions because it is a grid world, but the agent's internal machinery works with vectors, activations, competitions, learned weights, and gates.

The implementation now uses one dynamical principle throughout:

- perception, value, goals, action selection, expectation, and temporal prediction all inherit from `ARTField`,
- fields communicate through learned `AssociativeProjection` objects,
- vigilance is adjusted by a learned `VigilanceController`,
- action is selected by resonant learned action-schema categories,
- value categories learn scalar value, vigilance modulation, and perceptual attention associations,
- expectation performs prospective resonance and rejects failed imagined candidates,
- temporal categories learn ordered sequence chunks and expose mismatch/reset traces.

## Current Limits

This is still a minimal research scaffold. Important limitations remain:

- sequence learning is closer to LIST/PREEMPT-style chunking but still compact,
- future unfolding uses ART transition categories rather than a rich world model,
- action schema vectors still have documented slots for perceptual, goal, value, temporal, motor, and outcome signals,
- the spatial module is lightly coupled rather than a full navigation subsystem,
- value learning is learned category association, not a full motivational system,
- the toy environment is intentionally small.

Those limits are deliberate: the code is meant to stay readable enough that each mechanism can be replaced with a more faithful model.

## Quick Development Checks

Syntax check:

```bash
python3 -m compileall main.py sovereign_ai tests scripts
```

Faithfulness tests:

```bash
python3 -m unittest discover -s tests
```

Trace demo:

```bash
python3 scripts/run_trace_demo.py
```

This writes `artifacts/trace_demo.json` and prints a machine-readable summary.

## Trace Data

Tracing is optional. Pass a `TraceRecorder` into `CognitiveArchitecture(..., trace_recorder=recorder)` to collect structured data without relying on debug prints.

Trace event types:

- `FieldTrace`: field name, step, convergence iteration, winning category, resonance flag, vigilance, best match, search path, novelty, and activation change.
- `ProjectionTrace`: learned projection name, step, update norm, source category count, and target category count.
- `BehaviorTrace`: step, selected action, reward, scalar value, goal alignment, temporal mismatch, and whether imagination accepted a candidate.

Summary fields:

- `resonance_rate`: fraction of field events that reached resonance. Lower rates mean more mismatch/reset pressure.
- `average_search_length`: average number of categories tried before resonance or fallback. Longer paths indicate more reset/search.
- `average_vigilance`: effective vigilance after learned modulation.
- `projection update norm`: magnitude of learned associative projection changes. Near-zero means little new coupling was learned.
- `temporal_mismatch`: sequence prediction mismatch. A falling trend suggests repeated transition structure is being learned.
- `imagination_acceptance_rate`: fraction of behavior steps where at least one imagined rollout candidate was accepted by resonance.
- `action_distribution`: selected action counts across the run.

Run a short smoke check:

```bash
python3 - <<'PY'
import numpy as np
from sovereign_ai.learning import GatedLearning
from sovereign_ai.perception import ARTPerception

prototypes = np.array([[0.8, 0.6, 0.2]])
GatedLearning(intersection_rate=1.0).update(prototypes, 0, np.array([0.5, 0.9, 0.1]))
assert np.allclose(prototypes[0], [0.5, 0.6, 0.1])

p = ARTPerception(input_dim=3, vigilance=0.95)
p.process(np.array([1.0, 0.5, 0.2]))
r = p.process(np.array([0.2, 1.0, 0.8]))
assert r.resonance and r.resonance_trace
print("ok")
PY
```

## Status

This repo is best treated as an experimental cognitive architecture sandbox, not a production ML library.
