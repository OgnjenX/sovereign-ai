# Sovereign AI

A small research prototype for a SOVEREIGN-like cognitive architecture in Python.

This is not an implementation of Stephen Grossberg's SOVEREIGN model. It is a compact, runnable system that borrows several of the important design pressures: pattern-based processing, distributed vector representations, recurrent field dynamics, competitive normalization, gated learning, reactive/planned action pathways, and a simple imagination loop.

The goal is to make the architecture easy to inspect, modify, and argue with.

## What It Does

The demo runs an agent in a tiny vector-observed grid world. On each environment step, the agent first runs an intra-step convergence loop across ART fields:

```text
PerceptualField <-> ValueField <-> ActionField
      ^               ^              ^
      |               |              |
GoalField ---- top-down bias --------+
      ^                              |
      |                              |
TemporalField -- prediction --> PerceptualField
      ^
      |
ExpectationField -- hypothesis --> PerceptualField/ActionField

learning gates open only after the fields settle
```

Inside that loop, perception, value, action, expectation, goals, and predicted future states repeatedly bias each other until the state change is small or the iteration budget is exhausted. The selected action is the result of a resonant action-schema category, not a one-pass scoring pipeline.

The printed trace is intentionally verbose so the internal dynamics are visible while experimenting.

## Architecture

The modules live in `sovereign_ai/` and are meant to be independently testable.

- `art_field.py`: shared ART field base class with categories, bottom-up input, top-down expectation, match, vigilance, reset/search, resonance, and prototype learning.
- `perception.py`: `ARTPerceptualField`, a perceptual ART field with top-down expectation and imagined category coupling.
- `learning.py`: gated prototype learning with an ART-style component-wise intersection hybrid.
- `memory.py`: STM, decaying traces, LTM binding, and short category/action sequence memory.
- `evaluation.py`: `ARTValueField`, whose categories are value contexts that modulate vigilance and category preference in other fields.
- `action_selection.py`: `ARTActionField`, whose categories are action schemas; action selection is resonance plus reset/search over schemas.
- `pathways.py`: compatibility adapters around ART action selection; direct reactive/planned vector scoring has been removed from the architecture.
- `imagination.py`: `ARTExpectationField`, which generates top-down hypotheses and tests whether they induce perceptual resonance without external input.
- `goal_system.py`: `ARTGoalField`, whose categories are persistent goal states that bias perception and action through top-down signals.
- `temporal_state.py`: present/future state buffer used for recurrent internal unfolding.
- `transition_model.py`: `ARTTemporalField`, whose categories are learned state/action/state transitions used for next-state prediction.
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
[value-dyn] category=1 scalar=0.767 change=0.1253 vigilance=0.700 activation=[...]
[expectation-coupled] input_norm=2.405 category_bias=[0.494 0.506] action_prior=[...]
[goal-dyn] category=0 alignment=0.915 activation=[0.911 0.04  0.049]
[composition] components=[0, 1] weights=[0.42 0.58]
[action-dyn] category=5 action=0 change=0.5245 vigilance=0.589 distribution=[...]
[convergence] iter=3 total_change=0.5247 p_change=0.0000 v_change=0.0000 a_change=0.5246 value_vigilance=0.017 ...
[learning] category=0 delta=0.1408 mode=art-hybrid-intersection
```

## Why This Shape

The code avoids symbolic task rules inside the agent. The environment has discrete actions because it is a grid world, but the agent's internal machinery works with vectors, activations, competitions, learned weights, and gates.

The implementation now uses one dynamical principle throughout:

- perception, value, goals, action selection, expectation, and temporal prediction all inherit from `ARTField`,
- fields communicate by category bias, top-down expectation, and vigilance modulation,
- action is selected by resonant action-schema categories,
- value contexts modulate vigilance and perceptual category preference,
- expectation generates top-down hypotheses and asks perception to resonate without real input,
- temporal categories learn state/action/state transitions and bias the next perceptual state.

## Current Limits

This is still a minimal research scaffold. Important limitations remain:

- sequence learning is category-based but still compact, not a full LIST/PREEMPT-style sequence controller,
- future unfolding uses ART transition categories rather than a rich world model,
- the spatial module is lightly coupled rather than a full navigation subsystem,
- value learning is a small ART value-context system, not a full motivational system,
- the toy environment is intentionally small.

Those limits are deliberate: the code is meant to stay readable enough that each mechanism can be replaced with a more faithful model.

## Quick Development Checks

Syntax check:

```bash
python3 -m compileall main.py sovereign_ai
```

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
