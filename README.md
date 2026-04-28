# Sovereign AI

A small research prototype for a SOVEREIGN-like cognitive architecture in Python.

This is not an implementation of Stephen Grossberg's SOVEREIGN model. It is a compact, runnable system that borrows several of the important design pressures: pattern-based processing, distributed vector representations, recurrent field dynamics, competitive normalization, gated learning, reactive/planned action pathways, and a simple imagination loop.

The goal is to make the architecture easy to inspect, modify, and argue with.

## What It Does

The demo runs an agent in a tiny vector-observed grid world. On each environment step, the agent first runs an intra-step convergence loop:

```text
real input + imagination -> perception field <-> value field <-> action field
                         ^             |              |              |
                         +-------------+--------------+--------------+
                 goals + temporal future state bias all fields

learning gates open only after the fields settle
```

Inside that loop, perception, value, action, imagination, goals, and predicted future states repeatedly bias each other until the state change is small or the iteration budget is exhausted. The selected action is the result of the settled field, not a one-pass pipeline.

The printed trace is intentionally verbose so the internal dynamics are visible while experimenting.

## Architecture

The modules live in `sovereign_ai/` and are meant to be independently testable.

- `perception.py`: ART-like category learning with vigilance, reset/search, top-down matching, resonance traces, and recurrent category-field updates.
- `learning.py`: gated prototype learning with an ART-style component-wise intersection hybrid.
- `memory.py`: STM, decaying traces, LTM binding, and short category/action sequence memory.
- `evaluation.py`: value field with reward, novelty, learned expectation, prediction error, temporal context, and shunting-style state updates.
- `action_selection.py`: basal-ganglia-like GO/STOP competition, learned category-action weights, and recurrent action-field updates.
- `pathways.py`: fast reactive pathway, slower planned pathway, and soft urgency-based pathway mixing.
- `imagination.py`: top-down category reconstruction, partial activation, candidate evaluation, and imagined action priors.
- `goal_system.py`: persistent distributed goal vectors with slow competitive updates.
- `temporal_state.py`: present/future state buffer used for recurrent internal unfolding.
- `transition_model.py`: online linear transition model used to unfold future states inside the main loop.
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
[value] reward=-0.005 expected=0.000 prediction_error=-0.005 ...
[imagination-coupled] input_norm=2.032 category_bias=[1.] action_prior=[0.2   0.221 0.319 0.261]
[goal] alignment=0.842 activation=[0.336 0.329 0.335]
[composition] components=[0, 1] weights=[0.42 0.58]
[convergence] iter=3 total_change=0.0775 p_change=0.0000 v_change=0.0670 a_change=0.0105 future_alignment=0.812 alpha=0.42 ...
[learning] category=0 delta=0.1408 mode=art-hybrid-intersection
```

## Why This Shape

The code avoids symbolic task rules inside the agent. The environment has discrete actions because it is a grid world, but the agent's internal machinery works with vectors, activations, competitions, learned weights, and gates.

The implementation also separates fast and slow behavior while keeping them coupled:

- the reactive path maps input directly to action drives,
- the planned path goes through perception, value, imagination, and action selection,
- urgency controls a continuous blend between the two instead of a hard switch,
- action and value fields feed top-down attention back into perception during convergence.
- present and predicted future states co-exist in a temporal buffer and feed the same convergence loop,
- the goal system contributes goal alignment to value and provides a persistent top-down bias.

## Current Limits

This is still a minimal research scaffold. Important limitations remain:

- sequence learning is a short trace, not a full LIST/PREEMPT-style sequence controller,
- future unfolding uses a shallow learned linear transition model rather than a rich world model,
- the spatial module is lightly coupled rather than a full navigation subsystem,
- value learning is simple prediction-error learning, not a full motivational system,
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
