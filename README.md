# Sovereign AI

A small research prototype for a SOVEREIGN-like cognitive architecture in Python.

This is not an implementation of Stephen Grossberg's SOVEREIGN model. It is a compact, runnable system that borrows several of the important design pressures: pattern-based processing, distributed vector representations, competitive dynamics, gated learning, reactive/planned action pathways, and a simple imagination loop.

The goal is to make the architecture easy to inspect, modify, and argue with.

## What It Does

The demo runs an agent in a tiny vector-observed grid world. On each step, the agent:

1. observes the environment as a continuous vector,
2. categorizes the input through an ART-like perception module,
3. evaluates the state using reward, novelty, prediction error, and temporal context,
4. samples imagined category reconstructions and turns them into an action prior,
5. mixes reactive and planned action pathways through continuous competition,
6. executes an action,
7. updates memory, spatial traces, action weights, and category prototypes when gates open.

The printed trace is intentionally verbose so the internal dynamics are visible while experimenting.

## Architecture

The modules live in `sovereign_ai/` and are meant to be independently testable.

- `perception.py`: ART-like category learning with vigilance, reset/search, top-down matching, and resonance traces.
- `learning.py`: gated prototype learning with an ART-style component-wise intersection hybrid.
- `memory.py`: STM, decaying traces, LTM binding, and short category/action sequence memory.
- `evaluation.py`: scalar value system with reward, novelty, learned expectation, prediction error, and temporal context.
- `action_selection.py`: basal-ganglia-like GO/STOP competition and learned category-action weights.
- `pathways.py`: fast reactive pathway, slower planned pathway, and soft urgency-based pathway mixing.
- `imagination.py`: top-down category reconstruction, partial activation, candidate evaluation, and imagined action priors.
- `spatial.py`: optional stripe/SOM-style spatial substrate, now lightly integrated into the action bias.
- `environment.py`: simple toy grid world used by the demo.
- `architecture.py`: control loop that wires the pieces together.

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
[imagination-action] prior=[0.204 0.224 0.308 0.265]
[action-mix] action=2 reactive_weight=0.001 planned_weight=0.999 ...
[learning] category=0 delta=0.1408 mode=art-hybrid-intersection
```

## Why This Shape

The code avoids symbolic task rules inside the agent. The environment has discrete actions because it is a grid world, but the agent's internal machinery works with vectors, activations, competitions, learned weights, and gates.

The implementation also separates fast and slow behavior:

- the reactive path maps input directly to action drives,
- the planned path goes through perception, value, imagination, and action selection,
- urgency controls a continuous blend between the two instead of a hard switch.

## Current Limits

This is still a minimal research scaffold. Important limitations remain:

- sequence learning is a short trace, not a full LIST/PREEMPT-style planning system,
- imagination is shallow and samples category reconstructions rather than multi-step futures,
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
