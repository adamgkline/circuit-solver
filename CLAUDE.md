# CLAUDE.md — notes for Claude / agents working on circuit-solver

Orientation notes for a coding agent picking up this repo cold. The README is
for humans; this file records the non-obvious mechanics, invariants, and known
rough edges that will bite you.

## What this is, in one paragraph

Physical learning machines: a resistor network is a `torch.nn.Module` whose
"forward pass" is not a feedforward evaluation but an **inner optimization** that
relaxes free-node voltages to an equilibrium. Training is **not** backprop
through that solver; it is a set of hand-derived **local** update rules
(coupled learning and relatives) that compare a free phase and a clamped phase
and adjust each edge's parameters from locally available quantities. Keep this
distinction front of mind: `autograd` is used *inside* the solver (to relax
voltages) and *inside* elements (to get `gamma`/derivatives from `rho`), but the
learning rules update `layer.theta.data` in place, outside the autograd graph.

## The data-flow spine

- **Nodes and edges are ordered by `list(graph.nodes())` / `list(graph.edges())`.**
  Everything downstream indexes into these orders. `utils.nodes_to_inds` and
  `utils.edges_to_inds` convert node/edge labels → positional indices. Bugs in
  this project have historically been graph-ordering bugs (see git log: multiple
  "graph ordering" fixes) — be very careful mixing node *labels* with node
  *indices*.
- **Incidence matrix**: `circuit.DelT = nx.incidence_matrix(oriented=True)`,
  `Del = DelT.T`. In `CircuitModel`, `Del_dense` is the source of truth; sparse
  versions are derived when `use_sparse=True`.
- **Voltage sign convention**: `model.V_edge()` returns `-Del @ V_node`. The
  minus sign is baked in — several learning rules have comments reminding you of
  this. Don't "fix" it.
- **Element → edge routing**: `MaskedNonlinearity` applies each element's
  function only to that element's edge indices, so all edges are processed in
  one `(batch, N_edges)` tensor. `model.element_to_inds[name]` gives the indices;
  `model.element_layers[name]` gives the trainable layer.

## Elements (`elements.py`)

- An element is defined by **one function**: its cocontent `rho(x, theta)`.
  Everything else is autodiff of `rho`:
  - `gamma = d rho / dx`  (the i–V relation / current)
  - `d_gamma_d_x = d² rho / dx²`  (differential conductance)
  - `d_rho_d_theta`, `d_gamma_d_theta` — parameter gradients used by learning rules.
  So to add a component, you only write its `rho`. Use **torch ops** (`tc.exp`,
  `tc.relu`, …) so grad works.
- Prebuilt elements are **factory callables** made by `element_generator(...)`.
  `es.Resistor()` *creates a fresh instance*; call it, don't use the bare name.
  `theta` layout and `param_ranges` are per-element (e.g. Diode = `[conductance,
  steepness]`).
- `element.name` must be unique within a circuit. `Circuit.__init__` auto-renames
  duplicates (with a warning) and identifies elements by `name`, not by object.
- Pickling matters here (experiments are pickled). That's why reversed functions
  use the `ReversedFunction` class instead of a lambda, and why
  `circuit.elements` is stored as a `set`.

## The solver (`CircuitModel`, `circuit.py`)

- **`forward(*tensors)`** = clamp inputs → build a fresh optimizer over
  `self.V_free` → run `N_optim_steps` of LBFGS (default) or Adam minimizing the
  physical objective → return `(V_node, obj_history)`. `V_free` is re-zeroed
  every `clamp()` call, so each solve starts from scratch (no warm start).
- **`set_inputs(*groups)`** decides which nodes are clamped vs. free. Reference
  nodes (`GROUND`/`HIGH`/`LOW`) are *always* clamped. Accepts either node-index
  lists or string keys into `node_type_dict`. Must be called before a solve to
  select the phase (free = inputs only; clamped = inputs + outputs). It rebuilds
  `Del_clamped`/`Del_free`.
- **Two physical objectives** (`mode`): `'cocontent'` (default; minimize summed
  `rho`) and `'current'` (minimize squared node current = enforce KCL). They
  should agree at equilibrium; cocontent is the smoother objective.
- **LBFGS default `N_optim_steps=3`** — that's intentional (LBFGS does many
  internal evals per step). Adam default is 200 steps, lr 0.1. LBFGS is noted to
  struggle at large batch size.
- **Parameter constraints**: `layer.clip_parameters()` clamps `theta` into
  `element.param_ranges` and is called after every learning update. Absorbing
  boundaries (`set_absorbing()` / `set_reflecting()`) freeze a parameter once it
  hits a range limit via `absorb_mask`, which zeroes that parameter's effective
  learning rate.
- `ElementLayer.vectorize` uses nested `tc.vmap` over (batch, edge). Shapes:
  inputs `(batch, N_edges)`, params `theta` `(N_params, N_edges)`, and e.g.
  `d_rho_d_theta` returns `(batch, N_params, N_edges)`.

## Learning rules (`learning.py`)

All training functions share the signature `Fn(X, Y, model, config=None,
**params)` and the same skeleton:

1. resolve `x_nodes`/`y_nodes` (default keys `'x'`/`'y'` in `node_type_dict`),
2. per batch: **free phase** `set_inputs(x_nodes); model(x)` → read `V_edge_F`;
   **clamped phase** `set_inputs(x_nodes, y_nodes); model(x, y_C)` → read
   `V_edge_C`, where `y_C` comes from a **clamping function** (`clamping_functions`
   dict: MSE, overclamping, cross-entropy, …),
3. for each element layer: `layer.theta.data += compute_d_theta(...)` then
   `layer.clip_parameters()`,
4. log `theta`, loss, accuracy; return via `finalize_history`.

The rules differ **only** in `compute_d_theta` (the local update) and in how many
clamped states they compute (symmetric uses ±; noise-contrastive adds per-class
negative states). `eta` = clamp/nudge strength, `alpha` = learning rate, and
`layer.learning_rates` is a per-parameter multiplier.

**Parameter update methods**: `cfg.parameter_update_method` selects how the raw
`compute_d_theta` step is applied — `None`/`'sgd'` adds it directly, `'adam'`
preconditions it with a stateful `AdamUpdater` (per-layer moment tensors shaped
like `theta`; step size set by `cfg.adam_config.lr`, since Adam normalizes away
`alpha`). Wired into the three core rules (`CoupledLearning`,
`CoupledLearningEtaZero`, `SymmetricCoupledLearning`) only. The functions below
the "Experimental learning rule training functions" header are the author's
scratch space — leave them alone unless explicitly asked to change them.

`history[element_name]` is the parameter trajectory with shape
`(N_params, N_batches+1, N_edges)` (note: **param, time, edge** — see
`restore_model_to_t`). `finalize_history` stacks along `axis=1`.

## Known broken / stub code (don't call without fixing)

- `utils.generate_layer_graph` — uses `coloring.greedy_color` but `coloring` is
  never imported (should be `nx.coloring.greedy_color`); also reads
  `circuit.graph` / `circuit.inputs` / `circuit.outputs` attributes that
  `Circuit` doesn't set. Will `NameError`/`AttributeError`.
- `learning.hinge_clamping`, `learning.constant_clamping`,
  `learning.distance_classification_accuracy` — empty stubs (return `None`).

These are annotated inline with `# TODO`/`# BUG`. If asked to work on
layer-graph or hinge loss, these are the starting points.

## Repo layout notes

- `circuit_solver/dev/` and `__pycache__/*_old*/*_copy*` are scratch/older
  variants — not part of the public API. Prefer the top-level modules.
- `circuit_solver/examples/` has notebooks; `examples/development/` are dated
  working notebooks (`YYMMDD ...`). Per the author, the example notebooks may be
  stale — trust the source modules over the notebooks, and ignore the
  pandas/experiment-index and MNIST-preprocessing scaffolding when learning the
  core API.
- `venv/` is checked in-ish and noisy — ignore it.

## Fast ways to verify a change

- Build a tiny bipartite circuit (`utils.bipartite_network`, or the crossbar in
  the README quickstart) with a handful of nodes, `es.Resistor()`, and run a
  few batches of `CoupledLearning` on random `(X, Y)`. If loss doesn't error and
  parameters stay within `param_ranges`, the plumbing is intact.
- For solver changes, compare `mode='cocontent'` vs `mode='current'` equilibria —
  they should give near-identical `V_node`.
- Watch for graph-ordering regressions: test with nodes added in a
  non-sorted/non-contiguous order, since that's where past bugs lived.
