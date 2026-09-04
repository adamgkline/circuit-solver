# circuit-solver

Model nonlinear analog resistive circuits and train them as physical neural
networks using **contrastive local learning rules** (coupled learning and
relatives).

A circuit is a graph: nodes are junctions, edges are nonlinear resistive
components. The circuit "computes" by physically relaxing to an
energy-minimizing equilibrium, and it "learns" by making *local* adjustments to
each component's parameters based on the difference between two physical states
(a free state and a target-nudged "clamped" state). No backpropagation through
the solver is required.

This is a research codebase for studying physical learning machines.

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.7. Core dependencies: `numpy`, `networkx`, `scipy`,
`torch`, `matplotlib`, `tqdm` (see `pyproject.toml`).

## The three modules

| Module | What it holds |
|---|---|
| `elements.py` | Nonlinear component definitions. Each `ResistiveElement` is defined by its **cocontent** `rho(x, θ)` (the integral of the i–V curve); current `gamma`, conductance derivatives, and parameter gradients are derived automatically via `torch.func.grad`. Ships prebuilt elements: `Resistor`, `Diode`, `IdealDiode`, `AdjDiode`, `Cubic`, symmetric variants, etc. |
| `circuit.py` | `Circuit` (a `networkx.DiGraph` + element/edge bookkeeping + incidence matrix) and `CircuitModel` (a `torch.nn.Module` that **solves the equilibrium**: clamp input nodes, then optimize free-node voltages to minimize a physical objective — cocontent or squared node current). |
| `learning.py` | The training rules. Each runs a free phase and a clamped phase, then applies a local parameter update. Includes `CoupledLearning` and variants, plus `TestClassificationAccuracy`. |
| `utils.py` | Graph builders (`bipartite_network`, `grid_network`, …), node/edge index helpers, plotting, and experiment bookkeeping. |

## Quickstart

The core workflow is: **build a graph → assign elements to edges → wrap in a
`Circuit` → build a `CircuitModel` → call a learning rule** on your `(X, Y)`
data.

```python
import networkx as nx
from circuit_solver import circuit as ct
from circuit_solver import elements as es
from circuit_solver import utils
from circuit_solver.learning import CoupledLearning, CLTrainingConfig

# X: (P, N) inputs, Y: (P, M) targets  (as numpy arrays or torch tensors)
P, N, M = X.shape[0], X.shape[1], Y.shape[1]

# 1. Topology: a directed graph whose nodes we partition by role.
N_nodes = N + M
graph = nx.DiGraph()
graph.add_nodes_from(range(N_nodes))

# Reserve reference nodes, then N input ("x") and M output ("y") nodes.
GROUND, HIGH, LOW, x_nodes, y_nodes = utils.segment_array(
    graph.nodes, (None, None, None, N, M)
)  # here HIGH and LOW come out empty

# Connect every input/reference node to every output node (a crossbar).
fixed_nodes = GROUND + HIGH + LOW + x_nodes
all_edges = utils.all_pairs_between(fixed_nodes, y_nodes)
graph.add_edges_from(all_edges)

# 2. Assign a circuit element to those edges.
element = es.Resistor()                 # a trainable linear conductance
element_dict = {element: all_edges}     # {element: [edges where it lives]}
circuit = ct.Circuit(graph, element_dict)

# 3. Tell the model which node lists are which role. Values are graph nodes
#    (labels in the DiGraph), not positional indices into list(circuit.nodes()).
node_type_dict = {
    'GROUND': GROUND, 'HIGH': HIGH, 'LOW': LOW,
    'x': x_nodes, 'y': y_nodes,
}
model = ct.CircuitModel(circuit, node_type_dict, ct.CircuitModelConfig())

# 4. Train with a contrastive learning rule.
config = CLTrainingConfig(batch_size=50, N_batches=1000, eta=1.0, alpha=0.01)
history = CoupledLearning(X, Y, model, config)   # x_nodes/'x', y_nodes/'y' by default

print(history['accuracy'][-1], history['loss'][-1])
```

`history` is a dict with `'loss'`, `'accuracy'`, and one entry per element name
holding its full parameter trajectory (shape `(N_params, N_batches+1, N_edges)`).

### Running a solve directly

`model(x)` clamps the `x` nodes and relaxes the rest, returning
`(node_voltages, objective_history)`:

```python
model.set_inputs('x')          # clamp only inputs -> "free" phase
V_node, _ = model(x)           # x: (batch, N)
model.set_inputs('x', 'y')     # clamp inputs and outputs -> "clamped" phase
V_node, _ = model(x, y)
```

## Key configuration

- **`CircuitModelConfig`** — solver settings: `optimizer` (`'lbfgs'` default, or
  `'adam'`), `N_optim_steps`, `optim_lr`, `mode` (`'cocontent'` or `'current'`),
  `HIGH_voltage` / `LOW_voltage`, `use_sparse`, `use_gpu`.
- **`CLTrainingConfig`** (and per-rule variants) — learning settings:
  `batch_size`, `N_epochs` *or* `N_batches`, `eta` (nudge strength), `alpha`
  (learning rate), `clamp_method` (`'MSE'`, `'overclamping'`, `'cross_entropy'`,
  …), and `parameter_update_method` (`None`/`'sgd'` applies raw learning-rule
  steps; `'adam'` preconditions them with Adam, configured via
  `adam_config=AdamConfig(lr=...)`).
- **Elements** carry `param_ranges` (parameters are clamped to these each step),
  `init_mode` (`'geometric_mean'`, `'constant'`, `'uniform'`, `'normal'`), and
  per-parameter `learning_rates`.

## Available learning rules (`learning.py`)

`CoupledLearning` (main), `CoupledLearningEtaZero`, `SymmetricCoupledLearning`,
`GeoCoupledLearning`, `AdjointLearning`, `InvariantLearning`,
`NCCoupledLearning` (noise-contrastive). All share the same
`(X, Y, model, config)` signature. `TestClassificationAccuracy` evaluates a
trained model.

## Status / caveats

Research code under active development. Some experimental branches are
incomplete: `generate_layer_graph` in `utils.py` references undefined names
and will error if called; `hinge_clamping` and
`distance_classification_accuracy` are stubs. The actively
used paths listed above work. See `CLAUDE.md` for a deeper map.
