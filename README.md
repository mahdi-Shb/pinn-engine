# PINN Engine

An experimental engine for composing PyTorch models using quantities and named
sample axes. This repository runs independently of the original PINNlab package.

## Setup

Use Python 3.10 or newer. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,notebook]"
python -m pytest
jupyter lab Temperature_Tutorial.ipynb
```

Install the project in the same Python environment used by pytest and the
notebook kernel. The src layout intentionally does not inject the checkout into
sys.path. After editing Python source, an editable install uses the changes
without reinstalling.

The engine itself only requires PyTorch; `dev` installs the test dependency and
`notebook` installs the interactive tutorial dependencies. Graphviz's system
`dot` executable is optional: the notebook has a Matplotlib graph fallback.

## Files

- [src/pinn_engine/](src/pinn_engine/): implementation modules:
  - `geometry.py`: the abstract geometry interface.
  - `quantities.py`: `Quantity` and `Field` declarations.
  - `values.py`: `Value`, named axes, and runtime shape checks.
  - `model.py`: `Signature`, abstract `Model`, and automatic `vmap`.
  - `solver.py`: the nn.Module host, runtime value checks, and public API.
  - `_graph.py`: dependency metadata, producer registration, ordering, and selection.
  - `derivatives.py`: derivative declarations and DerivativeEngine, which owns
    virtual nodes, backend-declared requirements, and request coordination.
  - `derivative_paths.py`: optional model-path analysis used by AutogradBackend.
  - `derivative_backends.py`: the small DerivativeBackend execution interface
    and explicit method selection.
  - `_autograd.py`: AutogradBackend, its AD checks and gradient-mode handling,
    per-call forward tape, and grouped local VJPs; independent of Solver.
  - `_visualization.py`: Mermaid/DOT rendering and Graphviz file output.
- [src/contracts.py](src/contracts.py): compatibility imports and the runnable example.
  Both `from contracts import Model` and `from pinn_engine import Model` resolve
  to the same class. Existing notebook imports remain valid.
- [Temperature_Tutorial.ipynb](Temperature_Tutorial.ipynb): a DeepONet temperature
  example. Its branch takes the left boundary temperature, its trunk takes the
  coordinate, and the model learns from the analytical solution.
- [tests/test_contracts.py](tests/test_contracts.py): executable contract tests.
- [Current contract documentation, in Persian](docs/contracts.fa.md).
- [Geometry lists and automatic batching, in Persian](docs/geometry_lists.fa.md).
- [Design and research notes](docs/README.md).

## Current implementation boundary

Ordinary model evaluation, named-axis `vmap`, parameter training, dependency
planning, and graph visualization are implemented. A model can consume a full
point axis and return values on that same axis. `Value` also accepts a flat list
of Tensors for a named outer geometry axis with variable-size members. Models
either consume that list in full or run per geometry; common dense axes can
still use `vmap`. This representation adds no padding or parallel job scheduling.

Select the numerical method in the host constructor:

```python
solver = Solver(models, outputs=loss, derivative_backend="autograd")
```

`autograd` is the default and currently the only implemented built-in method.
Unknown or unimplemented names raise an error. A custom DerivativeBackend object
can also be supplied; it owns its capability checks and runtime requirements.

The inheritance tree is intentionally shallow:

```text
DerivativeBackend (abstract)
└── AutogradBackend
```

Both classes are public imports. Supplying an object is equivalent to choosing
the built-in method by name:

```python
from pinn_engine import AutogradBackend, DerivativeBackend, Solver

solver = Solver(models, outputs=loss, derivative_backend=AutogradBackend())
```

Each backend implements `name` and `execute`. Optional `dependencies(graph,
request)` declares required values (default: `of` and `wrt`); a mesh-based method
can add connectivity, while an analytical method can omit the primal output.
Optional `prepare(graph, requests)` returns static method-specific metadata;
the default is an empty mapping and performs no model-path analysis. Runtime
mesh/grid-dependent work belongs in `execute`, after values become available.
`execute` receives raw derivative requests and its own `preparation` keyword.
`validate` can reject unsupported requests, including unsupported holding rules.

`solver.derivative_preparation` exposes that metadata. `solver.derivative_plans`
is an optional path inspection view: AutogradBackend supplies it through
`path_plans`; value-based backends leave it empty. Graph diagrams show the actual
declared requirements, including extra mesh/grid inputs. Solver still declares
external inputs for the entire model graph; skipping a source forward does not
change that input contract.

DerivativeEngine holds a backend rather than inheriting from it. The base
contract requires no tape, grid, function replay, or model-path plan.
Finite-difference, spectral, and FEM backends can be added as direct subclasses
when they are implemented; these numerical methods are not included yet.

First-order total and partial derivatives are implemented with grouped local
reverse-mode AD. Model inputs are recorded as distinct identity views during a
single forward. Each request has independent `wrt` and `holding` stops; equal
incoming adjoints share local VJPs across requests. Other directions are batched
and chunked with `Solver(..., derivative_chunk_size=64)`. Derivatives remain
connected to parameter training, including parameters producing held values.
Shared independent sample axes are aligned without a full cross-sample Jacobian;
distinct output/input axis names expose cross-axis Jacobians explicitly. Coupled
shared sample axes are rejected. Real Tensor lists are supported as well.

The notebook still trains on analytical data, then evaluates and backpropagates
through the physics residual. It does not claim convergence from physics training.
See [the AD contract](docs/contracts.fa.md) for scope and limitations.

Geometry compatibility beyond object identity, sampler implementations, higher-order
derivative requests, non-AD backends, and persistent cache policy remain open design work. The historical
notes describe proposals as well as decisions; the current contract document and
code describe the implemented behavior.
