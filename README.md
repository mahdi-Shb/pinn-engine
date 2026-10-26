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
  - `derivatives.py`: nested derivative declarations, virtual producer nodes,
    backend-declared requirements, and topological execution coordination.
  - `derivative_backends.py`: value-based numerical methods and method selection.
  - `_autograd.py`: direct PyTorch AD, target preparation, named-axis layout
    checks, and chunked tensor-output directions. No per-model backward or tape.
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

Each backend implements `name` and `compute(request, values, *, chunk_size)`.
The engine executes required models and derivative nodes in topological order;
the backend computes one derivative from its declared values. Default
`dependencies(graph, request)` requires only `of` and `wrt`. A mesh-based method
can additionally request mesh/connectivity quantities, including model outputs.
An analytical method can omit an unnecessary primal value. Optional hooks cover
execution context, target preparation, and method-specific validation.
There are no DerivativePlan objects, path plans, or per-model reverse scheduling.

AD needs the connected tensors, not just numerical copies of the two values.
It uses `torch.autograd.grad` on the existing forward graph, with
`create_graph=True` and graph retention for further derivatives and training.
Models run once per selected forward call. Separate requests run sequentially;
shared backward portions may be computed again. Tensor-output directions within
one request can be batched and bounded by `derivative_chunk_size`.

```python
Tx = Derivative(T, x)
Txx = Derivative(Tx, x)
solver = Solver(models, outputs=(T, Tx, Txx))
result = solver(values)
loss = result[Txx].tensor.square().mean()
loss.backward()
```

Nested, mixed, and vector-valued derivatives are supported, including constant
higher derivatives and ragged Tensor lists. Independent shared sample axes are
aligned without full cross-sample Jacobians. A shared axis consumed in full on
a dependent path is conservatively rejected; distinct input/output axis names
request explicit cross-axis Jacobians. Nonempty `holding` is explicitly rejected
by direct AD. `holding=()` is accepted and numerically behaves like a total
request. Spatial partials with independent x/y inputs remain ordinary valid AD.

The former local reverse implementation, holding support, and cross-request
VJP grouping are preserved on branch `codex/grouped-local-ad` at `12d97a1`.
Main continues from that commit using the simpler direct-AD implementation.

The notebook trains on analytical data, then evaluates and backpropagates through
a physics residual. It does not claim convergence from physics training.
See [the derivative contract](docs/contracts.fa.md) for details and limitations.

Geometry compatibility beyond object identity, sampler implementations, non-AD
built-in methods, persistent cache policy, and CUDA stream scheduling remain
open work. Ordinary Solver fullgraph capture is tested; derivative fullgraph
capture, Inductor speedups, and GPU concurrency are not claimed.
