# PINN Engine

An experimental engine for composing PyTorch models using quantities and named
sample axes. Quantities are identified by name; Model and Solver do not
carry geometry/domain attributes. This repository runs independently of the
original PINNlab package.

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

The engine itself only requires PyTorch. GeoCore is bundled under `src/geocore`;
install `.[geometry]` for its NumPy/SciPy/Numba dependencies, and `.[mesh]`
for optional trimesh operations. `dev` installs the test dependency and
`notebook` installs the interactive tutorial dependencies. Graphviz's system
`dot` executable is optional: the notebook has a Matplotlib graph fallback.

## Files

- [Signature and wildcard axes](docs/signature.fa.md): preserved axis names
  default to sample independence, with explicit `independence` overrides. Signatures containing `"..."` let the model
  handle named Values directly, with no automatic mapping or permutation.
- [GeoCore wrapper](docs/geocore.fa.md): sequential CPU generation of design
  variants as named Tensor lists, with [a runnable example](examples/geocore_design.py).
- [src/geocore/](src/geocore/): independent geometry package (`import geocore`),
  including its native runtime libraries.
- [src/pinn_engine/networks/](src/pinn_engine/networks/): native network Models
  and their shared input/output contract.
- [src/pinn_engine/networks/normalization.py](src/pinn_engine/networks/normalization.py): shared
  dense/list/masked BatchNorm and per-case point normalization.
  [Normalization details](docs/normalization.fa.md) and
  [upstream change audit](docs/network_upstream_changes.fa.md) document the boundary.
- [third_party/](third_party/): provenance, original documentation, benchmark
  utilities, examples, and tests retained as reference; active runtime packages
  live in `src/`.
- [src/pinn_engine/](src/pinn_engine/): implementation modules:
  - `quantities.py`: `Quantity` declarations.
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
- [Tensor lists and automatic batching, in Persian](docs/tensor_lists.fa.md).
- [Design and research notes](docs/README.md).

## Current implementation boundary

Ordinary model evaluation, named-axis `vmap`, parameter training, dependency
planning, and graph visualization are implemented. A model can consume a full
point axis and return values on that same axis. `Value` also accepts a flat list
of Tensors for a named outer sample axis with variable-size members. Models
either consume that list in full or run per member; common dense axes can
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
The engine executes ready models and derivative groups in dependency order;
the backend computes derivatives from their declared values. Default
`dependencies(graph, request)` requires only `of` and `wrt`. A mesh-based method
can additionally request mesh/connectivity quantities, including model outputs.
An analytical method can omit an unnecessary primal value. Optional hooks cover
execution context, target preparation, method-specific validation, and grouping.
There are no DerivativePlan objects, path plans, or per-model reverse scheduling.

AD needs the connected tensors, not just numerical copies of the two values.
It uses `torch.autograd.grad` on the existing forward graph, with
`create_graph=True` and graph retention for further derivatives and training.
Models run once per selected forward call. Requests with the same source and
compatible output directions share a reverse sweep over all target tensors.
Different sources or incompatible point/scenario contractions stay separate;
there is no global stacking with zero lanes for unrelated outputs. Tensor-output
directions are batched and bounded by `derivative_chunk_size`.

Grouping belongs to each backend. Optional `group_key(request, values)` selects
compatible ready requests; None keeps a singleton. `compute_group(requests,
inputs, *, chunk_size, prepared=None)` receives a separate dependency mapping
for each request and returns exactly their Value results. `prepare_group` can
build common numeric data once per key per Solver call, such as a mesh operator,
and reuse it in later dependency frontiers. No persistent cache is introduced.
The default backend hooks keep existing single-request implementations working.

See [grouping rules and backend examples, in Persian](docs/derivative_grouping.fa.md).
Forward-mode, Taylor-mode, and FEM are not new built-in methods: small executable
test backends verify shared-input JVPs, fused polynomial orders, and reuse of
triangle geometry operators with this interface.

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
VJP grouping are preserved on branch `partial-derivative-and-geometry-domain-support` at `12d97a1`.
Main continues from that commit using the simpler direct-AD implementation.

The notebook trains on analytical data, then evaluates and backpropagates through
a physics residual. It does not claim convergence from physics training.
See [the derivative contract](docs/contracts.fa.md) for details and limitations.

Geometry/domain interfaces are deferred; the GeoCore and field-domain ideas are
recorded in [the design notes](docs/architecture_open_decisions.fa.md).

The eleven Neural3DBench networks are native `Model` subclasses under
`pinn_engine.networks`, installed with the engine. Their constructors accept
`inputs={role: quantity}` and `outputs={role: quantity}`. Actual BatchNorm modules
consume the complete batch; graph networks pack disjoint meshes internally.
Other networks use the engine's named-axis automatic batching. See
[network layouts and examples](docs/networks.fa.md) and the runnable
[GeoCore-to-mesh-network example](examples/geocore_mesh_network.py).

Sampler implementations, non-AD built-in methods, persistent cache policy, and CUDA stream scheduling remain
open work. Ordinary Solver fullgraph capture is tested; derivative fullgraph
capture, Inductor speedups, and GPU concurrency are not claimed.
