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

The engine itself only requires PyTorch; `dev` installs the test dependency and
`notebook` installs the interactive tutorial dependencies. Graphviz's system
`dot` executable is optional: the notebook has a Matplotlib graph fallback.

## Files

- [pinn_engine/](pinn_engine/): implementation modules:
  - `geometry.py`: the abstract geometry interface.
  - `quantities.py`: `Quantity` and `Field` declarations.
  - `values.py`: `Value`, named axes, and runtime shape checks.
  - `model.py`: `Signature`, abstract `Model`, and automatic `vmap`.
  - `derivatives.py`: symbolic derivatives, derivative plans, AD checks, and the
    numerical derivative execution boundary.
  - `solver.py`: graph wiring, execution, and visualization; derivative-specific
    work is delegated to `derivatives.py`.
- [contracts.py](contracts.py): compatibility imports and the runnable example.
  Both `from contracts import Model` and `from pinn_engine import Model` resolve
  to the same class. Existing notebook imports remain valid.
- [Temperature_Tutorial.ipynb](Temperature_Tutorial.ipynb): a DeepONet temperature
  example. Its branch takes the left boundary temperature, its trunk takes the
  coordinate, and the model learns from the analytical solution.
- [tests/test_contracts.py](tests/test_contracts.py): executable contract tests.
- [Current contract documentation, in Persian](docs/contracts.fa.md).
- [Design and research notes](docs/README.md).

## Current implementation boundary

Ordinary model evaluation, named-axis `vmap`, parameter training, dependency
planning, and graph visualization are implemented. A model can consume a full
point axis and return values on that same axis.

Numerical derivative execution is intentionally **not implemented**. The default
AD boundary rejects requests that would incorrectly assume independence along
a coupled shared sample axis. Scalar reductions, held branches, and branches
independent of the differentiated input are treated separately. A physics loss
that requests a derivative still raises `NotImplementedError`; the notebook
catches that expected error and does not claim to solve the PDE with a PINN.

Geometry compatibility beyond object identity, sampler implementations, numerical
derivative backends, and cache policy remain open design work. The historical
notes describe proposals as well as decisions; the current contract document and
code describe the implemented behavior.
