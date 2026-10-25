"""Symbolic first derivatives, backend requirements, and request coordination.

Numerical execution is selected through DerivativeBackend. Autograd-specific
capability checks, tape creation, and gradient mode live in _autograd.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from .geometry import Geometry
from .quantities import Quantity
from .derivative_backends import resolve_backend
from .derivative_paths import DerivativePlan  # Compatibility export, not mandatory planning.

@dataclass(frozen=True, eq=False, init=False)
class Derivative(Quantity):
    """A first-order, full-Quantity derivative, independent of any AD backend.

    Derivative(a, b) is total along the hosting Solver's computation graph.
    Derivative(a, b, holding=(z,)) is partial: hold z's current value fixed
    while varying b. holding=None and holding=() deliberately differ in kind.
    Holding concerns this derivative only, NOT freezing or detaching training.

    Intrinsic shape is a.shape + b.shape, output components first. Runtime
    sample axes are NOT guessed here. Independent mapped samples can align;
    The selected numerical backend validates its own sample-axis capabilities.
    No component indexing or higher-order derivatives are supported yet.
    """

    of: Quantity
    wrt: Quantity
    holding: tuple[Quantity, ...] | None

    def __init__(
        self, of: Quantity, wrt: Quantity, *,
        holding: Sequence[Quantity] | None = None,
    ) -> None:
        if not isinstance(of, Quantity) or not isinstance(wrt, Quantity):
            raise TypeError("Derivative operands must be complete Quantity objects")
        fixed = None if holding is None else tuple(holding)
        operands = (of, wrt, *(fixed or ()))
        if any(not isinstance(q, Quantity) for q in operands):
            raise TypeError("holding must contain Quantity objects")
        if any(isinstance(q, Derivative) for q in operands):
            raise NotImplementedError("Only first-order derivative declarations are supported")
        if fixed is not None:
            if len(set(fixed)) != len(fixed):
                raise ValueError("holding cannot repeat quantities")
            if wrt in fixed or of in fixed:
                raise ValueError("Do not hold the differentiated input or output fixed")
        object.__setattr__(self, "of", of)
        object.__setattr__(self, "wrt", wrt)
        object.__setattr__(self, "holding", fixed)
        suffix = "" if fixed is None else "; holding=" + ",".join(q.name for q in fixed)
        name = f"{self.kind}({of.name}, {wrt.name}{suffix})"
        Quantity.__init__(self, name, of.shape + wrt.shape)

    @property
    def kind(self) -> str:
        return "total" if self.holding is None else "partial"

    @property
    def _domain(self) -> Geometry | None:
        return self.of._domain

    @property
    def _identity(self) -> tuple:
        # Symbolic operands, not generated display text, define identity.
        # The order of held quantities is mathematically irrelevant.
        return ("derivative", self.kind, self.of, self.wrt,
                frozenset(self.holding or ()))


def TotalDerivative(of: Quantity, wrt: Quantity) -> Derivative:
    """Readable constructor for a total derivative; there is only one class."""
    return Derivative(of, wrt)


def PartialDerivative(
    of: Quantity, wrt: Quantity, *, holding: Sequence[Quantity],
) -> Derivative:
    """Require an explicit declaration of quantities held fixed."""
    return Derivative(of, wrt, holding=holding)


def _related_quantities(quantity: Quantity):
    """Include symbolic operands when checking repeated shape declarations."""
    yield quantity
    if isinstance(quantity, Derivative):
        yield quantity.of
        yield quantity.wrt
        yield from quantity.holding or ()


class DerivativeEngine:
    """Own derivative graph integration, requirements, and backend coordination.

    The host supplies a computation graph plus two callbacks: run a model, and
    store validated values. No nn.Module registration or mutable runtime state
    is retained here. The numerical backend never needs the hosting Solver.
    """

    def __init__(self, chunk_size=64, *, backend="autograd"):
        self.backend = resolve_backend(backend)
        self.chunk_size = chunk_size
        self.requests = ()
        self.preparation = {}
        self.requirements = {}

    @property
    def plans(self):
        """Compatibility inspection of paths, without interpreting backend data."""
        return self.backend.path_plans(self.preparation)

    @property
    def chunk_size(self):
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, size):
        if type(size) is not int or size < 1:
            raise ValueError("derivative_chunk_size must be a positive integer")
        self._chunk_size = size

    def install(self, graph, requested_outputs):
        """Discover requests and register their virtual producer nodes."""
        requests = {}
        for model in graph.models:
            for quantity in model.outputs:
                if isinstance(quantity, Derivative):
                    raise ValueError("Derivative quantities are produced by Solver, not by Models")
            for quantity in (*model.inputs, *model.outputs):
                self._collect(graph, quantity, requests)
        for quantity in requested_outputs or ():
            if not isinstance(quantity, Quantity):
                raise TypeError("Solver outputs must be Quantity objects")
            self._collect(graph, quantity, requests)
        self.requests = tuple(requests)
        self.requirements = {
            request: tuple(dict.fromkeys(self.backend.dependencies(graph, request)))
            for request in self.requests
        }
        for derivative in self.requests:
            graph.add_node(("derivative", derivative), self.dependencies(derivative), (derivative,))

    @staticmethod
    def _collect(graph, quantity, requests):
        for related in _related_quantities(quantity):
            graph.register_quantity(related)
        if isinstance(quantity, Derivative):
            requests[quantity] = None

    def dependencies(self, derivative):
        return self.requirements[derivative]

    def prepare(self, graph):
        """Keep backend-owned preparation; the default does no model-path work."""
        self.preparation = self.backend.prepare(graph, self.requests)

    def execute(self, graph, needed, values, run_model, store):
        """Handle a derivative call; return False for ordinary host execution."""
        requests = [key for kind, key in graph.ordered
                 if kind == "derivative" and (kind, key) in needed]
        if not requests:
            return False

        self.backend.execute(graph, requests, needed, values, run_model, store,
                             self.chunk_size, preparation=self.preparation)
        return True

    def validate(self, graph, request, values):
        self.backend.validate(graph, request, values, preparation=self.preparation)

    def node_id(self, node):
        return f"d{self.requests.index(node[1])}"

    def graph_inputs(self, derivative):
        """Render only actual backend requirements, including additional data."""
        required = self.requirements[derivative]
        roles = [(derivative.of, "of"), (derivative.wrt, "wrt")]
        roles.extend((q, "holding") for q in derivative.holding or ())
        for quantity, role in roles:
            if quantity in required:
                yield quantity, role
        named = {q for q, _ in roles}
        for quantity in required:
            if quantity not in named:
                yield quantity, "input"

    def graph_detail(self, derivative):
        return self.backend.describe_request(derivative, preparation=self.preparation)
