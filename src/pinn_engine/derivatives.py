"""Symbolic first derivatives, backend-independent planning, and coordination.

Numerical execution is selected through DerivativeBackend. Autograd-specific
capability checks, tape creation, and gradient mode live in _autograd.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence
from .geometry import Geometry
from .quantities import Quantity
from .derivative_backends import resolve_backend

if TYPE_CHECKING:
    from ._graph import ComputationGraph

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


@dataclass(frozen=True)
class DerivativePlan:
    """Describe the source path and cut points for a derivative request.

    model_indices are dependency-ordered indices into Solver.models. Traversal
    stops at wrt and held quantities; their current values become function
    inputs. Other external inputs also remain fixed during this derivative.
    This is structural reachability, not proof of a nonzero numerical derivative.
    Numerical backends are responsible for preserving training where supported.
    """

    derivative: Derivative
    producer_index: int | None
    model_indices: tuple[int, ...]
    function_inputs: tuple[Quantity, ...]
    depends_on_wrt: bool

    @property
    def dependencies(self) -> tuple[Quantity, ...]:
        # These values must exist before grouped derivative execution.
        return tuple(dict.fromkeys((self.derivative.of, self.derivative.wrt,
                                    *(self.derivative.holding or ()))))


def plan_derivative(graph: ComputationGraph, derivative: Derivative) -> DerivativePlan:
    """Plan the source path using the declared computation graph."""
    stops = {derivative.wrt, *(derivative.holding or ())}
    leaves, indices, visited = {}, [], set()

    def visit(quantity):
        if quantity in visited:
            return
        visited.add(quantity)
        if isinstance(quantity, Derivative):
            raise NotImplementedError("Differentiating paths containing derivatives is not implemented")
        if quantity in stops or quantity not in graph.model_producers:
            leaves[quantity] = None
            return
        index = graph.model_producers[quantity]
        for required in graph.models[index].inputs:
            visit(required)
        if index not in indices:
            indices.append(index)

    visit(derivative.of)
    # Preserve global dependency order, including models with multiple outputs.
    ordered = tuple(key for kind, key in graph.ordered
                    if kind == "model" and key in indices)
    return DerivativePlan(derivative, graph.model_producers.get(derivative.of),
                          ordered, tuple(leaves), derivative.wrt in leaves)


class DerivativeEngine:
    """Own derivative graph integration, plans, and backend coordination.

    The host supplies a computation graph plus two callbacks: run a model, and
    store validated values. No nn.Module registration or mutable runtime state
    is retained here. The numerical backend never needs the hosting Solver.
    """

    def __init__(self, chunk_size=64, *, backend="autograd"):
        self.backend = resolve_backend(backend)
        self.chunk_size = chunk_size
        self.requests = ()
        self.plans = {}

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
        for derivative in self.requests:
            graph.add_node(("derivative", derivative), self.dependencies(derivative), (derivative,))

    @staticmethod
    def _collect(graph, quantity, requests):
        for related in _related_quantities(quantity):
            graph.register_quantity(related)
        if isinstance(quantity, Derivative):
            requests[quantity] = None

    @staticmethod
    def dependencies(derivative):
        return tuple(dict.fromkeys((derivative.of, derivative.wrt, *(derivative.holding or ()))))

    def build_plans(self, graph):
        """Resolve paths only after the complete graph has been ordered."""
        self.plans = {request: plan_derivative(graph, request) for request in self.requests}

    def execute(self, graph, needed, values, run_model, store):
        """Handle a derivative call; return False for ordinary host execution."""
        plans = [self.plans[key] for kind, key in graph.ordered
                 if kind == "derivative" and (kind, key) in needed]
        if not plans:
            return False

        self.backend.execute(graph, plans, needed, values, run_model, store, self.chunk_size)
        return True

    def validate(self, graph, plan, values):
        self.backend.validate(graph, plan, values)

    def node_id(self, node):
        return f"d{self.requests.index(node[1])}"

    @staticmethod
    def graph_inputs(derivative):
        yield derivative.of, "of"
        yield derivative.wrt, "wrt"
        for quantity in derivative.holding or ():
            yield quantity, "holding"

    def graph_detail(self, derivative):
        return f"source models: {self.plans[derivative].model_indices}; {self.backend.description}"
