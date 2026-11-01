"""Nested derivative declarations and ordinary graph execution coordination.

Numerical execution is selected through DerivativeBackend. Autograd-specific
capability checks and gradient mode live in _autograd.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from .quantities import Quantity
from .derivative_backends import resolve_backend

@dataclass(frozen=True, eq=False, init=False)
class Derivative(Quantity):
    """A full-Quantity derivative, optionally of another derivative.

    Derivative(a, b) is total along the hosting Solver's computation graph.
    Derivative(a, b, holding=(z,)) is partial: hold z's current value fixed
    while varying b. holding=None and holding=() deliberately differ in kind.
    Holding concerns this derivative only, NOT freezing or detaching training.

    Intrinsic shape is a.shape + b.shape, output components first. Runtime
    sample axes are NOT guessed here. Independent mapped samples can align;
    The selected numerical backend validates its own sample-axis capabilities.
    Nest of to request higher orders. Component indexing is not implemented.
    A backend may reject holding; direct AD supports only an empty holding set.
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
    """Include nested symbolic operands when checking repeated shapes."""
    yield quantity
    if isinstance(quantity, Derivative):
        for operand in (quantity.of, quantity.wrt, *(quantity.holding or ())):
            yield from _related_quantities(operand)


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
        self.requirements = {}

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
            if isinstance(related, Derivative):
                requests[related] = None

    def dependencies(self, derivative):
        return self.requirements[derivative]

    def execute(self, graph, needed, values, run_model, store):
        """Handle a derivative call; return False for ordinary host execution."""
        requests = [key for kind, key in graph.ordered
                 if kind == "derivative" and (kind, key) in needed]
        if not requests:
            return False

        targets = {request.wrt for request in requests}
        # An intermediate target may be used inside its producer before being
        # returned. Enable its floating external ancestors before that forward.
        ancestors = graph.needed_nodes(tuple(targets))
        roots = targets | {q for node in ancestors for q in graph.dependencies[node]}

        def prepare_targets(quantities, *, external=False):
            for quantity in quantities:
                if quantity in (roots if external else targets):
                    value = values[quantity]
                    leaves = value.tensor if isinstance(value.tensor, list) else (value.tensor,)
                    if external and quantity not in targets and not all(
                        t.is_floating_point() or t.is_complex() for t in leaves
                    ):
                        continue
                    # Separate aliases, but preserve a genuine intermediate
                    # tensor already used by sibling outputs of the same model.
                    aliases = {id(t) for q, v in values.items() if q != quantity
                               for t in (v.tensor if isinstance(v.tensor, list) else (v.tensor,))}
                    separate = external or any(id(t) in aliases for t in leaves)
                    values[quantity] = self.backend.prepare_target(value, separate=separate)

        # Finish ready models before collecting a derivative frontier. This
        # exposes compatible requests even when their producers are far apart
        # in the declared graph, without replay or executing unselected nodes.
        with self.backend.execution_context():
            prepare_targets(tuple(values), external=True)
            pending = set(needed)
            prepared_groups = {}  # Per-call numeric reuse, never persistent state.
            while pending:
                # graph.ordered already places upstream models first, so one
                # pass completes every currently reachable ordinary model.
                for node in graph.ordered:
                    if node in pending and node[0] == "model" and all(
                        q in values for q in graph.dependencies[node]
                    ):
                        run_model(node[1])
                        prepare_targets(graph.node_outputs[node])
                        pending.remove(node)
                ready = [key for kind, key in graph.ordered
                         if kind == "derivative" and (kind, key) in pending
                         and all(q in values for q in self.dependencies(key))]
                if not ready:
                    if pending:
                        raise RuntimeError("No ready derivative or model; a declared dependency is missing")
                    break
                inputs, groups = {}, {}
                for request in ready:
                    self.validate(graph, request, values)
                    inputs[request] = {q: values[q] for q in self.dependencies(request)}
                    key = self.backend.group_key(request, inputs[request])
                    # None means singleton, not one large default group.
                    key = (False, request) if key is None else (True, key)
                    groups.setdefault(key, []).append(request)
                for key, group in groups.items():
                    group_inputs = {request: inputs[request] for request in group}
                    if key not in prepared_groups:
                        prepared_groups[key] = self.backend.prepare_group(tuple(group), group_inputs)
                    result = self.backend.compute_group(
                        tuple(group), group_inputs, chunk_size=self.chunk_size,
                        prepared=prepared_groups[key],
                    )
                    if set(result) != set(group):
                        raise ValueError("Derivative backend returned requests different from its group")
                    store(result)
                    prepare_targets(group)
                    pending.difference_update(("derivative", request) for request in group)
        return True

    def validate(self, graph, request, values):
        self.backend.validate(graph, request, values)

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
        return self.backend.description
