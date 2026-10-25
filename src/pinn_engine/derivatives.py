"""Symbolic derivative requests, path planning, and the numerical backend boundary.

Numerical differentiation is deliberately unimplemented. Keep derivative-specific
changes here; Solver delegates to these functions while retaining its override hooks.
Imports of Solver and Value are type-only to avoid circular runtime imports.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence
from .geometry import Geometry
from .quantities import Quantity

if TYPE_CHECKING:
    from .solver import Solver
    from .values import Value

@dataclass(frozen=True, eq=False, init=False)
class Derivative(Quantity):
    """A first-order, full-Quantity derivative, independent of any AD backend.

    Derivative(a, b) is total along the hosting Solver's computation graph.
    Derivative(a, b, holding=(z,)) is partial: hold z's current value fixed
    while varying b. holding=None and holding=() deliberately differ in kind.
    Holding concerns this derivative only, NOT freezing or detaching training.

    Intrinsic shape is a.shape + b.shape, output components first. Runtime
    sample axes are NOT guessed here. Independent mapped samples can align;
    Solver validates that assumption before the default AD execution boundary.
    Full coupled Jacobians and numerical derivative backends remain pending.
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
    """Describe the function to differentiate, without implementing its backend.

    model_indices are dependency-ordered indices into Solver.models. Traversal
    stops at wrt and held quantities; their current values become function
    inputs. Other external inputs also remain fixed during this derivative.
    This is structural reachability, not proof of a nonzero numerical derivative.
    A future backend must preserve the outer parameter-training graph.
    """

    derivative: Derivative
    producer_index: int | None
    model_indices: tuple[int, ...]
    function_inputs: tuple[Quantity, ...]
    depends_on_wrt: bool

    @property
    def dependencies(self) -> tuple[Quantity, ...]:
        # These values must exist before the derivative execution placeholder.
        return tuple(dict.fromkeys((self.derivative.of, self.derivative.wrt,
                                    *(self.derivative.holding or ()))))


def plan_derivative(solver: Solver, derivative: Derivative) -> DerivativePlan:
    """Plan the source path using the hosting solver's dependency graph."""
    stops = {derivative.wrt, *(derivative.holding or ())}
    leaves, indices, visited = {}, [], set()

    def visit(quantity):
        if quantity in visited:
            return
        visited.add(quantity)
        if isinstance(quantity, Derivative):
            raise NotImplementedError("Differentiating paths containing derivatives is not implemented")
        if quantity in stops or quantity not in solver._producers:
            leaves[quantity] = None
            return
        index = solver._producers[quantity]
        for required in solver.models[index].inputs:
            visit(required)
        if index not in indices:
            indices.append(index)

    visit(derivative.of)
    # Preserve global dependency order, including models with multiple outputs.
    ordered = tuple(key for kind, key in solver._ordered
                    if kind == "model" and key in indices)
    return DerivativePlan(derivative, solver._producers.get(derivative.of),
                          ordered, tuple(leaves), derivative.wrt in leaves)


def evaluate_derivative(
    solver: Solver, plan: DerivativePlan, values: Mapping[Quantity, Value],
) -> Value:
    """Default AD boundary; numerical differentiation is still unimplemented.

    Re-evaluating paths at perturbed wrt values and preserving held values,
    parameter gradients, and mapped/core-axis semantics are NOT implemented.
    Do not return a zero, detached Tensor, or externally supplied substitute.
    A future non-AD backend can override Solver._evaluate_derivative and apply
    its own capability checks instead of the AD-only validation below.
    """
    solver.validate_autograd(plan, values)
    raise NotImplementedError(
        f"Derivative execution is not implemented: {plan.derivative.name}; "
        f"source model indices={plan.model_indices}"
    )


def validate_autograd(
    solver: Solver, plan: DerivativePlan, values: Mapping[Quantity, Value],
) -> None:
    """Reject unsupported alignment of dependent samples for this AD request.

    values contains the current source values, including of and wrt. Their
    layouts tell us which axes an independent-sample AD backend would align.
    Consuming any such axis on a wrt-dependent path removes that guarantee.
    This is conservative: evaluate's internals are not inspected.

    A reduction to a scalar has no shared sample axis and remains legal.
    A reduction followed by broadcasting back onto that axis is not legal.
    Held and wrt-independent branches do not count as differentiated paths.
    This check does not detach values or restrict ordinary loss.backward().
    """
    derivative = plan.derivative
    shared = set(values[derivative.of].axes) & set(values[derivative.wrt].axes)
    if not shared or not plan.depends_on_wrt:
        return

    # model_indices also contains branches needed to evaluate constants.
    # Only propagate coordinate dependence from wrt, stopping at holding.
    dependent = {derivative.wrt}
    held = set(derivative.holding or ())
    for index in plan.model_indices:
        model = solver.models[index]
        signature = model.signature
        active_inputs = [q for q in signature.inputs if q in dependent]
        if not active_inputs:
            continue
        consumed = {axis for q in active_inputs for axis in signature.inputs[q]}
        unsupported = shared & consumed
        if unsupported:
            raise NotImplementedError(
                f"Autograd across non-independent sample axes is not supported: "
                f"{sorted(unsupported)}; model {index} ({type(model).__name__}) "
                f"consumes these axes on the path for {derivative.name}. "
                "For independent computation, leave mapped axes out of the "
                "signature; otherwise use a suitable non-AD backend or an "
                "appropriate partial derivative with holding."
            )
        dependent.update(q for q in signature.outputs if q not in held)
