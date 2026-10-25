"""Optional model-path analysis, separate from derivative requests and backends.

Value-based methods need not import or call this helper. AD uses it to select
local reverse paths; callable-based finite differences may reuse it later.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from .quantities import Quantity

if TYPE_CHECKING:
    from ._graph import ComputationGraph
    from .derivatives import Derivative


@dataclass(frozen=True)
class DerivativePlan:
    """Optional model-path preparation for AD or callable-based differentiation.

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
    """Plan a callable path only when a numerical backend requests it."""
    from .derivatives import Derivative
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

