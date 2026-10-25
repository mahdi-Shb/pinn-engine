"""Per-call forward tape and grouped local VJPs; no model forward is replayed.

Each model input gets a distinct identity view. Local VJPs stop at those views,
while their original parameter-training connections stay intact. Requests may
have different sources, targets, and held quantities. They share a local VJP
when their incoming symbolic adjoints match, and otherwise use batched VJPs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Mapping

import torch

from .values import Value
from .derivative_backends import DerivativeBackend

if TYPE_CHECKING:
    from ._graph import ComputationGraph
    from .derivatives import DerivativePlan
    from .quantities import Quantity


def _leaves(value):
    return tuple(value.tensor) if isinstance(value.tensor, list) else (value.tensor,)


class ForwardTrace:
    """Own only this call's input views and outputs, never a persistent cache."""

    def __init__(self):
        self.records = {}

    def run(self, index, model, values):
        marked = {}
        for quantity, value in values.items():
            tensors = []
            for tensor in _leaves(value):
                if tensor.is_floating_point() or tensor.is_complex():
                    # A fresh consumer edge, not a copy or a training detach.
                    tensor = tensor.view_as(tensor)
                    if not tensor.requires_grad:
                        tensor.requires_grad_(True)
                tensors.append(tensor)
            payload = tensors if isinstance(value.tensor, list) else tensors[0]
            marked[quantity] = Value(payload, value.axes)
        result = model(marked)
        self.records[index] = (marked, result)
        return result


@dataclass
class _Adjoint:
    tensors: tuple
    key: object


class _Task:
    """One request, or one outer-list member of a request.

    Shared independent sample axes are summed by the VJP seed and aligned at
    extraction. Only output components and unshared output axes need separate
    directions. This avoids a full sample-by-sample Jacobian for pointwise AD.
    """

    def __init__(self, plan, values, outer=None, member=None):
        self.plan = plan
        self.outer, self.member = outer, member
        self.source = values[plan.derivative.of]
        self.target = values[plan.derivative.wrt]
        self.y, self.y_axes = self.select(self.source)
        self.x, self.x_axes = self.select(self.target)
        if not self.y.is_floating_point() or not self.x.is_floating_point():
            raise TypeError("Derivative operands must contain real floating-point Tensors")
        self.shared = set(self.y_axes) & set(self.x_axes)
        self.extra = tuple(a for a in self.y_axes if a not in self.shared)
        self.extra_shape = tuple(self.y.shape[self.y_axes.index(a)] for a in self.extra)
        self.of_shape = plan.derivative.of.shape
        self.count = prod(self.extra_shape + self.of_shape)
        self.axes = self.y_axes + tuple(a for a in self.x_axes if a not in self.y_axes)
        self.parts = []

    def select(self, value, tensors=None, batched=False):
        leaves = _leaves(value) if tensors is None else tensors
        if isinstance(value.tensor, list):
            return leaves[self.member], value.axes[1:]
        tensor, axes = leaves[0], value.axes
        if self.outer in axes:
            position = axes.index(self.outer)
            tensor = tensor.select(position + int(batched), self.member)
            axes = axes[:position] + axes[position + 1:]
        return tensor, axes

    def seed(self, start, stop):
        count = stop - start
        # Allocate a chunk of basis rows, never the whole square identity.
        basis = self.y.new_zeros((count, self.count))
        basis.scatter_(1, torch.arange(start, stop, device=self.y.device)[:, None], 1)
        shape = (count,) + tuple(
            1 if a in self.shared else self.y.shape[i] for i, a in enumerate(self.y_axes)
        ) + self.of_shape
        local = basis.reshape(shape).expand((count,) + tuple(self.y.shape))
        if isinstance(self.source.tensor, list):
            tensors = tuple(local if j == self.member else t.new_zeros((count,) + tuple(t.shape))
                            for j, t in enumerate(_leaves(self.source)))
            slot = self.member
        elif self.outer in self.source.axes:
            tensors = (self.source.tensor.new_zeros((count,) + tuple(self.source.tensor.shape)),)
            tensors[0].select(1 + self.source.axes.index(self.outer), self.member).copy_(local)
            slot = self.member
        else:
            tensors = (local,)
            slot = None
        key = ("seed", self.plan.derivative.of, tuple(a for a in self.y_axes if a in self.shared),
               self.outer if slot is not None else None, slot, start, stop)
        return _Adjoint(tuple(tensors), key)

    def extract(self, adjoint, count):
        if adjoint is not None:
            result = self.select(self.target, adjoint.tensors, batched=True)[0]
            if result.requires_grad:
                return result
        else:
            result = self.x.new_zeros((count,) + tuple(self.x.shape))
        # A real structural zero, still differentiable for an outer training loss.
        # An empty view gives zero without multiplying possibly large values.
        for tensor in (*_leaves(self.source), self.x):
            if tensor.is_floating_point():
                result = result + tensor.unsqueeze(0)[:0].sum()
        return result

    def finish(self):
        if not self.parts:
            flat = self.x.new_empty((0,) + tuple(self.x.shape))
        else:
            flat = self.parts[0] if len(self.parts) == 1 else torch.cat(self.parts, dim=0)
        tensor = flat.reshape(self.extra_shape + self.of_shape + tuple(self.x.shape))
        # Current dimensions: extra output axes, output components, input axes,
        # input components. Put all semantic sample axes before components.
        positions = {a: i for i, a in enumerate(self.extra)}
        offset = len(self.extra) + len(self.of_shape)
        positions.update({a: offset + i for i, a in enumerate(self.x_axes)})
        permutation = tuple(positions[a] for a in self.axes)
        permutation += tuple(range(len(self.extra), offset))
        permutation += tuple(range(offset + len(self.x_axes), tensor.ndim))
        return tensor.permute(permutation)


def _depends(graph, quantity, derivative, memo):
    if quantity == derivative.wrt:
        return True
    if quantity in (derivative.holding or ()):
        return False
    if quantity not in memo:
        index = graph.model_producers.get(quantity)
        memo[quantity] = index is not None and any(
            _depends(graph, q, derivative, memo) for q in graph.models[index].inputs
        )
    return memo[quantity]


def _add(state, quantity, adjoint, intern):
    previous = state.get(quantity)
    if previous is None:
        state[quantity] = adjoint
    else:
        key = intern(("sum", tuple(sorted((previous.key, adjoint.key)))))
        state[quantity] = _Adjoint(
            tuple(a + b for a, b in zip(previous.tensors, adjoint.tensors)), key,
        )


def _reverse(graph, trace, blocks):
    """Traverse model boundaries once per direction chunk, with per-request cuts."""
    provenance = {}

    def intern(key):
        # Small expression IDs avoid recursively growing keys on deep graphs.
        return provenance.setdefault(key, len(provenance))

    states, memos = [], [{} for _ in blocks]
    for task, start, stop in blocks:
        seed = task.seed(start, stop)
        states.append({task.plan.derivative.of: _Adjoint(seed.tensors, intern(seed.key))})

    for kind, index in reversed(graph.ordered):
        if kind != "model" or index not in trace.records:
            continue
        inputs, outputs = trace.records[index]
        groups = {}
        for i, (task, start, stop) in enumerate(blocks):
            derivative = task.plan.derivative
            # Stop at held quantities and at this request's own target.
            active = {q: states[i][q] for q in outputs if q in states[i]
                      and q != derivative.wrt
                      and _depends(graph, q, derivative, memos[i])}
            if active:
                key = tuple((q, active[q].key) for q in outputs if q in active)
                groups.setdefault(key, []).append((i, active))
        if not groups:
            continue

        # Identical incoming adjoints share computation even when the requests
        # have different wrt/holding. Their policies apply when distributing it.
        groups = list(groups.items())
        counts = [blocks[members[0][0]][2] - blocks[members[0][0]][1]
                  for _, members in groups]
        output_leaves = [(q, j, t) for q, value in outputs.items()
                         for j, t in enumerate(_leaves(value)) if t.requires_grad
                         and any(q in members[0][1] for _, members in groups)]
        input_leaves = [(q, j, t) for q, value in inputs.items()
                        for j, t in enumerate(_leaves(value)) if t.requires_grad
                        and any(_depends(graph, q, blocks[i][0].plan.derivative, memos[i])
                                for _, members in groups for i, _ in members)]
        if not output_leaves or not input_leaves:
            continue
        seeds = []
        for q, j, tensor in output_leaves:
            parts = [
                members[0][1][q].tensors[j] if q in members[0][1]
                else tensor.new_zeros((count,) + tuple(tensor.shape))
                for (_, members), count in zip(groups, counts)
            ]
            seeds.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=0))
        gradients = torch.autograd.grad(
            tuple(t for _, _, t in output_leaves), tuple(t for _, _, t in input_leaves),
            grad_outputs=tuple(seeds), is_grads_batched=True,
            create_graph=True, retain_graph=True, allow_unused=True,
        )
        position = 0
        for (key, members), count in zip(groups, counts):
            contributions = {}
            for (q, j, tensor), gradient in zip(input_leaves, gradients):
                if gradient is not None:
                    contributions.setdefault(q, {})[j] = gradient[position:position + count]
            for q, leaves in contributions.items():
                tensors = tuple(leaves[j] if j in leaves else t.new_zeros((count,) + tuple(t.shape))
                                for j, t in enumerate(_leaves(inputs[q])))
                adjoint = _Adjoint(tensors, intern(("pull", index, key, q)))
                for i, _ in members:
                    if _depends(graph, q, blocks[i][0].plan.derivative, memos[i]):
                        _add(states[i], q, adjoint, intern)
            position += count
        for i, active in (item for _, members in groups for item in members):
            for q in active:
                states[i].pop(q)

    for state, (task, start, stop) in zip(states, blocks):
        task.parts.append(task.extract(state.get(task.plan.derivative.wrt), stop - start))


def execute(graph, plans, values, trace, chunk_size):
    """Evaluate first derivatives using this forward's tape, retaining training AD."""
    tasks, by_request = [], {}
    for plan in plans:
        pair = (values[plan.derivative.of], values[plan.derivative.wrt])
        outer_axes = {v.axes[0] for v in pair if isinstance(v.tensor, list)}
        if len(outer_axes) > 1:
            raise NotImplementedError("Derivative operands must share one outer list axis")
        outer = next(iter(outer_axes), None)
        count = next((len(v.tensor) for v in pair if isinstance(v.tensor, list)), 1)
        current = [_Task(plan, values, outer, i if outer is not None else None)
                   for i in range(count)]
        by_request[plan.derivative] = current
        tasks.extend(current)

    # Chunk the combined directions, not a dense Jacobian of the shared model.
    blocks, used = [], 0
    for task in tasks:
        start = 0
        while start < task.count:
            stop = min(task.count, start + chunk_size - used)
            blocks.append((task, start, stop))
            used += stop - start
            start = stop
            if used == chunk_size:
                _reverse(graph, trace, blocks)
                blocks, used = [], 0
    if blocks:
        _reverse(graph, trace, blocks)

    result = {}
    for derivative, current in by_request.items():
        tensors = [task.finish() for task in current]
        axes = current[0].axes
        if current[0].outer is not None:
            result[derivative] = Value(tensors, (current[0].outer,) + axes)
        else:
            result[derivative] = Value(tensors[0], axes)
    return result


def validate_autograd(
    graph: ComputationGraph, plan: DerivativePlan, values: Mapping[Quantity, Value],
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
        model = graph.models[index]
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


class AutogradBackend(DerivativeBackend):
    """Grouped local reverse-mode AD using one recorded forward per model."""

    name = "autograd"
    description = "local reverse-mode AD"

    def validate(self, graph, plan, values):
        validate_autograd(graph, plan, values)

    def execute(self, graph, plans, needed, values, run_model, store, chunk_size):
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Derivative requests cannot run under torch.inference_mode(); use torch.no_grad() instead")
        trace = ForwardTrace()
        recorded = {index for plan in plans for index in plan.model_indices}
        prerequisites = graph.needed_nodes(tuple(q for plan in plans for q in plan.dependencies))
        if any(kind == "derivative" for kind, _ in prerequisites):
            raise NotImplementedError("Differentiating paths containing derivatives is not implemented")
        with torch.enable_grad():
            # Complete source values before batching requests. Residuals wait.
            for kind, index in graph.ordered:
                if (kind, index) in prerequisites:
                    run_model(index, trace.run if index in recorded else None)
            for plan in plans:
                self.validate(graph, plan, values)
            store(execute(graph, plans, values, trace, chunk_size))
            for kind, index in graph.ordered:
                if kind == "model" and (kind, index) in needed and (kind, index) not in prerequisites:
                    run_model(index)
