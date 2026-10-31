"""Direct PyTorch AD with shared-source reverse sweeps on one forward graph.

There is no per-model backward, tape, or path plan. Compatible requests share
one source direction and differentiate with respect to several targets at once.
Different sources stay separate; tensor-output directions are chunked.
"""

from math import prod
import torch
from .values import Value
from .derivative_backends import DerivativeBackend


def _leaves(value):
    return tuple(value.tensor) if isinstance(value.tensor, list) else (value.tensor,)


def _validate_alignment(graph, request, values):
    """Reject compressed shared axes when the declared graph may mix samples.

    This checks signature metadata only; it neither schedules nor differentiates
    individual models. Distinct axis names request the full cross-axis Jacobian.
    """
    shared = set(values[request.of].axes) & set(values[request.wrt].axes)
    if not shared:
        return
    memo = {}

    def dependence(quantity):
        if quantity == request.wrt:
            return True, set()
        if quantity in memo:
            return memo[quantity]
        node = graph.producer_node(quantity)
        if node is None:
            return False, set()
        if node == graph.producer_node(request.wrt) and node[0] == "model":
            # Shared output-axis names promise independence, including siblings.
            # Renamed or reduced axes lose that guarantee.
            return True, set(values[request.wrt].axes) - set(values[quantity].axes)
        active, mixed = False, set()
        for incoming in graph.dependencies[node]:
            depends, consumed = dependence(incoming)
            if depends:
                active = True
                mixed.update(consumed)
                if node[0] == "model":
                    # Preserved names promise independence regardless of whether
                    # evaluate saw a full core axis, a list, or a wildcard group.
                    # Never forget a reduction if a later node broadcasts it back.
                    mixed.update(set(values[incoming].axes) - set(values[quantity].axes))
        memo[quantity] = active, mixed
        return active, mixed

    _, mixed = dependence(request.of)
    if shared & mixed:
        raise NotImplementedError(
            "Autograd across non-independent sample axes is not supported: "
            f"{sorted(shared & mixed)}. Use distinct input/output axis names "
            "to request the full cross-axis Jacobian."
        )


class _TensorDerivative:
    """Layout handling for a dense request or one ragged outer member."""

    def __init__(self, request, values, outer=None, member=None):
        self.request, self.outer, self.member = request, outer, member
        self.source, self.target = values[request.of], values[request.wrt]
        self.y, self.y_axes = self.select(self.source)
        self.x, self.x_axes = self.select(self.target)
        self.shared = set(self.y_axes) & set(self.x_axes)
        self.extra = tuple(a for a in self.y_axes if a not in self.shared)
        self.extra_shape = tuple(self.y.shape[self.y_axes.index(a)] for a in self.extra)
        self.count = prod(self.extra_shape + request.of.shape)
        self.axes = self.y_axes + tuple(a for a in self.x_axes if a not in self.y_axes)
        self.parts = []

    def select(self, value):
        leaves = _leaves(value)
        if isinstance(value.tensor, list):
            return leaves[self.member], value.axes[1:]
        tensor, axes = leaves[0], value.axes
        if self.outer in axes:
            position = axes.index(self.outer)
            tensor = tensor.select(position, self.member)
            axes = axes[:position] + axes[position + 1:]
        return tensor, axes

    def seed(self, start, stop):
        # Never allocate the whole square identity for a large output.
        count = stop - start
        basis = self.y.new_zeros((count, self.count))
        basis.scatter_(1, torch.arange(start, stop, device=self.y.device)[:, None], 1)
        shape = (count,) + tuple(
            1 if a in self.shared else self.y.shape[i] for i, a in enumerate(self.y_axes)
        ) + self.request.of.shape
        return basis.reshape(shape).expand((count,) + tuple(self.y.shape))

    @property
    def target_tensor(self):
        # Select actual list leaves, never a dense slice made after forward.
        return self.target.tensor[self.member] if isinstance(self.target.tensor, list) else self.target.tensor

    def record(self, gradients, count):
        result = gradients.get(id(self.target_tensor))
        if result is None:
            result = self.x.new_zeros((count,) + tuple(self.x.shape))
        elif not isinstance(self.target.tensor, list) and self.outer in self.target.axes:
            result = result.select(1 + self.target.axes.index(self.outer), self.member)
        if not result.requires_grad:
            # Preserve zero training gradients and arbitrarily many higher orders.
            for tensor in (self.y, self.target_tensor):
                result = result + tensor.unsqueeze(0)[:0].sum()
        self.parts.append(result)

    def finish(self):
        if not self.parts:
            flat = self.x.new_empty((0,) + tuple(self.x.shape))
            for tensor in (self.y, self.target_tensor):
                flat = flat + tensor.unsqueeze(0)[:0].sum()
        else:
            flat = self.parts[0] if len(self.parts) == 1 else torch.cat(self.parts, dim=0)
        tensor = flat.reshape(self.extra_shape + self.request.of.shape + tuple(self.x.shape))
        positions = {a: i for i, a in enumerate(self.extra)}
        offset = len(self.extra) + len(self.request.of.shape)
        positions.update({a: offset + i for i, a in enumerate(self.x_axes)})
        permutation = tuple(positions[a] for a in self.axes)
        permutation += tuple(range(len(self.extra), offset))
        permutation += tuple(range(offset + len(self.x_axes), tensor.ndim))
        return tensor.permute(permutation), self.axes


def _reverse_group(tasks, chunk_size):
    """Use one source/seed and all distinct targets; no zero output lanes."""
    source = tasks[0]
    targets = tuple({id(task.target_tensor): task.target_tensor for task in tasks}.values())
    for start in range(0, source.count, chunk_size):
        stop = min(start + chunk_size, source.count)
        seed = source.seed(start, stop)
        gradients = {}
        if source.y.requires_grad:
            batched = stop - start > 1
            raw = torch.autograd.grad(
                (source.y,), targets,
                grad_outputs=(seed if batched else seed[0],),
                is_grads_batched=batched, create_graph=True,
                retain_graph=True, allow_unused=True,
            )
            gradients = {id(t): g if batched else g.unsqueeze(0)
                         for t, g in zip(targets, raw) if g is not None}
        for task in tasks:
            task.record(gradients, stop - start)


class AutogradBackend(DerivativeBackend):
    """Total derivatives through the existing PyTorch graph, including higher orders."""

    name = "autograd"
    description = "direct PyTorch AD"

    def execution_context(self):
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Derivative requests cannot run under torch.inference_mode(); use torch.no_grad() instead")
        return torch.enable_grad()

    def prepare_target(self, value, *, separate=False):
        tensors = []
        for tensor in _leaves(value):
            if not (tensor.is_floating_point() or tensor.is_complex()):
                raise TypeError("Derivative operands must contain real floating-point Tensors")
            # A distinct Quantity boundary, with no copy/detach and no mutation
            # of caller-owned requires_grad flags. Consumers use this exact view.
            if separate or not tensor.requires_grad:
                tensor = tensor.view_as(tensor)
            if not tensor.requires_grad:
                tensor.requires_grad_(True)
            tensors.append(tensor)
        return Value(tensors if isinstance(value.tensor, list) else tensors[0], value.axes)

    def validate(self, graph, request, values):
        self._check_values(request, values)
        _validate_alignment(graph, request, values)

    @staticmethod
    def _check_values(request, values):
        if request.holding:
            raise NotImplementedError("Direct AD does not support nonempty holding; use the partial-derivative-and-geometry-domain-support branch")
        for quantity in (request.of, request.wrt):
            if any(not t.is_floating_point() for t in _leaves(values[quantity])):
                raise TypeError("Derivative operands must contain real floating-point Tensors")
        if any(not t.requires_grad for t in _leaves(values[request.wrt])):
            raise ValueError("AD targets must require grad before the source forward runs")

    @staticmethod
    def _outer_layout(request, values):
        pair = values[request.of], values[request.wrt]
        outer_axes = {v.axes[0] for v in pair if isinstance(v.tensor, list)}
        if len(outer_axes) > 1:
            raise NotImplementedError("Derivative operands must share one outer list axis")
        outer = next(iter(outer_axes), None)
        count = next((len(v.tensor) for v in pair if isinstance(v.tensor, list)), 1)
        return outer, count

    def group_key(self, request, values):
        source, target = values[request.of], values[request.wrt]
        outer, _ = self._outer_layout(request, values)
        split = outer if isinstance(source.tensor, list) or outer in source.axes else None
        source_axes = tuple(a for a in source.axes if a != split)
        target_axes = tuple(a for a in target.axes if a != outer)
        shared = tuple(a for a in source_axes if a in target_axes)
        # Sharing of alone is insufficient: point/scenario contractions can
        # require different seeds. Tensor identity also protects direct callers
        # supplying different values for the same symbolic source.
        return (request.of, tuple(id(t) for t in _leaves(source)),
                source.axes, request.of.shape, split, shared)

    def compute(self, request, values, *, chunk_size=64):
        return self.compute_group((request,), {request: values}, chunk_size=chunk_size)[request]

    def compute_group(self, requests, inputs, *, chunk_size=64, prepared=None):
        tasks_by_request, groups = {}, {}
        for request in requests:
            values = inputs[request]
            self._check_values(request, values)
            outer, count = self._outer_layout(request, values)
            key = self.group_key(request, values)
            tasks = []
            for member in range(count):
                task = _TensorDerivative(request, values, outer, member if outer is not None else None)
                tasks.append(task)
                # If y has no outer axis (e.g. energy reduced over a list), one
                # reverse sweep can cover all target members too. A split source
                # instead gets one sweep per member, without padded zero seeds.
                split_source = isinstance(task.source.tensor, list) or outer in task.source.axes
                slot = member if split_source else None
                groups.setdefault((key, slot), []).append(task)
            tasks_by_request[request] = outer, tasks
        for tasks in groups.values():
            _reverse_group(tasks, chunk_size)
        result = {}
        for request, (outer, tasks) in tasks_by_request.items():
            completed = [task.finish() for task in tasks]
            axes = completed[0][1]
            tensors = [tensor for tensor, _ in completed]
            result[request] = (Value(tensors, (outer,) + axes) if outer is not None
                               else Value(tensors[0], axes))
        return result
