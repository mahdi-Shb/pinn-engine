"""Direct PyTorch AD on one forward graph, one derivative request at a time.

There is no per-model backward, tape, path plan, or cross-request VJP batching.
Only directions within a tensor-valued request are batched and chunked.
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
            # Sibling outputs may depend on each other inside evaluate. Treat
            # its full core axes conservatively; the tensor graph decides AD.
            return True, {a for axes in graph.models[node[1]].signature.inputs.values() for a in axes}
        active, mixed = False, set()
        for incoming in graph.dependencies[node]:
            depends, consumed = dependence(incoming)
            if depends:
                active = True
                mixed.update(consumed)
                if node[0] == "model":
                    mixed.update(graph.models[node[1]].signature.inputs[incoming])
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
        # Never allocate the whole square identity for a large output.
        count = stop - start
        basis = self.y.new_zeros((count, self.count))
        basis.scatter_(1, torch.arange(start, stop, device=self.y.device)[:, None], 1)
        shape = (count,) + tuple(
            1 if a in self.shared else self.y.shape[i] for i, a in enumerate(self.y_axes)
        ) + self.request.of.shape
        return basis.reshape(shape).expand((count,) + tuple(self.y.shape))

    def compute(self, chunk_size):
        parts = []
        # Differentiate with respect to the original target tensors, never a
        # slice created after forward: that slice is not an ancestor of y.
        targets = _leaves(self.target)
        for start in range(0, self.count, chunk_size):
            stop = min(start + chunk_size, self.count)
            seed = self.seed(start, stop)
            if self.y.requires_grad:
                # One direction uses normal reverse mode. Multiple component
                # directions within THIS request can use PyTorch's batched VJP.
                batched = stop - start > 1
                gradients = torch.autograd.grad(
                    (self.y,), targets,
                    grad_outputs=(seed if batched else seed[0],),
                    is_grads_batched=batched, create_graph=True,
                    retain_graph=True, allow_unused=True,
                )
                gradients = tuple(
                    (g if batched else g.unsqueeze(0)) if g is not None
                    else t.new_zeros((stop - start,) + tuple(t.shape))
                    for t, g in zip(targets, gradients)
                )
                result = self.select(self.target, gradients, batched=True)[0]
            else:
                result = self.x.new_zeros((stop - start,) + tuple(self.x.shape))
            if not result.requires_grad:
                # Constant/unused derivatives must remain differentiable at all
                # subsequent orders, and allow training with a zero gradient.
                for tensor in (self.y, *targets):
                    result = result + tensor.unsqueeze(0)[:0].sum()
            parts.append(result)
        flat = torch.cat(parts, dim=0) if parts else self.x.new_empty((0,) + tuple(self.x.shape))
        tensor = flat.reshape(self.extra_shape + self.request.of.shape + tuple(self.x.shape))
        positions = {a: i for i, a in enumerate(self.extra)}
        offset = len(self.extra) + len(self.request.of.shape)
        positions.update({a: offset + i for i, a in enumerate(self.x_axes)})
        permutation = tuple(positions[a] for a in self.axes)
        permutation += tuple(range(len(self.extra), offset))
        permutation += tuple(range(offset + len(self.x_axes), tensor.ndim))
        return tensor.permute(permutation), self.axes


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
            raise NotImplementedError("Direct AD does not support nonempty holding; use the partial-derivatives branch")
        for quantity in (request.of, request.wrt):
            if any(not t.is_floating_point() for t in _leaves(values[quantity])):
                raise TypeError("Derivative operands must contain real floating-point Tensors")
        if any(not t.requires_grad for t in _leaves(values[request.wrt])):
            raise ValueError("AD targets must require grad before the source forward runs")

    def compute(self, request, values, *, chunk_size=64):
        self._check_values(request, values)
        pair = values[request.of], values[request.wrt]
        outer_axes = {v.axes[0] for v in pair if isinstance(v.tensor, list)}
        if len(outer_axes) > 1:
            raise NotImplementedError("Derivative operands must share one outer list axis")
        outer = next(iter(outer_axes), None)
        count = next((len(v.tensor) for v in pair if isinstance(v.tensor, list)), 1)
        tensors = []
        for member in range(count):
            tensor, axes = _TensorDerivative(
                request, values, outer, member if outer is not None else None,
            ).compute(chunk_size)
            tensors.append(tensor)
        return Value(tensors, (outer,) + axes) if outer is not None else Value(tensors[0], axes)
