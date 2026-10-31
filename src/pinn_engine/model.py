"""Model signatures and automatic named-axis vectorization."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence
import torch
from torch import nn
from .quantities import Quantity
from .values import Value, _axes, _check_value
from .derivatives import _related_quantities

@dataclass(frozen=True)
class Signature:
    """Axes consumed or produced in full by one call to the user function.

    Each Quantity maps to the axes required by the function.
    Accept axis names either alone or in a sequence, and normalize them to
    tuples of strings. A bare string names one axis, not its letters.
    () passes no sample axes; the Quantity's intrinsic shape still applies.
    (point,) passes the full point axis, for example for attention or integration.
    Core axes may also appear in outputs, for example for a graph network that
    predicts at its input points. Reusing a name preserves sample identity and
    length AND sample independence along that axis. A function mixing samples
    must rename that output axis. This is the author's promise, not something
    the engine can prove by inspecting arbitrary evaluate implementations.
    Rename an output axis when its sample identity, ordering, or length changes.
    AD compatibility belongs to the requested derivative path, not this class.
    Additional runtime batch axes are not included in this signature.

    "..." is a wildcard axis group. A signature containing it passes the
    original Value mapping to evaluate, once, without automatic batching or
    permutation. evaluate returns Values with explicit axes. Wildcard outputs
    retain the captured axes; omitting the wildcard reduces those axes away.
    Concrete-only inputs in this mode must already have exactly those axes.

    This structured representation is an initial notation for the skeleton.
    Text parsing is not implemented. Model.forward uses the structured
    signature to construct vmap calls directly.
    """

    inputs: Mapping[Quantity, str | Sequence[str]]
    outputs: Mapping[Quantity, str | Sequence[str]]

    def __post_init__(self) -> None:
        # Validate declarations locally instead of retaining a global registry.
        known_quantities: dict[Quantity, Quantity] = {}
        for side in ("inputs", "outputs"):
            declarations = {}
            for quantity, axes in getattr(self, side).items():
                if not isinstance(quantity, Quantity):
                    raise TypeError("Signature keys must be Quantity objects")
                for related in _related_quantities(quantity):
                    previous = known_quantities.get(related)
                    if previous is not None and previous.shape != related.shape:
                        raise ValueError(f"Conflicting shapes for quantity {related.name!r}")
                    known_quantities[related] = related

                declarations[quantity] = _axes(axes)

            # Copy declarations so changes to the original dictionary cannot
            # change the model's signature.
            # A plain copied dict also lets torch.compile trace Quantity keys.
            object.__setattr__(self, side, declarations)

        if any("..." in axes for axes in self.outputs.values()) and not any(
            "..." in axes for axes in self.inputs.values()
        ):
            raise ValueError("An output wildcard requires an input wildcard")

    @property
    def has_ellipsis(self) -> bool:
        return any("..." in axes for side in (self.inputs, self.outputs) for axes in side.values())

    def wildcard_axes(self, values: Mapping[Quantity, Value]) -> tuple[str, ...]:
        """Resolve the shared wildcard group from actual input layouts.

        Explicit axis names anywhere in the signature are not part of the
        wildcard group. Inputs may share a subset of captured axes, like named
        broadcasting. First input occurrence determines the output group order.
        """
        explicit = {a for side in (self.inputs, self.outputs)
                    for axes in side.values() for a in axes if a != "..."}
        return tuple(dict.fromkeys(
            a for q, pattern in self.inputs.items() if "..." in pattern
            for a in values[q].axes if a not in explicit
        ))

    @classmethod
    def from_string(cls, text: str) -> Signature:
        """Parse a text signature; syntax and name resolution are still pending."""
        raise NotImplementedError("Signature text parsing is not implemented yet")


def _batch_axes(signature: Signature, layouts) -> tuple[str, ...]:
    """Derive mapped axes from layouts, checking whole-axis requirements."""
    core_axes = set(
        axis for side in (signature.inputs, signature.outputs)
        for axes in side.values() for axis in axes
    )
    batch = tuple(dict.fromkeys(
        axis for quantity in signature.inputs for axis in layouts[quantity]
        if axis not in core_axes
    ))
    for quantity, core in signature.inputs.items():
        actual = set(layouts[quantity])
        if not set(core) <= actual:
            raise ValueError(f"Missing core axes for {quantity.name!r}")
        if (actual - set(core)) & core_axes:
            raise ValueError(
                f"An axis of {quantity.name!r} is core elsewhere in this model; "
                "declare it in this input's signature too"
            )
    return batch


class Model(nn.Module, ABC):
    """An abstract computation, not necessarily a neural or trainable model.

    evaluate follows the core signature and receives raw Tensors, or lists
    when a list axis is explicitly consumed in full.
    forward accepts Value objects and handles extra axes automatically.
    Call model(values) to preserve PyTorch hooks. Subclass constructors must call
    super().__init__() before assigning parameters or child modules.
    Subclasses provide signature through a property or class attribute. Their
    constructors are free to accept whatever they need.
    """

    @property
    @abstractmethod
    def signature(self) -> Signature:
        """Declare the inputs, outputs, and core axes of one function call."""
        raise NotImplementedError

    @property
    def inputs(self) -> tuple[Quantity, ...]:
        """Derive input quantities from signature instead of declaring them twice."""
        return tuple(self.signature.inputs)

    @property
    def outputs(self) -> tuple[Quantity, ...]:
        """Derive output quantities from signature instead of declaring them twice."""
        return tuple(self.signature.outputs)

    @abstractmethod
    def evaluate(
        self, values: Mapping[Quantity, torch.Tensor | list[torch.Tensor] | Value]
    ) -> dict[Quantity, torch.Tensor | list[torch.Tensor] | Value]:
        """Evaluate one call according to signature using the subclass algorithm.

        For a pointwise call, each Quantity value has only its intrinsic shape.
        For a full-point call, the point axis precedes the Quantity's intrinsic shape.
        A full list axis must be first in that input's core signature. evaluate
        then receives a list of Tensors, without Value wrappers. A returned list
        likewise represents the first declared output core axis.
        If any signature entry contains "...", receive the original Values and
        return Values; no automatic batching or axis permutation is performed.
        Returned Quantity keys must match outputs.
        """
        raise NotImplementedError

    def forward(self, values: Mapping[Quantity, Value]) -> dict[Quantity, Value]:
        """Apply evaluate to named batches, returning Tensors with their axes.

        Axes declared in signature are passed to evaluate in full and in the
        declared order. Other axes are mapped independently; an input missing
        a mapped axis is shared along it. Equal axis names declare matched
        samples and require equal lengths, rather than inferring matches by size.
        """
        signature = self.signature
        if set(values) != set(signature.inputs):
            raise ValueError(f"Expected input quantities: {[q.name for q in signature.inputs]}")
        declarations = {quantity: quantity for quantity in signature.inputs}
        sizes = {}
        for quantity, value in values.items():
            if quantity.shape != declarations[quantity].shape:
                raise ValueError(f"Conflicting shapes for {quantity.name!r}")
            _check_value(quantity, value, sizes)

        if signature.has_ellipsis:
            return self._evaluate_ellipsis(values, sizes)

        batch = _batch_axes(signature, {q: value.axes for q, value in values.items()})
        list_axes = {v.axes[0] for v in values.values() if isinstance(v.tensor, list)}
        if len(list_axes) > 1:
            raise ValueError("Only one outer list axis is supported in a model call")
        list_axis = next(iter(list_axes), None)
        if list_axis in batch:
            # Select a list member before vectorizing its own dense sample axes.
            # Other inputs may share that axis densely, or be shared by all list members.
            return self._map_list_axis(values, list_axis, sizes[list_axis])

        # A retained Tensor list can be vmapped over common dense axes, but
        # vmap cannot pair 20 points in one member with 40 in another.
        for axis in batch:
            length = sizes[axis]
            if isinstance(length, tuple) and len(set(length)) != 1:
                raise ValueError(
                    f"Cannot map ragged axis {axis!r} while retaining list axis "
                    f"{list_axis!r}; consume both axes in the signature"
                )

        arguments, layouts = [], []
        for quantity, core in signature.inputs.items():
            value = values[quantity]
            is_list = isinstance(value.tensor, list)
            if is_list and (not core or core[0] != list_axis):
                raise ValueError("A retained list axis must be first in the core signature")
            actual = value.axes[1:] if is_list else value.axes
            leaf_core = tuple(core[1:]) if is_list else tuple(core)
            ordered = tuple(axis for axis in batch if axis in actual) + leaf_core
            members = value.tensor if is_list else (value.tensor,)
            permutation = tuple(actual.index(axis) for axis in ordered)
            permutation += tuple(range(len(actual), members[0].ndim))
            tensors = [tensor.permute(permutation) for tensor in members]
            arguments.append(tensors if is_list else tensors[0])
            layouts.append(actual)

        def call(*tensors):
            # The list structure is preserved; vmap maps Tensor leaves inside it.
            result = self.evaluate(dict(zip(signature.inputs, tensors)))
            if set(result) != set(signature.outputs):
                raise ValueError("Model returned quantities different from its signature")
            for quantity, tensor in result.items():
                if isinstance(tensor, list):
                    if not signature.outputs[quantity] or not tensor:
                        raise ValueError("A list output requires members and a declared core axis")
                    if any(not isinstance(t, torch.Tensor) for t in tensor):
                        raise TypeError("Model.evaluate must return Tensors or flat Tensor lists")
                elif not isinstance(tensor, torch.Tensor):
                    raise TypeError("Model.evaluate must return raw Tensor values or Tensor lists")
            return tuple(result[quantity] for quantity in signature.outputs)

        # Nested in_dims describes both dense arguments and lists of Tensor leaves.
        mapped = call
        for axis in reversed(batch):
            in_dims = []
            for argument, actual in zip(arguments, layouts):
                dim = 0 if axis in actual else None
                in_dims.append([dim] * len(argument) if isinstance(argument, list) else dim)
            mapped = torch.func.vmap(mapped, in_dims=tuple(in_dims), out_dims=0)

        outputs = {}
        for quantity, tensor in zip(signature.outputs, mapped(*arguments)):
            core = tuple(signature.outputs[quantity])
            # A list axis stays outermost, including when dense batch axes are mapped.
            axes = (core[0],) + batch + core[1:] if isinstance(tensor, list) else batch + core
            value = Value(tensor, axes)
            _check_value(quantity, value, sizes)
            outputs[quantity] = value
        return outputs

    def _map_list_axis(self, values, axis, count) -> dict[Quantity, Value]:
        """Run one list member at a time, keeping reference slices and autograd.

        Output lists are deliberately not stacked, even if member sizes match.
        There is no padding, job scheduling, or implicit Cartesian matching.
        """
        collected, output_axes = {}, {}
        for index in range(count):
            selected = {}
            for quantity, value in values.items():
                if axis not in value.axes:
                    selected[quantity] = value
                    continue
                position = value.axes.index(axis)
                tensor = (value.tensor[index] if isinstance(value.tensor, list)
                          else value.tensor.select(position, index))
                axes = value.axes[:position] + value.axes[position + 1:]
                selected[quantity] = Value(tensor, axes)
            result = self.forward(selected)
            for quantity, value in result.items():
                if isinstance(value.tensor, list):
                    raise ValueError("Nested output list axes are not supported")
                if quantity in output_axes and output_axes[quantity] != value.axes:
                    raise ValueError("Output axes must agree across list members")
                output_axes[quantity] = value.axes
                collected.setdefault(quantity, []).append(value.tensor)
        return {q: Value(tensors, (axis,) + output_axes[q]) for q, tensors in collected.items()}


    def _evaluate_ellipsis(self, values, sizes) -> dict[Quantity, Value]:
        """Pass original named Values to a model that owns its batching.

        Wildcards capture runtime axis names, not Tensor dimensions or intrinsic
        component dimensions. Their absence in an output declares a reduction.
        """
        signature = self.signature
        captured = signature.wildcard_axes(values)

        def expand(pattern, available):
            group = tuple(axis for axis in available if axis in captured)
            return tuple(a for token in pattern for a in (group if token == "..." else (token,)))

        for quantity, pattern in signature.inputs.items():
            # Validate layout instead of silently permuting an object promised
            # to be passed untouched. The function can read Value.axes itself.
            actual = values[quantity].axes
            if actual != expand(pattern, actual):
                raise ValueError(f"Input axes for {quantity.name!r} do not match wildcard signature")
        result = self.evaluate(values)
        if set(result) != set(signature.outputs):
            raise ValueError("Model returned quantities different from its signature")
        declarations = {q: q for q in signature.outputs}
        for quantity, value in result.items():
            if quantity.shape != declarations[quantity].shape:
                raise ValueError(f"Conflicting shapes for {quantity.name!r}")
            _check_value(quantity, value, sizes)
            expected = expand(signature.outputs[quantity], captured)
            if value.axes != expected:
                raise ValueError(f"Output axes for {quantity.name!r} do not match wildcard signature")
        return result
