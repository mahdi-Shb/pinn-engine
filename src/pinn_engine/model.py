"""Model signatures and automatic named-axis vectorization."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence
import torch
from torch import nn
from .geometry import Geometry
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
    length, NOT independence: consuming a whole axis can mix its samples.
    Rename an output axis when its sample identity, ordering, or length changes.
    AD compatibility belongs to the requested derivative path, not this class.
    Additional runtime batch axes are not included in this signature.

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

    evaluate follows the core signature and receives raw Tensors.
    forward accepts Value objects and handles extra axes automatically.
    Call model(values) to preserve PyTorch hooks. Subclass constructors must call
    super().__init__() before assigning parameters or child modules.
    Subclasses provide signature and geometry through properties or class
    attributes. Their constructors are free to accept whatever they need.
    """

    @property
    @abstractmethod
    def signature(self) -> Signature:
        """Declare the inputs, outputs, and core axes of one function call."""
        raise NotImplementedError

    @property
    @abstractmethod
    def geometry(self) -> Geometry | None:
        """Declare the execution domain, or None for a nonspatial model.

        Compatibility checks with Field domains will use geometry relationships;
        that validation is not implemented by this declaration-only skeleton.
        """
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
        self, values: Mapping[Quantity, torch.Tensor]
    ) -> dict[Quantity, torch.Tensor]:
        """Evaluate one call according to signature using the subclass algorithm.

        For a pointwise call, each Field value has only its intrinsic shape.
        For a full-point call, the point axis precedes the Field's intrinsic shape.
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

        batch = _batch_axes(signature, {q: value.axes for q, value in values.items()})
        arguments = []
        for quantity, core in signature.inputs.items():
            value = values[quantity]
            actual = value.axes
            ordered = tuple(axis for axis in batch if axis in actual) + tuple(core)
            permutation = tuple(actual.index(axis) for axis in ordered)
            permutation += tuple(range(len(actual), value.tensor.ndim))
            arguments.append(value.tensor.permute(permutation))

        def call(*tensors):
            # User code sees only the core Tensor shapes, with no Value wrappers.
            result = self.evaluate(dict(zip(signature.inputs, tensors)))
            if set(result) != set(signature.outputs):
                raise ValueError("Model returned quantities different from its signature")
            for tensor in result.values():
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError("Model.evaluate must return raw Tensor values")
            return tuple(result[quantity] for quantity in signature.outputs)

        # Inputs share batch-axis order after permutation. Wrapping inside-out
        # makes each mapped axis leading when its corresponding vmap sees it.
        mapped = call
        for axis in reversed(batch):
            in_dims = tuple(0 if axis in values[q].axes else None for q in signature.inputs)
            mapped = torch.func.vmap(mapped, in_dims=in_dims, out_dims=0)

        outputs = {}
        for quantity, tensor in zip(signature.outputs, mapped(*arguments)):
            value = Value(tensor, batch + tuple(signature.outputs[quantity]))
            _check_value(quantity, value, sizes)
            outputs[quantity] = value
        return outputs
