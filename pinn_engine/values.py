"""Named Tensor values and runtime shape validation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from .quantities import Quantity

# Axis identity is its semantic name, independent of size or Tensor position.
Axis = str

def _axes(names: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize axis declarations without splitting a bare string into letters."""
    if isinstance(names, str):
        names = (names,)
    result = tuple(names)
    if any(not isinstance(axis, str) for axis in result):
        raise TypeError("Axis names must be strings")
    if any(not axis.strip() for axis in result):
        raise ValueError("Axis names must be non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError("An axis declaration cannot repeat the same sample axis")
    return result


@dataclass(frozen=True, eq=False)
class Value:
    """A Tensor, or a geometry list of Tensors, with named sample axes.

    A list represents the first named axis. Each member represents one geometry
    and has all remaining sample axes followed by intrinsic Quantity dimensions.
    For example, axes=("geometry", "point") can hold [(20, 3), (40, 3)]
    for a Quantity with shape=(3,). Lists must be nonempty and non-nested.
    The outer axis name is user-defined; it need not literally be "geometry".
    Equal-sized outputs remain lists too; stacking is an explicit model choice.

    Example: Value(temperatures, axes=("scenario", "point")) for scalar data
    shaped (S, P). Intrinsic Quantity dimensions follow these axes and are not
    named here: a vector value can have shape (S, P, 3) with the same axes.

    The Tensor is kept by reference, with no copying or detaching. Frozen fields
    protect the metadata but do not make Tensor storage immutable. Versions,
    caching, and other execution metadata may be considered later.
    """

    tensor: torch.Tensor | list[torch.Tensor]
    axes: str | Sequence[str] = ()

    def __post_init__(self) -> None:
        axes = _axes(self.axes)
        if isinstance(self.tensor, list):
            if not self.tensor or not axes:
                raise ValueError("A Tensor list requires members and an outer sample axis")
            members, sample_rank = self.tensor, len(axes) - 1
        else:
            members, sample_rank = (self.tensor,), len(axes)
        for tensor in members:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Value.tensor must be a Tensor or a flat list of Tensors")
            if sample_rank > tensor.ndim:
                raise ValueError("Value cannot have more sample axes than Tensor dimensions")
        object.__setattr__(self, "axes", axes)


def _record_size(sizes, axis, size) -> None:
    """Retain per-geometry lengths instead of forcing ragged axes to be equal.

    An integer is a shared dense length; a tuple contains one length per geometry.
    Shared dense axes must agree with every corresponding per-geometry length.
    """
    if axis in sizes:
        previous = sizes[axis]
        if isinstance(previous, tuple) and isinstance(size, tuple):
            matches = previous == size
        elif isinstance(previous, tuple):
            matches = all(n == size for n in previous)
        elif isinstance(size, tuple):
            matches = all(n == previous for n in size)
        else:
            matches = previous == size
        if not matches:
            raise ValueError(f"Inconsistent size for axis {axis!r}")
        if isinstance(previous, tuple):
            return
    sizes[axis] = size


def _check_value(quantity: Quantity, value: Value, sizes: dict) -> None:
    """Check intrinsic shapes and sample lengths, including ragged geometry axes."""
    if not isinstance(value, Value):
        raise TypeError(f"Value for {quantity.name!r} must be a Value object")
    is_list = isinstance(value.tensor, list)
    members = value.tensor if is_list else (value.tensor,)
    axes = value.axes[1:] if is_list else value.axes
    for tensor in members:
        if tensor.ndim != len(axes) + len(quantity.shape):
            raise ValueError(f"Rank mismatch for {quantity.name!r}; check its sample axes")
        if tuple(tensor.shape[len(axes):]) != quantity.shape:
            raise ValueError(f"Intrinsic shape mismatch for {quantity.name!r}")
    if is_list:
        _record_size(sizes, value.axes[0], len(members))
    for index, axis in enumerate(axes):
        size = tuple(t.shape[index] for t in members) if is_list else members[0].shape[index]
        _record_size(sizes, axis, size)
