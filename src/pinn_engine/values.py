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
    """A Tensor together with its actual, ordered sample-axis names.

    Example: Value(temperatures, axes=("scenario", "point")) for scalar data
    shaped (S, P). Intrinsic Quantity dimensions follow these axes and are not
    named here: a vector value can have shape (S, P, 3) with the same axes.

    The Tensor is kept by reference, with no copying or detaching. Frozen fields
    protect the metadata but do not make Tensor storage immutable. Versions,
    caching, and other execution metadata may be considered later.
    """

    tensor: torch.Tensor
    axes: str | Sequence[str] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("Value.tensor must be a Tensor")
        axes = _axes(self.axes)
        if len(axes) > self.tensor.ndim:
            raise ValueError("Value cannot have more sample axes than Tensor dimensions")
        object.__setattr__(self, "axes", axes)


def _check_value(quantity: Quantity, value: Value, sizes: dict[str, int]) -> None:
    """Validate a value against its Quantity and shared sample-axis sizes."""
    if not isinstance(value, Value):
        raise TypeError(f"Value for {quantity.name!r} must be a Value object")
    if value.tensor.ndim != len(value.axes) + len(quantity.shape):
        raise ValueError(f"Rank mismatch for {quantity.name!r}; check its sample axes")
    if tuple(value.tensor.shape[len(value.axes):]) != quantity.shape:
        raise ValueError(f"Intrinsic shape mismatch for {quantity.name!r}")
    for axis, size in zip(value.axes, value.tensor.shape):
        if axis in sizes and sizes[axis] != size:
            raise ValueError(f"Inconsistent size for axis {axis!r}")
        sizes[axis] = size
