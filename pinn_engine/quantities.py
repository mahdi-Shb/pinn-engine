"""Semantic quantity declarations, independent of runtime data and execution."""

from __future__ import annotations
from dataclasses import dataclass, field
from .geometry import Geometry

@dataclass(frozen=True, eq=False)
class Quantity:
    """Declare a nonspatial quantity without a runtime value, version, or cache.

    shape is intrinsic: scalar (), vector (3,), stress tensor (3, 3).
    Scenario and point counts are excluded from shape.
    Definitions with the same name and domain identify the same dictionary key.
    Construct new instances freely; there is no global registry or get factory.
    Signature checks shape consistency when declarations are used together.
    """

    name: str
    shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Quantity.name must be a non-empty string")

        shape = tuple(self.shape)
        if any(type(size) is not int or size < 0 for size in shape):
            raise ValueError("Quantity.shape must contain non-negative integers")
        object.__setattr__(self, "shape", shape)

    @property
    def _domain(self) -> Geometry | None:
        # Nonspatial quantities have no domain; Field overrides this property.
        return None

    @property
    def _identity(self) -> tuple:
        # Geometry identity currently means the same object, regardless of shape
        # equality or whether two geometries belong to the same class.
        domain = self._domain
        return (None if domain is None else id(domain), self.name)

    def __hash__(self) -> int:
        return hash(self._identity)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return NotImplemented
        return self._identity == other._identity


@dataclass(frozen=True, eq=False)
class Field(Quantity):
    """A spatial quantity whose values are defined at points of a geometry.

    Example: Field("T", shape=(), geometry=solid).
    Reuse the same geometry object when redeclaring the same field.
    """

    geometry: Geometry = field(kw_only=True)

    def __post_init__(self) -> None:
        if self.geometry is None:
            raise ValueError("Field.geometry is required")
        super().__post_init__()

    @property
    def _domain(self) -> Geometry:
        return self.geometry
