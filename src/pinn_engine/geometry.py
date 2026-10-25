"""Abstract geometry operations; geometric algorithms are supplied by users."""

from __future__ import annotations
from abc import ABC, abstractmethod

class Geometry(ABC):
    """Abstract domain operations for the new design, without an implementation.

    Concrete geometry implementations are supplied by the application.
    Geometry identity remains object identity, even when shapes are equivalent.
    """

    @abstractmethod
    def boundary(self) -> Geometry:
        """Return the boundary as a geometry."""
        raise NotImplementedError

    @abstractmethod
    def union(self, other: Geometry) -> Geometry:
        """Return the union of this domain and other."""
        raise NotImplementedError

    @abstractmethod
    def intersection(self, other: Geometry) -> Geometry:
        """Return the intersection, which may be an empty domain."""
        raise NotImplementedError

    @abstractmethod
    def difference(self, other: Geometry) -> Geometry:
        """Return the portion of this domain outside other."""
        raise NotImplementedError

    @abstractmethod
    def is_subset_of(self, other: Geometry) -> bool:
        """Test inclusion, including equality; raise if it cannot be determined.

        An unsupported or unknown relationship must not silently return False.
        """
        raise NotImplementedError

    def is_superset_of(self, other: Geometry) -> bool:
        """Test the reverse inclusion using the same subset contract."""
        return other.is_subset_of(self)

    def __or__(self, other: Geometry) -> Geometry:
        """Support a | b as shorthand for a.union(b)."""
        return self.union(other)

    def __and__(self, other: Geometry) -> Geometry:
        """Support a & b as shorthand for a.intersection(b)."""
        return self.intersection(other)

    def __sub__(self, other: Geometry) -> Geometry:
        """Support a - b as shorthand for a.difference(b)."""
        return self.difference(other)
