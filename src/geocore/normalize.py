"""Coordinate normalization shared by all representations.

``x_normalized = (x - center) / scale``  (one isotropic scale, so distances,
SDF values and edge lengths simply divide by ``scale``).

``normalization_id`` names the preprocessing.  When you trained a model with
a fixed normalization, rebuild it with the same ``center``/``scale``/``id``
(``Normalizer.from_dict``) so new inputs match training."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from .implicit import BBox


@dataclass
class Normalizer:
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: float = 1.0
    method: str = "none"
    id: str | None = None

    def __post_init__(self):
        self.center = np.asarray(self.center, np.float64).reshape(3)
        self.scale = float(self.scale)
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.id is None:
            if self.method == "none" and self.scale == 1.0 and not self.center.any():
                self.id = "none"
            else:
                key = json.dumps([self.method, np.round(self.center, 9).tolist(),
                                  round(self.scale, 12)])
                self.id = f"{self.method}:{hashlib.sha1(key.encode()).hexdigest()[:12]}"

    # ---------------------------------------------------------------- build
    @staticmethod
    def identity() -> "Normalizer":
        return Normalizer()

    @staticmethod
    def from_bounds(bounds: BBox, method: str = "unit_cube", margin: float = 0.0,
                    id: str | None = None) -> "Normalizer":
        """``unit_cube``: box centre -> 0, largest half extent -> 1 (fits
        [-1,1]^3).  ``unit_sphere``: box centre -> 0, half diagonal -> 1.
        ``margin`` adds a relative margin (0.05 = 5%)."""
        c = bounds.center
        if method == "unit_cube":
            s = 0.5 * float(np.max(bounds.size))
        elif method == "unit_sphere":
            s = 0.5 * float(np.linalg.norm(bounds.size))
        elif method == "none":
            return Normalizer()
        else:
            raise ValueError(f"unknown normalization method {method}")
        return Normalizer(c, max(s, 1e-12) * (1.0 + margin), method, id)

    @staticmethod
    def from_points(p, method: str = "unit_sphere", id: str | None = None) -> "Normalizer":
        """Centroid -> 0; max radius (unit_sphere) or max abs coord
        (unit_cube) -> 1 (PointNet style)."""
        p = np.asarray(p, np.float64).reshape(-1, 3)
        c = p.mean(0)
        q = p - c
        s = float(np.max(np.linalg.norm(q, axis=1))) if method == "unit_sphere" else float(np.abs(q).max())
        return Normalizer(c, max(s, 1e-12), "centroid_" + method, id)

    @staticmethod
    def from_dict(d: dict) -> "Normalizer":
        return Normalizer(d["center"], d["scale"], d.get("method", "custom"), d.get("id"))

    def to_dict(self) -> dict:
        return {"center": self.center.tolist(), "scale": self.scale,
                "method": self.method, "id": self.id}

    # ---------------------------------------------------------------- apply
    def points(self, p) -> np.ndarray:
        return (np.asarray(p, np.float64) - self.center) / self.scale

    def inverse_points(self, q) -> np.ndarray:
        return np.asarray(q, np.float64) * self.scale + self.center

    def lengths(self, d) -> np.ndarray:
        """Distances, SDF values, displacements."""
        return np.asarray(d, np.float64) / self.scale

    def areas(self, a) -> np.ndarray:
        return np.asarray(a, np.float64) / self.scale ** 2

    def bounds(self, b: BBox) -> BBox:
        return BBox(self.points(b.min), self.points(b.max))

    @property
    def is_identity(self) -> bool:
        return self.id == "none"
