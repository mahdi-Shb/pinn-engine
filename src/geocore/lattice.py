"""Beam lattices (PicoGK ``Lattice``): beams = round cones, spheres = beams of
zero length.  Exact union-of-beams SDF with a BVH (numba)."""
from __future__ import annotations

from typing import Callable, Union as TUnion

import numpy as np

from . import _kernels as K
from .implicit import BBox, Implicit, _as_points

Radius = TUnion[float, Callable[[np.ndarray], np.ndarray]]

# unit cell beam lists in [0,1]^3 coordinates
_C = np.array([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], float)
_CUBE_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8)
               if np.sum(np.abs(_C[a] - _C[b])) == 1]
_FACE_CENTERS = np.array([[0.5, 0.5, 0], [0.5, 0.5, 1], [0.5, 0, 0.5],
                          [0.5, 1, 0.5], [0, 0.5, 0.5], [1, 0.5, 0.5]])


def _cell_segments(kind: str) -> np.ndarray:
    """[S, 2, 3] segments of one unit cell."""
    k = kind.lower()
    segs = []
    center = np.array([0.5, 0.5, 0.5])
    if k in ("cubic", "simple_cubic", "sc"):
        segs = [(_C[a], _C[b]) for a, b in _CUBE_EDGES]
    elif k == "bcc":
        segs = [(center, c) for c in _C]
    elif k in ("bccz", "bcc_z"):
        segs = [(center, c) for c in _C] + [(_C[a], _C[b]) for a, b in _CUBE_EDGES
                                             if _C[a][2] != _C[b][2]]
    elif k == "fcc":
        for f in _FACE_CENTERS:
            ax = int(np.argmax([f[0] in (0, 1), f[1] in (0, 1), f[2] in (0, 1)]))
            for c in _C:
                if c[ax] == f[ax]:
                    segs.append((f, c))
    elif k == "octet":
        segs = [s for s in _cell_segments("fcc")]
        for i in range(6):
            for j in range(i + 1, 6):
                if np.isclose(np.linalg.norm(_FACE_CENTERS[i] - _FACE_CENTERS[j]), np.sqrt(0.5)):
                    segs.append((_FACE_CENTERS[i], _FACE_CENTERS[j]))
        return np.asarray(segs, float).reshape(-1, 2, 3)
    elif k == "diamond":
        # FCC sites (incl. the 14 on the cell boundary) bonded to the 4 inner
        # tetrahedral sites
        fcc = np.concatenate([_C, _FACE_CENTERS])
        inner = np.array([[.25, .25, .25], [.75, .75, .25], [.75, .25, .75], [.25, .75, .75]])
        for t in inner:
            for s in fcc:
                if np.isclose(np.linalg.norm(s - t), np.sqrt(3) / 4):
                    segs.append((t, s))
    else:
        raise ValueError(f"unknown cell type '{kind}'")
    return np.asarray(segs, float).reshape(-1, 2, 3)


class Lattice:
    """Collection of beams ``(a, ra) -> (b, rb)`` and spheres."""

    def __init__(self):
        self._beams: list = []
        self._caps: list = []

    # PicoGK API --------------------------------------------------------
    def add_beam(self, a, b, ra: float, rb: float | None = None, round_cap: bool = True):
        rb = ra if rb is None else rb
        self._beams.append([*np.asarray(a, float), *np.asarray(b, float), float(ra), float(rb)])
        self._caps.append(bool(round_cap))
        return self

    def add_sphere(self, c, r: float):
        c = np.asarray(c, float)
        return self.add_beam(c, c, r, r)

    AddBeam = add_beam
    AddSphere = add_sphere

    def add_beams(self, a, b, ra, rb=None):
        a = np.asarray(a, float).reshape(-1, 3)
        b = np.asarray(b, float).reshape(-1, 3)
        ra = np.broadcast_to(np.asarray(ra, float), (len(a),))
        rb = ra if rb is None else np.broadcast_to(np.asarray(rb, float), (len(a),))
        self._beams.extend(np.column_stack([a, b, ra, rb]).tolist())
        self._caps.extend([True] * len(a))
        return self

    @property
    def beams(self) -> np.ndarray:
        return np.asarray(self._beams, float).reshape(-1, 8)

    @property
    def round_caps(self) -> np.ndarray:
        return np.asarray(self._caps, bool)

    def __len__(self):
        return len(self._beams)

    @property
    def bounds(self) -> BBox:
        b = self.beams
        r = np.maximum(b[:, 6], b[:, 7])[:, None]
        lo = np.minimum(b[:, 0:3], b[:, 3:6]) - r
        hi = np.maximum(b[:, 0:3], b[:, 3:6]) + r
        return BBox(lo.min(0), hi.max(0))

    def as_implicit(self) -> "LatticeImplicit":
        return LatticeImplicit(self)

    def output(self, representation: str = "MESH_GRAPH", **kwargs):
        """NN input of this geometry: ``"MESH_GRAPH"``, ``"POINT_CLOUD"``,
        ``"VOXEL_GRID"``, ``"PARAMETRIC_SURFACE"`` or ``"PINN_COLLOCATION"``.
        ``name=True`` keyword arguments select arrays, other keywords are
        builder options (see ``geocore.outputs``)."""
        from .outputs import output
        return output(self, representation, **kwargs)

    def outputs(self, representations=("MESH_GRAPH", "POINT_CLOUD", "VOXEL_GRID"), **kwargs):
        """{representation: output} for several representations."""
        from .outputs import outputs_all
        return outputs_all(self, representations, **kwargs)

    # generators ----------------------------------------------------------
    @staticmethod
    def from_cells(cell_type: str, vmin, vmax, cell_size, radius: Radius) -> "Lattice":
        """Periodic strut lattice filling the box [vmin, vmax].

        ``cell_type``: cubic | bcc | bccz | fcc | octet | diamond.
        ``radius``: constant, or callable f(points [N,3]) -> radii (graded
        lattices, like the PicoGK BeamLattice demo)."""
        vmin = np.asarray(vmin, float)
        vmax = np.asarray(vmax, float)
        cs = np.broadcast_to(np.asarray(cell_size, float), (3,))
        n = np.maximum(np.round((vmax - vmin) / cs).astype(int), 1)
        seg = _cell_segments(cell_type)
        ijk = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1).reshape(-1, 1, 1, 3)
        allseg = (seg[None] + ijk) * cs + vmin            # [cells, S, 2, 3]
        allseg = allseg.reshape(-1, 2, 3)
        # dedupe edges shared by neighbouring cells (orientation independent)
        fwd = _lex_less(allseg[:, 0], allseg[:, 1])[:, None]
        canon = np.where(fwd, allseg.reshape(-1, 6),
                         np.concatenate([allseg[:, 1], allseg[:, 0]], 1))
        key = np.round(canon / (1e-6 * cs.max())).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        allseg = allseg[np.sort(idx)]
        a, b = allseg[:, 0], allseg[:, 1]
        if callable(radius):
            ra = np.asarray(radius(a), float).reshape(-1)
            rb = np.asarray(radius(b), float).reshape(-1)
        else:
            ra = rb = np.full(len(a), float(radius))
        return Lattice().add_beams(a, b, ra, rb)


def _lex_less(a, b):
    """Row-wise lexicographic a < b."""
    out = np.zeros(len(a), bool)
    decided = np.zeros(len(a), bool)
    for k in range(3):
        lt = (a[:, k] < b[:, k]) & ~decided
        gt = (a[:, k] > b[:, k]) & ~decided
        out |= lt
        decided |= lt | gt
    return out


class LatticeImplicit(Implicit):
    """Exact SDF of a union of round-cone beams (outside), numba accelerated."""

    def __init__(self, lattice: Lattice):
        self.lattice = lattice
        b = lattice.beams
        if len(b) == 0:
            raise ValueError("empty lattice")
        self._b = np.ascontiguousarray(b)
        r = np.maximum(b[:, 6], b[:, 7])[:, None]
        lo = np.minimum(b[:, 0:3], b[:, 3:6]) - r
        hi = np.maximum(b[:, 0:3], b[:, 3:6]) + r
        self._bvh = K.build_bvh(lo, hi)
        self._bb = BBox(lo.min(0), hi.max(0))

    def _eval(self, p):
        p = np.ascontiguousarray(p)
        out = np.empty(len(p))
        K.beams_sdf_bvh(p, self._b, *self._bvh, out)
        return out

    def _bounds(self):
        return self._bb
