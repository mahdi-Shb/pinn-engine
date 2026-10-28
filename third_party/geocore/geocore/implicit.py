"""Implicit (signed distance) geometry, PicoGK conventions.

* ``sdf(p) < 0`` inside, ``> 0`` outside, ``0`` on the surface, millimetres.
* Every node evaluates batches of points ``p`` of shape ``[N, 3]`` (float64).
* Every node can also report which *tagged* surface owns the value
  (``sdf_tagged``), which is how boundary-condition masks are produced.

Primitive SDFs are exact (Inigo Quilez formulas, the same ones the PicoGK web
modeler and the C# demos use).  CSG nodes return the usual min/max bounds,
which are exact outside unions and on the zero level set; ``Voxels`` turns any
tree into an exact distance field by redistancing.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from . import _kernels as K

Array = np.ndarray
_CHUNK = 1 << 20
_COMPILE_MIN = 4096      # batches at least this large use the compiled path


def _as_points(p) -> Array:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        p = p[None, :]
    if p.shape[-1] != 3:
        raise ValueError(f"points must have shape [N,3], got {p.shape}")
    return np.ascontiguousarray(p.reshape(-1, 3))


def _vec(v) -> Array:
    return np.asarray(v, dtype=np.float64).reshape(3)


class BBox:
    """Axis aligned box (PicoGK BBox3)."""

    __slots__ = ("min", "max")

    def __init__(self, vmin, vmax):
        self.min = _vec(vmin)
        self.max = _vec(vmax)

    @property
    def size(self) -> Array:
        return self.max - self.min

    @property
    def center(self) -> Array:
        return 0.5 * (self.min + self.max)

    def is_empty(self) -> bool:
        return bool(np.any(self.max < self.min))

    def expanded(self, d: float) -> "BBox":
        return BBox(self.min - d, self.max + d)

    def union(self, o: "BBox") -> "BBox":
        return BBox(np.minimum(self.min, o.min), np.maximum(self.max, o.max))

    def intersection(self, o: "BBox") -> "BBox":
        return BBox(np.maximum(self.min, o.min), np.minimum(self.max, o.max))

    def as_array(self) -> Array:
        return np.stack([self.min, self.max])

    def __repr__(self):
        return f"BBox(min={self.min.tolist()}, max={self.max.tolist()})"


# ==========================================================================
# base class
# ==========================================================================
class Implicit:
    """Base class.  Subclasses implement ``_eval`` (and ``_bounds``)."""

    exact: bool = True          # true Euclidean distance everywhere?
    tag: Optional[str] = None

    # ------------------------------------------------------------- evaluation
    def _eval(self, p: Array) -> Array:
        raise NotImplementedError

    def _eval_t(self, p: Array, table: dict) -> tuple[Array, Array]:
        d = self._eval(p)
        return d, np.full(d.shape, table[id(self)], np.int32)

    def _bounds(self) -> Optional[BBox]:
        return None

    def _children(self) -> Sequence["Implicit"]:
        return ()

    def sdf(self, p) -> Array:
        """Signed distance at points [N,3] -> [N] (float64).

        Large batches of CSG trees run as one compiled, multi-threaded numba
        pass (``geocore._compiled``); small ones use numpy."""
        p = _as_points(p)
        if len(p) >= _COMPILE_MIN and self._compilable():
            prog = self._program()
            if prog is not None:
                from . import _compiled
                out = np.empty(len(p))
                for s in range(0, len(p), 4 * _CHUNK):
                    out[s:s + 4 * _CHUNK] = _compiled.evaluate(prog, p[s:s + 4 * _CHUNK])
                return out
        if len(p) <= _CHUNK:
            return self._eval(p)
        out = np.empty(len(p))
        for s in range(0, len(p), _CHUNK):
            out[s:s + _CHUNK] = self._eval(p[s:s + _CHUNK])
        return out

    __call__ = sdf

    def _compilable(self) -> bool:
        from . import _compiled
        return bool(self._children()) or type(self) in _compiled._native_types()

    def _program(self):
        """Compiled stack program of this tree (cached; trees are immutable)."""
        try:
            return self.__dict__["_prog_cache"]
        except KeyError:
            from . import _compiled
            prog = _compiled.compile_tree(self)
            self.__dict__["_prog_cache"] = prog
            return prog

    # ------------------------------------------------------------- tags
    def tagged(self, name: str) -> "Implicit":
        """Name the surface(s) of this node (used for boundary masks)."""
        self.tag = name
        self.__dict__.pop("_geometry_cache", None)
        return self

    def _tag_leaves(self) -> list["Implicit"]:
        """Nodes that own a surface: tagged nodes and untagged leaves."""
        out = []

        def walk(n):
            if n.tag is not None or not n._children():
                out.append(n)
                return
            for c in n._children():
                walk(c)
        walk(self)
        return out

    def tag_names(self) -> list[str]:
        names, seen = [], {}
        for n in self._tag_leaves():
            nm = n.tag if n.tag is not None else type(n).__name__.lower()
            if nm not in seen:
                seen[nm] = len(names)
                names.append(nm)
        return names

    def _tag_table(self) -> dict:
        names = self.tag_names()
        table = {}
        for n in self._tag_leaves():
            nm = n.tag if n.tag is not None else type(n).__name__.lower()
            table[id(n)] = names.index(nm)
        return table

    def sdf_tagged(self, p) -> tuple[Array, Array]:
        """(sdf [N], tag index [N]) - tag index into ``tag_names()``."""
        p = _as_points(p)
        table = self._tag_table()
        d = np.empty(len(p))
        t = np.empty(len(p), np.int32)
        for s in range(0, len(p), _CHUNK):
            d[s:s + _CHUNK], t[s:s + _CHUNK] = self._eval_tagged(p[s:s + _CHUNK], table)
        return d, t

    def _eval_tagged(self, p, table):
        if id(self) in table:          # tagged node or leaf: owns its surface
            d = self._eval(p)
            return d, np.full(d.shape, table[id(self)], np.int32)
        return self._eval_t(p, table)

    # ------------------------------------------------------------- geometry
    @property
    def bounds(self) -> Optional[BBox]:
        return self._bounds()

    def gradient(self, p, eps: float | None = None) -> Array:
        """Central difference gradient [N,3] (unit length for exact SDFs)."""
        p = _as_points(p)
        if eps is None:
            b = self.bounds
            scale = float(np.max(b.size)) if b is not None else 1.0
            eps = 1e-5 * max(scale, 1e-3)
        g = np.empty_like(p)
        for a in range(3):
            q = p.copy()
            q[:, a] += eps
            fp = self.sdf(q)
            q[:, a] -= 2 * eps
            fm = self.sdf(q)
            g[:, a] = (fp - fm) / (2 * eps)
        return g

    def normal(self, p, eps: float | None = None) -> Array:
        g = self.gradient(p, eps)
        n = np.linalg.norm(g, axis=1, keepdims=True)
        return g / np.maximum(n, 1e-30)

    def project(self, p, iters: int = 8, tol: float = 1e-9) -> Array:
        """Newton projection of points onto the zero level set."""
        q = _as_points(p).copy()
        for _ in range(iters):
            d = self.sdf(q)
            if np.max(np.abs(d)) < tol:
                break
            g = self.gradient(q)
            g2 = np.maximum(np.einsum("ij,ij->i", g, g), 1e-30)
            q -= (d / g2)[:, None] * g
        return q

    def contains(self, p) -> Array:
        return self.sdf(p) <= 0.0

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

    # ------------------------------------------------------------- CSG sugar
    def __or__(self, o):  return Union(self, o)
    def __add__(self, o): return Union(self, o)
    def __sub__(self, o): return Difference(self, o)
    def __and__(self, o): return Intersection(self, o)
    def __neg__(self):    return Complement(self)

    def union(self, *o):        return Union(self, *o)
    def subtract(self, *o):     return Difference(self, Union(*o) if len(o) > 1 else o[0])
    def intersect(self, *o):    return Intersection(self, *o)

    def smooth_union(self, o, k: float):        return SmoothUnion(self, o, k)
    def smooth_subtract(self, o, k: float):     return SmoothDifference(self, o, k)
    def smooth_intersect(self, o, k: float):    return SmoothIntersection(self, o, k)

    def offset(self, d: float):  return Offset(self, d)
    def shell(self, t: float):   return Shell(self, t)
    def trim(self, vmin, vmax):  return Intersection(self, Box.from_bounds(vmin, vmax))

    def translate(self, t):      return Transform(self, translate=t)
    def rotate(self, axis, angle_rad, center=(0, 0, 0)):
        return Transform(self, rotate=(axis, angle_rad), center=center)
    def scale(self, s: float, center=(0, 0, 0)):
        return Transform(self, scale=s, center=center)
    def transform(self, matrix):  return Transform(self, matrix=matrix)


# ==========================================================================
# primitives
# ==========================================================================
class Sphere(Implicit):
    def __init__(self, center, radius: float):
        self.c = _vec(center)
        self.r = float(radius)

    def _eval(self, p):
        return np.linalg.norm(p - self.c, axis=1) - self.r

    def _bounds(self):
        return BBox(self.c - self.r, self.c + self.r)


class Box(Implicit):
    """Centered box of full size ``size``; optional edge ``rounding``."""

    def __init__(self, center, size, rounding: float = 0.0):
        self.c = _vec(center)
        self.half = 0.5 * np.abs(_vec(size))
        self.r = float(rounding)

    @classmethod
    def from_bounds(cls, vmin, vmax, rounding=0.0):
        vmin, vmax = _vec(vmin), _vec(vmax)
        return cls(0.5 * (vmin + vmax), vmax - vmin, rounding)

    def _eval(self, p):
        q = np.abs(p - self.c) - (self.half - self.r)
        out = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        return out + np.minimum(q.max(axis=1), 0.0) - self.r

    def _bounds(self):
        return BBox(self.c - self.half, self.c + self.half)


class Cylinder(Implicit):
    """Flat-ended cylinder from ``a`` to ``b`` (exact; the WebModel Cylinder)."""

    def __init__(self, a, b, radius: float):
        self.a, self.b, self.r = _vec(a), _vec(b), float(radius)

    def _eval(self, p):
        ba = self.b - self.a
        pa = p - self.a
        baba = ba @ ba
        paba = pa @ ba
        x = np.linalg.norm(pa * baba - np.outer(paba, ba), axis=1) - self.r * baba
        y = np.abs(paba - baba * 0.5) - baba * 0.5
        x2 = x * x
        y2 = y * y * baba
        inside = np.maximum(x, y) < 0
        d = np.where(inside, -np.minimum(x2, y2),
                     np.where(x > 0, x2, 0.0) + np.where(y > 0, y2, 0.0))
        return np.sign(d) * np.sqrt(np.abs(d)) / baba

    def _bounds(self):
        d = self.b - self.a
        L = np.linalg.norm(d) or 1.0
        e = self.r * np.sqrt(np.maximum(0.0, 1.0 - (d / L) ** 2))
        return BBox(np.minimum(self.a, self.b) - e, np.maximum(self.a, self.b) + e)


class Beam(Implicit):
    """PicoGK lattice beam / ``Voxels.voxLatticeBeam``: a round cone with
    spherical caps (radius ``ra`` at ``a``, ``rb`` at ``b``)."""

    def __init__(self, a, b, ra: float, rb: float | None = None):
        self.a, self.b = _vec(a), _vec(b)
        self.ra = float(ra)
        self.rb = float(ra if rb is None else rb)

    def _eval(self, p):
        out = np.empty(len(p))
        beams = np.array([[*self.a, *self.b, self.ra, self.rb]])
        K.beams_sdf_brute(np.ascontiguousarray(p), beams, out)
        return out

    def _bounds(self):
        r = max(self.ra, self.rb)
        return BBox(np.minimum(self.a, self.b) - r, np.maximum(self.a, self.b) + r)


Capsule = Beam


class Torus(Implicit):
    """Torus around the local z axis (PicoGK demo ``Torus``)."""

    def __init__(self, center, major_radius: float, minor_radius: float):
        self.c = _vec(center)
        self.R, self.r = float(major_radius), float(minor_radius)

    def _eval(self, p):
        d = p - self.c
        q = np.hypot(d[:, 0], d[:, 1]) - self.R
        return np.hypot(q, d[:, 2]) - self.r

    def _bounds(self):
        e = np.array([self.R + self.r, self.R + self.r, self.r])
        return BBox(self.c - e, self.c + e)


class Plane(Implicit):
    """Half space ``dot(n, p - point) <= 0`` (unbounded)."""

    def __init__(self, point, normal):
        self.p0 = _vec(point)
        n = _vec(normal)
        self.n = n / np.linalg.norm(n)

    def _eval(self, p):
        return (p - self.p0) @ self.n


class Cone(Implicit):
    """Capped cone (frustum) from ``a`` (radius ra) to ``b`` (radius rb), flat
    ends.  Exact (Quilez)."""

    def __init__(self, a, b, ra: float, rb: float):
        self.a, self.b, self.ra, self.rb = _vec(a), _vec(b), float(ra), float(rb)

    def _eval(self, p):
        ra, rb = self.ra, self.rb
        ba = self.b - self.a
        pa = p - self.a
        rba = rb - ra
        baba = ba @ ba
        papa = np.einsum("ij,ij->i", pa, pa)
        paba = (pa @ ba) / baba
        x = np.sqrt(np.maximum(papa - paba * paba * baba, 0.0))
        cax = np.maximum(0.0, x - np.where(paba < 0.5, ra, rb))
        cay = np.abs(paba - 0.5) - 0.5
        k = rba * rba + baba
        f = np.clip((rba * (x - ra) + paba * baba) / k, 0.0, 1.0)
        cbx = x - ra - f * rba
        cby = paba - f
        s = np.where((cbx < 0.0) & (cay < 0.0), -1.0, 1.0)
        return s * np.sqrt(np.minimum(cax * cax + cay * cay * baba,
                                      cbx * cbx + cby * cby * baba))

    def _bounds(self):
        r = max(self.ra, self.rb)
        return BBox(np.minimum(self.a, self.b) - r, np.maximum(self.a, self.b) + r)


# ---------------------------------------------------------------- TPMS
class _TPMS(Implicit):
    """Sheet TPMS: ``(|f(k p)| - t/2 * k) / k`` like the PicoGK Gyroid demo.

    The result is an approximate distance (exact on the mid surface).
    Unbounded: intersect it with a bounded shape (PicoGK ``IntersectImplicit``).
    ``solid=True`` gives the network (one side of the surface) instead of the
    sheet."""

    exact = False

    def __init__(self, period: float, wall: float, solid: bool = False,
                 level: float = 0.0, origin=(0, 0, 0)):
        self.k = 2 * np.pi / float(period)
        self.half_t = 0.5 * float(wall) * self.k
        self.solid = solid
        self.level = float(level)
        self.o = _vec(origin)

    def _field(self, x, y, z):
        raise NotImplementedError

    def _eval(self, p):
        q = (p - self.o) * self.k
        g = self._field(q[:, 0], q[:, 1], q[:, 2]) - self.level
        if self.solid:
            return g / self.k
        return (np.abs(g) - self.half_t) / self.k


class Gyroid(_TPMS):
    def _field(self, x, y, z):
        return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)


class SchwarzP(_TPMS):
    def _field(self, x, y, z):
        return np.cos(x) + np.cos(y) + np.cos(z)


class Diamond(_TPMS):
    def _field(self, x, y, z):
        sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
        cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
        return sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz


class CubicStrutLattice(Implicit):
    """The web modeler 'lattice' feature: cubic strut lattice with cell size
    ``cell`` and strut radius ``r`` clipped to a box."""

    def __init__(self, center, size, cell: float, radius: float):
        self.box = Box(center, size)
        self.cell, self.r = float(cell), float(radius)
        self.o = self.box.c - self.box.half

    def _eval(self, p):
        a = self.cell
        q = np.mod(p - self.o, a)
        q = np.minimum(q, a - q)
        d = np.minimum(np.minimum(np.hypot(q[:, 1], q[:, 2]), np.hypot(q[:, 0], q[:, 2])),
                       np.hypot(q[:, 0], q[:, 1])) - self.r
        return np.maximum(d, self.box._eval(p))

    exact = False

    def _bounds(self):
        return self.box._bounds()


# ---------------------------------------------------------------- user fns
class FunctionImplicit(Implicit):
    """Wrap a vectorised ``f(points [N,3]) -> [N]``."""

    def __init__(self, fn: Callable[[Array], Array], bounds=None, exact=False):
        self.fn = fn
        self._bb = None if bounds is None else BBox(*bounds)
        self.exact = exact

    def _eval(self, p):
        return np.asarray(self.fn(p), dtype=np.float64).reshape(len(p))

    def _bounds(self):
        return self._bb


# ==========================================================================
# CSG nodes
# ==========================================================================
class _NAry(Implicit):
    exact = False

    def __init__(self, *children: Implicit):
        flat = []
        for c in children:
            if isinstance(c, (list, tuple)):
                flat.extend(c)
            else:
                flat.append(c)
        if not flat:
            raise ValueError("need at least one operand")
        self.children = flat

    def _children(self):
        return self.children


class Union(_NAry):
    def _eval(self, p):
        d = self.children[0]._eval(p)
        for c in self.children[1:]:
            d = np.minimum(d, c._eval(p))
        return d

    def _eval_t(self, p, table):
        d, t = self.children[0]._eval_tagged(p, table)
        for c in self.children[1:]:
            d2, t2 = c._eval_tagged(p, table)
            m = d2 < d
            d = np.where(m, d2, d)
            t = np.where(m, t2, t)
        return d, t

    def _bounds(self):
        bs = [c.bounds for c in self.children]
        if any(b is None for b in bs):
            return None
        out = bs[0]
        for b in bs[1:]:
            out = out.union(b)
        return out


class Intersection(_NAry):
    def _eval(self, p):
        d = self.children[0]._eval(p)
        for c in self.children[1:]:
            d = np.maximum(d, c._eval(p))
        return d

    def _eval_t(self, p, table):
        d, t = self.children[0]._eval_tagged(p, table)
        for c in self.children[1:]:
            d2, t2 = c._eval_tagged(p, table)
            m = d2 > d
            d = np.where(m, d2, d)
            t = np.where(m, t2, t)
        return d, t

    def _bounds(self):
        out = None
        for c in self.children:
            b = c.bounds
            if b is not None:
                out = b if out is None else out.intersection(b)
        return out


class Difference(Implicit):
    """``a - b`` (PicoGK BoolSubtract)."""

    exact = False

    def __init__(self, a: Implicit, b: Implicit):
        self.a, self.b = a, b

    def _children(self):
        return (self.a, self.b)

    def _eval(self, p):
        return np.maximum(self.a._eval(p), -self.b._eval(p))

    def _eval_t(self, p, table):
        da, ta = self.a._eval_tagged(p, table)
        db, tb = self.b._eval_tagged(p, table)
        m = -db > da
        return np.where(m, -db, da), np.where(m, tb, ta)

    def _bounds(self):
        return self.a.bounds


class Complement(Implicit):
    def __init__(self, a: Implicit):
        self.a = a
        self.exact = a.exact

    def _children(self):
        return (self.a,)

    def _eval(self, p):
        return -self.a._eval(p)

    def _eval_t(self, p, table):
        d, t = self.a._eval_tagged(p, table)
        return -d, t


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


class SmoothUnion(Implicit):
    """Polynomial smooth minimum with blend radius ``k`` (mm)."""

    exact = False

    def __init__(self, a, b, k: float):
        self.a, self.b, self.k = a, b, float(k)

    def _children(self):
        return (self.a, self.b)

    def _eval(self, p):
        return _smin(self.a._eval(p), self.b._eval(p), self.k)

    def _eval_t(self, p, table):
        da, ta = self.a._eval_tagged(p, table)
        db, tb = self.b._eval_tagged(p, table)
        return _smin(da, db, self.k), np.where(db < da, tb, ta)

    def _bounds(self):
        a, b = self.a.bounds, self.b.bounds
        return None if a is None or b is None else a.union(b)


class SmoothDifference(SmoothUnion):
    def _eval(self, p):
        return -_smin(-self.a._eval(p), self.b._eval(p), self.k)

    def _eval_t(self, p, table):
        da, ta = self.a._eval_tagged(p, table)
        db, tb = self.b._eval_tagged(p, table)
        return -_smin(-da, db, self.k), np.where(-db > da, tb, ta)

    def _bounds(self):
        return self.a.bounds


class SmoothIntersection(SmoothUnion):
    def _eval(self, p):
        return -_smin(-self.a._eval(p), -self.b._eval(p), self.k)

    def _eval_t(self, p, table):
        da, ta = self.a._eval_tagged(p, table)
        db, tb = self.b._eval_tagged(p, table)
        return -_smin(-da, -db, self.k), np.where(db > da, tb, ta)

    def _bounds(self):
        a, b = self.a.bounds, self.b.bounds
        if a is None:
            return b
        return a if b is None else a.intersection(b)


class Offset(Implicit):
    """Move the surface outward (d > 0) or inward (d < 0)  (PicoGK Offset).

    On a non exact tree this is only approximate; use ``Voxels.offset`` for
    PicoGK-identical results."""

    def __init__(self, a: Implicit, d: float):
        self.a, self.d = a, float(d)
        self.exact = a.exact

    def _children(self):
        return (self.a,)

    def _eval(self, p):
        return self.a._eval(p) - self.d

    def _eval_t(self, p, table):
        d, t = self.a._eval_tagged(p, table)
        return d - self.d, t

    def _bounds(self):
        b = self.a.bounds
        return None if b is None else b.expanded(max(self.d, 0.0))


class Shell(Implicit):
    """PicoGK ``voxShell``: t < 0 keeps the outer size and hollows inward,
    t > 0 grows a wall of thickness t outside the object."""

    exact = False

    def __init__(self, a: Implicit, t: float):
        self.a, self.t = a, float(t)

    def _children(self):
        return (self.a,)

    def _eval(self, p):
        d = self.a._eval(p)
        if self.t < 0:
            return np.maximum(d, -(d - self.t))
        return np.maximum(d - self.t, -d)

    def _bounds(self):
        b = self.a.bounds
        return None if b is None else b.expanded(max(self.t, 0.0))


class Transform(Implicit):
    """Rigid transform + uniform scale (keeps distances exact)."""

    def __init__(self, a: Implicit, translate=None, rotate=None, scale=None,
                 matrix=None, center=(0, 0, 0)):
        self.a = a
        self.exact = a.exact
        c = _vec(center)
        M = np.eye(4)
        if matrix is not None:
            M = np.asarray(matrix, np.float64).reshape(4, 4)
        else:
            T1 = np.eye(4); T1[:3, 3] = -c
            T2 = np.eye(4); T2[:3, 3] = c
            R = np.eye(4)
            if rotate is not None:
                axis, ang = rotate
                R[:3, :3] = _rotation(axis, ang)
            S = np.eye(4)
            if scale is not None:
                S[:3, :3] *= float(scale)
            T = np.eye(4)
            if translate is not None:
                T[:3, 3] = _vec(translate)
            M = T @ T2 @ R @ S @ T1
        A = M[:3, :3]
        s = np.cbrt(abs(np.linalg.det(A)))
        if not np.allclose(A.T @ A, s * s * np.eye(3), atol=1e-9 * max(s * s, 1)):
            raise ValueError("only rigid transforms with uniform scale keep an SDF")
        self.M = M
        self.Minv = np.linalg.inv(M)
        self.s = s

    def _children(self):
        return (self.a,)

    def _local(self, p):
        return p @ self.Minv[:3, :3].T + self.Minv[:3, 3]

    def _eval(self, p):
        return self.a._eval(self._local(p)) * self.s

    def _eval_t(self, p, table):
        d, t = self.a._eval_tagged(self._local(p), table)
        return d * self.s, t

    def _bounds(self):
        b = self.a.bounds
        if b is None:
            return None
        corners = np.array([[x, y, z] for x in (b.min[0], b.max[0])
                            for y in (b.min[1], b.max[1]) for z in (b.min[2], b.max[2])])
        w = corners @ self.M[:3, :3].T + self.M[:3, 3]
        return BBox(w.min(0), w.max(0))


def _rotation(axis, ang):
    axis = _vec(axis)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(ang), np.sin(ang)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def union(*a):        return Union(*a)
def intersection(*a): return Intersection(*a)
def difference(a, b): return Difference(a, b)
