"""PicoGK ``Voxels``: a signed distance voxel field.

Two interchangeable backends behind one API:

* ``"picogk"`` - the PicoGK C++ runtime (OpenVDB sparse narrow band, TBB
  parallel).  Same results as the user's C# PicoGK code.
* ``"dense"``  - numba/numpy dense grids on the same global voxel lattice
  (``index * voxel_size``).  Works everywhere (Linux, GPU boxes, ...).

Both export an *exact, full* signed distance field (``to_dense``) - PicoGK
only stores a ±1.5 voxel narrow band, geocore redistances it.
"""
from __future__ import annotations

import ctypes as C
import math
from typing import Iterable, Optional

import numpy as np

from . import _native as N
from .fields import GridImplicit, GridSpec, mesh_from_grid, redistance, sample_implicit
from .implicit import BBox, Box, Implicit, Sphere, Beam, _as_points
from .lattice import Lattice, LatticeImplicit
from .mesh import Mesh

_BIG = 1e6        # "far outside" value for dense padding
_PAD = 3          # voxels of margin around dense content


# ==========================================================================
# Library: voxel size + backend (PicoGK ``Library``)
# ==========================================================================
class Library:
    """Holds the voxel size (mm) and backend.  PicoGK requires one voxel size
    per library instance; use several libraries for several resolutions."""

    def __init__(self, voxel_size: float, backend: str = "auto"):
        self.voxel_size = float(voxel_size)
        if backend == "auto":
            backend = "picogk" if N.AVAILABLE else "dense"
        if backend not in ("picogk", "dense"):
            raise ValueError("backend must be 'auto', 'picogk' or 'dense'")
        if backend == "picogk":
            self.h = N.dll().Library_hCreateInstance(self.voxel_size)
            if not self.h:
                raise RuntimeError("PicoGK: could not create library instance")
        else:
            self.h = 0
        self.backend = backend

    def __repr__(self):
        return f"Library(voxel_size={self.voxel_size}, backend='{self.backend}')"

    def __del__(self):  # pragma: no cover - interpreter shutdown order
        try:
            if self.h and N.AVAILABLE:
                N.dll().Library_DestroyInstance(self.h)
                self.h = 0
        except Exception:
            pass


_default: Optional[Library] = None


def set_voxel_size(voxel_size: float, backend: str = "auto") -> Library:
    """Set the default library (like ``PicoGK.Library(fVoxelSizeMM)``)."""
    global _default
    _default = Library(voxel_size, backend)
    return _default


def default_library() -> Library:
    global _default
    if _default is None:
        _default = Library(0.5)
    return _default


# ==========================================================================
# Voxels
# ==========================================================================
class Voxels:
    """Signed distance voxel field (negative inside, mm).

    Construct with an implicit (bounded), a ``Mesh``, a ``Lattice``, or
    nothing (empty).  Operators: ``a + b`` (BoolAdd), ``a - b`` (BoolSubtract),
    ``a & b`` (BoolIntersect); in-place variants ``+=, -=, &=``."""

    def __init__(self, source=None, *, lib: Library | None = None, bounds=None):
        self.lib = lib or default_library()
        self._h = 0                      # PicoGK handle
        self._phi = None                 # dense: float32 [X,Y,Z]
        self._i0 = np.zeros(3, np.int64)  # dense: index of node (0,0,0)
        self._exact = True
        self._dense_cache = None         # (phi exact, GridSpec) export cache
        if self.lib.backend == "picogk":
            self._h = N.dll().Voxels_hCreate(self.lib.h)
        if source is None:
            return
        if isinstance(source, Implicit):
            self.render_implicit(source, bounds)
        elif isinstance(source, Mesh):
            self.render_mesh(source)
        elif isinstance(source, Lattice):
            self.render_lattice(source)
        else:
            raise TypeError(f"cannot build Voxels from {type(source).__name__}")

    # ---------------------------------------------------------- lifetime
    def __del__(self):  # pragma: no cover
        try:
            if self._h and N.AVAILABLE and self.lib.h:
                N.dll().Voxels_Destroy(self.lib.h, self._h)
                self._h = 0
        except Exception:
            pass

    @property
    def backend(self) -> str:
        return self.lib.backend

    @property
    def voxel_size(self) -> float:
        return self.lib.voxel_size

    def _changed(self):
        self._dense_cache = None
        self.__dict__.pop("_geometry_cache", None)

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

    def copy(self) -> "Voxels":
        out = Voxels(lib=self.lib)
        if self._h:
            N.dll().Voxels_Destroy(self.lib.h, out._h)
            out._h = N.dll().Voxels_hCreateCopy(self.lib.h, self._h)
        else:
            out._phi = None if self._phi is None else self._phi.copy()
            out._i0 = self._i0.copy()
            out._exact = self._exact
        return out

    duplicate = copy

    # ---------------------------------------------------------- factories
    @staticmethod
    def sphere(center, radius: float, lib: Library | None = None) -> "Voxels":
        lib = lib or default_library()
        if lib.backend == "picogk":
            v = Voxels(lib=lib)
            N.dll().Voxels_Destroy(lib.h, v._h)
            v._h = N.dll().Voxels_hCreateSphere(lib.h, N.v3(center), float(radius))
            return v
        return Voxels(Sphere(center, radius), lib=lib)

    @staticmethod
    def beam(a, ra: float, b, rb: float, lib: Library | None = None) -> "Voxels":
        """``Voxels.voxLatticeBeam``: round cone with spherical caps."""
        lib = lib or default_library()
        if lib.backend == "picogk":
            v = Voxels(lib=lib)
            N.dll().Voxels_Destroy(lib.h, v._h)
            v._h = N.dll().Voxels_hCreateCapsule(lib.h, N.v3(a), N.v3(b), float(ra), float(rb))
            return v
        return Voxels(Beam(a, b, ra, rb), lib=lib)

    @staticmethod
    def box(vmin, vmax, lib: Library | None = None) -> "Voxels":
        """Box (PicoGK: voxels of ``Utils.mshCreateCube``)."""
        lib = lib or default_library()
        if lib.backend == "picogk":
            return Voxels(Mesh.box(vmin, vmax), lib=lib)
        return Voxels(Box.from_bounds(vmin, vmax), lib=lib)

    @staticmethod
    def mesh_shell(mesh: Mesh, thickness: float, lib: Library | None = None) -> "Voxels":
        """``voxMeshShell``: hollow skin of radius ``thickness`` around a mesh."""
        lib = lib or default_library()
        if lib.backend == "picogk":
            m = N.mesh_push(lib.h, mesh.vertices, mesh.faces)
            v = Voxels(lib=lib)
            N.dll().Voxels_Destroy(lib.h, v._h)
            v._h = N.dll().Voxels_hCreateMeshShell(lib.h, m, float(thickness))
            N.dll().Mesh_Destroy(lib.h, m)
            return v
        from .implicit import FunctionImplicit
        f = FunctionImplicit(lambda p: mesh.unsigned_distance(p) - thickness,
                             bounds=(mesh.bounds.min - thickness, mesh.bounds.max + thickness))
        return Voxels(f, lib=lib)

    @staticmethod
    def combine_all(items: Iterable["Voxels"]) -> "Voxels":
        items = list(items)
        out = items[0].copy()
        for v in items[1:]:
            out.bool_add(v)
        return out

    # ---------------------------------------------------------- rendering
    def render_implicit(self, imp: Implicit, bounds=None):
        """Render a (bounded) implicit and union it into this field."""
        bb = bounds if bounds is not None else imp.bounds
        if bb is None:
            raise ValueError("unbounded implicit: give bounds= or intersect it "
                             "with a bounded shape (intersect_implicit)")
        if not isinstance(bb, BBox):
            bb = BBox(*bb)
        h = self.voxel_size
        if self.backend == "picogk":
            other = Voxels(lib=self.lib)
            g = GridSpec.lattice(bb, h, pad=2)
            phi = sample_implicit(imp, g)
            addr, keep = N.grid_callback(phi, g.origin, h, outside=float(3 * h))
            lo = g.origin
            hi = g.origin + (np.asarray(g.shape) - 1) * h
            box = N.BBox3(N.v3(lo), N.v3(hi))
            N.dll().Voxels_RenderImplicit(self.lib.h, other._h, C.byref(box), addr)
            del keep
            self.bool_add(other)
        else:
            g = GridSpec.lattice(bb, h, pad=_PAD)
            phi = sample_implicit(imp, g)
            self._dense_union(phi, _lattice_index(g, h), exact=imp.exact)
        self._changed()
        return self

    def render_mesh(self, mesh: Mesh):
        """Voxelize a closed mesh and union it in."""
        if self.backend == "picogk":
            m = N.mesh_push(self.lib.h, mesh.vertices, mesh.faces)
            N.dll().Voxels_RenderMesh(self.lib.h, self._h, m)
            N.dll().Mesh_Destroy(self.lib.h, m)
        else:
            h = self.voxel_size
            g = GridSpec.lattice(mesh.bounds, h, pad=_PAD)
            phi = mesh.sdf(g.points()).astype(np.float32).reshape(g.shape)
            self._dense_union(phi, _lattice_index(g, h), exact=True)
        self._changed()
        return self

    def render_lattice(self, lat: Lattice):
        if self.backend == "picogk":
            hl = N.lattice_push(self.lib.h, lat.beams, lat.round_caps)
            N.dll().Voxels_RenderLattice(self.lib.h, self._h, hl)
            N.dll().Lattice_Destroy(self.lib.h, hl)
        else:
            imp = LatticeImplicit(lat)
            h = self.voxel_size
            g = GridSpec.lattice(imp.bounds, h, pad=_PAD)
            phi = sample_implicit(imp, g)
            self._dense_union(phi, _lattice_index(g, h), exact=False)
        self._changed()
        return self

    def intersect_implicit(self, imp: Implicit):
        """Keep only the part inside ``imp`` (PicoGK ``IntersectImplicit``),
        e.g. a gyroid infill.  Works with unbounded implicits."""
        if self.is_empty():
            return self
        h = self.voxel_size
        if self.backend == "picogk":
            # PicoGK 26.2's Voxels_IntersectImplicit throws for thin features
            # (e.g. a 0.8 mm gyroid at 0.3 mm voxels), so render the implicit
            # over this field's extent and BoolIntersect instead (same result)
            i0, size = self._pg_dims()
            lo = i0 * h
            hi = (i0 + size - 1) * h
            mask = Voxels(lib=self.lib)
            mask.render_implicit(imp, BBox(lo, hi))
            N.dll().Voxels_BoolIntersect(self.lib.h, self._h, mask._h)
        else:
            g = self._grid()
            self._phi = np.maximum(self._phi, sample_implicit(imp, g))
            self._exact = False
        self._changed()
        return self

    # ---------------------------------------------------------- booleans
    def bool_add(self, other: "Voxels"):
        self._check(other)
        if self.backend == "picogk":
            N.dll().Voxels_BoolAdd(self.lib.h, self._h, other._h)
        elif other._phi is not None:
            self._dense_union(other._phi, other._i0, exact=other._exact)
        self._changed()
        return self

    def bool_subtract(self, other: "Voxels"):
        self._check(other)
        if self.backend == "picogk":
            N.dll().Voxels_BoolSubtract(self.lib.h, self._h, other._h)
        elif self._phi is not None and other._phi is not None:
            b = other._resampled_to(self._i0, self._phi.shape, fill=_BIG)
            self._phi = np.maximum(self._phi, -b)
            self._exact = False
        self._changed()
        return self

    def bool_intersect(self, other: "Voxels"):
        self._check(other)
        if self.backend == "picogk":
            N.dll().Voxels_BoolIntersect(self.lib.h, self._h, other._h)
        elif self._phi is not None:
            if other._phi is None:
                self._phi = None
            else:
                b = other._resampled_to(self._i0, self._phi.shape, fill=_BIG)
                self._phi = np.maximum(self._phi, b)
                self._exact = False
        self._changed()
        return self

    def __add__(self, o):  return self.copy().bool_add(o)
    def __sub__(self, o):  return self.copy().bool_subtract(o)
    def __and__(self, o):  return self.copy().bool_intersect(o)
    def __iadd__(self, o): return self.bool_add(o)
    def __isub__(self, o): return self.bool_subtract(o)
    def __iand__(self, o): return self.bool_intersect(o)

    # PicoGK spellings
    BoolAdd = bool_add
    BoolSubtract = bool_subtract
    BoolIntersect = bool_intersect

    # ---------------------------------------------------------- offsets
    def offset(self, dist: float):
        """Move the surface outward (+) or inward (-) by ``dist`` mm."""
        if self.backend == "picogk":
            N.dll().Voxels_Offset(self.lib.h, self._h, float(dist))
        elif self._phi is not None:
            self._dense_grow(max(dist, 0.0))
            self._make_exact()
            self._phi -= np.float32(dist)
            # phi - d is only a bound of the offset shape's distance (rays
            # converge in concave regions): that is what makes fillets work
            self._exact = False
            self._dense_crop()
        self._changed()
        return self

    def double_offset(self, d1: float, d2: float):
        if self.backend == "picogk":
            N.dll().Voxels_DoubleOffset(self.lib.h, self._h, float(d1), float(d2))
            self._changed()
            return self
        return self.offset(d1).offset(d2)

    def triple_offset(self, d: float):
        """Inward by d, outward by 2d, inward by d: removes detail < d."""
        if self.backend == "picogk":
            N.dll().Voxels_TripleOffset(self.lib.h, self._h, float(d))
            self._changed()
            return self
        return self.offset(-d).offset(2 * d).offset(-d)

    smoothen = triple_offset

    def over_offset(self, first: float, final: float = 0.0):
        return self.double_offset(first, final - first)

    def fillet(self, rounding: float):
        """Round concave edges with radius ``rounding`` (OverOffset(r, 0))."""
        return self.over_offset(rounding, 0.0)

    def shell(self, offset: float) -> "Voxels":
        """``voxShell``: negative -> hollow inward (outer size kept);
        positive -> wall of that thickness grown outside.  Returns new field."""
        if offset < 0:
            return self - self.copy().offset(offset)
        return self.copy().offset(offset) - self

    def shell2(self, neg_offset: float, pos_offset: float, smooth_inner: float = 0.0) -> "Voxels":
        lo, hi = sorted((neg_offset, pos_offset))
        inner = self.copy().offset(lo)
        if smooth_inner > 0:
            inner.triple_offset(smooth_inner)
        return self.copy().offset(hi) - inner

    def trim(self, vmin, vmax):
        """Remove everything outside the box (PicoGK ``Trim``)."""
        return self.bool_intersect(Voxels.box(vmin, vmax, lib=self.lib))

    Offset = offset
    DoubleOffset = double_offset
    TripleOffset = triple_offset
    Smoothen = smoothen
    OverOffset = over_offset
    Fillet = fillet
    Trim = trim

    # ---------------------------------------------------------- queries
    def is_empty(self) -> bool:
        if self.backend == "picogk":
            return bool(N.dll().Voxels_bIsEmpty(self.lib.h, self._h))
        return self._phi is None or not bool((self._phi < 0).any())

    def to_mesh(self) -> Mesh:
        """Surface mesh (PicoGK: OpenVDB volume-to-mesh; dense: surface nets
        projected onto the zero level set)."""
        if self.is_empty():
            return Mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int64))
        if self.backend == "picogk":
            m = N.dll().Mesh_hCreateFromVoxels(self.lib.h, self._h)
            V, T = N.mesh_pull(self.lib.h, m)
            N.dll().Mesh_Destroy(self.lib.h, m)
            return Mesh(V.astype(np.float64), T.astype(np.int64))
        g = self._grid()
        return mesh_from_grid(self._phi, g.origin, g.spacing)

    mesh = to_mesh

    def calculate_properties(self):
        """(volume mm^3, BBox) - like PicoGK ``CalculateProperties``.

        Volume is the exact enclosed volume of the surface mesh (divergence
        theorem).  PicoGK's own recipe (mesh -> voxels -> volume) fills
        internal cavities, so it is not used."""
        m = self.to_mesh()
        if m.n_faces == 0:
            return 0.0, BBox(np.zeros(3), np.zeros(3))
        return m.volume(), m.bounds

    def volume(self) -> float:
        return self.calculate_properties()[0]

    def bounding_box(self) -> BBox:
        return self.calculate_properties()[1]

    def is_inside(self, p) -> np.ndarray:
        return self.sdf(p) <= 0.0

    def sdf(self, p) -> np.ndarray:
        """Signed distance at arbitrary points (trilinear on the exact grid)."""
        return self.as_implicit().sdf(p)

    def surface_normal(self, p) -> np.ndarray:
        return self.as_implicit().normal(p, eps=0.25 * self.voxel_size)

    def closest_point_on_surface(self, p) -> np.ndarray:
        imp = self.as_implicit()
        p = _as_points(p)
        q = p - imp.sdf(p)[:, None] * imp.normal(p, eps=0.25 * self.voxel_size)
        return imp.project(q, iters=4)

    def raycast_to_surface(self, origins, directions, max_dist: float | None = None):
        """Sphere tracing; returns (hit points [N,3], hit mask [N])."""
        imp = self.as_implicit()
        o = _as_points(origins)
        d = _as_points(directions)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        g = self._export_grid()
        if max_dist is None:
            max_dist = float(np.linalg.norm(g.bounds.size)) * 2
        t = np.zeros(len(o))
        hit = np.zeros(len(o), bool)
        alive = np.ones(len(o), bool)
        eps = 0.05 * self.voxel_size
        for _ in range(512):
            if not alive.any():
                break
            q = o[alive] + t[alive, None] * d[alive]
            s = imp.sdf(q)
            idx = np.nonzero(alive)[0]
            close = np.abs(s) < eps
            hit[idx[close]] = True
            alive[idx[close]] = False
            t[idx[~close]] += np.maximum(np.abs(s[~close]), eps)
            alive[idx[t[idx] > max_dist]] = False
        return o + t[:, None] * d, hit

    # ---------------------------------------------------------- export
    def to_dense(self, grid: GridSpec | None = None, *, exact: bool = True):
        """(phi, GridSpec): full signed distance field, float32 [X,Y,Z], mm.

        Without ``grid`` the field is returned on its own voxel lattice
        (exact values, no interpolation).  With ``grid`` the voxel lattice is
        first extended to cover that grid (so distances stay exact far from
        the part) and then resampled trilinearly onto it."""
        if grid is None:
            return self._exact_dense() if exact else self._raw_dense()
        phi, g = self._raw_dense()
        h = self.voxel_size
        need = grid.bounds.expanded(2 * h)
        lo = np.floor(need.min / h).astype(np.int64)
        hi = np.ceil(need.max / h).astype(np.int64) + 1
        i0 = np.round(g.origin / h).astype(np.int64)
        pad_lo = np.maximum(i0 - lo, 0)
        pad_hi = np.maximum(hi - (i0 + np.asarray(phi.shape)), 0)
        if np.any(pad_lo) or np.any(pad_hi):
            fill = _BIG if self.backend == "dense" else float(phi.max())
            phi = np.pad(phi, list(zip(pad_lo, pad_hi)), constant_values=fill)
            g = GridSpec((i0 - pad_lo) * h, g.spacing, phi.shape)
            phi = redistance(phi, g.spacing, g.origin)
        else:
            phi, g = self._exact_dense()
        imp = GridImplicit(phi, g.origin, g.spacing)
        return imp.sdf(grid.points()).astype(np.float32).reshape(grid.shape), grid

    def as_implicit(self) -> GridImplicit:
        phi, g = self._exact_dense()
        return GridImplicit(phi, g.origin, g.spacing)

    def save_stl(self, path: str):
        self.to_mesh().save_stl(path)

    def __repr__(self):
        return f"Voxels(backend='{self.backend}', voxel_size={self.voxel_size})"

    # ======================================================== internals
    def _check(self, other):
        if other.lib.voxel_size != self.lib.voxel_size or other.backend != self.backend:
            raise ValueError("voxel fields must share voxel size and backend")
        if self.backend == "picogk" and other.lib.h != self.lib.h:
            raise ValueError("PicoGK voxel fields must come from the same Library")

    # -- picogk helpers
    def _pg_dims(self):
        ints = [C.c_int() for _ in range(6)]
        N.dll().Voxels_GetVoxelDimensions(self.lib.h, self._h, *[C.byref(x) for x in ints])
        v = np.array([x.value for x in ints], np.int64)
        return v[:3], v[3:]

    def _raw_dense(self):
        """Narrow band field on the voxel lattice (picogk: ±1.5 voxels)."""
        h = self.voxel_size
        if self.backend == "picogk":
            i0, size = self._pg_dims()
            if np.any(size <= 0):
                raise ValueError("empty voxel field")
            nx, ny, nz = (int(s) for s in size)
            phi = np.empty((nx, ny, nz), np.float32)
            sl = np.empty((ny, nx), np.float32)
            bg = C.c_float()
            for k in range(nz):
                N.dll().Voxels_GetZSlice(self.lib.h, self._h, k, sl.ctypes.data, C.byref(bg))
                phi[:, :, k] = sl.T
            phi *= np.float32(h)              # voxel units -> mm
            # pad so the outside band is complete everywhere
            pad = _PAD
            band = float(bg.value) * h
            phi = np.pad(phi, pad, constant_values=band)
            i0 = i0 - pad
            return phi, GridSpec(i0 * h, np.full(3, h), phi.shape)
        if self._phi is None:
            raise ValueError("empty voxel field")
        return self._phi, self._grid()

    def _exact_dense(self):
        if self._dense_cache is None:
            if self.backend == "dense":
                self._make_exact()
                self._dense_cache = (self._phi, self._grid())
            else:
                phi, g = self._raw_dense()
                self._dense_cache = (redistance(phi, g.spacing, g.origin), g)
        return self._dense_cache

    def _export_grid(self) -> GridSpec:
        return self._exact_dense()[1]

    # -- dense helpers
    def _grid(self) -> GridSpec:
        h = self.voxel_size
        return GridSpec(self._i0 * h, np.full(3, h), self._phi.shape)

    def _make_exact(self):
        if not self._exact and self._phi is not None:
            g = self._grid()
            self._phi = redistance(self._phi, g.spacing, g.origin)
            self._exact = True

    def _resampled_to(self, i0, shape, fill):
        """This field's values on another lattice window (same voxel size)."""
        out = np.full(shape, fill, np.float32)
        if self._phi is None:
            return out
        a0 = np.maximum(i0, self._i0)
        a1 = np.minimum(i0 + np.asarray(shape), self._i0 + np.asarray(self._phi.shape))
        if np.any(a1 <= a0):
            return out
        dst = tuple(slice(int(a - b), int(c - b)) for a, c, b in zip(a0, a1, i0))
        src = tuple(slice(int(a - b), int(c - b)) for a, c, b in zip(a0, a1, self._i0))
        out[dst] = self._phi[src]
        return out

    def _dense_union(self, phi, i0, exact):
        phi = np.asarray(phi, np.float32)
        i0 = np.asarray(i0, np.int64)
        if self._phi is None:
            self._phi, self._i0, self._exact = phi, i0, exact
            return
        lo = np.minimum(self._i0, i0)
        hi = np.maximum(self._i0 + self._phi.shape, i0 + np.asarray(phi.shape))
        shape = tuple(int(v) for v in hi - lo)
        a = self._resampled_to(lo, shape, _BIG)
        tmp = Voxels.__new__(Voxels)
        tmp._phi, tmp._i0 = phi, i0
        b = Voxels._resampled_to(tmp, lo, shape, _BIG)
        self._phi = np.minimum(a, b)
        self._i0 = lo
        self._exact = False

    def _dense_grow(self, d: float):
        """Make sure there are >= d + _PAD mm of margin around the surface."""
        h = self.voxel_size
        need = int(math.ceil(d / h)) + _PAD
        inside = self._phi < 0
        if not inside.any():
            return
        idx = np.argwhere(inside)
        lo = idx.min(0) - need
        hi = idx.max(0) + need + 1
        pad_lo = np.maximum(-lo, 0)
        pad_hi = np.maximum(hi - np.asarray(self._phi.shape), 0)
        if np.any(pad_lo) or np.any(pad_hi):
            if not self._exact:
                self._make_exact()
            self._phi = np.pad(self._phi, list(zip(pad_lo, pad_hi)), constant_values=_BIG)
            self._i0 = self._i0 - pad_lo
            self._exact = False

    def _dense_crop(self):
        inside = self._phi < 0
        if not inside.any():
            self._phi = None
            return
        idx = np.argwhere(inside)
        lo = np.maximum(idx.min(0) - _PAD - 1, 0)
        hi = np.minimum(idx.max(0) + _PAD + 2, self._phi.shape)
        self._phi = np.ascontiguousarray(self._phi[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        self._i0 = self._i0 + lo


def _lattice_index(g: GridSpec, h: float) -> np.ndarray:
    return np.round(g.origin / h).astype(np.int64)
