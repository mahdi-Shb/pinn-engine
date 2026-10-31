"""One adapter over every geometry source + samplers for PINN collocation.

``Geometry(src)`` accepts an ``Implicit``, ``Voxels``, ``Mesh`` or
``Lattice`` and offers: exact-as-possible ``sdf``, normals, boundary tags,
a surface mesh, and surface / volume / band samplers."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.stats import qmc

from .fields import GridSpec, mesh_from_grid, sample_implicit
from .implicit import BBox, Implicit, _as_points
from .lattice import Lattice, LatticeImplicit
from .mesh import Mesh, MeshImplicit
from .voxels import Voxels


class Geometry:
    """Uniform view of a geometry source.

    Parameters
    ----------
    src : Implicit | Voxels | Mesh | Lattice
    resolution : voxel size used to mesh implicits (default: extent / 160)
    face_tags, tag_names : optional per-face boundary labels for meshes
    tag_source : optional tagged ``Implicit`` that names the surfaces of a
        non-implicit source - e.g. the CSG tree a filleted ``Voxels`` field
        was made from (each point takes the tag of the closest owner)
    tag_order : fixed list of tag names (e.g. every tag a parametric design
        declares, even disabled ones) so tag channels/indices are identical
        for all variants; tags not present in the geometry are simply empty
    """

    def __init__(self, src, resolution: float | None = None,
                 face_tags: Optional[np.ndarray] = None,
                 tag_names: Optional[Sequence[str]] = None,
                 tag_source: Optional[Implicit] = None,
                 tag_order: Optional[Sequence[str]] = None):
        if isinstance(src, Geometry):
            if tag_source is None:
                tag_source = src.tag_source
            if tag_order is None and src._tag_map is not None:
                tag_order = src.tag_names
            src = src.src
        self.tag_source = tag_source
        self.src = src
        self._mesh: Mesh | None = None
        self._face_tags = None if face_tags is None else np.asarray(face_tags, np.int64)
        if isinstance(src, Lattice):
            self.imp: Implicit = LatticeImplicit(src)
        elif isinstance(src, Mesh):
            self.imp = MeshImplicit(src)
            self._mesh = src
        elif isinstance(src, Voxels):
            self.imp = src.as_implicit()
        elif isinstance(src, Implicit):
            self.imp = src
        else:
            raise TypeError(f"unsupported geometry source {type(src).__name__}")
        b = self.imp.bounds
        if b is None:
            raise ValueError("geometry must be bounded (intersect unbounded "
                             "implicits like Gyroid with a bounded shape)")
        self.bounds: BBox = b
        self.resolution = float(resolution) if resolution else float(np.max(b.size)) / 160.0
        if isinstance(src, Implicit):
            self._tag_names = src.tag_names()
        elif tag_source is not None:
            self._tag_names = tag_source.tag_names()
        elif tag_names is not None:
            self._tag_names = list(tag_names)
        else:
            self._tag_names = ["surface"]
        self._tag_map = None
        if tag_order is not None:
            order = list(tag_order)
            for nm in self._tag_names:          # undeclared (untagged leaves) go last
                if nm not in order:
                    order.append(nm)
            self._tag_map = np.array([order.index(nm) for nm in self._tag_names], np.int64)
            self._tag_names = order

    # ------------------------------------------------------------ fields
    @property
    def tag_names(self) -> list[str]:
        return list(self._tag_names)

    def sdf(self, p) -> np.ndarray:
        return self.imp.sdf(p)

    def normal(self, p) -> np.ndarray:
        """Outward unit normal / SDF gradient direction."""
        if isinstance(self.src, Mesh):
            return self.imp.normal(p, eps=1e-4 * float(np.max(self.bounds.size)))
        return self.imp.normal(p, eps=0.05 * self.resolution if isinstance(self.src, Voxels) else None)

    def tags(self, p) -> np.ndarray:
        """Boundary tag index for (surface) points."""
        p = _as_points(p)
        if isinstance(self.src, Implicit):
            return self._remap(self.src.sdf_tagged(p)[1].astype(np.int64))
        if self.tag_source is not None:
            return self._remap(self.tag_source.sdf_tagged(p)[1].astype(np.int64))
        if self._face_tags is not None and isinstance(self.src, Mesh):
            return self._face_tags[self.src.closest_points(p)[1]]
        return np.zeros(len(p), np.int64)

    def output(self, representation: str = "MESH_GRAPH", **kwargs):
        """NN input of this geometry (see ``geocore.outputs``)."""
        from .outputs import output
        return output(self, representation, **kwargs)

    def _remap(self, t):
        return t if self._tag_map is None else self._tag_map[t]

    # ------------------------------------------------------------ mesh
    def surface_mesh(self, resolution: float | None = None) -> Mesh:
        """Watertight surface mesh.  Implicits: manifold surface nets with
        vertices projected onto the exact surface."""
        h = float(resolution) if resolution else self.resolution
        if self._mesh is not None and (resolution is None or isinstance(self.src, Mesh)):
            return self._mesh
        if isinstance(self.src, Voxels):
            m = self.src.to_mesh()
        else:
            g = GridSpec.lattice(self.bounds, h, pad=2)
            phi = sample_implicit(self.imp, g)
            m = mesh_from_grid(phi, g.origin, g.spacing, project=self.imp)
        if resolution is None:
            self._mesh = m
        return m

    # ------------------------------------------------------------ samplers
    def sample_surface(self, n: int, rng=None, project: bool = True):
        """(points [n,3], normals [n,3], tags [n]) uniformly by area."""
        rng = np.random.default_rng(rng)
        m = self.surface_mesh()
        if isinstance(self.src, Mesh):
            p, nrm, f = m.sample_surface(n, rng, return_face=True)
            t = self._face_tags[f] if self._face_tags is not None else np.zeros(n, np.int64)
            return p, nrm, t
        p, _ = m.sample_surface(n, rng)
        if project:
            p = self.imp.project(p, iters=6)
        return p, self.normal(p), self.tags(p)

    def sample_volume(self, n: int, rng=None, method: str = "sobol",
                      inside: bool = True, bounds: BBox | None = None,
                      max_rounds: int = 64) -> np.ndarray:
        """Points strictly inside (``inside=True``) or anywhere in ``bounds``.

        method: 'sobol' (scrambled, low discrepancy), 'halton', 'random',
        'grid' (jittered stratified)."""
        rng = np.random.default_rng(rng)
        b = bounds or self.bounds
        if not inside:
            return _box_samples(b, n, method, rng)
        out = []
        have = 0
        frac = 0.5
        for _ in range(max_rounds):
            need = n - have
            m = int(min(max(need / max(frac, 1e-3) * 1.2, 1024), 1 << 22))
            p = _box_samples(b, m, method, rng)
            keep = p[self.sdf(p) < 0]
            frac = max(len(keep) / m, 1e-4)
            out.append(keep)
            have += len(keep)
            if have >= n:
                break
        p = np.concatenate(out)
        if len(p) < n:
            raise RuntimeError("could not sample enough interior points "
                               "(is the geometry empty?)")
        return p[rng.permutation(len(p))[:n]] if method in ("random", "grid") else p[:n]

    def sample_band(self, n: int, width: float, rng=None) -> np.ndarray:
        """Points with |sdf| < width (near-wall refinement for PINNs)."""
        rng = np.random.default_rng(rng)
        p, nrm, _ = self.sample_surface(n, rng)
        return p + nrm * rng.uniform(-width, width, (n, 1))


def collocation(geom, n_interior: int, n_boundary: int, *, rng=None,
                method: str = "sobol", band: tuple | None = None, normalizer=None) -> dict:
    """PINN collocation set (float32 arrays, ready for torch.from_numpy).

    Returns dict with ``x_interior`` [Ni,3], ``sdf_interior`` [Ni],
    ``x_boundary`` [Nb,3], ``normal_boundary`` [Nb,3], ``tag_boundary``
    [Nb] (int64 into ``tag_names``) and optionally ``x_band`` (``band=(n,
    width)``).  ``-sdf_interior`` is an exact distance function for hard
    Dirichlet constraints (u = g + d(x) * NN(x))."""
    g = geom if isinstance(geom, Geometry) else Geometry(geom)
    rng = np.random.default_rng(rng)
    xi = g.sample_volume(n_interior, rng, method)
    di = g.sdf(xi)
    xb, nb, tb = g.sample_surface(n_boundary, rng)
    out = {"x_interior": xi, "sdf_interior": di, "x_boundary": xb,
           "normal_boundary": nb, "tag_boundary": tb, "tag_names": np.array(g.tag_names)}
    if band is not None:
        xs = g.sample_band(band[0], band[1], rng)
        out["x_band"] = xs
        out["sdf_band"] = g.sdf(xs)
    if normalizer is not None:
        for k in ("x_interior", "x_boundary", "x_band"):
            if k in out:
                out[k] = normalizer.points(out[k])
        for k in ("sdf_interior", "sdf_band"):
            if k in out:
                out[k] = normalizer.lengths(out[k])
        out["normalization_center"] = normalizer.center
        out["normalization_scale"] = np.array(normalizer.scale)
    for k, v in out.items():
        if isinstance(v, np.ndarray) and v.dtype == np.float64:
            out[k] = v.astype(np.float32)
    return out


# --------------------------------------------------------------------------
def _box_samples(b: BBox, n: int, method: str, rng) -> np.ndarray:
    if method in ("sobol", "halton"):
        seed = int(rng.integers(0, 2 ** 31 - 1))
        eng = qmc.Sobol(3, scramble=True, seed=seed) if method == "sobol" else \
            qmc.Halton(3, scramble=True, seed=seed)
        u = eng.random_base2(max(int(np.ceil(np.log2(max(n, 2)))), 1))[:n]
    elif method == "random":
        u = rng.random((n, 3))
    elif method == "grid":
        k = max(1, int(np.ceil(n ** (1 / 3))))
        ijk = np.stack(np.meshgrid(*[np.arange(k)] * 3, indexing="ij"), -1).reshape(-1, 3)
        u = (ijk + rng.random(ijk.shape)) / k
    else:
        raise ValueError(f"unknown sampling method {method}")
    return b.min + u * b.size
