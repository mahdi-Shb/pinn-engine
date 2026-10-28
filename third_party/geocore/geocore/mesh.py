"""Triangle meshes (numpy), STL I/O, exact mesh signed distance."""
from __future__ import annotations

import os
import struct

import numpy as np

from . import _kernels as K
from .implicit import BBox, Implicit, _as_points


class Mesh:
    """Indexed triangle mesh.  ``vertices`` [V,3] float64, ``faces`` [F,3] int64.

    Faces are counter-clockwise seen from outside (PicoGK / STL convention)."""

    def __init__(self, vertices, faces):
        self.vertices = np.ascontiguousarray(vertices, np.float64).reshape(-1, 3)
        self.faces = np.ascontiguousarray(faces, np.int64).reshape(-1, 3)
        self._cache = {}

    # ------------------------------------------------------------ basics
    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def __repr__(self):
        return f"Mesh(V={self.n_vertices}, F={self.n_faces})"

    def copy(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.faces.copy())

    @property
    def bounds(self) -> BBox:
        return BBox(self.vertices.min(0), self.vertices.max(0))

    def triangles(self) -> np.ndarray:
        return self.vertices[self.faces]

    def face_normals(self, normalize=True) -> np.ndarray:
        t = self.triangles()
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        if normalize:
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
        return n

    def face_areas(self) -> np.ndarray:
        return 0.5 * np.linalg.norm(self.face_normals(normalize=False), axis=1)

    def area(self) -> float:
        return float(self.face_areas().sum())

    def volume(self) -> float:
        """Enclosed volume (divergence theorem; closed meshes)."""
        t = self.triangles()
        return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)

    def vertex_normals(self) -> np.ndarray:
        """Angle weighted vertex normals."""
        if "vn" in self._cache:
            return self._cache["vn"]
        vn = _angle_weighted(self.vertices, self.faces)
        self._cache["vn"] = vn
        return vn

    def vertex_areas(self) -> np.ndarray:
        """Lumped (barycentric) area per vertex: one third of adjacent faces."""
        a = np.zeros(self.n_vertices)
        np.add.at(a, self.faces.ravel(), np.repeat(self.face_areas() / 3.0, 3))
        return a

    def edges(self) -> np.ndarray:
        """Unique undirected edges [E,2] (i < j)."""
        if "edges" in self._cache:
            return self._cache["edges"]
        f = self.faces
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        e = np.sort(e, axis=1)
        e = np.unique(e, axis=0)
        self._cache["edges"] = e
        return e

    def is_watertight(self) -> bool:
        f = self.faces
        e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        _, c = np.unique(e, axis=0, return_counts=True)
        return bool(np.all(c == 2))

    def mean_curvature(self) -> np.ndarray:
        """Signed mean curvature per vertex (cotangent Laplacian), 1/mm.
        Positive on convex parts (sphere radius R -> 1/R)."""
        V, F = self.vertices, self.faces
        L = np.zeros_like(V)
        for i in range(3):
            i0, i1, i2 = F[:, i], F[:, (i + 1) % 3], F[:, (i + 2) % 3]
            u = V[i1] - V[i0]
            v = V[i2] - V[i0]
            cot = np.einsum("ij,ij->i", u, v) / np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-300)
            # angle at i0 is opposite edge (i1, i2)
            w = 0.5 * cot
            d = V[i2] - V[i1]
            np.add.at(L, i1, w[:, None] * d)
            np.add.at(L, i2, -w[:, None] * d)
        A = np.maximum(self.vertex_areas(), 1e-300)
        Hn = L / A[:, None]                       # = -2 H n  (mean curvature normal)
        H = -0.5 * np.einsum("ij,ij->i", Hn, self.vertex_normals())
        return H

    # ------------------------------------------------------------ editing
    def transformed(self, matrix) -> "Mesh":
        M = np.asarray(matrix, np.float64).reshape(4, 4)
        V = self.vertices @ M[:3, :3].T + M[:3, 3]
        F = self.faces if np.linalg.det(M[:3, :3]) > 0 else self.faces[:, ::-1]
        return Mesh(V, F.copy())

    def translated(self, t) -> "Mesh":
        return Mesh(self.vertices + np.asarray(t, np.float64), self.faces.copy())

    def scaled(self, s) -> "Mesh":
        return Mesh(self.vertices * np.asarray(s, np.float64), self.faces.copy())

    def welded(self, tol: float | None = None) -> "Mesh":
        """Merge duplicate vertices (STL soups) and drop degenerate faces."""
        if tol is None:
            ext = float(np.max(np.ptp(self.vertices, axis=0))) if self.n_vertices else 1.0
            tol = 1e-7 * max(ext, 1e-12)
        key = np.round(self.vertices / tol).astype(np.int64)
        _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        F = inv[self.faces]
        ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])
        return Mesh(self.vertices[idx], F[ok])

    @staticmethod
    def concatenate(meshes) -> "Mesh":
        V, F, off = [], [], 0
        for m in meshes:
            V.append(m.vertices)
            F.append(m.faces + off)
            off += m.n_vertices
        return Mesh(np.concatenate(V), np.concatenate(F))

    # ------------------------------------------------------------ sampling
    def sample_surface(self, n: int, rng=None, return_face=False):
        """Area weighted uniform samples: (points [n,3], normals [n,3])."""
        rng = np.random.default_rng(rng)
        a = self.face_areas()
        cdf = np.cumsum(a)
        f = np.searchsorted(cdf, rng.random(n) * cdf[-1], side="right")
        f = np.minimum(f, self.n_faces - 1)
        r1 = np.sqrt(rng.random(n))
        r2 = rng.random(n)
        t = self.triangles()[f]
        p = (1 - r1)[:, None] * t[:, 0] + (r1 * (1 - r2))[:, None] * t[:, 1] + (r1 * r2)[:, None] * t[:, 2]
        nrm = self.face_normals()[f]
        if return_face:
            return p, nrm, f
        return p, nrm

    # ------------------------------------------------------------ distance
    def _sdf_struct(self):
        if "sdf" not in self._cache:
            V = self.vertices
            F = self.faces
            fn = self.face_normals()
            vn = self.vertex_normals()
            en = _edge_pseudo_normals(F, fn)
            t = V[F]
            lo, hi = t.min(1), t.max(1)
            pad = 1e-9 * max(float(np.max(np.ptp(V, axis=0))), 1e-12)
            bvh = K.build_bvh(lo - pad, hi + pad)
            self._cache["sdf"] = (V, F, fn, vn, en, bvh)
        return self._cache["sdf"]

    def sdf(self, p, return_closest=False):
        """Exact signed distance (negative inside) for closed, oriented meshes.

        Sign uses angle-weighted pseudo normals (Baerentzen & Aanaes), which is
        robust at edges and vertices.  ``return_closest=True`` also returns the
        closest surface points [N,3] and closest face indices [N]."""
        return self._query(p, True, return_closest)

    def unsigned_distance(self, p):
        return self._query(p, False, False)

    def closest_points(self, p):
        """(closest surface points [N,3], face index [N], unsigned distance)."""
        d, cp, f = self._query(p, False, True)
        return cp, f, d

    def _query(self, p, signed, full):
        p = _as_points(p)
        V, F, fn, vn, en, bvh = self._sdf_struct()
        out = np.empty(len(p))
        cp = np.empty((len(p), 3))
        face = np.empty(len(p), np.int64)
        K.mesh_sdf_bvh(p, V, F, fn, vn, en, *bvh, out, cp, face, signed)
        return (out, cp, face) if full else out

    def as_implicit(self) -> "MeshImplicit":
        return MeshImplicit(self)

    def is_manifold(self) -> bool:
        """Every edge shared by exactly two faces (no holes, no fins)."""
        return self.manifold_report()["ok"]

    def manifold_report(self) -> dict:
        f = self.faces
        e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        _, c = np.unique(e, axis=0, return_counts=True)
        rep = {"open_edges": int((c == 1).sum()), "non_manifold_edges": int((c > 2).sum())}
        rep["ok"] = rep["open_edges"] == 0 and rep["non_manifold_edges"] == 0
        return rep

    def winding_number(self, p) -> np.ndarray:
        """Generalized winding number at points: ~1 inside, ~0 outside, and
        a smooth value across holes (robust inside test for open meshes)."""
        p = _as_points(p)
        out = np.empty(len(p))
        K.winding_numbers(p, self.vertices, self.faces, out)
        return out

    def winding_number_grid(self, g, max_work: float = 3e9) -> np.ndarray:
        """Winding numbers at every node of GridSpec ``g`` (computed on a
        coarser grid when the mesh is large, then trilinearly upsampled)."""
        from .fields import GridSpec, GridImplicit
        n_coarse = max(int(max_work / max(self.n_faces, 1)), 4096)
        f = max(1, int(np.ceil((g.n / n_coarse) ** (1 / 3))))
        if f == 1:
            return self.winding_number(g.points()).reshape(g.shape)
        cg = GridSpec(g.origin, g.spacing * f,
                      tuple(int(np.ceil((s - 1) / f)) + 1 for s in g.shape))
        wc = self.winding_number(cg.points()).reshape(cg.shape)
        out = np.empty(g.n)
        K.trilinear(np.ascontiguousarray(wc), cg.origin, cg.spacing, g.points(), 0, out)
        return out.reshape(g.shape)

    def remeshed(self, resolution: float | None = None) -> "Mesh":
        """Watertight, manifold copy: exact signed distance sampled on a grid
        (``resolution`` mm; default: the mean edge length of this mesh, so the
        detail level is kept) and re-meshed with geocore's manifold surface
        nets.  Removes non-manifold edges, tiny holes in the sign field and
        sub-voxel spikes."""
        from .fields import GridSpec, mesh_from_grid, redistance
        if resolution:
            h = float(resolution)
        else:
            e = self.edges()
            mean_edge = float(np.linalg.norm(self.vertices[e[:, 0]] - self.vertices[e[:, 1]], axis=1).mean())
            ext = float(np.max(self.bounds.size))
            h = min(max(mean_edge, ext / 400.0), ext / 60.0)
        g = GridSpec.lattice(self.bounds, h, pad=3)
        P = g.points()
        phi = self.sdf(P).reshape(g.shape)
        if self.manifold_report()["open_edges"]:
            # holes: the pseudo-normal sign leaks through them.  Away from the
            # surface use the generalized winding number (smooth across
            # holes, computed on a coarse grid), near the surface keep the
            # local sign.
            w = self.winding_number_grid(g)
            far = np.abs(phi) > 1.5 * h
            inside = w > 0.5
            phi = np.where(far, np.where(inside, -np.abs(phi), np.abs(phi)), phi)
            # across a hole there is no surface: make the w = 0.5 level the
            # zero crossing there
            near_hole = (np.abs(w - 0.5) < 0.35) & far
            phi = np.where(near_hole, (0.5 - w) * 4 * h, phi)
        phi = redistance(phi.astype(np.float32), g.spacing, g.origin)
        m = mesh_from_grid(phi, g.origin, g.spacing)
        # snap every new vertex onto the original surface (closest point):
        # keeps sharp edges sharp instead of voxel-rounded
        cp, _, d = self.closest_points(m.vertices)
        ok = d < 1.5 * h
        V = m.vertices.copy()
        V[ok] = cp[ok]
        # (triangles that collapse onto an edge keep their place: removing
        # them would open the surface again)
        return Mesh(V, m.faces)

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

    # ------------------------------------------------------------ I/O
    @staticmethod
    def load(path: str, weld: bool = True) -> "Mesh":
        ext = os.path.splitext(path)[1].lower()
        if ext == ".stl":
            m = _read_stl(path)
        elif ext == ".obj":
            m = _read_obj(path)
        else:
            try:
                import trimesh
            except ImportError as e:
                raise ValueError(f"unsupported mesh format {ext}") from e
            t = trimesh.load(path, force="mesh")
            m = Mesh(t.vertices, t.faces)
        return m.welded() if weld else m

    def save_stl(self, path: str):
        t = self.triangles().astype(np.float32)
        n = self.face_normals().astype(np.float32)
        rec = np.zeros(self.n_faces, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"] = n
        rec["v"] = t
        with open(path, "wb") as f:
            f.write(b"geocore binary STL".ljust(80, b" "))
            f.write(struct.pack("<I", self.n_faces))
            f.write(rec.tobytes())

    # ------------------------------------------------------------ factories
    @staticmethod
    def box(vmin=(-0.5, -0.5, -0.5), vmax=(0.5, 0.5, 0.5)) -> "Mesh":
        """PicoGK ``Utils.mshCreateCube`` equivalent."""
        a, b = np.asarray(vmin, float), np.asarray(vmax, float)
        V = np.array([[a[0], a[1], a[2]], [b[0], a[1], a[2]], [b[0], b[1], a[2]], [a[0], b[1], a[2]],
                      [a[0], a[1], b[2]], [b[0], a[1], b[2]], [b[0], b[1], b[2]], [a[0], b[1], b[2]]])
        F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                      [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
        return Mesh(V, F)


class MeshImplicit(Implicit):
    """Use a closed mesh as an exact implicit (CSG with primitives works)."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    def _eval(self, p):
        return self.mesh.sdf(p)

    def _bounds(self):
        return self.mesh.bounds


# ---------------------------------------------------------------- helpers
def _angle_weighted(V, F):
    t = V[F]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-300)
    vn = np.zeros_like(V)
    for i in range(3):
        u = t[:, (i + 1) % 3] - t[:, i]
        v = t[:, (i + 2) % 3] - t[:, i]
        cosang = np.einsum("ij,ij->i", u, v) / np.maximum(
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)
        ang = np.arccos(np.clip(cosang, -1, 1))
        np.add.at(vn, F[:, i], ang[:, None] * fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-300)
    return vn


def _edge_pseudo_normals(F, fn):
    """Per face edge (ab, bc, ca): sum of the two adjacent face normals."""
    nf = len(F)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])   # [3F,2]
    key = np.sort(e, axis=1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    acc = np.zeros((inv.max() + 1, 3))
    np.add.at(acc, inv, np.tile(fn, (3, 1)))
    en = acc[inv]
    en /= np.maximum(np.linalg.norm(en, axis=1, keepdims=True), 1e-300)
    return np.ascontiguousarray(en.reshape(3, nf, 3).transpose(1, 0, 2))


def _read_stl(path) -> Mesh:
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")],
                                count=n, offset=84)
            V = rec["v"].reshape(-1, 3).astype(np.float64)
            return Mesh(V, np.arange(3 * n).reshape(-1, 3))
    # ascii: 'vertex x y z'
    txt = data.decode("ascii", errors="ignore").split()
    pts = [(float(txt[i + 1]), float(txt[i + 2]), float(txt[i + 3]))
           for i, tok in enumerate(txt) if tok == "vertex"]
    V = np.asarray(pts, np.float64)
    return Mesh(V, np.arange(len(V)).reshape(-1, 3))


def _read_obj(path) -> Mesh:
    V, F = [], []
    with open(path) as f:
        for line in f:
            s = line.split()
            if not s:
                continue
            if s[0] == "v":
                V.append([float(x) for x in s[1:4]])
            elif s[0] == "f":
                idx = [int(x.split("/")[0]) for x in s[1:]]
                idx = [i - 1 if i > 0 else len(V) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    F.append([idx[0], idx[k], idx[k + 1]])
    return Mesh(np.asarray(V), np.asarray(F))
