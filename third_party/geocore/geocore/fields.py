"""Dense grids: sampling, exact redistancing, meshing.

Arrays are stored ``[X, Y, Z]`` internally (x slowest) and transposed to the
NN layout ``[Z, Y, X]`` (``D, H, W``) only on export.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numba as nb
import numpy as np
from scipy import ndimage

from . import _kernels as K
from .implicit import BBox, Implicit, _as_points

_JIT = dict(cache=True)


@dataclass
class GridSpec:
    """Node-centred grid: node (i,j,k) sits at ``origin + (i,j,k) * spacing``."""

    origin: np.ndarray     # [3] x,y,z of node (0,0,0)
    spacing: np.ndarray    # [3]
    shape: tuple           # (nx, ny, nz)

    @staticmethod
    def from_bounds(bounds: BBox, *, spacing=None, shape=None) -> "GridSpec":
        """Grid whose cells tile ``bounds`` exactly; nodes are cell centres."""
        lo, hi = bounds.min, bounds.max
        ext = hi - lo
        if shape is not None:
            shape = tuple(int(s) for s in np.broadcast_to(shape, (3,)))
            sp = ext / np.asarray(shape, float)
        else:
            if spacing is None:
                raise ValueError("give spacing or shape")
            sp = np.broadcast_to(np.asarray(spacing, float), (3,)).copy()
            shape = tuple(int(max(1, math.ceil(e / s - 1e-9))) for e, s in zip(ext, sp))
            # keep the requested spacing, centre the grid on the bounds
            lo = 0.5 * (lo + hi) - 0.5 * np.asarray(shape) * sp
        origin = lo + 0.5 * sp
        return GridSpec(np.asarray(origin, float), np.asarray(sp, float), shape)

    @staticmethod
    def lattice(bounds: BBox, h: float, pad: int = 3) -> "GridSpec":
        """PicoGK style grid: nodes on the global lattice ``index * h``."""
        i0 = np.floor(bounds.min / h).astype(np.int64) - pad
        i1 = np.ceil(bounds.max / h).astype(np.int64) + pad
        return GridSpec(i0 * float(h), np.full(3, float(h)), tuple(int(v) for v in (i1 - i0 + 1)))

    @property
    def bounds(self) -> BBox:
        """Box covered by the cells (node +- spacing/2)."""
        lo = self.origin - 0.5 * self.spacing
        return BBox(lo, lo + np.asarray(self.shape) * self.spacing)

    @property
    def n(self) -> int:
        return int(np.prod(self.shape))

    def axes(self):
        return [self.origin[a] + np.arange(self.shape[a]) * self.spacing[a] for a in range(3)]

    def points(self) -> np.ndarray:
        """All node coordinates [nx*ny*nz, 3] in [X,Y,Z] (C) order."""
        x, y, z = self.axes()
        P = np.empty(self.shape + (3,))
        P[..., 0] = x[:, None, None]
        P[..., 1] = y[None, :, None]
        P[..., 2] = z[None, None, :]
        return P.reshape(-1, 3)


def sample_implicit(imp: Implicit, grid: GridSpec, tagged=False):
    """Evaluate an implicit at every grid node, in slabs of ~2M points
    (compiled multi-threaded evaluation, bounded memory)."""
    nx, ny, nz = grid.shape
    x, y, z = grid.axes()
    phi = np.empty(grid.shape, np.float32)
    tags = np.empty(grid.shape, np.int16) if tagged else None
    YZ = np.empty((ny * nz, 3))
    YZ[:, 1] = np.repeat(y, nz)
    YZ[:, 2] = np.tile(z, ny)
    step = max(1, (1 << 21) // max(ny * nz, 1))
    for i0 in range(0, nx, step):
        i1 = min(nx, i0 + step)
        P = np.repeat(YZ[None], i1 - i0, axis=0)
        P[:, :, 0] = x[i0:i1, None]
        P = P.reshape(-1, 3)
        if tagged:
            d, t = imp.sdf_tagged(P)
            tags[i0:i1] = t.reshape(i1 - i0, ny, nz)
        else:
            d = imp.sdf(P)
        phi[i0:i1] = d.reshape(i1 - i0, ny, nz)
    return (phi, tags) if tagged else phi


# --------------------------------------------------------------------------
# closest points of interface nodes: Newton on the trilinear field
# --------------------------------------------------------------------------
@nb.njit(inline="always", **_JIT)
def _tri(phi, ox, oy, oz, sx, sy, sz, x, y, z):
    nx, ny, nz = phi.shape
    fx = min(max((x - ox) / sx, 0.0), nx - 1.000001)
    fy = min(max((y - oy) / sy, 0.0), ny - 1.000001)
    fz = min(max((z - oz) / sz, 0.0), nz - 1.000001)
    i = int(fx); j = int(fy); k = int(fz)
    tx = fx - i; ty = fy - j; tz = fz - k
    i1 = min(i + 1, nx - 1); j1 = min(j + 1, ny - 1); k1 = min(k + 1, nz - 1)
    c00 = phi[i, j, k] * (1 - tx) + phi[i1, j, k] * tx
    c10 = phi[i, j1, k] * (1 - tx) + phi[i1, j1, k] * tx
    c01 = phi[i, j, k1] * (1 - tx) + phi[i1, j, k1] * tx
    c11 = phi[i, j1, k1] * (1 - tx) + phi[i1, j1, k1] * tx
    return (c00 * (1 - ty) + c10 * ty) * (1 - tz) + (c01 * (1 - ty) + c11 * ty) * tz


@nb.njit(parallel=True, **_JIT)
def _interface_closest(phi, origin, sp, ii, jj, kk, iters, cp, ok):
    """Foot points of band nodes on the trilinear zero level set.  ``ok``
    marks the ones that really reached it (nodes in a clamped / flat part of
    a narrow band field, e.g. PicoGK's +-1.5 voxel band, do not)."""
    ox, oy, oz = origin[0], origin[1], origin[2]
    sx, sy, sz = sp[0], sp[1], sp[2]
    nx, ny, nz = phi.shape
    hmin = min(sx, min(sy, sz))
    for t in nb.prange(len(ii)):
        i, j, k = ii[t], jj[t], kk[t]
        x = ox + i * sx; y = oy + j * sy; z = oz + k * sz
        # start: central/one-sided differences at the node
        ip = min(i + 1, nx - 1); im = max(i - 1, 0)
        jp = min(j + 1, ny - 1); jm = max(j - 1, 0)
        kp = min(k + 1, nz - 1); km = max(k - 1, 0)
        gx = (phi[ip, j, k] - phi[im, j, k]) / ((ip - im) * sx) if ip != im else 0.0
        gy = (phi[i, jp, k] - phi[i, jm, k]) / ((jp - jm) * sy) if jp != jm else 0.0
        gz = (phi[i, j, kp] - phi[i, j, km]) / ((kp - km) * sz) if kp != km else 0.0
        g2 = gx * gx + gy * gy + gz * gz
        d = phi[i, j, k]
        if g2 > 1e-24:
            x -= d * gx / g2; y -= d * gy / g2; z -= d * gz / g2
        ex = 0.5 * sx; ey = 0.5 * sy; ez = 0.5 * sz
        x0 = ox + i * sx; y0 = oy + j * sy; z0 = oz + k * sz
        for it in range(max(iters, 8)):
            d = _tri(phi, ox, oy, oz, sx, sy, sz, x, y, z)
            if it >= iters and abs(d) < 1e-3 * hmin:
                break
            gx = (_tri(phi, ox, oy, oz, sx, sy, sz, x + ex, y, z)
                  - _tri(phi, ox, oy, oz, sx, sy, sz, x - ex, y, z)) / (2 * ex)
            gy = (_tri(phi, ox, oy, oz, sx, sy, sz, x, y + ey, z)
                  - _tri(phi, ox, oy, oz, sx, sy, sz, x, y - ey, z)) / (2 * ey)
            gz = (_tri(phi, ox, oy, oz, sx, sy, sz, x, y, z + ez)
                  - _tri(phi, ox, oy, oz, sx, sy, sz, x, y, z - ez)) / (2 * ez)
            g2 = gx * gx + gy * gy + gz * gz
            if g2 < 1e-24:
                break
            x -= d * gx / g2; y -= d * gy / g2; z -= d * gz / g2
            # stay local: the true foot point is within ~1.5 cells
            dx = x - x0; dy = y - y0; dz = z - z0
            lim = 4.0 * max(sx, max(sy, sz))
            L = math.sqrt(dx * dx + dy * dy + dz * dz)
            if L > lim:
                x = x0 + dx * lim / L; y = y0 + dy * lim / L; z = z0 + dz * lim / L
        cp[t, 0] = x; cp[t, 1] = y; cp[t, 2] = z
        ok[t] = abs(_tri(phi, ox, oy, oz, sx, sy, sz, x, y, z)) < 1e-2 * hmin


@nb.njit(parallel=True, **_JIT)
def _snap_refine(phi, r, origin, sp, iters, out):
    """Second pass: from each node step along grad(r) to the surface, snap
    onto the zero level of ``phi`` (Newton on trilinear) and keep the shorter
    of the two upper bounds |x - y| and |r|."""
    ox, oy, oz = origin[0], origin[1], origin[2]
    sx, sy, sz = sp[0], sp[1], sp[2]
    nx, ny, nz = r.shape
    hmin = min(sx, min(sy, sz))
    for i in nb.prange(nx):
        for j in range(ny):
            for k in range(nz):
                d0 = r[i, j, k]
                out[i, j, k] = d0
                if abs(d0) < 1e-12:
                    continue
                ip = min(i + 1, nx - 1); im = max(i - 1, 0)
                jp = min(j + 1, ny - 1); jm = max(j - 1, 0)
                kp = min(k + 1, nz - 1); km = max(k - 1, 0)
                gx = (r[ip, j, k] - r[im, j, k]) / ((ip - im) * sx) if ip != im else 0.0
                gy = (r[i, jp, k] - r[i, jm, k]) / ((jp - jm) * sy) if jp != jm else 0.0
                gz = (r[i, j, kp] - r[i, j, km]) / ((kp - km) * sz) if kp != km else 0.0
                gl = math.sqrt(gx * gx + gy * gy + gz * gz)
                if gl < 0.5:          # kink (medial axis): keep the JFA value
                    continue
                x = ox + i * sx; y = oy + j * sy; z = oz + k * sz
                qx = x - d0 * gx / gl; qy = y - d0 * gy / gl; qz = z - d0 * gz / gl
                ok = False
                ex = 0.5 * sx; ey = 0.5 * sy; ez = 0.5 * sz
                for _ in range(iters):
                    f0 = _tri(phi, ox, oy, oz, sx, sy, sz, qx, qy, qz)
                    if abs(f0) < 1e-4 * hmin:
                        ok = True
                        break
                    hx = (_tri(phi, ox, oy, oz, sx, sy, sz, qx + ex, qy, qz)
                          - _tri(phi, ox, oy, oz, sx, sy, sz, qx - ex, qy, qz)) / (2 * ex)
                    hy = (_tri(phi, ox, oy, oz, sx, sy, sz, qx, qy + ey, qz)
                          - _tri(phi, ox, oy, oz, sx, sy, sz, qx, qy - ey, qz)) / (2 * ey)
                    hz = (_tri(phi, ox, oy, oz, sx, sy, sz, qx, qy, qz + ez)
                          - _tri(phi, ox, oy, oz, sx, sy, sz, qx, qy, qz - ez)) / (2 * ez)
                    h2 = hx * hx + hy * hy + hz * hz
                    if h2 < 1e-24:
                        break
                    qx -= f0 * hx / h2; qy -= f0 * hy / h2; qz -= f0 * hz / h2
                if not ok:
                    f0 = _tri(phi, ox, oy, oz, sx, sy, sz, qx, qy, qz)
                    ok = abs(f0) < 1e-4 * hmin
                if ok:
                    dx = x - qx; dy = y - qy; dz = z - qz
                    dn = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dn < abs(d0):
                        out[i, j, k] = -dn if d0 < 0 else dn


def redistance(phi: np.ndarray, spacing, origin=(0.0, 0.0, 0.0), *, iters: int = 2,
               band: int = 2, refine: bool = True, jfa: str = "short") -> np.ndarray:
    """Turn any field with the right zero level set into an exact SDF.

    Algorithm (closest point propagation):
      1. nodes within ``band`` voxels of a sign change get their own
         sub-voxel foot point on the trilinear zero level set (Newton);
      2. every other node takes the foot point of its nearest band node
         (exact Euclidean distance transform, scipy);
      3. jump flooding passes pick better foot points (``jfa="short"``:
         strides 4, 2, 1; ``"full"``: N/2 ... 1, 2, 1 - slower, ~20% lower
         mean error);
      4. distance = |x - foot point|, sign from the input;
      5. snap pass: step along the gradient of that field onto the zero level
         (Newton) and keep the shorter upper bound.
    Typical error: ~1% of a voxel near smooth surfaces, < 0.1 voxel far away;
    sharp edges are rounded to about h/3 (as in any grid level set)."""
    phi = np.ascontiguousarray(phi, np.float32)
    sp = np.broadcast_to(np.asarray(spacing, float), (3,)).astype(np.float64)
    origin = np.asarray(origin, np.float64)
    m = K.interface_mask(phi)
    if not m.any():
        big = float(np.linalg.norm(np.asarray(phi.shape) * sp))
        return np.where(phi < 0, -big, big).astype(np.float32)
    if band > 0:
        m = ndimage.binary_dilation(m, iterations=band)
    ii, jj, kk = np.nonzero(m)
    cp = np.empty((len(ii), 3))
    ok = np.empty(len(ii), np.bool_)
    _interface_closest(phi.astype(np.float64), origin, sp, ii, jj, kk, iters, cp, ok)
    if not ok.all():
        m[ii[~ok], jj[~ok], kk[~ok]] = False
        ii, jj, kk, cp = ii[ok], jj[ok], kk[ok], cp[ok]
    lin = np.full(phi.shape, -1, np.int32)
    lin[ii, jj, kk] = np.arange(len(ii), dtype=np.int32)
    idx = np.empty((3,) + phi.shape, np.int32)
    ndimage.distance_transform_edt(~m, sampling=sp, return_distances=False,
                                   return_indices=True, indices=idx)
    owner = lin[idx[0], idx[1], idx[2]]
    del idx
    if refine:
        # jump flooding on top of the EDT seed: the nearest band node is not
        # always the one with the nearest foot point
        buf = np.empty_like(owner)
        steps = []
        if jfa == "full":
            s = 1 << int(np.log2(max(max(phi.shape) // 2, 1)))
            while s >= 1:
                steps.append(s)
                s //= 2
            steps += [2, 1]
        elif jfa == "short":
            steps = [4, 2, 1]
        for s in steps:
            K.jfa_pass(owner, cp, origin, sp, s, buf)
            owner, buf = buf, owner
    out = np.empty(phi.shape, np.float32)
    K.propagate_closest(phi, origin, sp, owner, cp, out, False)
    if refine:
        ref = np.empty_like(out)
        _snap_refine(phi.astype(np.float64), out, origin, sp, 6, ref)
        out = ref
    return out


# --------------------------------------------------------------------------
# meshing
# --------------------------------------------------------------------------
def mesh_from_grid(phi, origin, spacing, project: Implicit | None = None, project_iters=4):
    """Surface nets mesh of the zero level set; vertices are then projected
    onto the exact surface (``project`` implicit) or the trilinear field."""
    from .mesh import Mesh
    sp = np.broadcast_to(np.asarray(spacing, float), (3,))
    if not np.allclose(sp, sp[0]):
        raise ValueError("mesh_from_grid needs isotropic spacing")
    V, Q = K.surface_nets(np.ascontiguousarray(phi, np.float64), np.asarray(origin, float), float(sp[0]))
    if len(Q) == 0:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int64))
    # split each quad along its shorter diagonal
    d02 = np.linalg.norm(V[Q[:, 0]] - V[Q[:, 2]], axis=1)
    d13 = np.linalg.norm(V[Q[:, 1]] - V[Q[:, 3]], axis=1)
    a = d02 <= d13
    F = np.concatenate([
        np.where(a[:, None], Q[:, [0, 1, 2]], Q[:, [0, 1, 3]]),
        np.where(a[:, None], Q[:, [0, 2, 3]], Q[:, [1, 2, 3]]),
    ])
    if project is not None:
        V = _project_clamped(project, V, float(sp[0]), project_iters)
    else:
        f = GridImplicit(phi, origin, sp)
        V = _project_clamped(f, V, float(sp[0]), project_iters)
    return Mesh(V, F)


def _project_clamped(imp, V, h, iters):
    q = V.copy()
    for _ in range(iters):
        d = imp.sdf(q)
        g = imp.gradient(q, eps=1e-3 * h)
        g2 = np.maximum(np.einsum("ij,ij->i", g, g), 1e-30)
        step = (d / g2)[:, None] * g
        L = np.linalg.norm(step, axis=1, keepdims=True)
        step *= np.minimum(1.0, 0.5 * h / np.maximum(L, 1e-300))
        q -= step
    # never move a vertex out of its cell neighbourhood
    off = q - V
    L = np.linalg.norm(off, axis=1, keepdims=True)
    return V + off * np.minimum(1.0, h / np.maximum(L, 1e-300))


class GridImplicit(Implicit):
    """Trilinear interpolation of a dense SDF grid ([X,Y,Z])."""

    def __init__(self, phi, origin, spacing):
        self.phi = np.ascontiguousarray(phi, np.float64)
        self.origin = np.asarray(origin, float)
        self.sp = np.broadcast_to(np.asarray(spacing, float), (3,)).copy()

    def _eval(self, p):
        p = np.ascontiguousarray(p)
        out = np.empty(len(p))
        K.trilinear(self.phi, self.origin, self.sp, p, 1, out)
        hi = self.origin + (np.asarray(self.phi.shape) - 1) * self.sp
        outside = np.any((p < self.origin) | (p > hi), axis=1)
        if outside.any():
            # outside the grid: go to the closest grid point b, then to b's
            # foot point on the surface; |p - foot| is a much tighter upper
            # bound than the trilinear extrapolation (exact along normals)
            q = p[outside]
            b = np.clip(q, self.origin, hi)
            db = np.empty(len(b))
            K.trilinear(self.phi, self.origin, self.sp, b, 1, db)
            eps = 0.5 * self.sp
            g = np.empty_like(b)
            for a in range(3):
                bp = b.copy(); bp[:, a] = np.minimum(bp[:, a] + eps[a], hi[a])
                bm = b.copy(); bm[:, a] = np.maximum(bm[:, a] - eps[a], self.origin[a])
                fp = np.empty(len(b)); fm = np.empty(len(b))
                K.trilinear(self.phi, self.origin, self.sp, bp, 1, fp)
                K.trilinear(self.phi, self.origin, self.sp, bm, 1, fm)
                g[:, a] = (fp - fm) / np.maximum(bp[:, a] - bm[:, a], 1e-300)
            g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-300)
            foot = b - db[:, None] * g
            out[outside] = np.minimum(out[outside], np.linalg.norm(q - foot, axis=1))
        return out

    def _bounds(self):
        return BBox(self.origin, self.origin + (np.asarray(self.phi.shape) - 1) * self.sp)
