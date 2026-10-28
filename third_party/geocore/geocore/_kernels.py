"""Numba kernels: the hot loops of geocore.

Everything here works on plain float64/int64 arrays so the public layer can stay
pure numpy.  All kernels are compiled lazily, cached on disk and parallel where
the work is embarrassingly parallel.
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np

_JIT = dict(cache=True)
_PJIT = dict(cache=True, parallel=True)


# --------------------------------------------------------------------------
# BVH over axis aligned boxes (median split), flat arrays
# --------------------------------------------------------------------------
@nb.njit(**_JIT)
def bvh_build(lo, hi, leaf_size=4):
    """Returns (node_lo, node_hi, left, right, first, count, order).

    Leaves reference ``order[first:first+count]``; inner nodes have count 0."""
    m = lo.shape[0]
    cen = 0.5 * (lo + hi)
    order = np.arange(m)
    cap = max(2 * m, 1)
    nlo = np.empty((cap, 3))
    nhi = np.empty((cap, 3))
    left = -np.ones(cap, np.int64)
    right = -np.ones(cap, np.int64)
    first = np.zeros(cap, np.int64)
    count = np.zeros(cap, np.int64)
    st_n = np.empty(128, np.int64)
    st_s = np.empty(128, np.int64)
    st_e = np.empty(128, np.int64)
    st_n[0] = 0
    st_s[0] = 0
    st_e[0] = m
    sp = 1
    nn = 1
    while sp > 0:
        sp -= 1
        node = st_n[sp]
        s = st_s[sp]
        e = st_e[sp]
        bl = np.full(3, 1e300)
        bh = np.full(3, -1e300)
        cl = np.full(3, 1e300)
        ch = np.full(3, -1e300)
        for t in range(s, e):
            k = order[t]
            for a in range(3):
                bl[a] = min(bl[a], lo[k, a])
                bh[a] = max(bh[a], hi[k, a])
                cl[a] = min(cl[a], cen[k, a])
                ch[a] = max(ch[a], cen[k, a])
        nlo[node] = bl
        nhi[node] = bh
        if e - s <= leaf_size:
            first[node] = s
            count[node] = e - s
            continue
        axis = 0
        if ch[1] - cl[1] > ch[axis] - cl[axis]:
            axis = 1
        if ch[2] - cl[2] > ch[axis] - cl[axis]:
            axis = 2
        sub = order[s:e].copy()
        keys = np.empty(e - s)
        for t in range(e - s):
            keys[t] = cen[sub[t], axis]
        idx = np.argsort(keys)
        for t in range(e - s):
            order[s + t] = sub[idx[t]]
        mid = (s + e) // 2
        l = nn
        r = nn + 1
        nn += 2
        left[node] = l
        right[node] = r
        st_n[sp] = r; st_s[sp] = mid; st_e[sp] = e
        sp += 1
        st_n[sp] = l; st_s[sp] = s; st_e[sp] = mid
        sp += 1
    return (nlo[:nn].copy(), nhi[:nn].copy(), left[:nn].copy(), right[:nn].copy(),
            first[:nn].copy(), count[:nn].copy(), order)


@nb.njit(inline="always", **_JIT)
def _box_d2(nlo, nhi, n, px, py, pz):
    dx = max(nlo[n, 0] - px, 0.0, px - nhi[n, 0])
    dy = max(nlo[n, 1] - py, 0.0, py - nhi[n, 1])
    dz = max(nlo[n, 2] - pz, 0.0, pz - nhi[n, 2])
    return dx * dx + dy * dy + dz * dz


def build_bvh(lo, hi, leaf_size=4):
    return bvh_build(np.ascontiguousarray(lo, np.float64),
                     np.ascontiguousarray(hi, np.float64), leaf_size)


# --------------------------------------------------------------------------
# Round cone (PicoGK lattice beam) SDF  -- Inigo Quilez, exact
# --------------------------------------------------------------------------
@nb.njit(inline="always", **_JIT)
def _sd_round_cone(px, py, pz, ax, ay, az, bx, by, bz, r1, r2):
    bax = bx - ax
    bay = by - ay
    baz = bz - az
    l2 = bax * bax + bay * bay + baz * baz
    pax = px - ax
    pay = py - ay
    paz = pz - az
    if l2 < 1e-18:
        return math.sqrt(pax * pax + pay * pay + paz * paz) - max(r1, r2)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    y = pax * bax + pay * bay + paz * baz
    z = y - l2
    xx = pax * l2 - bax * y
    xy = pay * l2 - bay * y
    xz = paz * l2 - baz * y
    x2 = xx * xx + xy * xy + xz * xz
    y2 = y * y * l2
    z2 = z * z * l2
    k = math.copysign(1.0, rr) * rr * rr * x2 if rr != 0.0 else 0.0
    if a2 <= 0.0:
        # one sphere swallows the other: the shape is the larger sphere
        if r1 >= r2:
            return math.sqrt(pax * pax + pay * pay + paz * paz) - r1
        qx = px - bx
        qy = py - by
        qz = pz - bz
        return math.sqrt(qx * qx + qy * qy + qz * qz) - r2
    sz = 1.0 if z > 0 else (-1.0 if z < 0 else 0.0)
    sy = 1.0 if y > 0 else (-1.0 if y < 0 else 0.0)
    if sz * a2 * z2 > k:
        return math.sqrt(x2 + z2) * il2 - r2
    if sy * a2 * y2 < k:
        return math.sqrt(x2 + y2) * il2 - r1
    return (math.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1


@nb.njit(**_PJIT)
def beams_sdf_brute(p, beams, out):
    """beams: [M, 8] = ax ay az bx by bz ra rb."""
    n = p.shape[0]
    m = beams.shape[0]
    for i in nb.prange(n):
        best = 1e30
        for k in range(m):
            d = _sd_round_cone(p[i, 0], p[i, 1], p[i, 2],
                               beams[k, 0], beams[k, 1], beams[k, 2],
                               beams[k, 3], beams[k, 4], beams[k, 5],
                               beams[k, 6], beams[k, 7])
            if d < best:
                best = d
        out[i] = best


@nb.njit(**_PJIT)
def beams_sdf_bvh(p, beams, nlo, nhi, left, right, first, count, order, out):
    """Union of round cones; exact outside (min of exact SDFs)."""
    n = p.shape[0]
    for i in nb.prange(n):
        px, py, pz = p[i, 0], p[i, 1], p[i, 2]
        best = 1e30
        stack = np.empty(128, np.int64)
        sp = 1
        stack[0] = 0
        while sp > 0:
            sp -= 1
            nd = stack[sp]
            lim = best if best > 0.0 else 0.0
            if _box_d2(nlo, nhi, nd, px, py, pz) > lim * lim:
                continue
            if count[nd] > 0:
                for t in range(first[nd], first[nd] + count[nd]):
                    k = order[t]
                    d = _sd_round_cone(px, py, pz,
                                       beams[k, 0], beams[k, 1], beams[k, 2],
                                       beams[k, 3], beams[k, 4], beams[k, 5],
                                       beams[k, 6], beams[k, 7])
                    if d < best:
                        best = d
            else:
                l = left[nd]
                r = right[nd]
                if _box_d2(nlo, nhi, l, px, py, pz) < _box_d2(nlo, nhi, r, px, py, pz):
                    stack[sp] = r
                    stack[sp + 1] = l
                else:
                    stack[sp] = l
                    stack[sp + 1] = r
                sp += 2
        out[i] = best


# --------------------------------------------------------------------------
# Triangle mesh signed distance (angle weighted pseudo normals)
# --------------------------------------------------------------------------
@nb.njit(inline="always", **_JIT)
def _closest_on_triangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
    """Ericson, Real-Time Collision Detection 5.1.5.

    Returns (qx, qy, qz, feature) with feature 0..2 = vertex a,b,c,
    3..5 = edge ab, bc, ca, 6 = face."""
    abx = bx - ax; aby = by - ay; abz = bz - az
    acx = cx - ax; acy = cy - ay; acz = cz - az
    apx = px - ax; apy = py - ay; apz = pz - az
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return ax, ay, az, 0
    bpx = px - bx; bpy = py - by; bpz = pz - bz
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return bx, by, bz, 1
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return ax + v * abx, ay + v * aby, az + v * abz, 3
    cpx = px - cx; cpy = py - cy; cpz = pz - cz
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return cx, cy, cz, 2
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return ax + w * acx, ay + w * acy, az + w * acz, 5
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return bx + w * (cx - bx), by + w * (cy - by), bz + w * (cz - bz), 4
    den = 1.0 / (va + vb + vc)
    v = vb * den
    w = vc * den
    return (ax + abx * v + acx * w, ay + aby * v + acy * w,
            az + abz * v + acz * w, 6)


@nb.njit(**_PJIT)
def mesh_sdf_bvh(p, V, F, fn, vn, en, nlo, nhi, left, right, first, count, order,
                 out, closest, face, signed):
    """Signed distance from points to a closed, consistently oriented mesh.

    fn: face normals [F,3]; vn: vertex pseudo normals [V,3];
    en: edge pseudo normals per face edge [F,3,3] (ab, bc, ca)."""
    n = p.shape[0]
    for i in nb.prange(n):
        px, py, pz = p[i, 0], p[i, 1], p[i, 2]
        best2 = 1e300
        bq0 = 0.0; bq1 = 0.0; bq2 = 0.0
        bf = -1
        bfeat = 6
        stack = np.empty(128, np.int64)
        sp = 1
        stack[0] = 0
        while sp > 0:
            sp -= 1
            nd = stack[sp]
            if _box_d2(nlo, nhi, nd, px, py, pz) > best2:
                continue
            if count[nd] > 0:
                for t in range(first[nd], first[nd] + count[nd]):
                    f = order[t]
                    ia = F[f, 0]; ib = F[f, 1]; ic = F[f, 2]
                    qx, qy, qz, feat = _closest_on_triangle(
                        px, py, pz,
                        V[ia, 0], V[ia, 1], V[ia, 2],
                        V[ib, 0], V[ib, 1], V[ib, 2],
                        V[ic, 0], V[ic, 1], V[ic, 2])
                    dx = px - qx; dy = py - qy; dz = pz - qz
                    d2 = dx * dx + dy * dy + dz * dz
                    if d2 < best2:
                        best2 = d2
                        bq0 = qx; bq1 = qy; bq2 = qz
                        bf = f
                        bfeat = feat
            else:
                l = left[nd]
                r = right[nd]
                if _box_d2(nlo, nhi, l, px, py, pz) < _box_d2(nlo, nhi, r, px, py, pz):
                    stack[sp] = r
                    stack[sp + 1] = l
                else:
                    stack[sp] = l
                    stack[sp + 1] = r
                sp += 2
        d = math.sqrt(best2)
        closest[i, 0] = bq0
        closest[i, 1] = bq1
        closest[i, 2] = bq2
        face[i] = bf
        if signed and bf >= 0:
            if bfeat < 3:
                vi = F[bf, bfeat]
                nx_ = vn[vi, 0]; ny_ = vn[vi, 1]; nz_ = vn[vi, 2]
            elif bfeat < 6:
                e = bfeat - 3
                nx_ = en[bf, e, 0]; ny_ = en[bf, e, 1]; nz_ = en[bf, e, 2]
            else:
                nx_ = fn[bf, 0]; ny_ = fn[bf, 1]; nz_ = fn[bf, 2]
            s = (px - bq0) * nx_ + (py - bq1) * ny_ + (pz - bq2) * nz_
            if s < 0.0:
                d = -d
        out[i] = d


# --------------------------------------------------------------------------
# Trilinear interpolation on a dense grid stored [X, Y, Z]
# --------------------------------------------------------------------------
@nb.njit(**_PJIT)
def trilinear(field, origin, sp, p, fill, out):
    """sp: grid spacing per axis.  fill=1: SDF semantics outside the grid."""
    nx, ny, nz = field.shape
    n = p.shape[0]
    for i in nb.prange(n):
        fx = (p[i, 0] - origin[0]) / sp[0]
        fy = (p[i, 1] - origin[1]) / sp[1]
        fz = (p[i, 2] - origin[2]) / sp[2]
        if (fx < 0.0 or fy < 0.0 or fz < 0.0 or fx > nx - 1 or fy > ny - 1
                or fz > nz - 1):
            # outside the grid: clamp, then add the distance to the grid box
            # (an upper bound of |sdf| for SDF fields, exact along normals)
            cx = min(max(fx, 0.0), nx - 1.0)
            cy = min(max(fy, 0.0), ny - 1.0)
            cz = min(max(fz, 0.0), nz - 1.0)
            ddx = (fx - cx) * sp[0]; ddy = (fy - cy) * sp[1]; ddz = (fz - cz) * sp[2]
            extra = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
            fx, fy, fz = cx, cy, cz
        else:
            extra = 0.0
        i0 = min(int(fx), nx - 2) if nx > 1 else 0
        j0 = min(int(fy), ny - 2) if ny > 1 else 0
        k0 = min(int(fz), nz - 2) if nz > 1 else 0
        tx = fx - i0; ty = fy - j0; tz = fz - k0
        i1 = min(i0 + 1, nx - 1); j1 = min(j0 + 1, ny - 1); k1 = min(k0 + 1, nz - 1)
        c00 = field[i0, j0, k0] * (1 - tx) + field[i1, j0, k0] * tx
        c10 = field[i0, j1, k0] * (1 - tx) + field[i1, j1, k0] * tx
        c01 = field[i0, j0, k1] * (1 - tx) + field[i1, j0, k1] * tx
        c11 = field[i0, j1, k1] * (1 - tx) + field[i1, j1, k1] * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        v = c0 * (1 - tz) + c1 * tz
        if extra > 0.0 and fill == 1:      # SDF semantics: stay a lower bound
            v = v + extra if v > 0 else extra
        out[i] = v


# --------------------------------------------------------------------------
# Manifold surface nets on [X,Y,Z] grids
#
# Classic surface nets puts one vertex in every cell the surface crosses,
# which is non-manifold where a cell holds two sheets.  Here every cell gets
# one vertex per connected component of its crossed edges; components are
# found from the 6 cube faces (ambiguous faces resolved by the face centre
# value, identically in both neighbouring cells), so the mesh is manifold
# and watertight wherever the surface is closed inside the grid.
# --------------------------------------------------------------------------
# cube corners: bit0 = x, bit1 = y, bit2 = z
_E0 = np.array([0, 2, 4, 6, 0, 1, 4, 5, 0, 1, 2, 3], np.int64)
_E1 = np.array([1, 3, 5, 7, 2, 3, 6, 7, 4, 5, 6, 7], np.int64)


def _faces_table():
    edge_of = {}
    for e in range(12):
        edge_of[(int(_E0[e]), int(_E1[e]))] = e
        edge_of[(int(_E1[e]), int(_E0[e]))] = e
    fc = []
    for axis in range(3):
        for side in (0, 1):
            others = [a for a in range(3) if a != axis]
            corners = []
            for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                corners.append((side << axis) | (u << others[0]) | (v << others[1]))
            fc.append(corners)
    fc = np.array(fc, np.int64)
    fe = np.empty((6, 4), np.int64)
    for f in range(6):
        for m in range(4):
            fe[f, m] = edge_of[(int(fc[f, m]), int(fc[f, (m + 1) % 4]))]
    return fc, fe


_FC, _FE = _faces_table()


@nb.njit(inline="always", **_JIT)
def _uf_find(par, x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


@nb.njit(**_JIT)
def surface_nets(phi, origin, h):
    nx, ny, nz = phi.shape
    E0 = _E0
    E1 = _E1
    FC = _FC
    FE = _FE
    ncell = (nx - 1) * (ny - 1) * (nz - 1)
    slot = -np.ones(ncell, np.int32)          # cell -> row in emap
    emap = -np.ones((1024, 12), np.int64)     # local edge -> vertex id
    verts = np.empty((1024, 3), np.float64)
    nv = 0
    ns = 0
    cube = np.empty(8)
    par = np.empty(12, np.int64)
    cross = np.zeros(12, np.bool_)
    px = np.empty(12)
    py = np.empty(12)
    pz = np.empty(12)
    acc = np.zeros((12, 4))
    vid = np.empty(12, np.int64)
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                neg = 0
                for c in range(8):
                    v = phi[i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1)]
                    cube[c] = v
                    if v < 0:
                        neg += 1
                if neg == 0 or neg == 8:
                    continue
                for e in range(12):
                    par[e] = e
                    va = cube[E0[e]]
                    vb = cube[E1[e]]
                    cross[e] = (va < 0) != (vb < 0)
                    if cross[e]:
                        t = va / (va - vb)
                        a = E0[e]
                        b = E1[e]
                        px[e] = (a & 1) + t * ((b & 1) - (a & 1))
                        py[e] = ((a >> 1) & 1) + t * (((b >> 1) & 1) - ((a >> 1) & 1))
                        pz[e] = ((a >> 2) & 1) + t * (((b >> 2) & 1) - ((a >> 2) & 1))
                # connect crossed edges through the cube faces
                for f in range(6):
                    nc = 0
                    for m in range(4):
                        if cross[FE[f, m]]:
                            nc += 1
                    if nc == 2:
                        e1 = -1
                        e2 = -1
                        for m in range(4):
                            if cross[FE[f, m]]:
                                if e1 < 0:
                                    e1 = FE[f, m]
                                else:
                                    e2 = FE[f, m]
                        par[_uf_find(par, e1)] = _uf_find(par, e2)
                    elif nc == 4:
                        centre = 0.0
                        for m in range(4):
                            centre += cube[FC[f, m]]
                        # centre outside -> inside corners are isolated, and
                        # vice versa; each isolated corner joins its 2 edges
                        iso_neg = centre >= 0
                        for m in range(4):
                            if (cube[FC[f, m]] < 0) == iso_neg:
                                ea = FE[f, (m + 3) % 4]
                                eb = FE[f, m]
                                par[_uf_find(par, ea)] = _uf_find(par, eb)
                if ns >= emap.shape[0]:
                    ne = -np.ones((emap.shape[0] * 2, 12), np.int64)
                    ne[:ns] = emap[:ns]
                    emap = ne
                slot[(i * (ny - 1) + j) * (nz - 1) + k] = ns
                for e in range(12):
                    acc[e, 0] = 0.0
                    acc[e, 1] = 0.0
                    acc[e, 2] = 0.0
                    acc[e, 3] = 0.0
                for e in range(12):
                    if cross[e]:
                        r = _uf_find(par, e)
                        acc[r, 0] += px[e]
                        acc[r, 1] += py[e]
                        acc[r, 2] += pz[e]
                        acc[r, 3] += 1.0
                for e in range(12):
                    vid[e] = -1
                    if cross[e] and _uf_find(par, e) == e:
                        if nv >= verts.shape[0]:
                            nvv = np.empty((verts.shape[0] * 2, 3))
                            nvv[:nv] = verts[:nv]
                            verts = nvv
                        verts[nv, 0] = origin[0] + (i + acc[e, 0] / acc[e, 3]) * h
                        verts[nv, 1] = origin[1] + (j + acc[e, 1] / acc[e, 3]) * h
                        verts[nv, 2] = origin[2] + (k + acc[e, 2] / acc[e, 3]) * h
                        vid[e] = nv
                        nv += 1
                for e in range(12):
                    if cross[e]:
                        emap[ns, e] = vid[_uf_find(par, e)]
                ns += 1
    # one quad per sign-changing grid edge, from the 4 cells around it
    quads = np.empty((max(nv, 16) * 2, 4), np.int64)
    nq = 0
    cid = np.empty(4, np.int64)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v0 = phi[i, j, k]
                for axis in range(3):
                    ii = i + (1 if axis == 0 else 0)
                    jj = j + (1 if axis == 1 else 0)
                    kk = k + (1 if axis == 2 else 0)
                    if ii >= nx or jj >= ny or kk >= nz:
                        continue
                    if (v0 < 0) == (phi[ii, jj, kk] < 0):
                        continue
                    ok = True
                    # the 4 cells around the edge, ccw seen from +axis
                    for q in range(4):
                        du = 1 if (q == 1 or q == 2) else 0
                        dv = 1 if q >= 2 else 0
                        if axis == 0:      # cell offsets in (y, z)
                            ci = i
                            cj = j - 1 + du
                            ck = k - 1 + dv
                            le = (1 - du) + 2 * (1 - dv)
                        elif axis == 1:    # cell offsets in (z, x)
                            ci = i - 1 + dv
                            cj = j
                            ck = k - 1 + du
                            le = 4 + (1 - dv) + 2 * (1 - du)
                        else:              # cell offsets in (x, y)
                            ci = i - 1 + du
                            cj = j - 1 + dv
                            ck = k
                            le = 8 + (1 - du) + 2 * (1 - dv)
                        if (ci < 0 or cj < 0 or ck < 0 or ci >= nx - 1
                                or cj >= ny - 1 or ck >= nz - 1):
                            ok = False
                            break
                        sl = slot[(ci * (ny - 1) + cj) * (nz - 1) + ck]
                        if sl < 0:
                            ok = False
                            break
                        cid[q] = emap[sl, le]
                    if not ok:
                        continue
                    if nq >= quads.shape[0]:
                        nqq = np.empty((quads.shape[0] * 2, 4), np.int64)
                        nqq[:nq] = quads[:nq]
                        quads = nqq
                    if v0 < 0:
                        quads[nq, 0] = cid[0]
                        quads[nq, 1] = cid[1]
                        quads[nq, 2] = cid[2]
                        quads[nq, 3] = cid[3]
                    else:
                        quads[nq, 0] = cid[0]
                        quads[nq, 1] = cid[3]
                        quads[nq, 2] = cid[2]
                        quads[nq, 3] = cid[1]
                    nq += 1
    return verts[:nv].copy(), quads[:nq].copy()


# --------------------------------------------------------------------------
# Redistancing helpers (closest point propagation)
# --------------------------------------------------------------------------
@nb.njit(**_JIT)
def interface_mask(phi):
    """Grid nodes that have a 6-neighbour on the other side of the surface."""
    nx, ny, nz = phi.shape
    m = np.zeros(phi.shape, np.bool_)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                s = phi[i, j, k] < 0
                if i + 1 < nx and (phi[i + 1, j, k] < 0) != s:
                    m[i, j, k] = True; m[i + 1, j, k] = True
                if j + 1 < ny and (phi[i, j + 1, k] < 0) != s:
                    m[i, j, k] = True; m[i, j + 1, k] = True
                if k + 1 < nz and (phi[i, j, k + 1] < 0) != s:
                    m[i, j, k] = True; m[i, j, k + 1] = True
    return m


@nb.njit(**_PJIT)
def propagate_closest(phi, origin, sp, idx_map, cpx, out, refine):
    """out = sign(phi) * |x - cp[nearest interface node]|.

    idx_map: [X,Y,Z] int32 index into cpx (closest point list) for every node.
    refine: also test the closest points owned by the 26 neighbours (one
    jump-flooding style pass that removes most EDT ownership errors)."""
    nx, ny, nz = phi.shape
    for i in nb.prange(nx):
        for j in range(ny):
            for k in range(nz):
                x = origin[0] + i * sp[0]
                y = origin[1] + j * sp[1]
                z = origin[2] + k * sp[2]
                c = idx_map[i, j, k]
                dx = x - cpx[c, 0]; dy = y - cpx[c, 1]; dz = z - cpx[c, 2]
                best = dx * dx + dy * dy + dz * dz
                if refine:
                    for a in range(max(i - 1, 0), min(i + 2, nx)):
                        for b in range(max(j - 1, 0), min(j + 2, ny)):
                            for cc in range(max(k - 1, 0), min(k + 2, nz)):
                                c2 = idx_map[a, b, cc]
                                if c2 == c:
                                    continue
                                dx = x - cpx[c2, 0]; dy = y - cpx[c2, 1]
                                dz = z - cpx[c2, 2]
                                d2 = dx * dx + dy * dy + dz * dz
                                if d2 < best:
                                    best = d2
                d = math.sqrt(best)
                out[i, j, k] = -d if phi[i, j, k] < 0 else d


@nb.njit(**_PJIT)
def jfa_pass(owner, cpx, origin, sp, step, out_owner):
    """One jump flooding pass: adopt a neighbour's (at +-step) foot point when
    it is closer.  Double buffered (owner -> out_owner)."""
    nx, ny, nz = owner.shape
    for i in nb.prange(nx):
        for j in range(ny):
            for k in range(nz):
                x = origin[0] + i * sp[0]
                y = origin[1] + j * sp[1]
                z = origin[2] + k * sp[2]
                c = owner[i, j, k]
                dx = x - cpx[c, 0]; dy = y - cpx[c, 1]; dz = z - cpx[c, 2]
                best = dx * dx + dy * dy + dz * dz
                bc = c
                for a in range(-1, 2):
                    ia = i + a * step
                    if ia < 0 or ia >= nx:
                        continue
                    for b in range(-1, 2):
                        jb = j + b * step
                        if jb < 0 or jb >= ny:
                            continue
                        for cc in range(-1, 2):
                            kc = k + cc * step
                            if kc < 0 or kc >= nz:
                                continue
                            c2 = owner[ia, jb, kc]
                            if c2 == bc:
                                continue
                            dx = x - cpx[c2, 0]; dy = y - cpx[c2, 1]; dz = z - cpx[c2, 2]
                            d2 = dx * dx + dy * dy + dz * dz
                            if d2 < best:
                                best = d2
                                bc = c2
                out_owner[i, j, k] = bc


# --------------------------------------------------------------------------
# z-buffer rasterizer (for geocore.render)
# --------------------------------------------------------------------------
@nb.njit(**_JIT)
def rasterize(px, py, depth, F, vn, zbuf, nbuf):
    """Draw triangles into zbuf (larger = closer) and interpolated vertex
    normals into nbuf.  px/py are pixel coordinates of the vertices."""
    H, W = zbuf.shape
    for f in range(F.shape[0]):
        a = F[f, 0]; b = F[f, 1]; c = F[f, 2]
        x0 = px[a]; y0 = py[a]; x1 = px[b]; y1 = py[b]; x2 = px[c]; y2 = py[c]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        inv = 1.0 / area
        i0 = max(int(math.floor(min(x0, min(x1, x2)))), 0)
        i1 = min(int(math.ceil(max(x0, max(x1, x2)))), W - 1)
        j0 = max(int(math.floor(min(y0, min(y1, y2)))), 0)
        j1 = min(int(math.ceil(max(y0, max(y1, y2)))), H - 1)
        for j in range(j0, j1 + 1):
            yc = j + 0.5
            for i in range(i0, i1 + 1):
                xc = i + 0.5
                w0 = ((x1 - xc) * (y2 - yc) - (x2 - xc) * (y1 - yc)) * inv
                w1 = ((x2 - xc) * (y0 - yc) - (x0 - xc) * (y2 - yc)) * inv
                w2 = 1.0 - w0 - w1
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                z = w0 * depth[a] + w1 * depth[b] + w2 * depth[c]
                if z > zbuf[j, i]:
                    zbuf[j, i] = z
                    for k in range(3):
                        nbuf[j, i, k] = w0 * vn[a, k] + w1 * vn[b, k] + w2 * vn[c, k]


# --------------------------------------------------------------------------
# generalized winding number (Jacobson et al. 2013) - robust inside/outside
# for meshes with holes
# --------------------------------------------------------------------------
@nb.njit(**_PJIT)
def winding_numbers(p, V, F, out):
    n = p.shape[0]
    for i in nb.prange(n):
        px, py, pz = p[i, 0], p[i, 1], p[i, 2]
        acc = 0.0
        for f in range(F.shape[0]):
            ax = V[F[f, 0], 0] - px; ay = V[F[f, 0], 1] - py; az = V[F[f, 0], 2] - pz
            bx = V[F[f, 1], 0] - px; by = V[F[f, 1], 1] - py; bz = V[F[f, 1], 2] - pz
            cx = V[F[f, 2], 0] - px; cy = V[F[f, 2], 1] - py; cz = V[F[f, 2], 2] - pz
            la = math.sqrt(ax * ax + ay * ay + az * az)
            lb = math.sqrt(bx * bx + by * by + bz * bz)
            lc = math.sqrt(cx * cx + cy * cy + cz * cz)
            det = (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx))
            den = (la * lb * lc + (ax * bx + ay * by + az * bz) * lc
                   + (ax * cx + ay * cy + az * cz) * lb + (bx * cx + by * cy + bz * cz) * la)
            acc += 2.0 * math.atan2(det, den)
        out[i] = acc / (4.0 * math.pi)
