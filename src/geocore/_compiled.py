"""Compile an implicit CSG tree into a flat stack program run by numba.

``Implicit.sdf`` uses this for large batches: one parallel pass over the
points, no numpy temporaries per node.  Leaves the compiler does not know
(meshes, lattices, grids, user functions) are evaluated in numpy first and
enter the program as precomputed columns, so every tree still compiles.
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np

from . import _kernels as K

# opcodes ------------------------------------------------------------------
SPHERE, BOX, CYL, BEAM, TORUS, PLANE, CONE, GYROID, SCHWARZP, DIAMOND, CUBLAT = range(11)
LOADCOL = 20
MIN, MAX, NEG, SUBC, SMIN, SDIFF, SINT, SHELL = 30, 31, 32, 33, 34, 35, 36, 37
TBEGIN, TEND = 40, 41


class Program:
    __slots__ = ("ops", "par", "cols", "depth")

    def __init__(self):
        self.ops = []      # (opcode, param offset)
        self.par = []
        self.cols = []     # Implicit leaves evaluated in numpy
        self.depth = 0

    def emit(self, op, *params):
        self.ops.append((op, len(self.par)))
        for v in params:
            if np.ndim(v):
                self.par.extend(float(x) for x in np.ravel(v))
            else:
                self.par.append(float(v))


_NATIVE = None


def _native_types():
    global _NATIVE
    if _NATIVE is None:
        from . import implicit as I
        _NATIVE = {I.Sphere, I.Box, I.Cylinder, I.Beam, I.Torus, I.Plane, I.Cone, I.Gyroid,
                   I.SchwarzP, I.Diamond, I.CubicStrutLattice, I.Union, I.Intersection,
                   I.Difference, I.Complement, I.SmoothUnion, I.SmoothDifference,
                   I.SmoothIntersection, I.Offset, I.Shell, I.Transform}
    return _NATIVE


def _fully_native(n) -> bool:
    return type(n) in _native_types() and all(_fully_native(c) for c in n._children())


def compile_tree(imp):
    """(ops, offsets, params, numpy-leaves) or None if the tree is too deep."""
    from . import implicit as I
    prog = Program()

    def walk(n):
        t = type(n)
        if t is I.Sphere:
            prog.emit(SPHERE, n.c, n.r)
        elif t is I.Box:
            prog.emit(BOX, n.c, n.half, n.r)
        elif t is I.Cylinder:
            prog.emit(CYL, n.a, n.b, n.r)
        elif t is I.Beam:
            prog.emit(BEAM, n.a, n.b, n.ra, n.rb)
        elif t is I.Torus:
            prog.emit(TORUS, n.c, n.R, n.r)
        elif t is I.Plane:
            prog.emit(PLANE, n.p0, n.n)
        elif t is I.Cone:
            prog.emit(CONE, n.a, n.b, n.ra, n.rb)
        elif t in (I.Gyroid, I.SchwarzP, I.Diamond):
            op = {I.Gyroid: GYROID, I.SchwarzP: SCHWARZP, I.Diamond: DIAMOND}[t]
            prog.emit(op, n.k, n.half_t, 1.0 if n.solid else 0.0, n.level, n.o)
        elif t is I.CubicStrutLattice:
            prog.emit(CUBLAT, n.box.c, n.box.half, n.cell, n.r, n.o)
        elif t is I.Union or t is I.Intersection:
            walk(n.children[0])
            for c in n.children[1:]:
                walk(c)
                prog.emit(MIN if t is I.Union else MAX)
        elif t is I.Difference:
            walk(n.a); walk(n.b); prog.emit(NEG); prog.emit(MAX)
        elif t is I.Complement:
            walk(n.a); prog.emit(NEG)
        elif t is I.SmoothUnion:
            walk(n.a); walk(n.b); prog.emit(SMIN, n.k)
        elif t is I.SmoothDifference:
            walk(n.a); walk(n.b); prog.emit(SDIFF, n.k)
        elif t is I.SmoothIntersection:
            walk(n.a); walk(n.b); prog.emit(SINT, n.k)
        elif t is I.Offset:
            walk(n.a); prog.emit(SUBC, n.d)
        elif t is I.Shell:
            walk(n.a); prog.emit(SHELL, n.t)
        elif t is I.Transform and _fully_native(n.a):
            prog.emit(TBEGIN, n.Minv[:3, :3], n.Minv[:3, 3], n.s)
            walk(n.a)
            prog.emit(TEND, n.s)
        else:
            prog.cols.append(n)
            prog.emit(LOADCOL, len(prog.cols) - 1)

    walk(imp)
    # value / point stack depth check (the kernel has fixed size stacks)
    depth = maxd = tdepth = maxt = 0
    for op, _ in prog.ops:
        if op < 30:
            depth += 1
        elif op in (MIN, MAX, SMIN, SDIFF, SINT):
            depth -= 1
        elif op == TBEGIN:
            tdepth += 1
            maxt = max(maxt, tdepth)
        elif op == TEND:
            tdepth -= 1
        maxd = max(maxd, depth)
    if maxd > 60 or maxt > 15:
        return None
    ops = np.array([o for o, _ in prog.ops], np.int64)
    offs = np.array([p for _, p in prog.ops], np.int64)
    par = np.array(prog.par + [0.0], np.float64)
    return ops, offs, par, prog.cols


def evaluate(compiled, p: np.ndarray) -> np.ndarray:
    ops, offs, par, cols = compiled
    n = len(p)
    if cols:
        C = np.empty((len(cols), n))
        for k, c in enumerate(cols):
            C[k] = c._eval(p)
    else:
        C = np.empty((0, n))
    out = np.empty(n)
    _run(ops, offs, par, np.ascontiguousarray(p), C, out)
    return out


# --------------------------------------------------------------------------
@nb.njit(inline="always", cache=True)
def _smin(a, b, k):
    h = min(max(0.5 + 0.5 * (b - a) / k, 0.0), 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


@nb.njit(cache=True, parallel=True)
def _run(ops, offs, P, X, C, out):
    n = X.shape[0]
    nop = ops.shape[0]
    for i in nb.prange(n):
        st = np.empty(64)
        pst = np.empty((16, 3))
        sp = 0
        pp = 0
        x = X[i, 0]; y = X[i, 1]; z = X[i, 2]
        for q in range(nop):
            op = ops[q]
            o = offs[q]
            if op == SPHERE:
                dx = x - P[o]; dy = y - P[o + 1]; dz = z - P[o + 2]
                st[sp] = math.sqrt(dx * dx + dy * dy + dz * dz) - P[o + 3]; sp += 1
            elif op == BOX:
                r = P[o + 6]
                qx = abs(x - P[o]) - (P[o + 3] - r)
                qy = abs(y - P[o + 1]) - (P[o + 4] - r)
                qz = abs(z - P[o + 2]) - (P[o + 5] - r)
                ox = max(qx, 0.0); oy = max(qy, 0.0); oz = max(qz, 0.0)
                st[sp] = (math.sqrt(ox * ox + oy * oy + oz * oz)
                          + min(max(qx, max(qy, qz)), 0.0) - r); sp += 1
            elif op == CYL:
                bax = P[o + 3] - P[o]; bay = P[o + 4] - P[o + 1]; baz = P[o + 5] - P[o + 2]
                pax = x - P[o]; pay = y - P[o + 1]; paz = z - P[o + 2]
                baba = bax * bax + bay * bay + baz * baz
                paba = pax * bax + pay * bay + paz * baz
                cx = pax * baba - bax * paba
                cy = pay * baba - bay * paba
                cz = paz * baba - baz * paba
                xv = math.sqrt(cx * cx + cy * cy + cz * cz) - P[o + 6] * baba
                yv = abs(paba - baba * 0.5) - baba * 0.5
                x2 = xv * xv
                y2 = yv * yv * baba
                if max(xv, yv) < 0.0:
                    d = -min(x2, y2)
                else:
                    d = (x2 if xv > 0.0 else 0.0) + (y2 if yv > 0.0 else 0.0)
                s = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
                st[sp] = s * math.sqrt(abs(d)) / baba; sp += 1
            elif op == BEAM:
                st[sp] = K._sd_round_cone(x, y, z, P[o], P[o + 1], P[o + 2],
                                          P[o + 3], P[o + 4], P[o + 5], P[o + 6], P[o + 7]); sp += 1
            elif op == TORUS:
                dx = x - P[o]; dy = y - P[o + 1]; dz = z - P[o + 2]
                qq = math.sqrt(dx * dx + dy * dy) - P[o + 3]
                st[sp] = math.sqrt(qq * qq + dz * dz) - P[o + 4]; sp += 1
            elif op == PLANE:
                st[sp] = ((x - P[o]) * P[o + 3] + (y - P[o + 1]) * P[o + 4]
                          + (z - P[o + 2]) * P[o + 5]); sp += 1
            elif op == CONE:
                ra = P[o + 6]; rb = P[o + 7]
                bax = P[o + 3] - P[o]; bay = P[o + 4] - P[o + 1]; baz = P[o + 5] - P[o + 2]
                pax = x - P[o]; pay = y - P[o + 1]; paz = z - P[o + 2]
                rba = rb - ra
                baba = bax * bax + bay * bay + baz * baz
                papa = pax * pax + pay * pay + paz * paz
                paba = (pax * bax + pay * bay + paz * baz) / baba
                xx = math.sqrt(max(papa - paba * paba * baba, 0.0))
                cax = max(0.0, xx - (ra if paba < 0.5 else rb))
                cay = abs(paba - 0.5) - 0.5
                kk = rba * rba + baba
                f = min(max((rba * (xx - ra) + paba * baba) / kk, 0.0), 1.0)
                cbx = xx - ra - f * rba
                cby = paba - f
                s = -1.0 if (cbx < 0.0 and cay < 0.0) else 1.0
                st[sp] = s * math.sqrt(min(cax * cax + cay * cay * baba,
                                           cbx * cbx + cby * cby * baba)); sp += 1
            elif op == GYROID or op == SCHWARZP or op == DIAMOND:
                k = P[o]
                qx = (x - P[o + 4]) * k; qy = (y - P[o + 5]) * k; qz = (z - P[o + 6]) * k
                if op == GYROID:
                    g = (math.sin(qx) * math.cos(qy) + math.sin(qy) * math.cos(qz)
                         + math.sin(qz) * math.cos(qx))
                elif op == SCHWARZP:
                    g = math.cos(qx) + math.cos(qy) + math.cos(qz)
                else:
                    sx = math.sin(qx); sy = math.sin(qy); sz = math.sin(qz)
                    cx = math.cos(qx); cy = math.cos(qy); cz = math.cos(qz)
                    g = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
                g -= P[o + 3]
                if P[o + 2] > 0.5:
                    st[sp] = g / k
                else:
                    st[sp] = (abs(g) - P[o + 1]) / k
                sp += 1
            elif op == CUBLAT:
                a = P[o + 6]
                qx = (x - P[o + 8]) % a; qx = min(qx, a - qx)
                qy = (y - P[o + 9]) % a; qy = min(qy, a - qy)
                qz = (z - P[o + 10]) % a; qz = min(qz, a - qz)
                d = min(min(math.sqrt(qy * qy + qz * qz), math.sqrt(qx * qx + qz * qz)),
                        math.sqrt(qx * qx + qy * qy)) - P[o + 7]
                bx = abs(x - P[o]) - P[o + 3]
                by = abs(y - P[o + 1]) - P[o + 4]
                bz = abs(z - P[o + 2]) - P[o + 5]
                ox = max(bx, 0.0); oy = max(by, 0.0); oz = max(bz, 0.0)
                db = math.sqrt(ox * ox + oy * oy + oz * oz) + min(max(bx, max(by, bz)), 0.0)
                st[sp] = max(d, db); sp += 1
            elif op == LOADCOL:
                st[sp] = C[int(P[o]), i]; sp += 1
            elif op == MIN:
                sp -= 1; st[sp - 1] = min(st[sp - 1], st[sp])
            elif op == MAX:
                sp -= 1; st[sp - 1] = max(st[sp - 1], st[sp])
            elif op == NEG:
                st[sp - 1] = -st[sp - 1]
            elif op == SUBC:
                st[sp - 1] -= P[o]
            elif op == SMIN:
                sp -= 1; st[sp - 1] = _smin(st[sp - 1], st[sp], P[o])
            elif op == SDIFF:
                sp -= 1; st[sp - 1] = -_smin(-st[sp - 1], st[sp], P[o])
            elif op == SINT:
                sp -= 1; st[sp - 1] = -_smin(-st[sp - 1], -st[sp], P[o])
            elif op == SHELL:
                d = st[sp - 1]
                t = P[o]
                st[sp - 1] = max(d, -(d - t)) if t < 0 else max(d - t, -d)
            elif op == TBEGIN:
                pst[pp, 0] = x; pst[pp, 1] = y; pst[pp, 2] = z; pp += 1
                nx_ = P[o] * x + P[o + 1] * y + P[o + 2] * z + P[o + 9]
                ny_ = P[o + 3] * x + P[o + 4] * y + P[o + 5] * z + P[o + 10]
                nz_ = P[o + 6] * x + P[o + 7] * y + P[o + 8] * z + P[o + 11]
                x = nx_; y = ny_; z = nz_
            elif op == TEND:
                pp -= 1
                x = pst[pp, 0]; y = pst[pp, 1]; z = pst[pp, 2]
                st[sp - 1] *= P[o]
        out[i] = st[0]
