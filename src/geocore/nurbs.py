"""2D NURBS domains for Deep NURBS (PARAMETRIC_SURFACE).

A ``NURBSSurface`` maps (u, v) in [0,1]^2 to a 2D physical domain:

    S(u,v) = sum_ij R_ij(u,v) P_ij,   R_ij = N_i(u) M_j(v) w_ij / W(u,v)

* exact constructors: rectangle, bilinear quad, annulus sector (exact
  circles), plate with hole, ruled surfaces between NURBS curves, Coons
  patches of B-spline boundary curves, interpolation of any map;
* exact h/p refinement (knot insertion, degree elevation) - the refined
  surface is the *same* geometry;
* Deep NURBS coefficients: ``admissible_coefficients`` (1 inside, 0 on the
  four boundaries -> phi = sum R_ij a_ij vanishes on the boundary) and
  ``lifting_coefficients`` (a NURBS function equal to the Dirichlet data on
  the boundary, Coons-blended inside).

Index convention: ``control_points[i, j]`` - i along u, j along v.
Boundaries: bottom v=0 (j=0), top v=1 (j=-1), left u=0 (i=0), right u=1 (i=-1).
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Optional, Union

import numpy as np
from scipy.stats import qmc

Array = np.ndarray


# ==========================================================================
# B-spline basics (vectorised over evaluation points)
# ==========================================================================
def find_span(knots: Array, p: int, n_ctrl: int, u: Array) -> Array:
    u = np.asarray(u, np.float64)
    s = np.searchsorted(knots, u, side="right") - 1
    return np.clip(s, p, n_ctrl - 1)


def basis_ders(knots: Array, p: int, n_ctrl: int, u: Array, nder: int = 0):
    """(span [N], ders [nder+1, N, p+1]) - Piegl & Tiller A2.3, vectorised."""
    u = np.atleast_1d(np.asarray(u, np.float64))
    span = find_span(knots, p, n_ctrl, u)
    N = len(u)
    ndu = np.zeros((p + 1, p + 1, N))
    ndu[0, 0] = 1.0
    left = np.zeros((p + 1, N))
    right = np.zeros((p + 1, N))
    for j in range(1, p + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = np.zeros(N)
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            den = np.where(ndu[j, r] == 0, 1.0, ndu[j, r])
            temp = ndu[r, j - 1] / den
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved
    ders = np.zeros((nder + 1, N, p + 1))
    for j in range(p + 1):
        ders[0, :, j] = ndu[j, p]
    if nder:
        a = np.zeros((2, p + 1, N))
        for r in range(p + 1):
            s1, s2 = 0, 1
            a[0] = 0.0
            a[0, 0] = 1.0
            for k in range(1, nder + 1):
                d = np.zeros(N)
                rk, pk = r - k, p - k
                if r >= k:
                    a[s2, 0] = a[s1, 0] / _nz(ndu[pk + 1, rk])
                    d = a[s2, 0] * ndu[rk, pk]
                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if r - 1 <= pk else p - r
                for j in range(j1, j2 + 1):
                    a[s2, j] = (a[s1, j] - a[s1, j - 1]) / _nz(ndu[pk + 1, rk + j])
                    d = d + a[s2, j] * ndu[rk + j, pk]
                if r <= pk:
                    a[s2, k] = -a[s1, k - 1] / _nz(ndu[pk + 1, r])
                    d = d + a[s2, k] * ndu[r, pk]
                ders[k, :, r] = d
                s1, s2 = s2, s1
        fac = p
        for k in range(1, nder + 1):
            ders[k] *= fac
            fac *= p - k
    return span, ders


def _nz(x):
    return np.where(x == 0, 1.0, x)


def basis_matrix(knots: Array, p: int, n_ctrl: int, u: Array, nder: int = 0) -> Array:
    """Dense [nder+1, N, n_ctrl] B-spline basis (and derivatives)."""
    span, d = basis_ders(knots, p, n_ctrl, u, nder)
    out = np.zeros((nder + 1, len(span), n_ctrl))
    rows = np.arange(len(span))
    for j in range(p + 1):
        out[:, rows, span - p + j] = d[:, :, j]
    return out


def greville(knots: Array, p: int) -> Array:
    n = len(knots) - p - 1
    if p == 0:
        return 0.5 * (knots[:-1] + knots[1:])
    return np.array([knots[i + 1:i + p + 1].mean() for i in range(n)])


def clamped_uniform(n_ctrl: int, p: int) -> Array:
    inner = np.linspace(0, 1, n_ctrl - p + 1)[1:-1]
    return np.concatenate([np.zeros(p + 1), inner, np.ones(p + 1)])


def _distinct(knots, p):
    """Distinct interior knots and their multiplicities."""
    inner = knots[p + 1:len(knots) - p - 1]
    if len(inner) == 0:
        return np.zeros(0), np.zeros(0, int)
    vals, mult = np.unique(np.round(inner, 14), return_counts=True)
    return vals, mult


def _reprojection(knots_old, p_old, knots_new, p_new) -> Array:
    """Matrix T with c_new = T @ c_old, exact when the old spline space is
    contained in the new one (knot insertion / degree elevation): it solves
    the interpolation problem at the new Greville points."""
    n_old = len(knots_old) - p_old - 1
    n_new = len(knots_new) - p_new - 1
    g = greville(knots_new, p_new)
    g = np.clip(g, 0.0, 1.0)
    Bn = basis_matrix(knots_new, p_new, n_new, g)[0]
    Bo = basis_matrix(knots_old, p_old, n_old, g)[0]
    return np.linalg.solve(Bn, Bo)


def _elevated_knots(knots, p, t):
    vals, mult = _distinct(knots, p)
    inner = np.repeat(vals, mult + t)
    return np.concatenate([np.zeros(p + 1 + t), inner, np.ones(p + 1 + t)])


def _merge_knots(k1, k2, p):
    v1, m1 = _distinct(k1, p)
    v2, m2 = _distinct(k2, p)
    d = {}
    for v, m in list(zip(v1, m1)) + list(zip(v2, m2)):
        d[v] = max(d.get(v, 0), m)
    vals = sorted(d)
    inner = np.repeat(np.array(vals, float), [d[v] for v in vals]) if vals else np.zeros(0)
    return np.concatenate([np.zeros(p + 1), inner, np.ones(p + 1)])


def _refined_knots(knots, p, n_target):
    """Insert knots (split the longest spans) until n_target control points."""
    k = list(knots)
    n = len(k) - p - 1
    if n_target < n:
        raise ValueError(f"cannot coarsen: have {n} control points, asked {n_target}")
    while n < n_target:
        uniq = np.unique(k)
        spans = np.diff(uniq)
        i = int(np.argmax(spans))
        k.append(0.5 * (uniq[i] + uniq[i + 1]))
        k.sort()
        n += 1
    return np.array(k)


# ==========================================================================
# curves
# ==========================================================================
class NURBSCurve:
    """Rational B-spline curve in 2D (or any dimension)."""

    def __init__(self, degree: int, knots, ctrl, weights=None):
        self.p = int(degree)
        self.knots = np.asarray(knots, np.float64)
        self.ctrl = np.asarray(ctrl, np.float64)
        self.w = np.ones(len(self.ctrl)) if weights is None else np.asarray(weights, np.float64)
        if len(self.knots) != len(self.ctrl) + self.p + 1:
            raise ValueError("len(knots) must equal n_ctrl + degree + 1")

    @property
    def n(self) -> int:
        return len(self.ctrl)

    @property
    def rational(self) -> bool:
        return not np.allclose(self.w, self.w[0])

    def __call__(self, t) -> Array:
        B = basis_matrix(self.knots, self.p, self.n, np.atleast_1d(t))[0]
        Bw = B * self.w
        return (Bw @ self.ctrl) / Bw.sum(1, keepdims=True)

    def _homog(self):
        return np.concatenate([self.ctrl * self.w[:, None], self.w[:, None]], 1)

    @staticmethod
    def _from_homog(p, knots, H):
        w = H[:, -1]
        return NURBSCurve(p, knots, H[:, :-1] / w[:, None], w)

    def elevate(self, t: int = 1) -> "NURBSCurve":
        if t <= 0:
            return self
        k = _elevated_knots(self.knots, self.p, t)
        T = _reprojection(self.knots, self.p, k, self.p + t)
        return NURBSCurve._from_homog(self.p + t, k, T @ self._homog())

    def insert_knots(self, new_knots) -> "NURBSCurve":
        k = np.sort(np.concatenate([self.knots, np.asarray(new_knots, float)]))
        T = _reprojection(self.knots, self.p, k, self.p)
        return NURBSCurve._from_homog(self.p, k, T @ self._homog())

    def with_knots(self, k) -> "NURBSCurve":
        T = _reprojection(self.knots, self.p, k, self.p)
        return NURBSCurve._from_homog(self.p, np.asarray(k, float), T @ self._homog())

    def refine(self, n_ctrl: int) -> "NURBSCurve":
        return self.with_knots(_refined_knots(self.knots, self.p, n_ctrl))

    def reversed(self) -> "NURBSCurve":
        return NURBSCurve(self.p, 1.0 - self.knots[::-1], self.ctrl[::-1], self.w[::-1])

    # constructors -------------------------------------------------------
    @staticmethod
    def line(a, b, degree: int = 1) -> "NURBSCurve":
        c = NURBSCurve(1, [0, 0, 1, 1], [a, b])
        return c.elevate(degree - 1)

    @staticmethod
    def polyline(points, degree: int = 1) -> "NURBSCurve":
        """Polyline with one span per segment (parameter ~ segment index)."""
        P = np.asarray(points, float)
        n = len(P)
        k = np.concatenate([[0, 0], np.linspace(0, 1, n)[1:-1], [1, 1]])
        return NURBSCurve(1, k, P).elevate(degree - 1)

    @staticmethod
    def arc(center, radius: float, theta0: float, theta1: float) -> "NURBSCurve":
        """Exact circular arc (degree 2, rational), <= 90 degrees per span."""
        c = np.asarray(center, float)
        dth = theta1 - theta0
        nseg = max(1, int(math.ceil(abs(dth) / (0.5 * math.pi) - 1e-9)))
        h = dth / nseg
        wm = math.cos(0.5 * h)
        P, W = [], []
        for s in range(nseg):
            a0 = theta0 + s * h
            a1 = a0 + h
            p0 = c + radius * np.array([math.cos(a0), math.sin(a0)])
            p2 = c + radius * np.array([math.cos(a1), math.sin(a1)])
            pm = c + radius / wm * np.array([math.cos(a0 + 0.5 * h), math.sin(a0 + 0.5 * h)])
            if s == 0:
                P.append(p0); W.append(1.0)
            P += [pm, p2]
            W += [wm, 1.0]
        inner = np.repeat(np.arange(1, nseg) / nseg, 2)
        k = np.concatenate([[0, 0, 0], inner, [1, 1, 1]])
        return NURBSCurve(2, k, np.array(P), np.array(W))

    @staticmethod
    def interpolate(fn: Callable[[Array], Array], n_ctrl: int, degree: int = 3) -> "NURBSCurve":
        """Non-rational B-spline interpolating ``fn(t)`` at Greville points."""
        k = clamped_uniform(n_ctrl, degree)
        g = greville(k, degree)
        B = basis_matrix(k, degree, n_ctrl, g)[0]
        return NURBSCurve(degree, k, np.linalg.solve(B, np.asarray(fn(g), float)))


def make_compatible(*curves: NURBSCurve) -> list[NURBSCurve]:
    """Same degree and knot vector (exact elevation + knot insertion)."""
    p = max(c.p for c in curves)
    cs = [c.elevate(p - c.p) for c in curves]
    k = cs[0].knots
    for c in cs[1:]:
        k = _merge_knots(k, c.knots, p)
    return [c.with_knots(k) for c in cs]


# ==========================================================================
# surfaces
# ==========================================================================
Dirichlet = Union[None, float, Callable[[Array], Array], Mapping[str, object]]


class NURBSSurface:
    """Tensor product NURBS map [0,1]^2 -> R^2 (Deep NURBS domain)."""

    def __init__(self, degree, knots_u, knots_v, ctrl, weights=None):
        self.pu, self.pv = (int(d) for d in np.broadcast_to(degree, (2,)))
        self.ku = np.asarray(knots_u, np.float64)
        self.kv = np.asarray(knots_v, np.float64)
        self.ctrl = np.asarray(ctrl, np.float64)
        nu, nv = self.ctrl.shape[:2]
        self.w = np.ones((nu, nv)) if weights is None else np.asarray(weights, np.float64)
        self.validate()

    # ------------------------------------------------------------ basics
    @property
    def shape(self):
        return self.ctrl.shape[:2]

    @property
    def degree(self):
        return (self.pu, self.pv)

    def validate(self):
        nu, nv = self.shape
        if self.ctrl.shape[2] != 2:
            raise ValueError("Deep NURBS needs 2D control points [Nu, Nv, 2]")
        if len(self.ku) != nu + self.pu + 1 or len(self.kv) != nv + self.pv + 1:
            raise ValueError("Ku = Nu + pu + 1 and Kv = Nv + pv + 1 must hold")
        for k, p in ((self.ku, self.pu), (self.kv, self.pv)):
            if np.any(np.diff(k) < 0):
                raise ValueError("knot vectors must be nondecreasing")
            if not (np.allclose(k[:p + 1], k[0]) and np.allclose(k[-p - 1:], k[-1])):
                raise ValueError("knot vectors must be clamped")
            if abs(k[0]) > 1e-12 or abs(k[-1] - 1) > 1e-12:
                raise ValueError("knot vectors must span [0, 1]")
        if np.any(self.w <= 0):
            raise ValueError("weights must be positive")

    def _homog(self):
        return np.concatenate([self.ctrl * self.w[..., None], self.w[..., None]], -1)

    @staticmethod
    def _from_homog(deg, ku, kv, H):
        w = H[..., -1]
        return NURBSSurface(deg, ku, kv, H[..., :2] / w[..., None], w)

    # ------------------------------------------------------------ evaluation
    def basis(self, uv, derivatives: bool = False):
        """Rational basis R [N, Nu*Nv] (C order over (i, j)); with
        ``derivatives`` also dR/du and dR/dv."""
        uv = np.atleast_2d(np.asarray(uv, np.float64))
        nu, nv = self.shape
        Bu = basis_matrix(self.ku, self.pu, nu, uv[:, 0], 1 if derivatives else 0)
        Bv = basis_matrix(self.kv, self.pv, nv, uv[:, 1], 1 if derivatives else 0)
        w = self.w
        A = np.einsum("ni,nj,ij->nij", Bu[0], Bv[0], w)
        W = A.sum((1, 2))
        R = (A / W[:, None, None]).reshape(len(uv), -1)
        if not derivatives:
            return R
        Au = np.einsum("ni,nj,ij->nij", Bu[1], Bv[0], w)
        Av = np.einsum("ni,nj,ij->nij", Bu[0], Bv[1], w)
        Wu = Au.sum((1, 2))
        Wv = Av.sum((1, 2))
        Ru = ((Au - A * (Wu / W)[:, None, None]) / W[:, None, None]).reshape(len(uv), -1)
        Rv = ((Av - A * (Wv / W)[:, None, None]) / W[:, None, None]).reshape(len(uv), -1)
        return R, Ru, Rv

    def __call__(self, uv) -> Array:
        """Physical points [N, 2]."""
        return self.basis(uv) @ self.ctrl.reshape(-1, 2)

    evaluate = __call__

    def jacobian(self, uv) -> Array:
        """dX/d(u,v) [N, 2, 2]: J[:, a, b] = d x_a / d u_b."""
        _, Ru, Rv = self.basis(uv, derivatives=True)
        P = self.ctrl.reshape(-1, 2)
        return np.stack([Ru @ P, Rv @ P], -1)

    def evaluate_field(self, coeffs, uv) -> Array:
        """sum R_ij c_ij for coefficient arrays [Nu, Nv] or [Nu, Nv, k]."""
        c = np.asarray(coeffs, np.float64)
        R = self.basis(uv)
        return R @ c.reshape(R.shape[1], -1) if c.ndim == 3 else R @ c.reshape(-1)

    def area(self) -> float:
        """Domain area (Gauss-Legendre on each knot span)."""
        # rational Jacobians are not polynomial: use a generous Gauss order
        x, wq = np.polynomial.legendre.leggauss(2 * max(self.pu, self.pv) + 14)
        tot = 0.0
        for a0, a1 in _spans(self.ku):
            for b0, b1 in _spans(self.kv):
                U, V = np.meshgrid(0.5 * (a1 - a0) * (x + 1) + a0, 0.5 * (b1 - b0) * (x + 1) + b0, indexing="ij")
                Wq = np.outer(wq, wq) * 0.25 * (a1 - a0) * (b1 - b0)
                J = self.jacobian(np.stack([U.ravel(), V.ravel()], 1))
                tot += float(np.sum(np.abs(np.linalg.det(J)) * Wq.ravel()))
        return tot

    def min_jacobian(self, n: int = 64) -> float:
        """Smallest det J on an n x n grid (>0 means a valid, non folded map)."""
        u = np.linspace(0, 1, n)
        U, V = np.meshgrid(u, u, indexing="ij")
        return float(np.linalg.det(self.jacobian(np.stack([U.ravel(), V.ravel()], 1))).min())

    def boundary_curve(self, side: str) -> NURBSCurve:
        if side == "bottom":
            return NURBSCurve(self.pu, self.ku, self.ctrl[:, 0], self.w[:, 0])
        if side == "top":
            return NURBSCurve(self.pu, self.ku, self.ctrl[:, -1], self.w[:, -1])
        if side == "left":
            return NURBSCurve(self.pv, self.kv, self.ctrl[0, :], self.w[0, :])
        if side == "right":
            return NURBSCurve(self.pv, self.kv, self.ctrl[-1, :], self.w[-1, :])
        raise ValueError(side)

    # ------------------------------------------------------------ refinement
    def _apply(self, Tu=None, Tv=None, deg=None, ku=None, kv=None):
        H = self._homog()
        if Tu is not None:
            H = np.einsum("ai,ijk->ajk", Tu, H)
        if Tv is not None:
            H = np.einsum("bj,ijk->ibk", Tv, H)
        return NURBSSurface._from_homog(deg or self.degree, self.ku if ku is None else ku,
                                        self.kv if kv is None else kv, H)

    def elevate(self, tu: int = 0, tv: int = 0) -> "NURBSSurface":
        """Exact degree elevation by (tu, tv)."""
        ku = _elevated_knots(self.ku, self.pu, tu) if tu else self.ku
        kv = _elevated_knots(self.kv, self.pv, tv) if tv else self.kv
        Tu = _reprojection(self.ku, self.pu, ku, self.pu + tu) if tu else None
        Tv = _reprojection(self.kv, self.pv, kv, self.pv + tv) if tv else None
        return self._apply(Tu, Tv, (self.pu + tu, self.pv + tv), ku, kv)

    def with_degree(self, pu: int, pv: int) -> "NURBSSurface":
        if pu < self.pu or pv < self.pv:
            raise ValueError("degree reduction is not exact; build at the lower degree")
        return self.elevate(pu - self.pu, pv - self.pv)

    def refine(self, n_u: int | None = None, n_v: int | None = None) -> "NURBSSurface":
        """Exact h-refinement (knot insertion) to n_u x n_v control points."""
        nu, nv = self.shape
        ku = _refined_knots(self.ku, self.pu, n_u) if n_u and n_u != nu else self.ku
        kv = _refined_knots(self.kv, self.pv, n_v) if n_v and n_v != nv else self.kv
        Tu = _reprojection(self.ku, self.pu, ku, self.pu) if ku is not self.ku else None
        Tv = _reprojection(self.kv, self.pv, kv, self.pv) if kv is not self.kv else None
        return self._apply(Tu, Tv, None, ku, kv)

    def insert_knots(self, u=(), v=()) -> "NURBSSurface":
        ku = np.sort(np.concatenate([self.ku, np.asarray(u, float)]))
        kv = np.sort(np.concatenate([self.kv, np.asarray(v, float)]))
        Tu = _reprojection(self.ku, self.pu, ku, self.pu) if len(u) else None
        Tv = _reprojection(self.kv, self.pv, kv, self.pv) if len(v) else None
        return self._apply(Tu, Tv, None, ku, kv)

    # ------------------------------------------------------------ constructors
    @staticmethod
    def rectangle(x0, x1, y0, y1, degree=(2, 2), n_ctrl=(None, None)) -> "NURBSSurface":
        return NURBSSurface.quad((x0, y0), (x1, y0), (x1, y1), (x0, y1), degree, n_ctrl)

    @staticmethod
    def quad(p00, p10, p11, p01, degree=(2, 2), n_ctrl=(None, None)) -> "NURBSSurface":
        """Bilinear map of the quadrilateral p00 (u=0,v=0), p10, p11, p01."""
        C = np.array([[p00, p01], [p10, p11]], float)
        s = NURBSSurface(1, [0, 0, 1, 1], [0, 0, 1, 1], C)
        pu, pv = np.broadcast_to(degree, (2,))
        s = s.elevate(int(pu) - 1, int(pv) - 1)
        return s.refine(*n_ctrl)

    @staticmethod
    def ruled(c0: NURBSCurve, c1: NURBSCurve, degree_v: int = 1, n_v: int | None = None) -> "NURBSSurface":
        """Linear blend between curve c0 (v=0) and c1 (v=1) - exact boundaries
        (curves are made compatible exactly first)."""
        a, b = make_compatible(c0, c1)
        C = np.stack([a.ctrl, b.ctrl], 1)
        W = np.stack([a.w, b.w], 1)
        s = NURBSSurface((a.p, 1), a.knots, [0, 0, 1, 1], C, W)
        s = s.elevate(0, degree_v - 1)
        return s.refine(None, n_v)

    @staticmethod
    def annulus(r_inner: float, r_outer: float, theta0: float = 0.0, theta1: float = 0.5 * math.pi,
                center=(0.0, 0.0), degree_r: int = 1) -> "NURBSSurface":
        """Exact annular sector: u along the angle, v along the radius."""
        ci = NURBSCurve.arc(center, r_inner, theta0, theta1)
        co = NURBSCurve.arc(center, r_outer, theta0, theta1)
        return NURBSSurface.ruled(ci, co, degree_r)

    @staticmethod
    def plate_with_hole(R: float = 1.0, L: float = 4.0) -> "NURBSSurface":
        """Classic IGA benchmark: quarter of a square plate [-L,0]x[0,L] with a
        hole of radius R at the origin.  v=0 is the (exact) hole arc, v=1 the
        two outer edges, u=0 lies on y=0, u=1 on x=0."""
        inner = NURBSCurve.arc((0, 0), R, math.pi, 0.5 * math.pi)
        outer = NURBSCurve.polyline([(-L, 0), (-L, L), (0, L)], degree=2)
        return NURBSSurface.ruled(inner, outer, 1)

    @staticmethod
    def coons(bottom: NURBSCurve, right: NURBSCurve, top: NURBSCurve, left: NURBSCurve,
              tol: float = 1e-9) -> "NURBSSurface":
        """Bilinearly blended Coons patch from four non-rational boundary
        B-spline curves: bottom (v=0, along +u), top (v=1, along +u),
        left (u=0, along +v), right (u=1, along +v).  Exact boundaries."""
        for c in (bottom, right, top, left):
            if c.rational:
                raise ValueError("coons() needs non-rational curves; use ruled() "
                                 "or NURBSCurve.interpolate to approximate")
        b, t = make_compatible(bottom, top)
        l, r = make_compatible(left, right)
        P00, P10 = b.ctrl[0], b.ctrl[-1]
        P01, P11 = t.ctrl[0], t.ctrl[-1]
        for a_, b_ in ((P00, l.ctrl[0]), (P10, r.ctrl[0]), (P01, l.ctrl[-1]), (P11, r.ctrl[-1])):
            if np.linalg.norm(a_ - b_) > tol * max(1.0, np.abs(a_).max()):
                raise ValueError("boundary curves do not meet at the corners")
        gu = greville(b.knots, b.p)[:, None, None]
        gv = greville(l.knots, l.p)[None, :, None]
        B = b.ctrl[:, None, :]
        T = t.ctrl[:, None, :]
        Lc = l.ctrl[None, :, :]
        Rc = r.ctrl[None, :, :]
        C = ((1 - gv) * B + gv * T + (1 - gu) * Lc + gu * Rc
             - ((1 - gu) * (1 - gv) * P00 + gu * (1 - gv) * P10
                + (1 - gu) * gv * P01 + gu * gv * P11))
        return NURBSSurface((b.p, l.p), b.knots, l.knots, C)

    @staticmethod
    def from_map(fn: Callable[[Array], Array], n_ctrl=(8, 8), degree=(3, 3)) -> "NURBSSurface":
        """Non-rational interpolation of a map (u,v) [N,2] -> (x,y) [N,2] at
        the Greville points (approximate for curved maps)."""
        nu, nv = n_ctrl
        pu, pv = degree
        ku, kv = clamped_uniform(nu, pu), clamped_uniform(nv, pv)
        gu, gv = greville(ku, pu), greville(kv, pv)
        U, V = np.meshgrid(gu, gv, indexing="ij")
        X = np.asarray(fn(np.stack([U.ravel(), V.ravel()], 1)), float).reshape(nu, nv, 2)
        Bu = basis_matrix(ku, pu, nu, gu)[0]
        Bv = basis_matrix(kv, pv, nv, gv)[0]
        C = np.einsum("ai,bj,ijk->abk", np.linalg.inv(Bu), np.linalg.inv(Bv), X)
        return NURBSSurface(degree, ku, kv, C)

    # ------------------------------------------------------------ Deep NURBS
    def output(self, representation: str = "PARAMETRIC_SURFACE", **kwargs):
        """PARAMETRIC_SURFACE of this domain, e.g.
        ``s.output("PARAMETRIC_SURFACE", output_dim=2, dirichlet=..., n_points=4096)``;
        ``name=True`` keywords select arrays (see ``geocore.outputs``)."""
        from .outputs import output
        return output(self, representation, **kwargs)

    def admissible_coefficients(self) -> Array:
        """[Nu, Nv]: 1 inside, 0 on the four boundary rows/columns, so
        phi(u,v) = sum R_ij a_ij is > 0 inside and = 0 on the boundary."""
        a = np.ones(self.shape)
        a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = 0.0
        return a

    def lifting_coefficients(self, dirichlet: Dirichlet = None, output_dim: int = 1) -> Array:
        """[Nu, Nv, output_dim] coefficients of a NURBS function that equals
        the Dirichlet data on the boundary (interpolated at the Greville
        points of each boundary curve, exact for data in the NURBS space) and
        is Coons-blended in the interior.

        ``dirichlet``: None (zeros), a constant, a callable g(xy [N,2]) ->
        [N] or [N, output_dim], or a dict {"bottom"|"top"|"left"|"right": same}
        (sides not given are 0; corners take the value of the side processed
        last in the order bottom, top, left, right)."""
        nu, nv = self.shape
        C = np.zeros((nu, nv, output_dim))
        if dirichlet is None:
            return C
        sides = dirichlet if isinstance(dirichlet, Mapping) else \
            {s: dirichlet for s in ("bottom", "top", "left", "right")}
        for side in ("bottom", "top", "left", "right"):
            if side not in sides:
                continue
            g = sides[side]
            cur = self.boundary_curve(side)
            t = greville(cur.knots, cur.p)
            xy = cur(t)
            val = _call(g, xy, output_dim)
            B = basis_matrix(cur.knots, cur.p, cur.n, t)[0] * cur.w
            Rm = B / B.sum(1, keepdims=True)
            c = np.linalg.solve(Rm, val)
            if side == "bottom":
                C[:, 0] = c
            elif side == "top":
                C[:, -1] = c
            elif side == "left":
                C[0, :] = c
            else:
                C[-1, :] = c
        # discrete Coons blend of the boundary coefficients
        gu = greville(self.ku, self.pu)[:, None, None]
        gv = greville(self.kv, self.pv)[None, :, None]
        B_, T_ = C[:, :1], C[:, -1:]
        L_, R_ = C[:1, :], C[-1:, :]
        c00, c10, c01, c11 = C[0, 0], C[-1, 0], C[0, -1], C[-1, -1]
        I = ((1 - gv) * B_ + gv * T_ + (1 - gu) * L_ + gu * R_
             - ((1 - gu) * (1 - gv) * c00 + gu * (1 - gv) * c10
                + (1 - gu) * gv * c01 + gu * gv * c11))
        C[1:-1, 1:-1] = I[1:-1, 1:-1]
        return C


def _spans(k):
    u = np.unique(k)
    return list(zip(u[:-1], u[1:]))


def _call(g, xy, k):
    if callable(g):
        v = np.asarray(g(xy), float)
    else:
        v = np.full(len(xy), float(g)) if np.ndim(g) == 0 else np.tile(np.asarray(g, float), (len(xy), 1))
    return v.reshape(len(xy), -1) if v.ndim > 1 or k > 1 else v.reshape(len(xy), 1)


# ==========================================================================
# UV sampling + representation dict
# ==========================================================================
def sample_uv(n: int, method: str = "sobol", rng=None, include_boundary: int = 0) -> Array:
    """[n, 2] parameters in [0,1]^2.  methods: grid | random | sobol | halton.
    ``include_boundary`` extra points placed uniformly on the 4 edges."""
    rng = np.random.default_rng(rng)
    if method == "grid":
        k = max(2, int(round(math.sqrt(n))))
        u = np.linspace(0, 1, k)
        U, V = np.meshgrid(u, u, indexing="ij")
        uv = np.stack([U.ravel(), V.ravel()], 1)
    elif method == "random":
        uv = rng.random((n, 2))
    elif method in ("sobol", "halton"):
        seed = int(rng.integers(0, 2 ** 31 - 1))
        eng = qmc.Sobol(2, scramble=True, seed=seed) if method == "sobol" else qmc.Halton(2, scramble=True, seed=seed)
        uv = eng.random_base2(max(int(np.ceil(np.log2(max(n, 2)))), 1))[:n]             if method == "sobol" else eng.random(n)
    else:
        raise ValueError(method)
    if include_boundary:
        t = rng.random(include_boundary)
        side = rng.integers(0, 4, include_boundary)
        b = np.stack([np.where(side == 0, t, np.where(side == 1, 1.0, np.where(side == 2, t, 0.0))),
                      np.where(side == 0, 0.0, np.where(side == 1, t, np.where(side == 2, 1.0, t)))], 1)
        uv = np.concatenate([uv, b])
    return np.clip(uv, 0.0, 1.0)


def parametric_surface(surface: NURBSSurface, *, output_dim: int = 1,
                       dirichlet: Dirichlet = None, n_points: int = 4096, batch: int = 1,
                       method: str = "sobol", uv=None, include_boundary: int = 0,
                       units: str = "mm", rng=None) -> dict:
    """Deep NURBS PARAMETRIC_SURFACE dict (``geocore.io.save`` ready)."""
    from .normalize import Normalizer
    from .representations import _metadata, _strs
    surface.validate()
    if uv is None:
        rng = np.random.default_rng(rng)
        uv = np.stack([sample_uv(n_points, method, rng, include_boundary) for _ in range(batch)])
    uv = np.asarray(uv, float)
    if uv.ndim == 2:
        uv = uv[None]
    phys = np.stack([surface(u) for u in uv])
    out = _metadata("PARAMETRIC_SURFACE", units, Normalizer.identity())
    out.update({
        "coordinate_system": np.array("cartesian_xy_2d"),
        "degree": np.array(surface.degree, np.int64),
        "knot_vector_u": surface.ku.astype(np.float32),
        "knot_vector_v": surface.kv.astype(np.float32),
        "control_points": surface.ctrl.astype(np.float32),
        "weights": surface.w.astype(np.float32),
        "admissible_coefficients": surface.admissible_coefficients().astype(np.float32),
        "lifting_coefficients": surface.lifting_coefficients(dirichlet, output_dim).astype(np.float32),
        "parametric_coordinates": uv.astype(np.float32),
        # extra (not required): physical points of the UV samples
        "physical_coordinates": phys.astype(np.float32),
    })
    return out
