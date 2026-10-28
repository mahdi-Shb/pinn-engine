"""Parametric designs: a YAML/JSON file describes parameters, a geometry
tree whose numbers are expressions of those parameters, optional PicoGK
voxel operations, and the NN outputs to produce.

    d = gc.Design.load("bracket.yaml")
    d.parameters                      # {'length': 60.0, 'fillet': 4.0, ...}
    d.set(length=80, fillet=2)        # checked against min/max/type/constraints
    geo = d.build()                   # Implicit (exact) or Voxels
    d.outputs("out/")                 # every output listed in the file
    d.sample(100, "dataset/")         # random variants -> one folder each
    d.sweep({"fillet": [0, 2, 4]}, "sweep/")

File layout (all sections except ``shape`` are optional)::

    name: bracket
    units: mm
    parameters:                       # value or {value, min, max, type, choices, description}
      length: {value: 60, min: 40, max: 100, description: plate length}
      holes:  {value: 3, min: 1, max: 6, type: int}
    derived:                          # evaluated in order, may use parameters
      half: length / 2
    constraints:                      # must all be true
      - hole_radius < boss_radius - 1
    shapes:                           # reusable named sub-shapes (use with {ref: name})
      hole: {cylinder: {a: [0, 0, -1], b: [0, 0, 20], radius: hole_radius}}
    shape:                            # the geometry tree
      difference:
        - union: [...]
        - {ref: hole, translate: [10, 0, 0]}
    voxels:                           # optional: voxelize, then PicoGK operations
      voxel_size: 0.3
      operations: [{fillet: fillet}, {smoothen: 0.5}]
    outputs:                          # NN inputs to write (options as in geocore)
      mesh_graph: {}
      point_cloud: {n_points: 4096, mode: mixed}
      voxel_grid: {shape: [48, 64, 96], channels: [occupancy, sdf, boundary, tags]}

Expressions use parameter/derived names, numbers, ``+ - * / // % **``,
comparisons, ``and or not``, ``a if c else b``, constants ``pi e tau`` and
functions ``sin cos tan asin acos atan atan2 sqrt exp log log10 abs min max
floor ceil round radians degrees hypot clamp``.  They are evaluated by a
restricted AST walker (no ``eval``).  A string ``"$name"`` is replaced by the
value of a parameter (e.g. a choice parameter for a lattice cell type).
"""
from __future__ import annotations

import ast
import copy
import csv
import itertools
import json
import math
import operator
import os
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from . import implicit as I
from .implicit import BBox, Implicit


class DesignError(ValueError):
    pass


# ==========================================================================
# safe expressions (scalars or numpy arrays)
# ==========================================================================
def _clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def _minmax(f, g):
    def h(*a):
        if len(a) == 1:
            return f(a[0])
        out = a[0]
        for b in a[1:]:
            out = g(out, b)
        return out
    return h


FUNCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin, "acos": np.arccos,
    "atan": np.arctan, "atan2": np.arctan2, "sqrt": np.sqrt, "exp": np.exp, "log": np.log,
    "log10": np.log10, "abs": np.abs, "floor": np.floor, "ceil": np.ceil, "round": np.round,
    "radians": np.radians, "degrees": np.degrees, "hypot": np.hypot, "clamp": _clamp,
    "min": _minmax(np.min, np.minimum), "max": _minmax(np.max, np.maximum),
    "int": lambda x: np.asarray(x).astype(np.int64) if np.ndim(x) else int(x),
}
CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "True": True, "False": False}

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow}
_CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
        ast.Eq: operator.eq, ast.NotEq: operator.ne}


def _names(expr: str) -> set:
    return {n.id for n in ast.walk(ast.parse(expr, mode="eval")) if isinstance(n, ast.Name)}


def evaluate(expr, env: Mapping[str, Any]):
    """Evaluate a number, list or expression string with ``env`` variables."""
    if isinstance(expr, bool) or expr is None:
        return expr
    if isinstance(expr, (int, float, np.ndarray, np.number)):
        return expr
    if isinstance(expr, (list, tuple)):
        return [evaluate(v, env) for v in expr]
    if not isinstance(expr, str):
        raise DesignError(f"cannot evaluate {expr!r}")
    s = expr.strip()
    if s.startswith("$"):
        key = s[1:]
        if key not in env:
            raise DesignError(f"unknown parameter '{key}' in '{expr}'")
        return env[key]
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as e:
        raise DesignError(f"bad expression '{expr}': {e.msg}") from None
    return _ev(tree.body, env, expr)


def _ev(n, env, src):
    if isinstance(n, ast.Constant):
        if isinstance(n.value, (int, float, bool)):
            return n.value
        raise DesignError(f"only numbers allowed in '{src}'")
    if isinstance(n, ast.Name):
        if n.id in env:
            return env[n.id]
        if n.id in CONSTANTS:
            return CONSTANTS[n.id]
        raise DesignError(f"unknown name '{n.id}' in '{src}'")
    if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
        return _BIN[type(n.op)](_ev(n.left, env, src), _ev(n.right, env, src))
    if isinstance(n, ast.UnaryOp):
        v = _ev(n.operand, env, src)
        if isinstance(n.op, ast.USub):
            return -v
        if isinstance(n.op, ast.UAdd):
            return +v
        if isinstance(n.op, ast.Not):
            return np.logical_not(v)
    if isinstance(n, ast.Compare):
        left = _ev(n.left, env, src)
        out = True
        for op, comp in zip(n.ops, n.comparators):
            if type(op) not in _CMP:
                raise DesignError(f"operator not allowed in '{src}'")
            right = _ev(comp, env, src)
            out = np.logical_and(out, _CMP[type(op)](left, right))
            left = right
        return out
    if isinstance(n, ast.BoolOp):
        vals = [_ev(v, env, src) for v in n.values]
        f = np.logical_and if isinstance(n.op, ast.And) else np.logical_or
        out = vals[0]
        for v in vals[1:]:
            out = f(out, v)
        return out
    if isinstance(n, ast.IfExp):
        c = _ev(n.test, env, src)
        if np.ndim(c) == 0:
            return _ev(n.body, env, src) if c else _ev(n.orelse, env, src)
        return np.where(c, _ev(n.body, env, src), _ev(n.orelse, env, src))
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCTIONS and not n.keywords:
        return FUNCTIONS[n.func.id](*[_ev(a, env, src) for a in n.args])
    if isinstance(n, (ast.List, ast.Tuple)):
        return [_ev(v, env, src) for v in n.elts]
    raise DesignError(f"not allowed in expression '{src}': {ast.dump(n)[:40]}")


def _is_expr_of(s: str, env) -> bool:
    try:
        return _names(s) <= set(env) | set(CONSTANTS) | set(FUNCTIONS)
    except SyntaxError:
        return False


# ==========================================================================
# parameters
# ==========================================================================
class Parameter:
    __slots__ = ("name", "value", "min", "max", "type", "choices", "description", "unit")

    def __init__(self, name, spec):
        if not isinstance(spec, Mapping):
            spec = {"value": spec}
        if "value" not in spec:
            raise DesignError(f"parameter '{name}' needs a value")
        self.name = name
        self.min = spec.get("min")
        self.max = spec.get("max")
        self.choices = spec.get("choices")
        self.description = spec.get("description", "")
        self.unit = spec.get("unit", "")
        v = spec["value"]
        t = spec.get("type")
        if t is None:
            t = ("choice" if self.choices else "bool" if isinstance(v, bool) else
                 "int" if isinstance(v, int) and not isinstance(v, bool) else
                 "str" if isinstance(v, str) else "float")
        self.type = t
        self.value = self.coerce(v)

    def coerce(self, v):
        t = self.type
        try:
            if t == "float":
                v = float(v)
            elif t == "int":
                if float(v) != int(round(float(v))):
                    raise DesignError(f"parameter '{self.name}' must be an integer, got {v}")
                v = int(round(float(v)))
            elif t == "bool":
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
            elif t in ("choice", "str"):
                v = str(v) if not isinstance(v, (int, float)) or self.choices is None else v
        except DesignError:
            raise
        except (TypeError, ValueError):
            raise DesignError(f"parameter '{self.name}': cannot use {v!r} as {t}") from None
        if self.choices is not None and v not in self.choices:
            raise DesignError(f"parameter '{self.name}' must be one of {self.choices}, got {v!r}")
        if self.min is not None and v < self.min:
            raise DesignError(f"parameter '{self.name}' = {v} is below its minimum {self.min}")
        if self.max is not None and v > self.max:
            raise DesignError(f"parameter '{self.name}' = {v} is above its maximum {self.max}")
        return v

    def to_dict(self):
        d = {"value": self.value}
        for k in ("min", "max", "choices", "description", "unit"):
            v = getattr(self, k)
            if v not in (None, ""):
                d[k] = v
        if self.type not in ("float",):
            d["type"] = self.type
        return d

    @property
    def sampleable(self) -> bool:
        return self.choices is not None or self.type == "bool" or (
            self.min is not None and self.max is not None)


# ==========================================================================
# geometry tree
# ==========================================================================
_PRIMS = ("sphere", "box", "cylinder", "cone", "beam", "capsule", "torus", "plane", "gyroid",
          "schwarzp", "diamond", "cubic_strut_lattice", "lattice", "mesh")
_OPS = ("union", "intersection", "difference", "complement", "smooth_union",
        "smooth_difference", "smooth_intersection", "linear_array", "grid_array",
        "polar_array", "ref")
NODE_TYPES = _PRIMS + _OPS
MODIFIERS = ("tag", "enabled", "offset", "shell", "scale", "rotate", "translate", "trim")


class _Builder:
    def __init__(self, design: "Design", env):
        self.d = design
        self.env = env
        self._ref_stack = []

    # ---- value helpers
    def num(self, v, what, env=None):
        x = evaluate(v, env or self.env)
        if isinstance(x, list) or np.ndim(x) != 0:
            raise DesignError(f"{what}: expected a number, got {x!r}")
        return float(x)

    def vec(self, v, what, env=None, n=3):
        env = env or self.env
        x = evaluate(v, env)
        if not isinstance(x, (list, tuple, np.ndarray)):
            raise DesignError(f"{what}: expected a list of {n} numbers, got {x!r}")
        x = [float(evaluate(c, env)) if not isinstance(c, (int, float)) else float(c) for c in x]
        if len(x) != n:
            raise DesignError(f"{what}: expected {n} numbers, got {len(x)}")
        return np.array(x)

    def text(self, v, env=None):
        if isinstance(v, str) and v.strip().startswith("$"):
            return str(evaluate(v, env or self.env))
        return str(v)

    # ---- nodes
    def node(self, spec, env=None, path="shape") -> Optional[Implicit]:
        env = env or self.env
        if isinstance(spec, list):
            return self.node({"union": spec}, env, path)
        if not isinstance(spec, Mapping):
            raise DesignError(f"{path}: a shape must be a mapping, got {spec!r}")
        kinds = [k for k in spec if k in NODE_TYPES]
        bad = [k for k in spec if k not in NODE_TYPES and k not in MODIFIERS]
        if bad:
            raise DesignError(f"{path}: unknown key(s) {bad}; shape types: {list(NODE_TYPES)}, "
                              f"modifiers: {list(MODIFIERS)}")
        if len(kinds) != 1:
            raise DesignError(f"{path}: needs exactly one shape type, got {kinds or 'none'}")
        if "enabled" in spec and not bool(evaluate(spec["enabled"], env)):
            return None
        kind = kinds[0]
        args = spec[kind]
        here = f"{path}.{kind}"
        s = self.primitive(kind, args, env, here) if kind in _PRIMS else self.op(kind, args, env, here)
        if s is None:
            return None
        return self.modify(s, spec, env, here)

    def _args(self, args, path, allowed, required=()):
        if not isinstance(args, Mapping):
            raise DesignError(f"{path}: arguments must be a mapping")
        bad = set(args) - set(allowed)
        if bad:
            raise DesignError(f"{path}: unknown argument(s) {sorted(bad)}; allowed: {list(allowed)}")
        miss = [k for k in required if k not in args]
        if miss:
            raise DesignError(f"{path}: missing {miss}")
        return args

    def primitive(self, kind, a, env, path):
        n, v = (lambda k: self.num(a[k], f"{path}.{k}", env)), (lambda k: self.vec(a[k], f"{path}.{k}", env))
        if kind == "sphere":
            self._args(a, path, ("center", "radius"), ("radius",))
            return I.Sphere(v("center") if "center" in a else (0, 0, 0), n("radius"))
        if kind == "box":
            self._args(a, path, ("center", "size", "min", "max", "rounding"))
            r = n("rounding") if "rounding" in a else 0.0
            if "min" in a:
                return I.Box.from_bounds(v("min"), v("max"), r)
            if "size" not in a:
                raise DesignError(f"{path}: give size (+center) or min/max")
            return I.Box(v("center") if "center" in a else (0, 0, 0), v("size"), r)
        if kind == "cylinder":
            self._args(a, path, ("a", "b", "radius", "center", "axis", "height"), ("radius",))
            if "a" in a:
                return I.Cylinder(v("a"), v("b"), n("radius"))
            ax = self._axis(a.get("axis", "z"), env, path)
            c = v("center") if "center" in a else np.zeros(3)
            hh = 0.5 * n("height")
            return I.Cylinder(c - hh * ax, c + hh * ax, n("radius"))
        if kind == "cone":
            self._args(a, path, ("a", "b", "ra", "rb"), ("a", "b", "ra", "rb"))
            return I.Cone(v("a"), v("b"), n("ra"), n("rb"))
        if kind in ("beam", "capsule"):
            self._args(a, path, ("a", "b", "ra", "rb", "radius"), ("a", "b"))
            ra = n("radius") if "radius" in a else n("ra")
            rb = n("rb") if "rb" in a else ra
            return I.Beam(v("a"), v("b"), ra, rb)
        if kind == "torus":
            self._args(a, path, ("center", "major", "minor"), ("major", "minor"))
            return I.Torus(v("center") if "center" in a else (0, 0, 0), n("major"), n("minor"))
        if kind == "plane":
            self._args(a, path, ("point", "normal"), ("normal",))
            return I.Plane(v("point") if "point" in a else (0, 0, 0), v("normal"))
        if kind in ("gyroid", "schwarzp", "diamond"):
            self._args(a, path, ("period", "wall", "solid", "level", "origin"), ("period", "wall"))
            cls = {"gyroid": I.Gyroid, "schwarzp": I.SchwarzP, "diamond": I.Diamond}[kind]
            return cls(n("period"), n("wall"), bool(evaluate(a.get("solid", False), env)),
                       n("level") if "level" in a else 0.0, v("origin") if "origin" in a else (0, 0, 0))
        if kind == "cubic_strut_lattice":
            self._args(a, path, ("center", "size", "cell", "radius"), ("size", "cell", "radius"))
            return I.CubicStrutLattice(v("center") if "center" in a else (0, 0, 0), v("size"),
                                       n("cell"), n("radius"))
        if kind == "lattice":
            from .lattice import Lattice
            self._args(a, path, ("cell", "min", "max", "cell_size", "radius"),
                       ("min", "max", "cell_size", "radius"))
            cs = evaluate(a["cell_size"], env)
            cs = self.vec(cs, f"{path}.cell_size", env) if isinstance(cs, list) else float(cs)
            rad = a["radius"]
            if isinstance(rad, str) and _names(rad) & {"x", "y", "z"}:
                expr = rad

                def radius(p, expr=expr, env=dict(env)):
                    e = dict(env, x=p[:, 0], y=p[:, 1], z=p[:, 2])
                    return np.broadcast_to(np.asarray(evaluate(expr, e), float), (len(p),))
            else:
                radius = n("radius")
            lat = Lattice.from_cells(self.text(a.get("cell", "bcc"), env), v("min"), v("max"), cs, radius)
            return lat.as_implicit()
        if kind == "mesh":
            from .mesh import Mesh
            self._args(a, path, ("file", "scale", "translate"), ("file",))
            f = self.text(a["file"], env)
            f = f if os.path.isabs(f) else os.path.join(self.d.base_dir, f)
            if not os.path.isfile(f):
                raise DesignError(f"{path}: mesh file not found: {f}")
            m = Mesh.load(f)
            if "scale" in a:
                m = m.scaled(n("scale"))
            if "translate" in a:
                m = m.translated(v("translate"))
            return m.as_implicit()
        raise DesignError(f"{path}: unknown primitive {kind}")

    def _axis(self, ax, env, path):
        if isinstance(ax, str) and ax.lower() in ("x", "y", "z"):
            return np.eye(3)["xyz".index(ax.lower())]
        v = self.vec(ax, f"{path}.axis", env)
        return v / np.linalg.norm(v)

    def _children(self, items, env, path):
        if not isinstance(items, list):
            raise DesignError(f"{path}: expects a list of shapes")
        out = [self.node(c, env, f"{path}[{i}]") for i, c in enumerate(items)]
        return [c for c in out if c is not None]

    def op(self, kind, a, env, path):
        if kind in ("union", "intersection"):
            ch = self._children(a, env, path)
            if not ch:
                return None
            return ch[0] if len(ch) == 1 else (I.Union(*ch) if kind == "union" else I.Intersection(*ch))
        if kind == "difference":
            ch = self._children(a, env, path)
            if not ch:
                return None
            out = ch[0]
            for c in ch[1:]:
                out = I.Difference(out, c)
            return out
        if kind == "complement":
            c = self.node(a, env, path)
            return None if c is None else I.Complement(c)
        if kind.startswith("smooth_"):
            self._args(a, path, ("k", "shapes"), ("k", "shapes"))
            ch = self._children(a["shapes"], env, f"{path}.shapes")
            if not ch:
                return None
            k = self.num(a["k"], f"{path}.k", env)
            cls = {"smooth_union": I.SmoothUnion, "smooth_difference": I.SmoothDifference,
                   "smooth_intersection": I.SmoothIntersection}[kind]
            out = ch[0]
            for c in ch[1:]:
                out = cls(out, c, k)
            return out
        if kind == "linear_array":
            self._args(a, path, ("shape", "count", "step", "index"), ("shape", "count", "step"))
            return self._array(a, env, path, lambda i: self.vec(a["step"], f"{path}.step",
                                                                dict(env, **{a.get("index", "i"): i})) * i)
        if kind == "grid_array":
            self._args(a, path, ("shape", "counts", "spacing"), ("shape", "counts", "spacing"))
            cnt = [int(self.num(c, f"{path}.counts", env)) for c in evaluate(a["counts"], env)]
            sp = self.vec(a["spacing"], f"{path}.spacing", env)
            parts = []
            for i, j, k in itertools.product(*[range(c) for c in cnt]):
                e = dict(env, i=i, j=j, k=k)
                s = self.node(a["shape"], e, f"{path}.shape")
                if s is not None:
                    parts.append(s.translate(sp * np.array([i, j, k])))
            return I.Union(*parts) if len(parts) > 1 else (parts[0] if parts else None)
        if kind == "polar_array":
            self._args(a, path, ("shape", "count", "axis", "center", "angle", "index"), ("shape", "count"))
            cnt = int(self.num(a["count"], f"{path}.count", env))
            ax = self._axis(a.get("axis", "z"), env, path)
            c = self.vec(a["center"], f"{path}.center", env) if "center" in a else np.zeros(3)
            total = self.num(a.get("angle", 360), f"{path}.angle", env)
            stepdeg = total / cnt if abs(total - 360) < 1e-9 else total / max(cnt - 1, 1)
            idx = a.get("index", "i")
            parts = []
            for i in range(cnt):
                s = self.node(a["shape"], dict(env, **{idx: i}), f"{path}.shape")
                if s is not None:
                    parts.append(s.rotate(ax, math.radians(stepdeg * i), center=c) if i else s)
            return I.Union(*parts) if len(parts) > 1 else (parts[0] if parts else None)
        if kind == "ref":
            name = self.text(a, env)
            if name not in self.d.shapes:
                raise DesignError(f"{path}: unknown shape '{name}' (defined: {list(self.d.shapes)})")
            if name in self._ref_stack:
                raise DesignError(f"{path}: recursive ref '{name}'")
            self._ref_stack.append(name)
            try:
                return self.node(self.d.shapes[name], env, f"shapes.{name}")
            finally:
                self._ref_stack.pop()
        raise DesignError(f"{path}: unknown operation {kind}")

    def _array(self, a, env, path, offset_of):
        cnt = int(self.num(a["count"], f"{path}.count", env))
        idx = a.get("index", "i")
        parts = []
        for i in range(cnt):
            s = self.node(a["shape"], dict(env, **{idx: i}), f"{path}.shape")
            if s is not None:
                off = offset_of(i)
                parts.append(s.translate(off) if np.any(off) else s)
        return I.Union(*parts) if len(parts) > 1 else (parts[0] if parts else None)

    def modify(self, s, spec, env, path):
        if "offset" in spec:
            s = s.offset(self.num(spec["offset"], f"{path}.offset", env))
        if "shell" in spec:
            s = s.shell(self.num(spec["shell"], f"{path}.shell", env))
        if "trim" in spec:
            t = spec["trim"]
            s = s.trim(self.vec(t["min"], f"{path}.trim.min", env), self.vec(t["max"], f"{path}.trim.max", env))
        if "scale" in spec:
            s = s.scale(self.num(spec["scale"], f"{path}.scale", env))
        if "rotate" in spec:
            r = spec["rotate"]
            if not isinstance(r, Mapping) or "angle" not in r:
                raise DesignError(f"{path}.rotate: use {{axis: z, angle: degrees, center: [..]}}")
            s = s.rotate(self._axis(r.get("axis", "z"), env, path),
                         math.radians(self.num(r["angle"], f"{path}.rotate.angle", env)),
                         center=self.vec(r["center"], f"{path}.rotate.center", env) if "center" in r else (0, 0, 0))
        if "translate" in spec:
            s = s.translate(self.vec(spec["translate"], f"{path}.translate", env))
        if "tag" in spec:
            s = s.tagged(self.text(spec["tag"], env))
        return s


# ==========================================================================
# Design
# ==========================================================================
_SECTIONS = ("name", "description", "units", "parameters", "derived", "constraints", "shapes",
             "shape", "voxels", "outputs")


class Design:
    """A parametric design (see module docstring for the file format)."""

    def __init__(self, data: Mapping, base_dir: str = "."):
        if not isinstance(data, Mapping):
            raise DesignError("a design must be a mapping")
        bad = set(data) - set(_SECTIONS)
        if bad:
            raise DesignError(f"unknown section(s) {sorted(bad)}; allowed: {list(_SECTIONS)}")
        if "shape" not in data:
            raise DesignError("a design needs a 'shape' section")
        self.data = copy.deepcopy(dict(data))
        self.base_dir = base_dir
        self.name = str(self.data.get("name", "design"))
        self.units = str(self.data.get("units", "mm"))
        self._params = {k: Parameter(k, v) for k, v in (self.data.get("parameters") or {}).items()}
        self.shapes = dict(self.data.get("shapes") or {})
        self._cache = None
        self.values()           # fail early on bad derived values / constraints

    # ------------------------------------------------------------ files
    @staticmethod
    def load(path: str) -> "Design":
        with open(path, encoding="utf8") as f:
            text = f.read()
        return Design.from_string(text, base_dir=os.path.dirname(os.path.abspath(path)),
                                  fmt="json" if path.lower().endswith(".json") else "yaml")

    @staticmethod
    def from_string(text: str, base_dir: str = ".", fmt: str = "yaml") -> "Design":
        if fmt == "json":
            data = json.loads(text)
        else:
            try:
                import yaml
            except ImportError as e:
                raise DesignError("YAML designs need PyYAML (pip install pyyaml); "
                                  "or write the design as .json") from e
            data = yaml.safe_load(text)
        return Design(data, base_dir)

    def to_dict(self) -> dict:
        d = copy.deepcopy(self.data)
        if self._params:
            d["parameters"] = {k: p.to_dict() for k, p in self._params.items()}
        return d

    def save(self, path: str):
        d = self.to_dict()
        with open(path, "w", encoding="utf8") as f:
            if path.lower().endswith(".json"):
                json.dump(d, f, indent=2)
            else:
                import yaml
                yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True)
        return path

    # ------------------------------------------------------------ parameters
    @property
    def parameters(self) -> dict:
        """Current parameter values."""
        return {k: p.value for k, p in self._params.items()}

    @property
    def parameter_info(self) -> dict:
        return {k: p.to_dict() for k, p in self._params.items()}

    def set(self, **values) -> "Design":
        """Change parameters (checked against type, min/max, choices and the
        design's constraints).  Returns self."""
        old = self.parameters
        for k in values:
            if k not in self._params:
                raise DesignError(f"unknown parameter '{k}'; parameters: {list(self._params)}")
        try:
            for k, v in values.items():
                self._params[k].value = self._params[k].coerce(v)
            self.values()
        except DesignError:
            for k, v in old.items():          # all-or-nothing
                self._params[k].value = v
            raise
        self._cache = None
        return self

    def with_params(self, **values) -> "Design":
        """A copy with changed parameters (this design is unchanged)."""
        d = Design(self.to_dict(), self.base_dir)
        return d.set(**values)

    def values(self, check: bool = True) -> dict:
        """Parameters + derived values (+ constraint check)."""
        env = dict(self.parameters)
        for k, expr in (self.data.get("derived") or {}).items():
            if k in env:
                raise DesignError(f"derived value '{k}' shadows a parameter")
            env[k] = evaluate(expr, env)
        if check:
            bad = self.violations(env)
            if bad:
                raise DesignError("constraint(s) violated: " + "; ".join(bad))
        return env

    def violations(self, env=None) -> list:
        env = env if env is not None else self.values(check=False)
        out = []
        for c in self.data.get("constraints") or []:
            if not bool(evaluate(c, env)):
                out.append(str(c))
        return out

    # ------------------------------------------------------------ geometry
    def build_shape(self) -> Implicit:
        """The tagged implicit geometry tree (exact, before voxel operations)."""
        b = _Builder(self, self.values())
        s = b.node(self.data["shape"])
        if s is None:
            raise DesignError("the shape is empty (every part disabled?)")
        return s

    def build(self):
        """The final geometry: the implicit, or ``Voxels`` when the design has
        a ``voxels`` section (then its operations are applied in order)."""
        if self._cache is not None:
            return self._cache
        shape = self.build_shape()
        vs = self.data.get("voxels")
        if vs is None:
            self._cache = shape
            return shape
        from .voxels import Library, Voxels
        env = self.values()
        b = _Builder(self, env)
        vs = dict(vs)
        bad = set(vs) - {"voxel_size", "backend", "operations"}
        if bad:
            raise DesignError(f"voxels: unknown key(s) {sorted(bad)}")
        lib = Library(b.num(vs.get("voxel_size", 0.5), "voxels.voxel_size"), vs.get("backend", "auto"))
        vox = Voxels(shape, lib=lib)
        for i, op in enumerate(vs.get("operations") or []):
            if not isinstance(op, Mapping) or len(op) != 1:
                raise DesignError(f"voxels.operations[{i}]: use one operation per item, e.g. {{fillet: 2}}")
            (k, a), = op.items()
            path = f"voxels.operations[{i}].{k}"
            if k in ("fillet", "smoothen", "offset"):
                getattr(vox, k)(b.num(a, path))
            elif k == "double_offset":
                vox.double_offset(*[b.num(x, path) for x in a])
            elif k == "shell":
                vox = vox.shell(b.num(a, path))
            elif k == "trim":
                vox.trim(b.vec(a["min"], path + ".min"), b.vec(a["max"], path + ".max"))
            elif k in ("add", "subtract", "intersect"):
                s = b.node(a, path=path)
                if s is None:
                    continue
                if k == "intersect":
                    vox.intersect_implicit(s)
                else:
                    other = Voxels(s, lib=lib)
                    vox = vox + other if k == "add" else vox - other
            else:
                raise DesignError(f"{path}: unknown voxel operation (fillet, smoothen, offset, "
                                  f"double_offset, shell, trim, add, subtract, intersect)")
        self._cache = vox
        self._shape_cache = shape
        return vox

    @property
    def tag_names(self) -> list:
        """Every surface tag the design declares (also tags of parts that are
        currently disabled), in order of appearance - fixed for all variants."""
        env = self.values(check=False)
        out = []

        def walk(x):
            if isinstance(x, Mapping):
                if "tag" in x:
                    t = x["tag"]
                    name = str(evaluate(t, env)) if isinstance(t, str) and t.strip().startswith("$") else str(t)
                    if name not in out:
                        out.append(name)
                for k, v in x.items():
                    if k == "ref" and isinstance(v, str) and v in self.shapes:
                        walk(self.shapes[v])
                    elif k != "tag":
                        walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(self.data["shape"])
        for op in (self.data.get("voxels") or {}).get("operations") or []:
            walk(op)
        return out

    def geometry(self, resolution: float | None = None):
        """``Geometry`` adapter of the final geometry.  Boundary tags come
        from the tagged shape tree (also after voxel operations) and use the
        fixed ``tag_names`` order, so every variant has the same tag
        channels."""
        from .sampling import Geometry
        g = self.build()
        order = self.tag_names or None
        if isinstance(g, Implicit):
            return Geometry(g, resolution, tag_order=order)
        return Geometry(g, resolution, tag_source=self.build_shape(), tag_order=order)

    def properties(self) -> dict:
        m = self.geometry().surface_mesh()
        return {"volume_mm3": float(m.volume()), "surface_area_mm2": float(m.area()),
                "bounds_mm": m.bounds.as_array().tolist(), "watertight": bool(m.is_watertight()),
                "tags": self.geometry().tag_names}

    # ------------------------------------------------------------ outputs
    def output(self, representation: str = "MESH_GRAPH", **kwargs):
        """One NN input of the design with its current parameters, e.g.
        ``d.output("VOXEL_GRID", shape=64)`` (see ``geocore.outputs``).
        ``PARAMETRIC_SURFACE`` uses the design's ``outputs.parametric_surface``."""
        from .outputs import output
        return output(self, representation, **kwargs)

    def outputs(self, out_dir: str | None = None, only: Sequence[str] | None = None,
                rng=None) -> dict:
        """Build the outputs listed in the ``outputs`` section (default: all
        four NN files) and save them into ``out_dir``.  Returns
        {output name: dict (NN representations / collocation) or file path}."""
        from . import io, representations as R
        from .nurbs import NURBSSurface, parametric_surface
        from .sampling import collocation
        spec = self.data.get("outputs")
        if spec is None:
            spec = {"mesh_graph": {}, "point_cloud": {}, "voxel_grid": {}}
        env = self.values()
        rng = np.random.default_rng(rng)
        names = list(spec) if only is None else list(only)
        unknown = set(names) - {"mesh_graph", "point_cloud", "voxel_grid", "parametric_surface",
                                "collocation", "photo", "stl", "summary"}
        if unknown:
            raise DesignError(f"outputs: unknown {sorted(unknown)}")
        g = self.geometry()
        res = {}
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        for name in names:
            opts = {k: _auto(v, env) for k, v in (spec.get(name) or {}).items()} \
                if isinstance(spec.get(name), Mapping) else {}
            if name == "mesh_graph":
                d = R.mesh_graph(g, rng=rng, **opts)
            elif name == "point_cloud":
                d = R.point_cloud(g, rng=rng, **opts)
            elif name == "voxel_grid":
                if "shape" not in opts and "spacing" not in opts:
                    opts["shape"] = 64
                d = R.voxel_grid(g, **opts)
            elif name == "parametric_surface":
                dom = opts.pop("domain", {"rectangle": {"x0": 0, "x1": 1, "y0": 0, "y1": 1}})
                (dk, da), = dom.items()
                da = {k: _auto(v, env) for k, v in (da or {}).items()}
                if dk == "annulus":
                    for key in ("theta0", "theta1"):
                        if key in da:
                            da[key] = math.radians(da[key])
                surf = getattr(NURBSSurface, dk)(**da)
                deg = opts.pop("degree", None)
                if deg is not None:
                    surf = surf.with_degree(*[int(x) for x in deg])
                nc = opts.pop("n_ctrl", None)
                if nc is not None:
                    surf = surf.refine(*[int(x) for x in nc])
                if "dirichlet" in opts:
                    opts["dirichlet"] = _dirichlet(opts["dirichlet"], env)
                d = parametric_surface(surf, rng=rng, units=self.units, **opts)
            elif name == "collocation":
                if "band" in opts:
                    opts["band"] = tuple(opts["band"])
                d = collocation(g, rng=rng, **opts)
                if out_dir:
                    np.savez_compressed(os.path.join(out_dir, "pinn_collocation.npz"), **d)
                res[name] = d
                continue
            elif name == "photo":
                from .render import render_mesh
                if out_dir:
                    p = os.path.join(out_dir, "object.png")
                    render_mesh(g.surface_mesh(), p, title=f"{self.name}", **opts)
                    res[name] = p
                continue
            elif name == "stl":
                if out_dir:
                    p = os.path.join(out_dir, "object.stl")
                    g.surface_mesh().save_stl(p)
                    res[name] = p
                continue
            elif name == "summary":
                res[name] = self.properties()
                continue
            if out_dir:
                io.save(out_dir, d)
            res[name] = d
        if out_dir:
            with open(os.path.join(out_dir, "parameters.json"), "w") as f:
                json.dump({"design": self.name, "parameters": self.parameters,
                           "derived": {k: _jsonable(v) for k, v in env.items() if k not in self._params},
                           **({"properties": res["summary"]} if "summary" in res else {})}, f, indent=2)
            self.save(os.path.join(out_dir, "design.yaml"))
        return res

    # ------------------------------------------------------------ variants
    def sample(self, n: int, out_dir: str | None = None, *, rng=None, method: str = "sobol",
               names: Sequence[str] | None = None, only: Sequence[str] | None = None,
               max_tries: int = 100) -> list:
        """``n`` random designs: every parameter with min/max (or choices /
        bool) is drawn (Sobol or random), integers are rounded, designs that
        break a constraint are redrawn.  With ``out_dir`` each variant's
        outputs go to ``out_dir/sample_0000`` ... and ``parameters.csv`` lists
        them all.  Returns the parameter dicts."""
        from scipy.stats import qmc
        rng = np.random.default_rng(rng)
        free = [k for k, p in self._params.items() if p.sampleable and (names is None or k in names)]
        if not free:
            raise DesignError("no parameter has min/max or choices to sample")
        if method == "sobol":
            eng = qmc.Sobol(len(free), scramble=True, seed=int(rng.integers(2 ** 31 - 1)))
            draw = lambda m: eng.random(m)
        else:
            draw = lambda m: rng.random((m, len(free)))
        rows = []
        tries = 0
        while len(rows) < n and tries < max_tries:
            tries += 1
            U = draw(max(2 * (n - len(rows)), 8))
            for u in U:
                vals = {k: self._from_unit(self._params[k], x) for k, x in zip(free, u)}
                try:
                    self.with_params(**vals)
                except DesignError:
                    continue
                rows.append(vals)
                if len(rows) == n:
                    break
        if len(rows) < n:
            raise DesignError(f"only {len(rows)} of {n} samples satisfy the constraints")
        return self._run_variants(rows, out_dir, only)

    def sweep(self, grid: Mapping[str, Sequence], out_dir: str | None = None,
              only: Sequence[str] | None = None) -> list:
        """Full factorial sweep over the given parameter values (variants that
        break a constraint are skipped)."""
        keys = list(grid)
        rows = []
        for combo in itertools.product(*[list(grid[k]) for k in keys]):
            vals = dict(zip(keys, combo))
            try:
                self.with_params(**vals)
            except DesignError:
                continue
            rows.append(vals)
        return self._run_variants(rows, out_dir, only)

    @staticmethod
    def _from_unit(p: Parameter, u: float):
        if p.choices is not None:
            return p.choices[min(int(u * len(p.choices)), len(p.choices) - 1)]
        if p.type == "bool":
            return bool(u >= 0.5)
        v = p.min + u * (p.max - p.min)
        return int(round(v)) if p.type == "int" else float(v)

    def _run_variants(self, rows, out_dir, only):
        if out_dir is None:
            return rows
        os.makedirs(out_dir, exist_ok=True)
        index = []
        for i, vals in enumerate(rows):
            d = self.with_params(**vals)
            sub = os.path.join(out_dir, f"sample_{i:04d}")
            d.outputs(sub, only=only)
            index.append({"sample": f"sample_{i:04d}", **d.parameters})
        with open(os.path.join(out_dir, "parameters.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(index[0]))
            w.writeheader()
            w.writerows(index)
        return rows

    def __repr__(self):
        return f"Design('{self.name}', parameters={self.parameters})"

    def describe(self) -> str:
        lines = [f"design: {self.name}  ({self.units})"]
        for k, p in self._params.items():
            rng_ = f"[{p.min}, {p.max}]" if p.min is not None or p.max is not None else ""
            ch = f" choices={p.choices}" if p.choices else ""
            lines.append(f"  {k:20s} = {p.value!r:<10} {p.type:6s} {rng_}{ch}  {p.description}")
        der = self.data.get("derived") or {}
        if der:
            env = self.values(check=False)
            lines.append("derived:")
            for k in der:
                lines.append(f"  {k:20s} = {_jsonable(env[k])}")
        cons = self.data.get("constraints") or []
        if cons:
            lines.append("constraints: " + "; ".join(map(str, cons)))
        lines.append("outputs: " + ", ".join(self.data.get("outputs") or
                                             ["mesh_graph", "point_cloud", "voxel_grid"]))
        return "\n".join(lines)


def _auto(v, env):
    """Output options: strings that are expressions of known names are
    evaluated, other strings (e.g. 'mixed', 'occupancy') stay literal."""
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("$") or _is_expr_of(s, env):
            return evaluate(s, env)
        return v
    if isinstance(v, list):
        return [_auto(x, env) for x in v]
    if isinstance(v, Mapping):
        return {k: _auto(x, env) for k, x in v.items()}
    return v


def _dirichlet(spec, env):
    """Dirichlet data: number, expression string in x, y (and parameters), or
    a list of them (one per output component), or {side: ...}."""
    if isinstance(spec, Mapping) and set(spec) <= {"bottom", "top", "left", "right"}:
        return {k: _dirichlet(v, env) for k, v in spec.items()}
    comps = spec if isinstance(spec, list) else [spec]

    def g(xy, comps=comps):
        e = dict(env, x=xy[:, 0], y=xy[:, 1])
        cols = [np.broadcast_to(np.asarray(evaluate(c, e), float), (len(xy),)) for c in comps]
        return np.stack(cols, 1)
    return g


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    return v
