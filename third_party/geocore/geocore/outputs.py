"""``part.output(...)``: one method that turns any geometry into any
ModelInput representation.

    part_mg = part.output("MESH_GRAPH")                          # every array of the schema
    part_mg = part.output("MESH_GRAPH", coordinates=True,
                          node_features=True)                    # only these arrays
    part_pc = part.output("POINT_CLOUD", n_points=4096, mode="mixed")
    part_vg = vox.output("VOXEL_GRID", shape=64, point_mask=False)
    part_mg.save("out/")                                         # -> out/mesh_graph.npz
    t = part_mg.to_torch("cuda")                                 # dict of torch tensors

``part`` can be anything geocore knows: an implicit shape / CSG tree, a
``Voxels`` field, a ``Mesh`` (e.g. a loaded STL), a ``Lattice``, a
parametric ``Design`` or a ``Geometry``.  2D ``NURBSSurface`` domains give
``PARAMETRIC_SURFACE``.

Keyword arguments are of two kinds:

* array names of the representation (``coordinates=True``, ``batch=False``
  ...) choose which arrays are returned;
* everything else is passed to the builder (``n_points``, ``mode``,
  ``shape``, ``channels``, ``kind``, ``normalization`` ...).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

COMMON = ("schema_version", "representation", "coordinate_system", "units", "normalization_id")
EXTRAS = ("normalization_center", "normalization_scale", "faces", "physical_coordinates")

#: arrays of each representation, in ModelInput.md order
SCHEMA = {
    "MESH_GRAPH": ("coordinates", "node_features", "edge_index", "edge_features", "batch",
                   "node_feature_names", "edge_feature_names"),
    "POINT_CLOUD": ("coordinates", "point_features", "point_mask", "point_feature_names"),
    "VOXEL_GRID": ("voxel_fields", "channel_names", "origin", "spacing", "bounds"),
    "PARAMETRIC_SURFACE": ("degree", "knot_vector_u", "knot_vector_v", "control_points", "weights",
                           "admissible_coefficients", "lifting_coefficients", "parametric_coordinates"),
    "PINN_COLLOCATION": ("x_interior", "sdf_interior", "x_boundary", "normal_boundary",
                         "tag_boundary", "tag_names", "x_band", "sdf_band"),
}
REQUIRED = {
    "MESH_GRAPH": ("node_features", "edge_index", "edge_features"),
    "POINT_CLOUD": ("coordinates",),
    "VOXEL_GRID": ("voxel_fields",),
    "PARAMETRIC_SURFACE": SCHEMA["PARAMETRIC_SURFACE"],
    "PINN_COLLOCATION": (),
}
ALIASES = {
    "MESH_GRAPH": "MESH_GRAPH", "MESHGRAPH": "MESH_GRAPH", "GRAPH": "MESH_GRAPH", "MESH": "MESH_GRAPH",
    "MGN": "MESH_GRAPH", "MESHGRAPHNET": "MESH_GRAPH",
    "POINT_CLOUD": "POINT_CLOUD", "POINTCLOUD": "POINT_CLOUD", "POINTS": "POINT_CLOUD",
    "POINTNET": "POINT_CLOUD", "PC": "POINT_CLOUD",
    "VOXEL_GRID": "VOXEL_GRID", "VOXELGRID": "VOXEL_GRID", "VOXEL": "VOXEL_GRID", "VOXELS": "VOXEL_GRID",
    "GRID": "VOXEL_GRID", "FNO": "VOXEL_GRID", "FNO3D": "VOXEL_GRID",
    "PARAMETRIC_SURFACE": "PARAMETRIC_SURFACE", "NURBS": "PARAMETRIC_SURFACE",
    "DEEP_NURBS": "PARAMETRIC_SURFACE", "PARAMETRIC": "PARAMETRIC_SURFACE",
    "PINN_COLLOCATION": "PINN_COLLOCATION", "COLLOCATION": "PINN_COLLOCATION", "PINN": "PINN_COLLOCATION",
}
FILE_NAMES = {"MESH_GRAPH": "mesh_graph.npz", "POINT_CLOUD": "point_cloud.npz",
              "VOXEL_GRID": "voxel_grid.npz", "PARAMETRIC_SURFACE": "parametric_surface.npz",
              "PINN_COLLOCATION": "pinn_collocation.npz"}


def representation_name(rep: str) -> str:
    key = str(rep).strip().upper().replace("-", "_").replace(" ", "_")
    if key not in ALIASES:
        raise ValueError(f"unknown representation '{rep}'; use one of {list(SCHEMA)}")
    return ALIASES[key]


# ==========================================================================
class ModelInput(dict):
    """The arrays of one representation (a ``dict`` of numpy arrays).

    Extra conveniences: attribute access (``mg.coordinates``),
    ``.representation``, ``.save(path)``, ``.to_torch(device)``,
    ``.validate()``, ``.summary()``."""

    def __init__(self, data: Mapping, representation: str):
        super().__init__(data)
        self.representation = representation

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(f"'{k}' is not in this {self.representation} output "
                                 f"(it has: {list(self)})") from None

    def __dir__(self):
        return list(super().__dir__()) + list(self)

    # ---------------------------------------------------------------- info
    def summary(self) -> str:
        lines = [f"{self.representation} ({len(self)} arrays)"]
        for k, v in self.items():
            a = np.asarray(v)
            if a.dtype.kind == "U":
                val = a.tolist() if a.size <= 12 else f"{a.size} strings"
                lines.append(f"  {k:26s} string  {list(a.shape)!s:18s} {val}")
            else:
                rng = f"[{a.min():.4g}, {a.max():.4g}]" if a.size and a.dtype.kind in "fiub" else ""
                lines.append(f"  {k:26s} {str(a.dtype):7s} {list(a.shape)!s:18s} {rng}")
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()

    @property
    def complete(self) -> bool:
        """True when every required array and the common metadata are present
        (then the output is a valid ModelInput file)."""
        rep = self.representation
        return all(k in self for k in REQUIRED[rep]) and (
            rep == "PINN_COLLOCATION" or all(k in self for k in COMMON))

    def validate(self, **kw) -> str:
        from .io import validate
        if self.representation == "PINN_COLLOCATION":
            return self.representation
        if not self.complete:
            raise ValueError("this output is a selection of arrays; only complete outputs "
                             "(all required arrays + metadata) can be validated")
        return validate(dict(self), **kw)

    # ---------------------------------------------------------------- export
    def save(self, path: str, compressed: bool = True) -> str:
        """Write ``.npz``.  ``path`` may be a folder (standard file name, e.g.
        ``mesh_graph.npz``).  Complete outputs are validated first."""
        import os
        if os.path.isdir(path) or path.endswith(("/", "\\")):
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, FILE_NAMES[self.representation])
        if self.complete and self.representation != "PINN_COLLOCATION":
            self.validate()
        arrays = {k: np.asarray(v) for k, v in self.items()}
        (np.savez_compressed if compressed else np.savez)(path, **arrays)
        return path

    def to_torch(self, device=None, *, float_dtype=None) -> dict:
        """Numpy arrays -> torch tensors (strings stay Python lists/str).
        ``float_dtype`` e.g. ``torch.float64`` casts float arrays."""
        try:
            import torch
        except Exception as e:     # ImportError or a broken torch install (DLL errors)
            raise ImportError(f"to_torch() needs a working PyTorch: {e}") from e
        out = {}
        for k, v in self.items():
            a = np.asarray(v)
            if a.dtype.kind == "U":
                out[k] = a.item() if a.ndim == 0 else a.tolist()
                continue
            t = torch.from_numpy(np.ascontiguousarray(a))
            if float_dtype is not None and t.is_floating_point():
                t = t.to(float_dtype)
            out[k] = t.to(device) if device is not None else t
        return out

    def numpy(self) -> dict:
        return {k: np.asarray(v) for k, v in self.items()}


# ==========================================================================
def output(source, representation: str = "MESH_GRAPH", *, metadata: bool | None = None,
           extras: bool = False, surface=None, **kwargs) -> ModelInput:
    """Build one representation of ``source`` (see module docstring).

    metadata : include the 5 common metadata strings.  Default: yes when all
        arrays are returned, no when you select arrays with ``name=True``.
    extras   : also return geocore extras (normalization_center/scale, faces,
        physical_coordinates).
    surface  : a 2D ``NURBSSurface`` for ``PARAMETRIC_SURFACE`` when the
        source is a 3D part.
    """
    rep = representation_name(representation)
    names = SCHEMA[rep]
    known = set(names) | set(COMMON) | set(EXTRAS)
    flags = {k: kwargs.pop(k) for k in list(kwargs) if k in known}
    for k, v in flags.items():
        if not isinstance(v, (bool, np.bool_)):
            raise TypeError(f"'{k}' selects an array: pass True or False (got {v!r})")
    data = _build(source, rep, surface, kwargs)

    chosen = [k for k, v in flags.items() if v]
    dropped = {k for k, v in flags.items() if not v}
    if chosen:
        order = [*COMMON, *names, *EXTRAS]           # schema order
        keys = [k for k in order if k in chosen and (metadata is not False or k not in COMMON)]
        if metadata:
            keys = [*COMMON, *[k for k in keys if k not in COMMON]]
    else:
        keys = ([*COMMON] if metadata is not False and rep != "PINN_COLLOCATION" else []) + \
               [k for k in names if k in data] + ([k for k in EXTRAS if k in data] if extras else [])
        keys = [k for k in keys if k not in dropped]
    out = {}
    for k in keys:
        if k not in data:
            data[k] = _fill_missing(rep, k, data)
        out[k] = data[k]
    return ModelInput(out, rep)


def _fill_missing(rep, k, data):
    """Arrays the schema marks optional and the builder skipped."""
    if rep == "POINT_CLOUD" and k == "point_mask":
        B, N = data["coordinates"].shape[:2]
        return np.ones((B, N), bool)
    if rep == "MESH_GRAPH" and k == "batch":
        return np.zeros(len(data["node_features"]), np.int64)
    if rep == "POINT_CLOUD" and k in ("point_features", "point_feature_names"):
        raise ValueError("this point cloud has no features: pass features=(...) "
                         "(e.g. ('nx','ny','nz','sdf','is_boundary')) to get point_features")
    raise ValueError(f"'{k}' is not produced for {rep} with these options")


# ==========================================================================
def geometry_of(source, resolution: float | None = None):
    """The cached ``Geometry`` adapter of any source."""
    from .design import Design
    from .lattice import Lattice
    from .mesh import Mesh
    from .implicit import Implicit
    from .sampling import Geometry
    from .voxels import Voxels
    if isinstance(source, Geometry):
        return source if resolution is None else Geometry(source, resolution)
    if isinstance(source, Design):
        key = ("design", tuple(sorted(source.parameters.items())), resolution)
        c = source.__dict__.setdefault("_geometry_cache", {})
        if key not in c:
            c.clear()
            c[key] = source.geometry(resolution)
        return c[key]
    if isinstance(source, (Implicit, Mesh, Lattice, Voxels)):
        c = source.__dict__.setdefault("_geometry_cache", {})
        if resolution not in c:
            c[resolution] = Geometry(source, resolution)
        return c[resolution]
    raise TypeError(f"cannot make NN inputs from {type(source).__name__}")


def _build(source, rep, surface, opts) -> dict:
    from . import representations as R
    from .nurbs import NURBSSurface, parametric_surface
    from .sampling import collocation
    if rep == "PARAMETRIC_SURFACE":
        surf = surface if surface is not None else (source if isinstance(source, NURBSSurface) else None)
        if surf is None:
            from .design import Design
            if isinstance(source, Design) and "parametric_surface" in (source.data.get("outputs") or {}):
                # the design file's Deep NURBS domain, options overridden by opts
                spec = dict(source.data["outputs"]["parametric_surface"] or {})
                spec.update(opts)
                data = source.to_dict()
                data["outputs"] = {"parametric_surface": spec}
                d2 = Design(data, source.base_dir)
                return dict(d2.outputs(only=["parametric_surface"])["parametric_surface"])
            raise ValueError(
                "PARAMETRIC_SURFACE (Deep NURBS) is a 2D domain: call it on a 2D "
                "gc.NURBSSurface (e.g. gc.NURBSSurface.plate_with_hole().output('PARAMETRIC_SURFACE')) "
                "or pass surface=<NURBSSurface>")
        return dict(parametric_surface(surf, **opts))
    if isinstance(source, NURBSSurface):
        raise ValueError(f"a 2D NURBSSurface only gives PARAMETRIC_SURFACE, not {rep}")
    res = opts.pop("geometry_resolution", None)
    if isinstance(source, (list, tuple)):
        # a batch of parts: B > 1 for POINT_CLOUD / VOXEL_GRID, one disjoint
        # graph (``batch`` = part index) for MESH_GRAPH
        from .sampling import Geometry
        gs = [geometry_of(s, res) for s in source]
        # one shared tag list, so every part has the same tag:* columns
        order = []
        for g in gs:
            order += [t for t in g.tag_names if t not in order]
        gs = [g if g.tag_names == order else Geometry(g, res, tag_order=order) for g in gs]
        if rep == "MESH_GRAPH":
            return dict(R.batch_graphs([R.mesh_graph(g, **opts) for g in gs]))
        if rep == "POINT_CLOUD":
            return dict(R.point_cloud(gs, **opts))
        if rep == "VOXEL_GRID":
            if "shape" not in opts and "spacing" not in opts:
                opts["shape"] = 64
            return dict(R.voxel_grid(gs, **opts))
        raise ValueError(f"{rep} takes one part, not a list")
    g = geometry_of(source, res)
    try:
        if rep == "MESH_GRAPH":
            return dict(R.mesh_graph(g, **opts))
        if rep == "POINT_CLOUD":
            return dict(R.point_cloud(g, **opts))
        if rep == "VOXEL_GRID":
            if "shape" not in opts and "spacing" not in opts:
                opts["shape"] = 64
            return dict(R.voxel_grid(g, **opts))
        if rep == "PINN_COLLOCATION":
            opts.setdefault("n_interior", 20000)
            opts.setdefault("n_boundary", 5000)
            return dict(collocation(g, **opts))
    except TypeError as e:
        if "unexpected keyword" in str(e):
            raise TypeError(f"{e}.  Array names for {rep}: {list(SCHEMA[rep])}; builder options: "
                            f"see help(gc.{_builder_name(rep)})") from None
        raise
    raise AssertionError(rep)


def _builder_name(rep):
    return {"MESH_GRAPH": "mesh_graph", "POINT_CLOUD": "point_cloud", "VOXEL_GRID": "voxel_grid",
            "PARAMETRIC_SURFACE": "parametric_surface", "PINN_COLLOCATION": "collocation"}[rep]


def outputs_all(source, representations=("MESH_GRAPH", "POINT_CLOUD", "VOXEL_GRID"), **common):
    """{name: ModelInput} for several representations at once."""
    return {r: output(source, r, **common) for r in representations}


# ==========================================================================
# loading 3D files into a part
# ==========================================================================
def load_geometry(path: str, *, repair: bool | str = "auto", resolution: float | None = None,
                  scale: float | None = None):
    """Read a geometry file into a geocore object that has ``.output()``.

    * ``.stl .obj`` (and ``.ply .off .glb .3mf`` ... with ``trimesh``) -> ``Mesh``
    * ``.yaml .yml .json`` -> parametric ``Design``

    repair : ``"auto"`` (default): re-mesh only meshes with holes (open
        edges), because their inside/outside is undefined.  Closed meshes with
        a few non-manifold edges (common in voxel-based exporters) are kept
        as they are - their signed distance is still reliable - with a
        warning.  ``True`` always re-meshes, ``False`` never.  Re-meshing =
        exact signed distance on a grid of ``resolution`` (default: the
        file's mean edge length) + geocore's manifold mesher, vertices snapped
        onto the original surface; sharp edges come out slightly jagged at
        the grid size, so pass a smaller ``resolution`` for crisper edges.
    scale : multiply coordinates (e.g. 1000 for files in metres -> mm).
    """
    import os
    import warnings
    from .design import Design
    from .mesh import Mesh
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml", ".json"):
        return Design.load(path)
    m = Mesh.load(path)
    if scale is not None:
        m = m.scaled(float(scale))
    if m.n_faces == 0:
        raise ValueError(f"{path}: no triangles")
    rep = m.manifold_report()
    if repair is True or (repair == "auto" and rep["open_edges"] > 0):
        if repair == "auto":
            warnings.warn(f"{os.path.basename(path)}: mesh has {rep['open_edges']} open edges "
                          f"(holes); re-meshing it to a closed surface (repair=False keeps it)")
        m = m.remeshed(resolution)
    elif rep["open_edges"]:
        warnings.warn(f"{os.path.basename(path)}: {rep['open_edges']} open edges (holes) kept "
                      f"because repair=False; inside/outside near the holes is unreliable")
    elif not rep["ok"]:
        warnings.warn(f"{os.path.basename(path)}: {rep['non_manifold_edges']} non-manifold edges "
                      f"(kept as is; signed distance is still valid; repair=True re-meshes)")
    m.name = os.path.basename(path)
    return m
