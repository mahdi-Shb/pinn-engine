"""Builders for the ModelInput representations (one dict per sample, ready for
``geocore.io.save``):

=====================  ====================  ===================================
function               representation        primary model
=====================  ====================  ===================================
``mesh_graph``         MESH_GRAPH            MeshGraphNet (GCN, GAT)
``point_cloud``        POINT_CLOUD           PointNet (PointNet++, Transolver..)
``voxel_grid``         VOXEL_GRID            FNO3D (3D U-Net)
``parametric_surface`` PARAMETRIC_SURFACE    Deep NURBS (see ``geocore.nurbs``)
=====================  ====================  ===================================

All float arrays are float32, all index arrays int64, string arrays are
numpy unicode arrays (load without ``allow_pickle``).
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np
from scipy.spatial import cKDTree

from .fields import GridSpec, redistance, sample_implicit
from .implicit import BBox, Implicit
from .mesh import Mesh
from .normalize import Normalizer
from .sampling import Geometry
from .voxels import Voxels

SCHEMA_VERSION = "1.0"
COORDINATE_SYSTEM = "cartesian_xyz_right_handed"

NormSpec = Union[str, Normalizer, None]


def _metadata(representation: str, units: str, norm: Normalizer) -> dict:
    return {
        "schema_version": np.array(SCHEMA_VERSION),
        "representation": np.array(representation),
        "coordinate_system": np.array(COORDINATE_SYSTEM),
        "units": np.array(units if norm.is_identity else f"{units}/normalized"),
        "normalization_id": np.array(norm.id),
        # extras (not required by the schema): enough to undo the normalization
        "normalization_center": norm.center.astype(np.float32),
        "normalization_scale": np.array(norm.scale, np.float32),
    }


def _norm(spec: NormSpec, bounds: BBox) -> Normalizer:
    if spec is None or spec == "none":
        return Normalizer.identity()
    if isinstance(spec, Normalizer):
        return spec
    return Normalizer.from_bounds(bounds, spec)


def _f32(a):
    return np.ascontiguousarray(a, np.float32)


def _strs(names):
    return np.array(list(names), dtype=np.str_)


def _tag_onehot(tags, names, mask=None):
    cols, labels = [], []
    for k, nm in enumerate(names):
        c = (tags == k)
        if mask is not None:
            c &= mask
        cols.append(c.astype(np.float32))
        labels.append(f"tag:{nm}")
    return cols, labels


# ==========================================================================
# 1. MESH_GRAPH
# ==========================================================================
SURFACE_NODE_FEATURES = ("nx", "ny", "nz", "vertex_area", "mean_curvature", "tags")
VOLUME_NODE_FEATURES = ("sdf", "is_boundary", "nx", "ny", "nz", "tags")


def mesh_graph(geometry, *, kind: str = "surface", resolution: float | None = None,
               n_interior: int = 20000, k: int = 12, radius: float | None = None,
               node_features: Sequence[str] | None = None,
               normalization: NormSpec = "unit_cube", units: str = "mm",
               face_tags=None, tag_names=None, rng=None) -> dict:
    """MeshGraphNet input.

    kind="surface": nodes = surface mesh vertices, edges = mesh edges.
        Features: ``nx ny nz`` (angle weighted normal), ``vertex_area``
        (lumped), ``mean_curvature``, ``tag:<name>`` one-hots, ``sdf``.
    kind="volume": nodes = interior collocation points (Sobol) + surface
        vertices; edges = symmetric k-NN (or ``radius``) graph.  Features:
        ``sdf``, ``is_boundary``, ``nx ny nz`` (SDF gradient, outward normal
        on the boundary), ``tag:<name>`` (boundary nodes only).

    Edge features: ``dx dy dz distance`` with d = x[receiver] - x[sender];
    every undirected edge appears in both directions.
    """
    g = geometry if isinstance(geometry, Geometry) else Geometry(
        geometry, resolution, face_tags=face_tags, tag_names=tag_names)
    m = g.surface_mesh(resolution)
    if m.n_faces == 0:
        raise ValueError("empty geometry")
    norm = _norm(normalization, g.bounds)
    names_tag = g.tag_names
    rng = np.random.default_rng(rng)

    if kind == "surface":
        feats = list(node_features or SURFACE_NODE_FEATURES)
        X = m.vertices
        vn = m.vertex_normals()
        e = m.edges()
        senders = np.concatenate([e[:, 0], e[:, 1]])
        receivers = np.concatenate([e[:, 1], e[:, 0]])
        cols, labels = [], []
        tags = None
        for f in feats:
            if f in ("nx", "ny", "nz"):
                cols.append(vn[:, "xyz".index(f[1])]); labels.append(f)
            elif f == "vertex_area":
                cols.append(norm.areas(m.vertex_areas())); labels.append(f)
            elif f == "mean_curvature":
                cols.append(m.mean_curvature() * norm.scale); labels.append(f)
            elif f == "sdf":
                cols.append(np.zeros(len(X))); labels.append(f)
            elif f == "is_boundary":
                cols.append(np.ones(len(X))); labels.append(f)
            elif f == "tags":
                if tags is None:
                    tags = _surface_vertex_tags(g, m)
                c, l = _tag_onehot(tags, names_tag)
                cols += c; labels += l
            else:
                raise ValueError(f"unknown surface node feature '{f}'")
    elif kind == "volume":
        feats = list(node_features or VOLUME_NODE_FEATURES)
        Xb = m.vertices
        Xi = g.sample_volume(n_interior, rng)
        X = np.concatenate([Xi, Xb])
        nb = len(Xb)
        is_b = np.concatenate([np.zeros(len(Xi), bool), np.ones(nb, bool)])
        tree = cKDTree(X)
        if radius is not None:
            pairs = tree.query_pairs(radius, output_type="ndarray")
        else:
            _, nn = tree.query(X, k=min(k + 1, len(X)), workers=-1)
            a = np.repeat(np.arange(len(X)), nn.shape[1] - 1)
            b = nn[:, 1:].reshape(-1)
            pairs = np.unique(np.sort(np.stack([a, b], 1), axis=1), axis=0)
        # keep the surface mesh edges too (exact boundary connectivity)
        e = m.edges() + len(Xi)
        pairs = np.unique(np.concatenate([pairs, e]), axis=0)
        senders = np.concatenate([pairs[:, 0], pairs[:, 1]])
        receivers = np.concatenate([pairs[:, 1], pairs[:, 0]])
        sdf = np.concatenate([g.sdf(Xi), np.zeros(nb)])
        nrm = np.concatenate([g.normal(Xi), m.vertex_normals()])
        cols, labels = [], []
        tags = None
        for f in feats:
            if f == "sdf":
                cols.append(norm.lengths(sdf)); labels.append(f)
            elif f == "is_boundary":
                cols.append(is_b.astype(np.float32)); labels.append(f)
            elif f in ("nx", "ny", "nz"):
                cols.append(nrm[:, "xyz".index(f[1])]); labels.append(f)
            elif f == "tags":
                if tags is None:
                    tags = np.full(len(X), -1, np.int64)
                    tags[len(Xi):] = _surface_vertex_tags(g, m)
                c, l = _tag_onehot(tags, names_tag, mask=is_b)
                cols += c; labels += l
            else:
                raise ValueError(f"unknown volume node feature '{f}'")
    else:
        raise ValueError("kind must be 'surface' or 'volume'")

    Xn = norm.points(X)
    d = Xn[receivers] - Xn[senders]
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    out = _metadata("MESH_GRAPH", units, norm)
    out.update({
        "coordinates": _f32(Xn),
        "node_features": _f32(np.stack(cols, 1) if cols else np.zeros((len(X), 0))),
        "edge_index": np.ascontiguousarray(np.stack([senders, receivers]), np.int64),
        "edge_features": _f32(np.concatenate([d, dist], 1)),
        "batch": np.zeros(len(X), np.int64),
        "node_feature_names": _strs(labels),
        "edge_feature_names": _strs(["dx", "dy", "dz", "distance"]),
    })
    if kind == "surface":
        out["faces"] = np.ascontiguousarray(m.faces, np.int64)   # extra: triangles
    return out


def _surface_vertex_tags(g: Geometry, m: Mesh) -> np.ndarray:
    if isinstance(g.src, Implicit):
        return g.tags(m.vertices)
    if g._face_tags is not None and m is g.src:
        # majority vote of the faces around each vertex
        t = np.zeros(m.n_vertices, np.int64)
        votes = np.zeros((m.n_vertices, max(len(g.tag_names), 1)))
        np.add.at(votes, (m.faces.ravel(), np.repeat(g._face_tags, 3)), 1.0)
        t = votes.argmax(1)
        return t
    return g.tags(m.vertices)


def batch_graphs(graphs: Sequence[dict]) -> dict:
    """Concatenate several MESH_GRAPH dicts into one disjoint batch graph."""
    out = dict(graphs[0])
    off = 0
    coords, nf, ei, ef, b = [], [], [], [], []
    for i, gr in enumerate(graphs):
        if list(gr["node_feature_names"]) != list(graphs[0]["node_feature_names"]):
            raise ValueError("graphs must share node features")
        n = len(gr["coordinates"])
        coords.append(gr["coordinates"]); nf.append(gr["node_features"])
        ei.append(gr["edge_index"] + off); ef.append(gr["edge_features"])
        b.append(np.full(n, i, np.int64))
        off += n
    out.update(coordinates=np.concatenate(coords), node_features=np.concatenate(nf),
               edge_index=np.concatenate(ei, 1), edge_features=np.concatenate(ef),
               batch=np.concatenate(b))
    out.pop("faces", None)
    return out


# ==========================================================================
# 2. POINT_CLOUD
# ==========================================================================
POINT_FEATURES = ("nx", "ny", "nz", "sdf", "is_boundary")


def point_cloud(geometries, n_points: int = 2048, *, mode: str = "surface",
                boundary_fraction: float = 0.5, features: Sequence[str] | None = POINT_FEATURES,
                normalization: NormSpec = "unit_sphere", shared_normalization: bool = False,
                pad_to: int | None = None, method: str = "sobol", units: str = "mm",
                resolution: float | None = None, rng=None) -> dict:
    """PointNet input [B, N, 3] (+ features [B, N, F]).

    mode: 'surface' (area uniform on the boundary), 'volume' (interior,
    Sobol), 'mixed' (``boundary_fraction`` on the surface, rest inside).
    features: any of nx ny nz (outward normal / SDF gradient), sdf,
    is_boundary, tags; ``None`` -> coordinates only (``3 + F == input_dim``).
    ``n_points`` may be a list (one per geometry): clouds are then padded to
    the longest and ``point_mask`` marks real points.
    """
    if not isinstance(geometries, (list, tuple)):
        geometries = [geometries]
    gs = [g if isinstance(g, Geometry) else Geometry(g, resolution) for g in geometries]
    ns = list(n_points) if isinstance(n_points, (list, tuple)) else [int(n_points)] * len(gs)
    rng = np.random.default_rng(rng)
    feats = list(features or ())
    if shared_normalization:
        bb = gs[0].bounds
        for g in gs[1:]:
            bb = bb.union(g.bounds)
        shared = _norm(normalization, bb)
    clouds, fcols, labels = [], [], None
    norms = []
    for g, n in zip(gs, ns):
        if mode == "surface":
            nbnd = n
        elif mode == "volume":
            nbnd = 0
        elif mode == "mixed":
            nbnd = int(round(boundary_fraction * n))
        else:
            raise ValueError("mode must be surface | volume | mixed")
        parts_p, parts_n, parts_t, parts_s, parts_b = [], [], [], [], []
        if nbnd:
            p, nrm, t = g.sample_surface(nbnd, rng)
            parts_p.append(p); parts_n.append(nrm); parts_t.append(t)
            parts_s.append(np.zeros(nbnd)); parts_b.append(np.ones(nbnd))
        if n - nbnd:
            p = g.sample_volume(n - nbnd, rng, method)
            parts_p.append(p); parts_n.append(g.normal(p)); parts_t.append(np.full(len(p), -1))
            parts_s.append(g.sdf(p)); parts_b.append(np.zeros(len(p)))
        P = np.concatenate(parts_p)
        if shared_normalization:
            norm = shared
        elif isinstance(normalization, str) and normalization in ("unit_sphere", "unit_cube"):
            norm = Normalizer.from_points(P, normalization)
        else:
            norm = _norm(normalization, g.bounds)
        norms.append(norm)
        cols, lab = [], []
        nrm = np.concatenate(parts_n)
        tags = np.concatenate(parts_t)
        for f in feats:
            if f in ("nx", "ny", "nz"):
                cols.append(nrm[:, "xyz".index(f[1])]); lab.append(f)
            elif f == "sdf":
                cols.append(norm.lengths(np.concatenate(parts_s))); lab.append(f)
            elif f == "is_boundary":
                cols.append(np.concatenate(parts_b)); lab.append(f)
            elif f == "tags":
                c, l = _tag_onehot(tags, g.tag_names)
                cols += c; lab += l
            else:
                raise ValueError(f"unknown point feature '{f}'")
        if labels is None:
            labels = lab
        elif lab != labels:
            raise ValueError("all geometries must produce the same feature columns "
                             "(tag names differ?)")
        clouds.append(norm.points(P))
        fcols.append(np.stack(cols, 1) if cols else np.zeros((len(P), 0)))
    Nmax = max(pad_to or 0, max(len(c) for c in clouds))
    B = len(clouds)
    coords = np.zeros((B, Nmax, 3), np.float32)
    F = fcols[0].shape[1]
    pf = np.zeros((B, Nmax, F), np.float32)
    mask = np.zeros((B, Nmax), bool)
    for b in range(B):
        n = len(clouds[b])
        coords[b, :n] = clouds[b]
        pf[b, :n] = fcols[b]
        mask[b, :n] = True
    out = _metadata("POINT_CLOUD", units, norms[0] if (B == 1 or shared_normalization) else
                    Normalizer(np.zeros(3), 1.0, "per_sample_" + str(normalization),
                               id="per_sample:" + str(normalization)))
    if B > 1 and not shared_normalization:
        out["normalization_center"] = np.stack([n.center for n in norms]).astype(np.float32)
        out["normalization_scale"] = np.array([n.scale for n in norms], np.float32)
    out["coordinates"] = coords
    if F:
        out["point_features"] = pf
        out["point_feature_names"] = _strs(labels)
    if not mask.all():
        out["point_mask"] = mask
    return out


# ==========================================================================
# 3. VOXEL_GRID
# ==========================================================================
VOXEL_CHANNELS = ("occupancy", "sdf", "boundary")


def voxel_grid(geometries, *, shape=None, spacing=None, bounds=None, margin: float = 0.05,
               channels: Sequence[str] = VOXEL_CHANNELS,
               fields: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None,
               truncation: float = 3.0, normalization: NormSpec = "unit_cube",
               units: str = "mm", exact_sdf: bool = True) -> dict:
    """FNO3D / 3D U-Net input ``voxel_fields`` [B, C, D, H, W] (D,H,W = Z,Y,X).

    Grid: give ``shape`` (D,H,W ints, or one int) and/or ``spacing``; the
    grid tiles ``bounds`` (default: union of the geometries' bounds plus
    ``margin``).  Nodes are cell centres: ``origin`` = centre of voxel
    [0,0,0] (x,y,z), ``bounds`` = cell-covered box, ``spacing`` = (dx,dy,dz).
    The layout matches ``torch.nn.functional.grid_sample(...,
    align_corners=False)`` over ``bounds``.

    Channels
      occupancy        filled volume, anti-aliased fraction in [0,1]
                       (from the exact SDF: clip(0.5 - sdf/h, 0, 1))
      occupancy_binary sdf <= 0
      sdf              exact signed distance (normalized units)
      tsdf             sdf / (truncation * h) clipped to [-1, 1]
      boundary         voxels whose cell the surface crosses (|sdf| <= h*sqrt3/2)
      tag:<name>       boundary voxels owned by that tagged surface (implicits)
      tags             shorthand for all tag:<name> channels
      <field name>     user fields: ``fields={"source": f(xyz [N,3]) -> [N]}``
    """
    if not isinstance(geometries, (list, tuple)):
        geometries = [geometries]
    gs = [g if isinstance(g, Geometry) else Geometry(g) for g in geometries]
    if bounds is None:
        bb = gs[0].bounds
        for g in gs[1:]:
            bb = bb.union(g.bounds)
        bb = bb.expanded(margin * float(np.max(bb.size)))
    else:
        bb = bounds if isinstance(bounds, BBox) else BBox(*np.asarray(bounds, float))
    if shape is not None:
        s = np.broadcast_to(np.asarray(shape, int), (3,))
        grid_shape_xyz = (int(s[2]), int(s[1]), int(s[0]))           # D,H,W -> X,Y,Z
        if spacing is not None:
            sp = np.broadcast_to(np.asarray(spacing, float), (3,))
            c = bb.center
            half = 0.5 * np.asarray(grid_shape_xyz) * sp
            bb = BBox(c - half, c + half)
        grid = GridSpec.from_bounds(bb, shape=grid_shape_xyz)
    elif spacing is not None:
        grid = GridSpec.from_bounds(bb, spacing=spacing)
    else:
        grid = GridSpec.from_bounds(bb, shape=64)
    norm = _norm(normalization, grid.bounds)
    h = float(np.max(grid.spacing))
    chans = []
    for c in channels:
        if c == "tags":
            names = gs[0].tag_names
            chans += [f"tag:{n}" for n in names]
        else:
            chans.append(c)
    vols = []
    for g in gs:
        phi, tag = _grid_sdf(g, grid, exact_sdf, need_tags=any(c.startswith("tag:") for c in chans))
        layers = []
        for c in chans:
            if c == "occupancy":
                layers.append(np.clip(0.5 - phi / h, 0.0, 1.0))
            elif c == "occupancy_binary":
                layers.append((phi <= 0).astype(np.float32))
            elif c == "sdf":
                layers.append(norm.lengths(phi))
            elif c == "tsdf":
                layers.append(np.clip(phi / (truncation * h), -1.0, 1.0))
            elif c == "boundary":
                layers.append((np.abs(phi) <= 0.5 * np.sqrt(3) * h).astype(np.float32))
            elif c.startswith("tag:"):
                name = c[4:]
                names = g.tag_names
                if name not in names:
                    raise ValueError(f"tag '{name}' not in {names}")
                bnd = np.abs(phi) <= 0.5 * np.sqrt(3) * h
                layers.append((bnd & (tag == names.index(name))).astype(np.float32))
            elif fields and c in fields:
                v = np.asarray(fields[c](grid.points()), np.float64).reshape(grid.shape)
                layers.append(v)
            else:
                raise ValueError(f"unknown channel '{c}' (add it to fields=)")
        vols.append(np.stack([np.ascontiguousarray(np.transpose(l, (2, 1, 0))) for l in layers]))
    out = _metadata("VOXEL_GRID", units, norm)
    out.update({
        "voxel_fields": _f32(np.stack(vols)),
        "channel_names": _strs(chans),
        "origin": _f32(norm.points(grid.origin)),
        "spacing": _f32(grid.spacing / norm.scale),
        "bounds": _f32(norm.points(grid.bounds.as_array())),
    })
    return out


def _grid_sdf(g: Geometry, grid: GridSpec, exact: bool, need_tags: bool):
    """Exact SDF (and owner tags) on an arbitrary node grid."""
    tags = None
    if isinstance(g.src, Voxels):
        phi, _ = g.src.to_dense(grid)
    elif isinstance(g.src, Mesh):
        phi = g.src.sdf(grid.points()).reshape(grid.shape).astype(np.float32)
    else:
        if need_tags:
            phi, tags = sample_implicit(g.imp, grid, tagged=True)
        else:
            phi = sample_implicit(g.imp, grid)
        if exact and not g.imp.exact:
            phi = redistance(phi, grid.spacing, grid.origin)
    if need_tags and tags is None:
        # owner of the closest surface point
        tags = g.tags(grid.points()).reshape(grid.shape)
    elif need_tags and isinstance(g.src, Implicit):
        # tags from the tree are the owner at the node; snap them to the
        # owner of the nearest surface point for boundary voxels
        bnd = np.abs(phi) <= 0.5 * np.sqrt(3) * float(np.max(grid.spacing))
        if bnd.any():
            P = grid.points().reshape(grid.shape + (3,))[bnd]
            q = g.imp.project(P, iters=4)
            tags = tags.copy()
            tags[bnd] = g.tags(q)
    return phi.astype(np.float64), tags
