"""``.npz`` I/O and schema validation for the ModelInput representations."""
from __future__ import annotations

import numpy as np

COMMON = ("schema_version", "representation", "coordinate_system", "units", "normalization_id")
FILENAMES = {
    "MESH_GRAPH": "mesh_graph.npz",
    "POINT_CLOUD": "point_cloud.npz",
    "VOXEL_GRID": "voxel_grid.npz",
    "PARAMETRIC_SURFACE": "parametric_surface.npz",
}


class SchemaError(ValueError):
    pass


def save(path: str, data: dict, *, compressed: bool = True, validate_first: bool = True) -> str:
    """Write one sample.  ``path`` may be a directory: the standard file name
    of the representation is used.  Returns the file path."""
    import os
    if validate_first:
        validate(data)
    if os.path.isdir(path):
        path = os.path.join(path, FILENAMES[str(data["representation"])])
    arrays = {k: np.asarray(v) for k, v in data.items()}
    (np.savez_compressed if compressed else np.savez)(path, **arrays)
    return path


def load(path: str, validate_after: bool = True) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if validate_after:
        validate(d)
    return d


def _s(d, k) -> str:
    return str(np.asarray(d[k]))


def _need(d, keys, rep):
    miss = [k for k in keys if k not in d]
    if miss:
        raise SchemaError(f"{rep}: missing {miss}")


def _dtype(d, k, kind, rep):
    a = np.asarray(d[k])
    ok = {"f32": a.dtype == np.float32, "i64": a.dtype == np.int64,
          "bool": a.dtype == np.bool_, "str": a.dtype.kind == "U"}[kind]
    if not ok:
        raise SchemaError(f"{rep}: '{k}' must be {kind}, got {a.dtype}")
    return a


def validate(d: dict, *, input_dim: int | None = None, input_channels: int | None = None,
             tol: float = 1e-6) -> str:
    """Check a representation dict against the schema; returns its name.

    ``input_dim`` (PointNet: 3 + F) and ``input_channels`` (FNO: C) are
    optional model checks."""
    _need(d, COMMON, "common")
    for k in COMMON:
        _dtype(d, k, "str", "common")
    rep = _s(d, "representation")
    if rep == "MESH_GRAPH":
        _need(d, ("node_features", "edge_index", "edge_features"), rep)
        nf = _dtype(d, "node_features", "f32", rep)
        ei = _dtype(d, "edge_index", "i64", rep)
        ef = _dtype(d, "edge_features", "f32", rep)
        N = nf.shape[0]
        if nf.ndim != 2 or ei.ndim != 2 or ei.shape[0] != 2 or ef.ndim != 2:
            raise SchemaError(f"{rep}: bad shapes")
        if ef.shape[0] != ei.shape[1]:
            raise SchemaError(f"{rep}: edge_features rows != number of edges")
        if ei.size and (ei.min() < 0 or ei.max() >= N):
            raise SchemaError(f"{rep}: edge_index out of range")
        fwd = set(map(tuple, ei.T.tolist()))
        if any((r, s) not in fwd for s, r in fwd):
            raise SchemaError(f"{rep}: undirected edges must appear in both directions")
        if "coordinates" in d:
            c = _dtype(d, "coordinates", "f32", rep)
            if c.shape != (N, 3):
                raise SchemaError(f"{rep}: coordinates must be [N,3]")
        if "batch" in d:
            b = _dtype(d, "batch", "i64", rep)
            if b.shape != (N,):
                raise SchemaError(f"{rep}: batch must be [N]")
        if "node_feature_names" in d and len(d["node_feature_names"]) != nf.shape[1]:
            raise SchemaError(f"{rep}: node_feature_names length != Fn")
        if "edge_feature_names" in d and len(d["edge_feature_names"]) != ef.shape[1]:
            raise SchemaError(f"{rep}: edge_feature_names length != Fe")
        if not (np.isfinite(nf).all() and np.isfinite(ef).all()):
            raise SchemaError(f"{rep}: non-finite features")
    elif rep == "POINT_CLOUD":
        _need(d, ("coordinates",), rep)
        c = _dtype(d, "coordinates", "f32", rep)
        if c.ndim != 3 or c.shape[2] != 3:
            raise SchemaError(f"{rep}: coordinates must be [B,N,3]")
        B, N = c.shape[:2]
        F = 0
        if "point_features" in d:
            pf = _dtype(d, "point_features", "f32", rep)
            if pf.shape[:2] != (B, N):
                raise SchemaError(f"{rep}: point_features must be [B,N,F]")
            F = pf.shape[2]
            if "point_feature_names" in d and len(d["point_feature_names"]) != F:
                raise SchemaError(f"{rep}: point_feature_names length != F")
        if "point_mask" in d:
            m = _dtype(d, "point_mask", "bool", rep)
            if m.shape != (B, N):
                raise SchemaError(f"{rep}: point_mask must be [B,N]")
        if input_dim is not None and 3 + F != input_dim:
            raise SchemaError(f"{rep}: 3 + F = {3 + F} != input_dim {input_dim}")
        if not np.isfinite(c).all():
            raise SchemaError(f"{rep}: non-finite coordinates")
    elif rep == "VOXEL_GRID":
        _need(d, ("voxel_fields",), rep)
        v = _dtype(d, "voxel_fields", "f32", rep)
        if v.ndim != 5:
            raise SchemaError(f"{rep}: voxel_fields must be [B,C,D,H,W]")
        if "channel_names" in d and len(d["channel_names"]) != v.shape[1]:
            raise SchemaError(f"{rep}: channel_names length != C")
        for k, shp in (("origin", (3,)), ("spacing", (3,)), ("bounds", (2, 3))):
            if k in d:
                a = _dtype(d, k, "f32", rep)
                if a.shape != shp:
                    raise SchemaError(f"{rep}: {k} must be {shp}")
        if "spacing" in d and "bounds" in d:
            D, H, W = v.shape[2:]
            ext = d["bounds"][1] - d["bounds"][0]
            if not np.allclose(ext, d["spacing"] * np.array([W, H, D]), rtol=1e-4, atol=1e-6):
                raise SchemaError(f"{rep}: bounds extent != spacing * (W,H,D)")
        if input_channels is not None and v.shape[1] != input_channels:
            raise SchemaError(f"{rep}: C = {v.shape[1]} != input_channels {input_channels}")
        if not np.isfinite(v).all():
            raise SchemaError(f"{rep}: non-finite voxel values")
    elif rep == "PARAMETRIC_SURFACE":
        keys = ("degree", "knot_vector_u", "knot_vector_v", "control_points", "weights",
                "admissible_coefficients", "lifting_coefficients", "parametric_coordinates")
        _need(d, keys, rep)
        deg = _dtype(d, "degree", "i64", rep)
        ku = _dtype(d, "knot_vector_u", "f32", rep)
        kv = _dtype(d, "knot_vector_v", "f32", rep)
        P = _dtype(d, "control_points", "f32", rep)
        w = _dtype(d, "weights", "f32", rep)
        a = _dtype(d, "admissible_coefficients", "f32", rep)
        L = _dtype(d, "lifting_coefficients", "f32", rep)
        uv = _dtype(d, "parametric_coordinates", "f32", rep)
        if deg.shape != (2,):
            raise SchemaError(f"{rep}: degree must be [2]")
        if P.ndim != 3 or P.shape[2] != 2:
            raise SchemaError(f"{rep}: control_points must be [Nu,Nv,2] (2D domain only)")
        Nu, Nv = P.shape[:2]
        pu, pv = int(deg[0]), int(deg[1])
        if len(ku) != Nu + pu + 1 or len(kv) != Nv + pv + 1:
            raise SchemaError(f"{rep}: Ku = Nu + degree_u + 1 / Kv = Nv + degree_v + 1 violated")
        for k, p, nm in ((ku, pu, "u"), (kv, pv, "v")):
            if np.any(np.diff(k) < 0):
                raise SchemaError(f"{rep}: knot_vector_{nm} not nondecreasing")
            if not (np.all(k[:p + 1] == k[0]) and np.all(k[-p - 1:] == k[-1])):
                raise SchemaError(f"{rep}: knot_vector_{nm} not clamped")
        if w.shape != (Nu, Nv) or np.any(w <= 0):
            raise SchemaError(f"{rep}: weights must be [Nu,Nv] and positive")
        if a.shape != (Nu, Nv):
            raise SchemaError(f"{rep}: admissible_coefficients must be [Nu,Nv]")
        if (np.abs(a[0]).max() > tol or np.abs(a[-1]).max() > tol or
                np.abs(a[:, 0]).max() > tol or np.abs(a[:, -1]).max() > tol):
            raise SchemaError(f"{rep}: admissible coefficients must be zero on all four boundaries")
        if L.ndim != 3 or L.shape[:2] != (Nu, Nv):
            raise SchemaError(f"{rep}: lifting_coefficients must be [Nu,Nv,output_dim]")
        if uv.ndim != 3 or uv.shape[2] != 2:
            raise SchemaError(f"{rep}: parametric_coordinates must be [B,N,2]")
        if uv.size and (uv.min() < 0 or uv.max() > 1):
            raise SchemaError(f"{rep}: UV coordinates must be within [0,1]")
    else:
        raise SchemaError(f"unknown representation '{rep}'")
    return rep
