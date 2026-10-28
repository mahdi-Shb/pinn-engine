"""Pictures of geometry and of the NN inputs (needs matplotlib).

* ``render_mesh``         photo of a mesh / any geometry (z-buffer rasterizer)
* ``render_image``        the same, as an RGB array
* ``render_point_cloud``  3D scatter of a POINT_CLOUD dict
* ``render_mesh_graph``   nodes + edges of a MESH_GRAPH dict
* ``render_voxel_grid``   mid-plane slices of every VOXEL_GRID channel
* ``render_parametric``   NURBS domain, control net and UV samples
"""
from __future__ import annotations

import numpy as np

from .mesh import Mesh


def _plt():
    import matplotlib
    if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
    import matplotlib.pyplot as plt
    return plt


def _set_equal(ax, lo, hi):
    c = 0.5 * (lo + hi)
    r = 0.5 * float(np.max(hi - lo))
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))


def render_mesh(geometry, path: str | None = None, *, color="#2A9DA6", elev: float = 25,
                azim: float = -60, title: str | None = None, views: int = 1,
                width: int = 1100, height: int = 900, supersample: int = 2,
                light=(-0.4, -0.6, 0.8), background=(1.0, 1.0, 1.0), dpi: int = 100):
    """Photo of a mesh or of any geocore geometry (implicit, Voxels, Lattice).

    Uses a z-buffer rasterizer (numba) with smooth shading, so the full
    resolution mesh is drawn with correct hidden surfaces.  ``views=4`` puts
    four camera angles in one image.  Returns the path (or the RGB image
    array when ``path`` is None)."""
    from matplotlib.colors import to_rgb
    plt = _plt()
    mesh = _photo_mesh(geometry)
    angles = [(elev, azim)] if views == 1 else [(elev, azim), (elev, azim + 90),
                                               (elev, azim + 180), (70, azim + 45)][:views]
    imgs = [render_image(mesh, elev=e, azim=a, width=width, height=height,
                         supersample=supersample, color=to_rgb(color), light=light,
                         background=background) for e, a in angles]
    if views == 1:
        img = imgs[0]
    else:
        rows = [np.concatenate(imgs[i:i + 2], 1) for i in range(0, len(imgs), 2)]
        if len(rows) > 1 and rows[-1].shape != rows[0].shape:
            pad = np.ones_like(rows[0]); pad[:, :rows[-1].shape[1]] = rows[-1]; rows[-1] = pad
        img = np.concatenate(rows, 0)
    if path is None and title is None:
        return img
    fig = plt.figure(figsize=(img.shape[1] / dpi, img.shape[0] / dpi + (0.4 if title else 0)), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, img.shape[0] / (img.shape[0] + (40 if title else 0))])
    ax.imshow(img)
    ax.set_axis_off()
    if title:
        fig.suptitle(title, y=0.995)
    return _finish(fig, path)


def _photo_mesh(geometry) -> Mesh:
    from .sampling import Geometry
    if isinstance(geometry, Mesh):
        return geometry
    return Geometry(geometry).surface_mesh()


def render_image(mesh: Mesh, *, elev=25.0, azim=-60.0, width=1100, height=900, supersample=2,
                 color=(0.16, 0.62, 0.65), light=(-0.4, -0.6, 0.8), background=(1, 1, 1),
                 margin: float = 0.06) -> np.ndarray:
    """RGB float image [H, W, 3] of a mesh, orthographic camera with the
    matplotlib elev/azim convention."""
    from . import _kernels as K
    e, a = np.radians(elev), np.radians(azim)
    eye = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.cross(np.array([0.0, 0.0, 1.0]), eye)
    if np.linalg.norm(right) < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(eye, right)
    V = mesh.vertices - mesh.vertices.mean(0)
    sx, sy, sz = V @ right, V @ up, V @ eye
    W, H = width * supersample, height * supersample
    ext = max((sx.max() - sx.min()) / W, (sy.max() - sy.min()) / H) * (1 + 2 * margin)
    px = (sx - 0.5 * (sx.max() + sx.min())) / ext + 0.5 * W
    py = 0.5 * H - (sy - 0.5 * (sy.max() + sy.min())) / ext
    vn = mesh.vertex_normals()
    zbuf = np.full((H, W), -np.inf)
    nbuf = np.zeros((H, W, 3))
    K.rasterize(np.ascontiguousarray(px), np.ascontiguousarray(py), np.ascontiguousarray(sz),
                mesh.faces, vn, zbuf, nbuf)
    hit = np.isfinite(zbuf)
    n = nbuf / np.maximum(np.linalg.norm(nbuf, axis=2, keepdims=True), 1e-12)
    facing = np.einsum("hwc,c->hw", n, eye)
    n = np.where((facing < 0)[..., None], -n, n)           # two sided
    Lk = np.asarray(light, float); Lk /= np.linalg.norm(Lk)
    Lf = np.array([0.6, 0.5, 0.3]); Lf /= np.linalg.norm(Lf)
    half = (Lk + eye) / np.linalg.norm(Lk + eye)
    diff = np.clip(n @ Lk, 0, 1)
    fill = np.clip(n @ Lf, 0, 1)
    spec = np.clip(n @ half, 0, 1) ** 40
    rim = (1 - np.clip(np.einsum("hwc,c->hw", n, eye), 0, 1)) ** 3
    base = np.asarray(color, float)
    rgb = base * (0.22 + 0.68 * diff + 0.18 * fill)[..., None] + (0.35 * spec + 0.08 * rim)[..., None]
    img = np.where(hit[..., None], np.clip(rgb, 0, 1), np.asarray(background, float))
    if supersample > 1:
        s = supersample
        img = img.reshape(height, s, width, s, 3).mean((1, 3))
    return img


def render_point_cloud(pc: dict, path: str | None = None, *, batch: int = 0,
                       color_by: str | None = "is_boundary", size=(7, 6), dpi=150):
    plt = _plt()
    X = pc["coordinates"][batch]
    if "point_mask" in pc:
        X = X[pc["point_mask"][batch]]
    c = None
    if color_by and "point_feature_names" in pc:
        names = [str(s) for s in pc["point_feature_names"]]
        if color_by in names:
            c = pc["point_features"][batch][:len(X), names.index(color_by)]
    fig = plt.figure(figsize=size, dpi=dpi)
    ax = fig.add_subplot(projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1.5, c=c, cmap="viridis")
    _set_equal(ax, X.min(0), X.max(0))
    ax.set_title(f"POINT_CLOUD  N={len(X)}" + (f"  (colour: {color_by})" if c is not None else ""))
    return _finish(fig, path)


def render_mesh_graph(g: dict, path: str | None = None, *, max_edges: int = 15000, size=(7, 6), dpi=150):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    plt = _plt()
    X = g["coordinates"]
    ei = g["edge_index"]
    sel = ei[0] < ei[1]
    e = ei[:, sel]
    if e.shape[1] > max_edges:
        e = e[:, np.random.default_rng(0).choice(e.shape[1], max_edges, replace=False)]
    fig = plt.figure(figsize=size, dpi=dpi)
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(Line3DCollection(np.stack([X[e[0]], X[e[1]]], 1), colors="#335", linewidths=0.15, alpha=0.6))
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=0.3, c="#d33")
    _set_equal(ax, X.min(0), X.max(0))
    ax.set_title(f"MESH_GRAPH  nodes={len(X)}  edges={ei.shape[1]} (directed)")
    return _finish(fig, path)


def render_voxel_grid(v: dict, path: str | None = None, *, batch: int = 0, dpi=130):
    """For every channel: the mid Z (D), mid Y (H) and mid X (W) slices."""
    plt = _plt()
    vol = v["voxel_fields"][batch]
    names = [str(s) for s in v.get("channel_names", [f"ch{i}" for i in range(len(vol))])]
    C, D, H, W = vol.shape
    fig, axs = plt.subplots(C, 3, figsize=(10, 2.8 * C), dpi=dpi, squeeze=False)
    for c in range(C):
        f = vol[c]
        sl = [(f[D // 2], "z mid (H x W)"), (f[:, H // 2], "y mid (D x W)"), (f[:, :, W // 2], "x mid (D x H)")]
        cmap = "RdBu" if names[c] in ("sdf", "tsdf") else "viridis"
        for k, (img, lab) in enumerate(sl):
            ax = axs[c, k]
            if cmap == "RdBu":
                m = float(np.abs(img).max()) or 1.0
                im = ax.imshow(img, origin="lower", cmap=cmap, vmin=-m, vmax=m)
                ax.contour(img, levels=[0], colors="k", linewidths=0.6)
            else:
                im = ax.imshow(img, origin="lower", cmap=cmap)
            ax.set_title(f"{names[c]} - {lab}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"VOXEL_GRID  [B,C,D,H,W] = {list(v['voxel_fields'].shape)}")
    fig.tight_layout()
    return _finish(fig, path)


def render_parametric(d: dict, path: str | None = None, *, batch: int = 0, size=(12, 5), dpi=150):
    from .nurbs import NURBSSurface
    plt = _plt()
    s = NURBSSurface(d["degree"], d["knot_vector_u"].astype(float), d["knot_vector_v"].astype(float),
                     d["control_points"].astype(float), d["weights"].astype(float))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=size, dpi=dpi)
    t = np.linspace(0, 1, 60)
    for u in np.unique(s.ku):
        xy = s(np.c_[np.full(60, u), t]); a1.plot(xy[:, 0], xy[:, 1], "k-", lw=0.5)
    for v in np.unique(s.kv):
        xy = s(np.c_[t, np.full(60, v)]); a1.plot(xy[:, 0], xy[:, 1], "k-", lw=0.5)
    P = s.ctrl
    a1.plot(P[..., 0], P[..., 1], "r.-", lw=0.4, ms=3)
    a1.plot(P[..., 0].T, P[..., 1].T, "r-", lw=0.4)
    uv = d["parametric_coordinates"][batch].astype(float)
    xy = s(uv)
    L = s.evaluate_field(d["lifting_coefficients"].astype(float)[..., 0], uv)
    sc = a1.scatter(xy[:, 0], xy[:, 1], s=2, c=L, cmap="viridis")
    fig.colorbar(sc, ax=a1, label="lifting (component 0)")
    a1.set_aspect("equal"); a1.set_title("physical domain: knot lines, control net, samples")
    A = s.evaluate_field(d["admissible_coefficients"].astype(float), uv)
    sc2 = a2.scatter(uv[:, 0], uv[:, 1], s=2, c=A, cmap="magma")
    fig.colorbar(sc2, ax=a2, label="admissible phi")
    a2.set_aspect("equal"); a2.set_xlabel("u"); a2.set_ylabel("v")
    a2.set_title("parametric domain [0,1]^2")
    fig.suptitle(f"PARAMETRIC_SURFACE  degree={[int(x) for x in d['degree']]}  control net={list(P.shape[:2])}")
    fig.tight_layout()
    return _finish(fig, path)


def _finish(fig, path):
    if path:
        fig.savefig(path, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return path
    return fig
