"""Build every ModelInput representation for one geometry, report the values,
and save pictures of the 3D object.

    python examples/model_inputs.py                       # demo bracket
    python examples/model_inputs.py --geometry part.stl   # any closed STL/OBJ
    python examples/model_inputs.py --geometry lattice --voxel-size 0.25 --out my_run

Writes into --out (default ``model_inputs/``):

    mesh_graph.npz          MESH_GRAPH          MeshGraphNet
    point_cloud.npz         POINT_CLOUD         PointNet
    voxel_grid.npz          VOXEL_GRID          FNO3D
    parametric_surface.npz  PARAMETRIC_SURFACE  Deep NURBS (2D domain)
    geometry_values.json    every array: dtype, shape, min/max/mean + metadata
    geometry_values.md      the same as a readable table
    object.png              shaded photo of the 3D object (4 views)
    object_front.png        one large view
    pinn_collocation.npz    PINN interior/boundary/near-wall points (+ sdf, normals, tags)
    preview_*.png           pictures of each representation
    object.stl              the surface mesh
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

import geocore as gc
from geocore import render


# ----------------------------------------------------------------- geometries
def bracket():
    """The PicoGK WebModel bracket, with named surfaces."""
    # in Python `-` binds tighter than `|`: bracket the union before subtracting
    return ((gc.Box((0, 0, 4), (60, 40, 8)).tagged("base")
             | gc.Box((-26, 0, 24), (8, 40, 40)).tagged("upright")
             | gc.Cylinder((12, 0, 0), (12, 0, 16), 9).tagged("boss"))
            - gc.Cylinder((12, 0, -5), (12, 0, 25), 4.5).tagged("bolt_hole")
            - gc.Cylinder((-40, 0, 30), (-10, 0, 30), 7).tagged("shaft_bore"))


def build_geometry(name: str, voxel_size: float):
    """Returns (geometry for the NN inputs, tagged implicit or None, label)."""
    gc.set_voxel_size(voxel_size)
    if os.path.isfile(name):
        mesh = gc.Mesh.load(name)
        if not mesh.is_watertight():
            print("warning: mesh is not watertight - SDF signs may be wrong")
        return mesh, None, os.path.basename(name)
    if name == "bracket":
        imp = bracket()
        vox = gc.Voxels(imp).fillet(4)                 # PicoGK OverOffset(4, 0)
        return vox, imp, "bracket (filleted)"
    if name == "lattice":
        lat = gc.Lattice.from_cells("bcc", (0, 0, 0), (40, 40, 40), 10,
                                    lambda p: 1.4 - 0.8 * p[:, 2] / 40)   # graded struts
        vox = gc.Voxels(lat) + gc.Voxels.box((0, 0, -0.75), (40, 40, 0.75)) \
            + gc.Voxels.box((0, 0, 39.25), (40, 40, 40.75))
        return vox, None, "graded BCC lattice panel"
    if name == "gyroid_sphere":
        vox = gc.Voxels.sphere((0, 0, 0), 25).shell(-1.2) \
            + gc.Voxels.sphere((0, 0, 0), 25).intersect_implicit(gc.Gyroid(10, 0.8))
        vox.trim((-30, -30, -30), (30, 0, 30))
        return vox, None, "gyroid sphere (cut)"
    raise SystemExit(f"unknown geometry '{name}' (bracket | lattice | gyroid_sphere | <file.stl>)")


# ----------------------------------------------------------------- reporting
def describe(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        a = np.asarray(v)
        if a.dtype.kind == "U":
            out[k] = {"dtype": "string", "shape": list(a.shape),
                      "value": a.tolist() if a.size <= 64 else f"{a.size} strings"}
        else:
            e = {"dtype": str(a.dtype), "shape": list(a.shape)}
            if a.size:
                f = a.astype(np.float64)
                e.update(min=float(f.min()), max=float(f.max()), mean=float(f.mean()))
            if a.size <= 12:
                e["value"] = a.tolist()
            out[k] = e
    return out


def to_markdown(report: dict) -> str:
    lines = ["# Geometry values", ""]
    for fname, info in report["files"].items():
        lines += [f"## {fname}  -  {info['representation']}", "",
                  "| array | dtype | shape | min | max | value |", "|---|---|---|---|---|---|"]
        for k, e in info["arrays"].items():
            val = e.get("value", "")
            if isinstance(val, list) and len(str(val)) > 60:
                val = str(val)[:57] + "..."
            mn = f"{e['min']:.4g}" if "min" in e else ""
            mx = f"{e['max']:.4g}" if "max" in e else ""
            lines.append(f"| `{k}` | {e['dtype']} | {e['shape']} | {mn} | {mx} | {val} |")
        lines.append("")
    g = report["geometry"]
    lines += ["## Geometry", "", *[f"- **{k}**: {v}" for k, v in g.items()], ""]
    return "\n".join(lines)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", default="bracket")
    ap.add_argument("--voxel-size", type=float, default=0.3)
    ap.add_argument("--out", default="model_inputs")
    ap.add_argument("--points", type=int, default=4096, help="POINT_CLOUD points")
    ap.add_argument("--grid", type=int, nargs="+", default=[48, 64, 96], help="VOXEL_GRID D H W")
    ap.add_argument("--graph-resolution", type=float, default=None, help="surface mesh edge length (mm)")
    ap.add_argument("--uv-points", type=int, default=4096, help="Deep NURBS UV samples")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    print(gc.info())

    t0 = time.time()
    geo, tagged, label = build_geometry(a.geometry, a.voxel_size)
    G = gc.Geometry(geo, resolution=a.graph_resolution)
    mesh = G.surface_mesh()
    vol = mesh.volume()
    print(f"geometry: {label}  ({time.time() - t0:.2f}s)  mesh {mesh.n_vertices} vertices,"
          f" {mesh.n_faces} faces, volume {vol:.1f} mm^3")
    mesh.save_stl(os.path.join(a.out, "object.stl"))

    # ---- the four representations of ModelInput.md ---------------------------
    reps = {}
    t = time.time()
    reps["mesh_graph.npz"] = gc.mesh_graph(G, resolution=a.graph_resolution)
    reps["point_cloud.npz"] = gc.point_cloud(G, a.points, mode="mixed", rng=rng)
    shape = a.grid if len(a.grid) == 3 else a.grid * 3
    channels = ("occupancy", "sdf", "boundary")
    reps["voxel_grid.npz"] = gc.voxel_grid(G, shape=shape, channels=channels)
    if tagged is not None:   # extra: per-surface boundary masks from the named implicit
        reps["voxel_grid_tagged.npz"] = gc.voxel_grid(tagged, shape=shape,
                                                       channels=("occupancy", "sdf", "tags"))
    # Deep NURBS is a 2D domain (schema limitation): plate with a hole, with
    # Dirichlet data u = (x, y) on the hole boundary
    plate = gc.NURBSSurface.plate_with_hole(R=1.0, L=4.0).with_degree(3, 3).refine(12, 10)
    reps["parametric_surface.npz"] = gc.parametric_surface(
        plate, output_dim=2, n_points=a.uv_points, dirichlet={"bottom": lambda xy: xy}, rng=rng)
    col = gc.collocation(G, 50_000, 20_000, band=(10_000, 2 * a.voxel_size), rng=rng)
    np.savez_compressed(os.path.join(a.out, "pinn_collocation.npz"), **col)
    print(f"representations built in {time.time() - t:.2f}s")

    report = {"geometry": {"name": label, "backend": getattr(geo, "backend", "mesh"),
                           "voxel_size_mm": a.voxel_size,
                           "bounds_mm": mesh.bounds.as_array().round(4).tolist(),
                           "volume_mm3": round(vol, 3), "surface_area_mm2": round(mesh.area(), 3),
                           "surface_vertices": mesh.n_vertices, "surface_faces": mesh.n_faces,
                           "watertight": mesh.is_watertight(),
                           "boundary_tags": tagged.tag_names() if tagged is not None else G.tag_names},
              "files": {}}
    for fname, d in reps.items():
        path = gc.save(os.path.join(a.out, fname), d)        # validated before writing
        d2 = gc.load(path)                                    # and after reading
        report["files"][fname] = {"representation": str(d2["representation"]), "arrays": describe(d2)}
        print(f"  {fname:26s} {str(d2['representation']):19s}",
              {k: tuple(v.shape) for k, v in d2.items() if v.ndim and v.dtype.kind != "U"})
    with open(os.path.join(a.out, "geometry_values.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(a.out, "geometry_values.md"), "w") as f:
        f.write(to_markdown(report))

    # ---- pictures -------------------------------------------------------------
    t = time.time()
    render.render_mesh(mesh, os.path.join(a.out, "object.png"), views=4, title=label,
                       width=800, height=650)
    render.render_mesh(mesh, os.path.join(a.out, "object_front.png"), title=label)
    render.render_point_cloud(reps["point_cloud.npz"], os.path.join(a.out, "preview_point_cloud.png"))
    render.render_mesh_graph(reps["mesh_graph.npz"], os.path.join(a.out, "preview_mesh_graph.png"))
    render.render_voxel_grid(reps["voxel_grid.npz"], os.path.join(a.out, "preview_voxel_grid.png"))
    render.render_parametric(reps["parametric_surface.npz"], os.path.join(a.out, "preview_parametric_surface.png"))
    print(f"pictures saved in {time.time() - t:.2f}s -> {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
