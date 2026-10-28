"""Speed / accuracy benchmark of the geocore hot paths (warm, after JIT)."""
import math
import time

import numpy as np

import geocore as gc


def timed(label, fn, repeat=3):
    fn()                                    # warm up (numba JIT / caches)
    best = min(_t(fn) for _ in range(repeat))
    print(f"{label:55s} {best * 1e3:9.1f} ms")
    return best


def _t(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


rng = np.random.default_rng(0)
P = rng.uniform(-15, 15, (1_000_000, 3))
sphere = gc.Sphere((0, 0, 0), 10)
part = (gc.Box((0, 0, 4), (60, 40, 8)) | gc.Box((-26, 0, 24), (8, 40, 40))
        | gc.Cylinder((12, 0, 0), (12, 0, 16), 9)) - gc.Cylinder((12, 0, -5), (12, 0, 25), 4.5)

print(gc.info())
print("-- point queries (1M points)")
timed("implicit CSG sdf (4 primitives)", lambda: part.sdf(P))
mesh = gc.Geometry(sphere, resolution=0.2).surface_mesh()
timed(f"exact mesh sdf, BVH ({mesh.n_faces} triangles)", lambda: mesh.sdf(P))
lat = gc.Lattice.from_cells("octet", (-15, -15, -15), (15, 15, 15), 5, 0.4)
li = lat.as_implicit()
timed(f"lattice sdf, BVH ({len(lat)} beams)", lambda: li.sdf(P))

print("-- voxel kernels")
for be in ["dense"] + (["picogk"] if gc.PICOGK_AVAILABLE else []):
    lib = gc.Library(0.3, be)
    timed(f"[{be}] Voxels(part) @0.3mm", lambda: gc.Voxels(part, lib=lib))
    v = gc.Voxels(part, lib=lib)
    timed(f"[{be}] fillet(4)", lambda: v.copy().fillet(4), repeat=1)
    timed(f"[{be}] to_mesh", lambda: v.to_mesh())
    timed(f"[{be}] exact dense SDF export (to_dense)", lambda: (v._changed(), v.to_dense()))

g = gc.GridSpec.lattice(sphere.bounds, 0.1, pad=4)
phi = gc.sample_implicit(sphere, g)
t = timed(f"redistance {g.shape} = {g.n / 1e6:.1f}M voxels", lambda: gc.redistance(phi, g.spacing, g.origin), 1)
r = gc.redistance(phi, g.spacing, g.origin)
band = np.abs(phi) < 0.3
print(f"   accuracy: band max {np.abs(r - phi)[band].max() / 0.1:.3f} h, mean {np.abs(r - phi).mean() / 0.1:.4f} h")

print("-- representations")
timed("mesh_graph(part) surface", lambda: gc.mesh_graph(part))
timed("point_cloud(part, 4096, mixed)", lambda: gc.point_cloud(part, 4096, mode="mixed"))
timed("voxel_grid(part, 64^3, occupancy+sdf+boundary)", lambda: gc.voxel_grid(part, shape=64))
timed("collocation(part, 100k interior, 20k boundary)", lambda: gc.collocation(part, 100_000, 20_000))
s = gc.NURBSSurface.plate_with_hole().with_degree(3, 3).refine(16, 12)
timed("parametric_surface(plate, 16k uv)", lambda: gc.parametric_surface(s, n_points=16384, output_dim=2))
