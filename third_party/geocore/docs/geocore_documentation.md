# geocore documentation

geocore is a Python library that builds 3D (and 2D NURBS) geometry and turns
it into the four neural-network input files of the ModelInput specification,
plus point sets for PINN training. Its voxel engine is the PicoGK C++ runtime;
everything else runs as compiled, multi-threaded numba code.

Version 0.1.0.

---

## Contents

1. [Install and check](#1-install-and-check)
2. [Core ideas](#2-core-ideas)
3. [Five-minute tour](#3-five-minute-tour)
4. [Building geometry](#4-building-geometry)
5. [Voxels (PicoGK operations)](#5-voxels-picogk-operations)
6. [Meshes and lattices](#6-meshes-and-lattices)
7. [The Geometry adapter and sampling](#7-the-geometry-adapter-and-sampling)
8. [The ModelInput representations](#8-the-modelinput-representations)
9. [Deep NURBS domains](#9-deep-nurbs-domains)
10. [Saving, loading and validating](#10-saving-loading-and-validating)
11. [Normalization](#11-normalization)
12. [PINN collocation](#12-pinn-collocation)
13. [Pictures](#13-pictures)
14. [The `model_inputs.py` script](#14-the-model_inputspy-script)
15. [Parametric design files](#15-parametric-design-files)
16. [Accuracy and speed](#16-accuracy-and-speed)
17. [How it works inside](#17-how-it-works-inside)
18. [Troubleshooting](#18-troubleshooting)
19. [API reference](#19-api-reference)

---

## 1. Install and check

```bash
pip install -e "E:/study/personal/ai/PINN/Z-lab/geometry - py/geocore[test]"
```

Requirements: Python ≥ 3.10, numpy, scipy and numba; matplotlib is only
needed for pictures. The first import compiles the numba kernels, which takes
about 20–40 s once; the compiled code is cached on disk.

```python
import geocore as gc
print(gc.info())
# {'geocore': '0.1.0', 'picogk_available': True, 'picogk_version': '26.2.0', 'picogk_error': None}
```

| Platform | Voxel engine |
|---|---|
| Windows x64 | PicoGK C++ (`geocore/_bin/win-x64`) |
| macOS arm64 | PicoGK C++ (`geocore/_bin/osx-arm64`) |
| Linux, others | numba "dense" engine (same API) |

Environment variables:

| Variable | Effect |
|---|---|
| `GEOCORE_BACKEND=dense` | never load PicoGK; always use the numba engine |
| `GEOCORE_PICOGK_DIR=<folder>` | load the PicoGK library from this folder instead (for example a Linux build) |

Run the tests with `pytest` from the `geocore` folder.

---

## 2. Core ideas

* **Signed distance (SDF).** Every geometry is a function `sdf(p)`: negative
  inside, zero on the surface, positive outside. Values are distances in
  millimetres, the PicoGK convention.
* **Points are `[N, 3]` arrays** (float64 in, float64 out). Every query is
  vectorized.
* **Parametric design files** (§15) describe a part by named parameters;
  change them from Python or the command line and regenerate every output,
  or sample whole datasets of variants.
* **Four kinds of geometry source**, all interchangeable:
  * `Implicit`: analytic shapes and CSG trees (exact, infinitely sharp).
  * `Voxels`: PicoGK voxel field; needed for fillets, smoothing, offsets of
    complex parts, lattices and gyroid infills.
  * `Mesh`: a closed triangle mesh, for example an STL from CAD.
  * `Lattice`: beam lattice (round cones), like PicoGK's `Lattice`.
* **Tags.** Name a surface with `.tagged("inlet")`. The name survives CSG
  operations and becomes a boundary mask (`tag:inlet`) in every output, which
  is what you need to assign boundary conditions.
* **One file per sample.** Each builder returns a plain `dict` of numpy
  arrays; `gc.save` writes it as the standard `.npz` after validating it.

---

## 3. Five-minute tour

```python
import numpy as np
import geocore as gc

gc.set_voxel_size(0.3)                               # like PicoGK.Library(0.3)

# 1. geometry (bracket the union: in Python `-` binds tighter than `|`)
part = ((gc.Box((0, 0, 4), (60, 40, 8)).tagged("base")
         | gc.Box((-26, 0, 24), (8, 40, 40)).tagged("upright")
         | gc.Cylinder((12, 0, 0), (12, 0, 16), 9).tagged("boss"))
        - gc.Cylinder((12, 0, -5), (12, 0, 25), 4.5).tagged("bolt_hole"))

vox = gc.Voxels(part).fillet(4)                      # PicoGK fillet

# 2. NN inputs (ModelInput.md)
gc.save("out/", gc.mesh_graph(vox))                  # out/mesh_graph.npz
gc.save("out/", gc.point_cloud(vox, 4096, mode="mixed"))
gc.save("out/", gc.voxel_grid(vox, shape=(48, 64, 96)))
plate = gc.NURBSSurface.plate_with_hole(1, 4).with_degree(3, 3).refine(12, 10)
gc.save("out/", gc.parametric_surface(plate, output_dim=1, dirichlet=0.0))

# 3. PINN points
col = gc.collocation(vox, n_interior=100_000, n_boundary=20_000)

# 4. a picture
from geocore import render
render.render_mesh(vox, "out/object.png", views=4)
```

---

## 4. Building geometry

### 4.1 Primitives

All primitives return exact signed distances.

| Class | Arguments | Notes |
|---|---|---|
| `Sphere(center, radius)` | | |
| `Box(center, size, rounding=0)` | full edge lengths | `Box.from_bounds(vmin, vmax)` |
| `Cylinder(a, b, radius)` | axis endpoints | flat ends (the WebModel `Cylinder`) |
| `Beam(a, b, ra, rb=None)` | radii at both ends | round cone with spherical caps = PicoGK `voxLatticeBeam`; alias `Capsule` |
| `Cone(a, b, ra, rb)` | | flat-ended frustum |
| `Torus(center, major_radius, minor_radius)` | | around local z |
| `Plane(point, normal)` | | half space `dot(n, p − point) ≤ 0`, unbounded |
| `Gyroid(period, wall, solid=False, level=0, origin=(0,0,0))` | | TPMS sheet (as in the PicoGK demo); unbounded |
| `SchwarzP(...)`, `Diamond(...)` | | other TPMS, same arguments |
| `CubicStrutLattice(center, size, cell, radius)` | | the web modeler's lattice feature |
| `FunctionImplicit(fn, bounds=None, exact=False)` | `fn(points [N,3]) → [N]` | your own function |

Unbounded shapes (`Plane` and the TPMS) must be intersected with a bounded
shape before they can be voxelized or sampled.

### 4.2 Combining shapes

| Expression | Meaning | PicoGK equivalent |
|---|---|---|
| `a \| b` or `a + b` | union | `BoolAdd` |
| `a - b` | difference | `BoolSubtract` |
| `a & b` | intersection | `BoolIntersect` |
| `-a` | complement (inside ↔ outside) | |
| `a.smooth_union(b, k)` | blended union, radius `k` mm | |
| `a.smooth_subtract(b, k)`, `a.smooth_intersect(b, k)` | blended | |
| `a.offset(d)` | grow by `d` (shrink if `d < 0`) | `Offset` (approximate on CSG; use `Voxels` for exact) |
| `a.shell(t)` | hollow: `t < 0` keeps the outer size | `voxShell` |
| `a.trim(vmin, vmax)` | cut to a box | `Trim` |
| `gc.union(a, b, c)`, `gc.intersection(...)`, `gc.difference(a, b)` | function forms | |

> **Operator precedence.** Python evaluates `-` before `|`, so
> `a | b - c` means `a | (b - c)`. Write `(a | b) - c`.

### 4.3 Transforms

`translate(t)`, `rotate(axis, angle_rad, center=(0,0,0))`,
`scale(s, center=(0,0,0))` and `transform(matrix4x4)`. Only rigid motions
with uniform scale are allowed, because they keep the field a true distance.

### 4.4 Queries on any implicit

```python
d = part.sdf(points)                 # [N]
n = part.normal(points)              # unit outward normals [N, 3]
g = part.gradient(points)            # raw gradient
q = part.project(points)             # Newton projection onto the surface
inside = part.contains(points)       # bool [N]
b = part.bounds                      # BBox(min, max) or None
d, t = part.sdf_tagged(points)       # t indexes part.tag_names()
```

Untagged leaves get their class name as tag (for example `"sphere"`).

---

## 5. Voxels (PicoGK operations)

A `Voxels` object is a signed distance voxel field, the same object as
PicoGK's `Voxels`.

### 5.1 Library and voxel size

```python
gc.set_voxel_size(0.3)                 # default library, backend "auto"
lib = gc.Library(0.2, backend="dense") # explicit library ("auto" | "picogk" | "dense")
v = gc.Voxels(shape, lib=lib)
```

Fields can only be combined if they share a library, that is the same voxel
size and backend.

### 5.2 Creating fields

```python
gc.Voxels(implicit)                    # any bounded implicit
gc.Voxels(implicit, bounds=(lo, hi))   # unbounded implicit, explicit box
gc.Voxels(mesh)                        # closed mesh
gc.Voxels(lattice)                     # beam lattice
gc.Voxels.sphere(center, r)
gc.Voxels.beam(a, ra, b, rb)           # voxLatticeBeam
gc.Voxels.box(vmin, vmax)
gc.Voxels.mesh_shell(mesh, thickness)  # voxMeshShell
gc.Voxels.combine_all([v1, v2, v3])
```

### 5.3 Operations

Each operation changes the field in place and returns it, so calls can be
chained. The operators `+ - &` return new fields.

| Method | PicoGK name | Effect |
|---|---|---|
| `bool_add(o)` / `+=` | `BoolAdd` | union |
| `bool_subtract(o)` / `-=` | `BoolSubtract` | difference |
| `bool_intersect(o)` / `&=` | `BoolIntersect` | intersection |
| `offset(d)` | `Offset` | move the surface by `d` mm |
| `double_offset(d1, d2)` | `DoubleOffset` | two offsets |
| `triple_offset(d)` / `smoothen(d)` | `TripleOffset` | −d, +2d, −d: removes detail smaller than `d` |
| `over_offset(first, final=0)` | `OverOffset` | |
| `fillet(r)` | `Fillet` | rounds concave edges with radius `r` |
| `shell(t)` | `voxShell` | returns a new hollow field |
| `shell2(neg, pos, smooth_inner=0)` | `voxShell(a, b, c)` | |
| `trim(vmin, vmax)` | `Trim` | keep only the box |
| `intersect_implicit(imp)` | `IntersectImplicit` | for example gyroid infill |
| `render_implicit / render_mesh / render_lattice` | `Render…` | union a shape in |

The PicoGK capitalized names (`BoolAdd`, `Offset`, `Fillet`, …) also work.

### 5.4 Queries and export

```python
vol, bbox = v.calculate_properties()   # exact mesh volume (mm^3), BBox
v.volume(); v.bounding_box(); v.is_empty()
mesh = v.to_mesh()                     # Mesh (PicoGK: OpenVDB mesher)
v.save_stl("part.stl")
v.sdf(points); v.is_inside(points); v.surface_normal(points)
v.closest_point_on_surface(points)
hits, mask = v.raycast_to_surface(origins, directions)
phi, grid = v.to_dense()               # full exact SDF on its voxel lattice, [X,Y,Z]
phi, grid = v.to_dense(gc.GridSpec.from_bounds(box, shape=64))   # on your own grid
```

PicoGK stores only a narrow band (±1.5 voxels). `to_dense` and `sdf` turn it
into a full, exact distance field by redistancing (§16).

---

## 6. Meshes and lattices

### 6.1 Mesh

```python
m = gc.Mesh.load("part.stl")           # STL (binary/ascii), OBJ; other formats via trimesh
m = gc.Mesh(vertices, faces)           # [V,3] float, [F,3] int, counter-clockwise from outside
m.save_stl("out.stl")
m.is_watertight(); m.volume(); m.area(); m.bounds
m.face_normals(); m.vertex_normals(); m.vertex_areas(); m.mean_curvature(); m.edges()
p, n = m.sample_surface(10_000)        # area-uniform points + face normals
d = m.sdf(points)                      # exact signed distance (BVH, pseudo-normal sign)
cp, face, dist = m.closest_points(points)
m.welded(); m.translated(t); m.scaled(s); m.transformed(M)
gc.Mesh.box(vmin, vmax); gc.Mesh.concatenate([m1, m2])
```

Signed distance requires a closed, consistently oriented mesh. `load` welds
duplicate STL vertices automatically.

### 6.2 Lattice

```python
lat = gc.Lattice()
lat.add_beam(a, b, ra, rb)             # PicoGK AddBeam
lat.add_sphere(c, r)                   # PicoGK AddSphere
lat = gc.Lattice.from_cells("octet", vmin, vmax, cell_size, radius)
# cell types: cubic | bcc | bccz | fcc | octet | diamond
# radius: number, or f(points [N,3]) -> radii for graded lattices
imp = lat.as_implicit()                # exact SDF, BVH accelerated
vox = gc.Voxels(lat)
```

---

## 7. The Geometry adapter and sampling

`gc.Geometry(src)` wraps any source (`Implicit`, `Voxels`, `Mesh`, `Lattice`)
behind one interface. All builders accept either the raw source or a
`Geometry`.

```python
g = gc.Geometry(src, resolution=None)  # resolution: mesh size for implicits (default: extent / 160)
g.bounds; g.tag_names
g.sdf(p); g.normal(p); g.tags(p)
g.surface_mesh()                       # watertight mesh
p, n, t = g.sample_surface(5000)       # points on the surface, normals, tag index
p = g.sample_volume(20000, method="sobol")    # inside points: sobol | halton | random | grid
p = g.sample_band(5000, width=0.5)     # points within 0.5 mm of the wall
```

Mesh-only boundary tags: `gc.Geometry(mesh, face_tags=labels_per_face,
tag_names=[...])`.

---

## 8. The ModelInput representations

> The short way to get any of these is `part.output("MESH_GRAPH")`,
> `part.output("POINT_CLOUD")`, `part.output("VOXEL_GRID")` or
> `part.output("PARAMETRIC_SURFACE")` on any geometry. You can select
> arrays with `name=True`. The complete reference for every array is
> [nn_outputs.md](nn_outputs.md).

Every file contains the common metadata of ModelInput.md:

| Key | Type | geocore value |
|---|---|---|
| `schema_version` | string | `"1.0"` |
| `representation` | string | `MESH_GRAPH`, `POINT_CLOUD`, `VOXEL_GRID` or `PARAMETRIC_SURFACE` |
| `coordinate_system` | string | `cartesian_xyz_right_handed` (2D: `cartesian_xy_2d`) |
| `units` | string | `mm`, or `mm/normalized` when coordinates were normalized |
| `normalization_id` | string | name of the normalization (§11) |
| `normalization_center`, `normalization_scale` | float32 | extras, enough to undo the normalization |

Strings are numpy unicode arrays, so files load without `allow_pickle`. Float
arrays are float32 and index arrays int64.

### 8.1 MESH_GRAPH → MeshGraphNet (GCN, GAT)

```python
g = gc.mesh_graph(src, kind="surface")                     # mesh vertices + mesh edges
g = gc.mesh_graph(src, kind="volume", n_interior=20000, k=12)   # interior points + surface, k-NN edges
```

| Array | Shape | Content |
|---|---|---|
| `coordinates` | float32 [N, 3] | node positions (normalized) |
| `node_features` | float32 [N, Fn] | see below |
| `edge_index` | int64 [2, E] | row 0 senders, row 1 receivers; every edge in both directions |
| `edge_features` | float32 [E, 4] | `dx dy dz distance`, with `d = x[receiver] − x[sender]` |
| `batch` | int64 [N] | zeros for one graph |
| `node_feature_names`, `edge_feature_names` | string | column names |
| `faces` | int64 [F, 3] | extra (surface graphs): the triangles |

Node features (choose with `node_features=[...]`):

* **Surface graph:** `nx ny nz` (angle-weighted normal), `vertex_area`,
  `mean_curvature` (1/length, in normalized units), `sdf`, `is_boundary` and
  `tags` (expands to `tag:<name>` one-hots).
* **Volume graph:** `sdf`, `is_boundary`, `nx ny nz` (SDF gradient; outward
  normal on the wall) and `tags` (wall nodes only).

`gc.batch_graphs([g1, g2])` merges graphs into one disjoint graph and fills
`batch`.

### 8.2 POINT_CLOUD → PointNet (PointNet++, Transolver, GINO, SIREN)

```python
pc = gc.point_cloud(src, 2048, mode="surface")             # surface | volume | mixed
pc = gc.point_cloud([a, b, c], [2048, 1500, 1800], mode="mixed", boundary_fraction=0.5)
pc = gc.point_cloud(src, 2048, features=None)              # coordinates only (input_dim = 3)
```

| Array | Shape | Content |
|---|---|---|
| `coordinates` | float32 [B, N, 3] | B = 1 for one geometry |
| `point_features` | float32 [B, N, F] | default `nx ny nz sdf is_boundary` (+ `tags`) |
| `point_feature_names` | string [F] | |
| `point_mask` | bool [B, N] | only when clouds of different sizes were padded |

Make sure `3 + F` equals the model's `input_dim`; you can check it with
`gc.validate(pc, input_dim=...)`. Normalization is centroid / unit sphere per
cloud by default; `shared_normalization=True` uses one transform for the
whole batch.

### 8.3 VOXEL_GRID → FNO3D (3D U-Net)

```python
v = gc.voxel_grid(src, shape=(48, 64, 96))                 # (D, H, W) = (Z, Y, X)
v = gc.voxel_grid(src, spacing=0.5, bounds=((x0,y0,z0), (x1,y1,z1)))
v = gc.voxel_grid([a, b], shape=64, channels=("occupancy", "sdf", "boundary", "tags"),
                  fields={"heat_source": lambda p: np.exp(-(p**2).sum(1) / 50)})
```

| Array | Shape | Content |
|---|---|---|
| `voxel_fields` | float32 [B, C, D, H, W] | D, H, W = Z, Y, X |
| `channel_names` | string [C] | |
| `origin` | float32 [3] | (x, y, z) centre of voxel [0,0,0] |
| `spacing` | float32 [3] | (dx, dy, dz) |
| `bounds` | float32 [2, 3] | box covered by the cells, `[min; max]` in (x, y, z) |

Channels:

| Channel | Meaning |
|---|---|
| `occupancy` | filled volume, anti-aliased fraction in [0, 1] |
| `occupancy_binary` | 1 inside, 0 outside |
| `sdf` | exact signed distance (normalized units) |
| `tsdf` | `sdf / (truncation × h)`, clipped to [−1, 1] |
| `boundary` | voxels the surface passes through |
| `tag:<name>` / `tags` | boundary voxels belonging to a named surface |
| any name in `fields=` | your own field `f(xyz [N,3]) → [N]` (material, source term, …) |

Geometry is rasterized as filled volume and SDF, never as occupied vertices.
`C` must equal the model's `input_channels`
(`gc.validate(v, input_channels=C)`). With `align_corners=False`, the grid
layout matches `torch.nn.functional.grid_sample` over `bounds`.

### 8.4 PARAMETRIC_SURFACE → Deep NURBS

See §9. Arrays: `degree`, `knot_vector_u/v`, `control_points [Nu,Nv,2]`,
`weights`, `admissible_coefficients`, `lifting_coefficients
[Nu,Nv,output_dim]` and `parametric_coordinates [B,N,2]`, plus the extra
`physical_coordinates`.

---

## 9. Deep NURBS domains

Deep NURBS works on a **2D** physical domain, mapped from the parameter
square (u, v) ∈ [0, 1]².

### 9.1 Constructors (all exact)

```python
S = gc.NURBSSurface.rectangle(0, 2, 0, 1, degree=(3, 3), n_ctrl=(8, 6))
S = gc.NURBSSurface.quad(p00, p10, p11, p01, degree=(2, 2))          # bilinear quadrilateral
S = gc.NURBSSurface.annulus(r_in, r_out, theta0, theta1)             # exact circles
S = gc.NURBSSurface.plate_with_hole(R=1, L=4)                        # classic IGA benchmark
S = gc.NURBSSurface.ruled(curve0, curve1)                            # between two NURBS curves
S = gc.NURBSSurface.coons(bottom, right, top, left)                  # 4 boundary B-splines
S = gc.NURBSSurface.from_map(lambda uv: xy, n_ctrl=(10, 10), degree=(3, 3))   # interpolate any map
```

Curves: `NURBSCurve.line`, `polyline`, `arc` (exact) and `interpolate(fn,
n_ctrl, degree)`; `gc.make_compatible(c1, c2)` gives them the same degree and
knots.

### 9.2 Refinement (exact, the geometry does not change)

```python
S = S.with_degree(3, 3)        # degree elevation
S = S.refine(12, 10)           # knot insertion up to 12 x 10 control points
S = S.insert_knots(u=[0.3], v=[0.5])
```

Elevating the degree adds control points, so elevate first and refine
afterwards when you need exact control-point counts.

### 9.3 Evaluation

```python
xy = S(uv)                     # physical points [N, 2]
J = S.jacobian(uv)             # [N, 2, 2]
R = S.basis(uv)                # rational basis [N, Nu*Nv]
f = S.evaluate_field(coeffs, uv)
S.area(); S.min_jacobian()     # min_jacobian > 0 means the map does not fold
```

### 9.4 Deep NURBS coefficients and the file

* **Admissible coefficients** are 1 inside and 0 on all four boundaries, so
  φ = Σ R_ij a_ij vanishes on the boundary.
* **Lifting coefficients** are a NURBS function equal to the Dirichlet data on
  each boundary (interpolated at the Greville points, so exact for data in
  the NURBS space), Coons-blended inside.

```python
d = gc.parametric_surface(
        S, output_dim=2,
        dirichlet={"bottom": lambda xy: xy, "top": 0.0},   # or one callable/constant for all sides
        n_points=4096, batch=1, method="sobol",            # grid | random | sobol | halton
        include_boundary=0)                                # extra UV points on the edges
gc.save("out/", d)
```

Sides are `bottom` (v = 0), `top` (v = 1), `left` (u = 0) and `right`
(u = 1). Because the geometry is model configuration rather than runtime
input (see ModelInput.md), your loader builds the model configuration from
this file before inference.

---

## 10. Saving, loading and validating

```python
path = gc.save("out/", d)              # a directory -> standard file name; validates first
path = gc.save("out/sample_017.npz", d)
d = gc.load(path)                      # validates after reading
gc.validate(d, input_dim=None, input_channels=None)   # raises gc.SchemaError
```

The validator checks every rule of ModelInput.md:

* dtypes and shapes;
* edges stored in both directions;
* `edge_index` within range;
* mask shapes;
* `bounds = spacing × (W, H, D)`;
* `Ku = Nu + degree_u + 1`;
* clamped, nondecreasing knots;
* positive weights;
* admissible coefficients zero on all four boundaries;
* UV values in [0, 1];
* finite values.

---

## 11. Normalization

Coordinates are normalized as `x_n = (x − center) / scale`, with one
isotropic `scale`, so lengths and SDF values simply divide by `scale`.

| `normalization=` | Effect |
|---|---|
| `"unit_cube"` (graph, voxel default) | bounds centre → 0, largest half extent → 1 |
| `"unit_sphere"` (point cloud default) | centroid → 0, farthest point → 1 |
| `None` / `"none"` | physical millimetres |
| `gc.Normalizer(...)` | your own fixed transform |

To reproduce exactly what the model was trained with:

```python
norm = gc.Normalizer.from_bounds(train_bounds, "unit_cube", id="train_v1")
saved = norm.to_dict()                       # store once
norm = gc.Normalizer.from_dict(saved)        # later
gc.voxel_grid(part, shape=64, normalization=norm)
```

`normalization_id` in each file names the transform.

---

## 12. PINN collocation

```python
col = gc.collocation(src, n_interior=100_000, n_boundary=20_000,
                     band=(10_000, 0.5), method="sobol", normalizer=None, rng=0)
```

| Key | Shape | Content |
|---|---|---|
| `x_interior`, `sdf_interior` | [Ni, 3], [Ni] | interior points (Sobol), distance to the wall |
| `x_boundary`, `normal_boundary`, `tag_boundary` | [Nb, 3], [Nb, 3], [Nb] | wall points, outward normals, tag index |
| `tag_names` | [T] | names for `tag_boundary` |
| `x_band`, `sdf_band` | [Nn, 3], [Nn] | near-wall refinement points |

All arrays are float32 and ready for `torch.from_numpy`.

Because `-sdf_interior` is an exact distance function, Dirichlet conditions
can be imposed exactly with `u(x) = g(x) + d(x)·NN(x)`. For SDF lookups during
training, a `voxel_grid(..., channels=("sdf",))` file can be read on the GPU
with `grid_sample`.

---

## 13. Pictures

`geocore.render` needs matplotlib.

```python
from geocore import render
render.render_mesh(src_or_mesh, "object.png", views=4, title="part",   # photo, 1 or 4 views
                   elev=25, azim=-60, width=1100, height=900, color="#2A9DA6")
img = render.render_image(mesh, elev=30, azim=45)  # RGB float array [H, W, 3]
render.render_point_cloud(pc, "pc.png", color_by="sdf")
render.render_mesh_graph(g, "graph.png")
render.render_voxel_grid(v, "voxels.png")          # mid slices of every channel
render.render_parametric(d, "nurbs.png")           # domain, control net, lifting, admissible
```

`render_mesh` uses geocore's own z-buffer rasterizer (numba, 2×
supersampled, smooth shading). It draws the full-resolution mesh with
correct hidden surfaces, including internal structures such as gyroids seen
through a cut. Without a path, `render_mesh` returns the image array and the
other functions return the matplotlib figure.

---

## 14. The `model_inputs.py` script

`examples/model_inputs.py` builds everything ModelInput.md describes for one
geometry, reports the values and photographs the object.

```bash
python examples/model_inputs.py                              # demo bracket (filleted)
python examples/model_inputs.py --geometry part.stl          # your closed STL/OBJ
python examples/model_inputs.py --geometry lattice           # graded BCC lattice panel
python examples/model_inputs.py --geometry gyroid_sphere --voxel-size 0.25 --out run2
```

Options: `--points` (point cloud size), `--grid D H W`, `--graph-resolution`
(mm), `--uv-points` and `--seed`.

Output folder contents:

| File | Content |
|---|---|
| `mesh_graph.npz`, `point_cloud.npz`, `voxel_grid.npz`, `parametric_surface.npz` | the four representations |
| `voxel_grid_tagged.npz` | per-surface boundary masks (named geometries only) |
| `pinn_collocation.npz` | PINN point sets |
| `geometry_values.json` / `.md` | every array: dtype, shape, min/max/mean, metadata, plus geometry facts (volume, area, bounds, watertightness, tags) |
| `object.png`, `object_front.png` | shaded photos of the 3D object |
| `preview_*.png` | one picture per representation |
| `object.stl` | the surface mesh |

The example output for the bracket is in `examples/output_bracket/`.

---

## 15. Parametric design files

A design file (`.yaml`, `.yml` or `.json`) describes a part by its
parameters. geocore reads the file and builds the geometry from the current
parameter values. You can change any parameter from Python or the command
line and get every output again, without editing the file. It can also
generate whole datasets of variants for training.

### 15.1 Using a design

```python
import geocore as gc

d = gc.Design.load("examples/designs/bracket.yaml")
print(d.describe())                    # parameters, ranges, derived values, constraints
d.parameters                           # {'length': 60.0, 'width': 40.0, ..., 'fillet': 4.0}

d.set(length=80, fillet=2, mount_holes=3)      # checked: type, min/max, choices, constraints
geo = d.build()                        # Implicit (exact) or Voxels (if the file has 'voxels')
g = d.geometry()                       # Geometry adapter with the design's fixed tag list
d.properties()                         # volume, area, bounds, watertight, tags
res = d.outputs("run1/")               # every output listed in the file -> run1/*.npz, photo, ...
res = d.outputs(only=["voxel_grid"])   # in memory, no files

long = d.with_params(length=100)       # a copy; d itself is unchanged
d.save("bracket_80mm.yaml")            # write the design with the current values

# datasets of variants (each folder: the .npz files, parameters.json, design.yaml)
d.sample(200, "dataset/", rng=0)                    # Sobol over all min/max ranges
d.sweep({"fillet": [0, 2, 4], "mount_holes": [0, 2, 4]}, "sweep/")
```

`sample` draws every parameter that has `min`/`max` (or `choices`, or is a
bool), rounds integers, and redraws variants that break a constraint. Both
`sample` and `sweep` write `parameters.csv`, with one row per variant. That
gives you the design parameters as labels next to the NN inputs.

### 15.2 Command line

```bash
python -m geocore examples/designs/bracket.yaml --info
python -m geocore examples/designs/bracket.yaml --out run1
python -m geocore examples/designs/bracket.yaml --set length=80 fillet=2 --out run2
python -m geocore examples/designs/bracket.yaml --set length=80 --only voxel_grid,photo --out run3
python -m geocore examples/designs/bracket.yaml --sample 200 --out dataset
python -m geocore examples/designs/bracket.yaml --sweep fillet=0,2,4 mount_holes=0,2 --out sweep
python -m geocore examples/designs/bracket.yaml --set length=90 --save-design bracket_90.yaml
```

### 15.3 File format

```yaml
name: bracket                  # optional
description: ...               # optional
units: mm                      # optional (default mm)

parameters:                    # each: a value, or a mapping
  length:  {value: 60, min: 40, max: 100, description: plate length, unit: mm}
  holes:   {value: 3, min: 0, max: 6, type: int}
  cell:    {value: bcc, choices: [bcc, fcc, octet]}
  ribs_on: {value: true}
  radius:  4.5                 # plain value: float, no range

derived:                       # computed in order from parameters (and earlier derived values)
  half: length / 2

constraints:                   # every expression must be true, otherwise the variant is rejected
  - radius < half - 2

shapes:                        # named sub-shapes, used with {ref: name}
  hole: {cylinder: {center: [0, 0, 0], axis: z, height: 20, radius: radius}}

shape: ...                     # the geometry tree (required), see 15.5

voxels:                        # optional: voxelize the shape, then apply PicoGK operations
  voxel_size: 0.3
  backend: auto                # auto | picogk | dense
  operations:
    - fillet: 2

outputs:                       # optional (default: mesh_graph, point_cloud, voxel_grid)
  voxel_grid: {shape: [48, 64, 96], channels: [occupancy, sdf, tags]}
```

Parameter types are `float` (default), `int`, `bool`, `choice` (with
`choices`) and `str`. The type is inferred from the value when not given.

### 15.4 Expressions

Every number in `shape`, `voxels` and `outputs` may be an expression string
using parameter and derived names. Vectors are lists of numbers or
expressions, such as `[0, 0, "t / 2"]`. In YAML you can usually write
`t / 2` without quotes, but quote it when it starts with `-` or `[`.

| Allowed | |
|---|---|
| arithmetic | `+ - * / // % **`, unary `-` |
| logic | `< <= > >= == !=`, `and or not`, `a if cond else b` |
| constants | `pi e tau True False` |
| functions | `sin cos tan asin acos atan atan2 sqrt exp log log10 abs min max floor ceil round radians degrees hypot clamp int` |
| substitution | `"$name"` gives the parameter value itself (for example a text choice like a lattice cell type) |

Expressions are evaluated by a restricted evaluator: no Python `eval`, no
attribute access, no imports. A design file cannot run code.

### 15.5 Shape tree

A shape is a mapping with exactly one shape type, plus optional modifiers.
A list of shapes means their union.

**Primitives** (arguments in braces):

| Type | Arguments |
|---|---|
| `sphere` | `center`, `radius` |
| `box` | `center`, `size` (full lengths), `rounding`; or `min`, `max` |
| `cylinder` | `a`, `b`, `radius`; or `center`, `axis` (`x`/`y`/`z` or vector), `height`, `radius` |
| `cone` | `a`, `b`, `ra`, `rb` |
| `beam` / `capsule` | `a`, `b`, `radius` or `ra`, `rb` |
| `torus` | `center`, `major`, `minor` |
| `plane` | `point`, `normal` (half space, unbounded) |
| `gyroid` / `schwarzp` / `diamond` | `period`, `wall`, `solid`, `level`, `origin` (unbounded) |
| `cubic_strut_lattice` | `center`, `size`, `cell`, `radius` |
| `lattice` | `cell` (`bcc bccz fcc octet cubic diamond` or `$param`), `min`, `max`, `cell_size`, `radius` (may use `x y z` for graded struts) |
| `mesh` | `file` (STL/OBJ, relative to the design file), `scale`, `translate` |

**Operations:**

| Type | Value |
|---|---|
| `union`, `intersection` | list of shapes |
| `difference` | list: the first shape minus all the others |
| `complement` | one shape |
| `smooth_union`, `smooth_difference`, `smooth_intersection` | `{k: blend radius, shapes: [...]}` |
| `linear_array` | `{shape, count, step: [dx, dy, dz], index: i}`: copy *i* is moved by `step × i` |
| `grid_array` | `{shape, counts: [nx, ny, nz], spacing: [sx, sy, sz]}`; indices `i j k` |
| `polar_array` | `{shape, count, axis, center, angle: 360}`: copies rotated about the axis |
| `ref` | the name of a shape in `shapes` |

Inside an array the copy index (`i`, or `i j k` for `grid_array`) is
available in the child's expressions, so copies can differ, for example fins
whose height depends on `i`.

**Modifiers** (on any shape, applied in this order): `offset: d`,
`shell: t`, `trim: {min, max}`, `scale: s`,
`rotate: {axis, angle (degrees), center}`, `translate: [x, y, z]`,
`tag: name`, `enabled: expression` (false drops the shape).

### 15.6 Voxel operations

Each operation in `voxels.operations` is a mapping with one key: `fillet: r`,
`smoothen: r`, `offset: d`, `double_offset: [d1, d2]`, `shell: t`,
`trim: {min, max}`, `add: <shape>`, `subtract: <shape>` and
`intersect: <shape>` (for example a gyroid infill).

Boundary tags still come from the shape tree after voxel operations. Each
wall point takes the tag of the closest tagged surface.

### 15.7 Outputs

| Key | Builds | Options |
|---|---|---|
| `mesh_graph` | `mesh_graph.npz` | as `gc.mesh_graph` (`kind`, `resolution`, `n_interior`, `k`, `node_features`, …) |
| `point_cloud` | `point_cloud.npz` | as `gc.point_cloud` (`n_points`, `mode`, `features`, …) |
| `voxel_grid` | `voxel_grid.npz` | as `gc.voxel_grid` (`shape`, `spacing`, `channels`, `fields` not available in files) |
| `parametric_surface` | `parametric_surface.npz` | `domain` (`rectangle`/`quad`/`annulus` (angles in degrees)/`plate_with_hole`), `degree`, `n_ctrl`, `output_dim`, `n_points`, `dirichlet` |
| `collocation` | `pinn_collocation.npz` | as `gc.collocation` (`n_interior`, `n_boundary`, `band: [n, width]`) |
| `photo` | `object.png` | as `render.render_mesh` (`views`, `width`, `height`, `elev`, `azim`, `color`) |
| `stl` | `object.stl` | |
| `summary` | volume, area, bounds, tags in `parameters.json` | |

In outputs, a string is evaluated only when it is an expression of known
names. Words such as `mixed` or `occupancy` stay text.

`dirichlet` for the Deep NURBS file takes:
* a number;
* an expression in `x`, `y` and the parameters;
* a list of those, one per output component;
* a mapping `{bottom|top|left|right: ...}`.

**Stable tag channels:** `Design.tag_names` lists every tag the file
declares, including tags of parts that are currently disabled. All variants
therefore have the same `tag:*` channels, in the same order; a disabled
part's mask is simply all zeros.

### 15.8 Example designs

The examples are in `examples/designs/`; a photo of each is in its
`output_<name>/` folder.

| File | Shows |
|---|---|
| `bracket.yaml` | the PicoGK WebModel bracket: derived positions, constraints, a `linear_array` of mounting holes switched by `enabled`, PicoGK fillet |
| `heatsink.yaml` | fin array (`count` is an int parameter), tapered fins from planes, `base`/`fins` tags for heat-transfer BCs, collocation with a near-wall band |
| `flange.yaml` | `polar_array` bolt circle, `smooth_union` hub, exact implicit (no voxels), a Deep NURBS annulus output |
| `lattice_panel.yaml` | `lattice` with a `$cell` choice parameter and a strut radius graded in `z`, face sheets, voxel `trim` |

### 15.9 Designs from Python

A design is plain data, so you can also build it in code:

```python
spec = {
    "parameters": {"r": {"value": 10, "min": 5, "max": 20}, "wall": 1.0},
    "shape": {"sphere": {"radius": "r"}, "shell": "-wall", "tag": "skin"},
    "outputs": {"point_cloud": {"n_points": 1024}},
}
d = gc.Design(spec)
for r in (6, 10, 14):
    d.set(r=r).outputs(f"sphere_r{r}/")
```

---

## 16. Accuracy and speed

| Quantity | Result |
|---|---|
| primitive SDFs, CSG queries | exact (compiled evaluation is bit-identical to numpy) |
| mesh SDF | exact for the mesh |
| `Voxels.sdf` near the surface, 10 mm sphere at 0.3 mm voxels | ≤ 0.03 mm |
| redistancing | ≈ 1% of a voxel near smooth surfaces, < 0.25 voxel far away; sharp edges rounded by about h/3 |
| volumes (sphere, offset, shell) | ≤ 0.3–0.5% |
| fillet, PicoGK vs dense engine | ≤ 0.3% |
| NURBS constructors and refinement | exact to machine precision |

Timings on the development machine (16 threads):

| Operation | Time |
|---|---|
| CSG SDF, 1M points | 16 ms |
| lattice SDF, 5.6k beams, 1M points | 0.1 s |
| exact mesh SDF, 94k triangles, 1M points | 1.5 s |
| `Voxels(part)` at 0.3 mm (≈ 4.5M voxels): dense / PicoGK | 0.07 s / 0.19 s |
| `fillet(4)` at 0.3 mm: dense / PicoGK | 3.3 s / 5.0 s |
| redistancing, 9.1M voxels | 1.9 s |
| `mesh_graph` / `point_cloud` / `voxel_grid` 64³ / `collocation` 120k | 0.86 / 0.23 / 0.06 / 0.24 s |

Reproduce these with `python examples/benchmark.py`.

---

## 17. How it works inside

| Module | Role |
|---|---|
| `_native.py` | ctypes binding to `picogk.26.2`. Mesh transfer runs as numba loops that call the C functions directly (1.5M triangles in 70 ms). Implicit rendering uses a numba `cfunc` as PicoGK's distance callback, reading a precomputed grid. |
| `_compiled.py` | compiles a CSG tree into a stack program executed per point in parallel; unknown leaves (meshes, lattices, user functions) enter as precomputed columns |
| `_kernels.py` | BVH (mesh and beam distance), trilinear interpolation, manifold surface nets, jump flooding |
| `fields.py` | grids, sampling, redistancing, meshing |
| `voxels.py` | `Voxels` with the two engines |
| `implicit.py`, `mesh.py`, `lattice.py`, `nurbs.py` | geometry |
| `sampling.py`, `representations.py`, `io.py`, `normalize.py` | outputs |
| `render.py` | pictures; z-buffer rasterizer in `_kernels.rasterize` |
| `design.py`, `__main__.py` | parametric design files, safe expression evaluator, variants, command line |

**Redistancing** turns any field with the right zero surface into an exact
distance field:

1. Nodes within 2 voxels of the surface get sub-voxel foot points (Newton on
   the trilinear field). Foot points that do not reach the surface, as in
   PicoGK's clamped band, are dropped.
2. Every other node inherits the foot point of the nearest such node
   (Euclidean distance transform).
3. Jump-flooding passes then pick better foot points.
4. A final pass steps along the gradient onto the surface and keeps the
   shorter of the two distances.

**Meshing** uses manifold surface nets: one vertex per surface component in
each cell, so the mesh is always watertight. Vertices are then projected onto
the exact surface.

**Workarounds for PicoGK 26.2:**

* `Voxels_IntersectImplicit` throws on thin features, so geocore renders the
  implicit and uses `BoolIntersect` instead.
* PicoGK's mesh → voxels → volume recipe fills cavities, so geocore uses the
  exact mesh volume.

---

## 18. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| a hole is missing | operator precedence: write `(a \| b) - c` |
| `unbounded implicit` error | intersect `Gyroid`/`Plane` with a bounded shape, or pass `bounds=` |
| `voxel fields must share voxel size and backend` | create both from the same `Library` |
| wrong SDF signs on an STL | mesh not closed or inconsistently oriented: check `mesh.is_watertight()` |
| slow first call | numba compiles once per machine; the code is cached afterwards |
| `picogk_available: False` on Windows | `GEOCORE_BACKEND=dense` is set, or `geocore/_bin/win-x64` is missing |
| point cloud `3 + F != input_dim` | pick `features=` to match the model, or `features=None` |

---

## 19. API reference

### Top level (`import geocore as gc`)

* **Geometry:** `Sphere`, `Box`, `Cylinder`, `Beam`, `Capsule`, `Cone`,
  `Torus`, `Plane`, `Gyroid`, `SchwarzP`, `Diamond`, `CubicStrutLattice`,
  `FunctionImplicit`
* **CSG:** `Union`, `Intersection`, `Difference`, `Complement`,
  `SmoothUnion`, `SmoothDifference`, `SmoothIntersection`, `Offset`, `Shell`,
  `Transform`, `union`, `intersection`, `difference`
* **Containers:** `BBox`, `Mesh`, `MeshImplicit`, `Lattice`,
  `LatticeImplicit`, `GridSpec`, `GridImplicit`
* **Voxels:** `Library`, `Voxels`, `set_voxel_size`, `default_library`
* **Fields:** `redistance`, `sample_implicit`, `mesh_from_grid`
* **Sampling:** `Geometry`, `collocation`
* **Representations:** `mesh_graph`, `point_cloud`, `voxel_grid`,
  `batch_graphs`, `parametric_surface`
* **NURBS:** `NURBSCurve`, `NURBSSurface`, `make_compatible`, `sample_uv`
* **I/O:** `save`, `load`, `validate`, `SchemaError`, `Normalizer`
* **Designs:** `Design`, `DesignError`
* **Outputs:** `obj.output(rep, **kw)` on every geometry, `output(src, rep)`,
  `ModelInput`, `SCHEMA`, `load_geometry` / `read`
* **Info:** `info()`, `PICOGK_AVAILABLE`, `PICOGK_VERSION`, `SCHEMA_VERSION`

### Signatures

```text
Implicit.sdf(p) / sdf_tagged(p) / gradient(p, eps=None) / normal(p, eps=None)
        / project(p, iters=8) / contains(p) / tagged(name) / tag_names() / bounds
        / union(*o) / subtract(*o) / intersect(*o) / smooth_union(o, k) / smooth_subtract(o, k)
        / smooth_intersect(o, k) / offset(d) / shell(t) / trim(vmin, vmax)
        / translate(t) / rotate(axis, angle_rad, center) / scale(s, center) / transform(M)

Voxels(source=None, *, lib=None, bounds=None)
Voxels.sphere(center, r, lib) / beam(a, ra, b, rb, lib) / box(vmin, vmax, lib)
      / mesh_shell(mesh, t, lib) / combine_all(list)
v.render_implicit(imp, bounds=None) / render_mesh(mesh) / render_lattice(lat) / intersect_implicit(imp)
v.bool_add(o) / bool_subtract(o) / bool_intersect(o) / offset(d) / double_offset(d1, d2)
 / triple_offset(d) / smoothen(d) / over_offset(first, final=0) / fillet(r) / shell(t)
 / shell2(neg, pos, smooth_inner=0) / trim(vmin, vmax) / copy()
v.to_mesh() / calculate_properties() / volume() / bounding_box() / is_empty() / save_stl(path)
v.sdf(p) / is_inside(p) / surface_normal(p) / closest_point_on_surface(p)
 / raycast_to_surface(origins, dirs, max_dist=None) / to_dense(grid=None, exact=True) / as_implicit()

Mesh(vertices, faces); Mesh.load(path, weld=True); Mesh.box(vmin, vmax); Mesh.concatenate(list)
m.sdf(p, return_closest=False) / unsigned_distance(p) / closest_points(p) / sample_surface(n, rng, return_face)
 / volume() / area() / face_normals() / vertex_normals() / vertex_areas() / mean_curvature() / edges()
 / is_watertight() / welded(tol) / translated(t) / scaled(s) / transformed(M) / save_stl(path) / as_implicit()

Lattice(); lat.add_beam(a, b, ra, rb=None, round_cap=True) / add_sphere(c, r) / add_beams(a, b, ra, rb)
Lattice.from_cells(cell_type, vmin, vmax, cell_size, radius); lat.as_implicit(); lat.beams; lat.bounds

Geometry(src, resolution=None, face_tags=None, tag_names=None)
g.sdf / normal / tags / surface_mesh(resolution) / sample_surface(n, rng, project=True)
 / sample_volume(n, rng, method='sobol', inside=True, bounds=None) / sample_band(n, width, rng)
collocation(geom, n_interior, n_boundary, *, rng, method='sobol', band=None, normalizer=None)

mesh_graph(geometry, *, kind='surface', resolution=None, n_interior=20000, k=12, radius=None,
           node_features=None, normalization='unit_cube', units='mm', face_tags=None, tag_names=None, rng=None)
point_cloud(geometries, n_points=2048, *, mode='surface', boundary_fraction=0.5,
            features=('nx','ny','nz','sdf','is_boundary'), normalization='unit_sphere',
            shared_normalization=False, pad_to=None, method='sobol', units='mm', resolution=None, rng=None)
voxel_grid(geometries, *, shape=None, spacing=None, bounds=None, margin=0.05,
           channels=('occupancy','sdf','boundary'), fields=None, truncation=3.0,
           normalization='unit_cube', units='mm', exact_sdf=True)
batch_graphs(graphs)
parametric_surface(surface, *, output_dim=1, dirichlet=None, n_points=4096, batch=1,
                   method='sobol', uv=None, include_boundary=0, units='mm', rng=None)

NURBSSurface(degree, knots_u, knots_v, ctrl[Nu,Nv,2], weights=None)
 .rectangle / quad / annulus / plate_with_hole / ruled / coons / from_map
 s(uv) / jacobian(uv) / basis(uv, derivatives=False) / evaluate_field(c, uv) / area() / min_jacobian()
 / elevate(tu, tv) / with_degree(pu, pv) / refine(n_u, n_v) / insert_knots(u, v) / boundary_curve(side)
 / admissible_coefficients() / lifting_coefficients(dirichlet, output_dim) / validate()
NURBSCurve(degree, knots, ctrl, weights=None).line / polyline / arc / interpolate
 c(t) / elevate(t) / insert_knots(k) / refine(n) / reversed()

GridSpec(origin, spacing, shape); GridSpec.from_bounds(bbox, spacing=None, shape=None); GridSpec.lattice(bbox, h, pad)
redistance(phi, spacing, origin, *, iters=2, band=2, refine=True, jfa='short')
Normalizer(center, scale, method, id); .from_bounds(bbox, method, margin, id) / from_points(p, method)
 / from_dict(d) / to_dict() / points(p) / inverse_points(q) / lengths(d) / areas(a)
save(path, data, compressed=True, validate_first=True); load(path, validate_after=True)
validate(d, input_dim=None, input_channels=None)

Design(data, base_dir='.'); Design.load(path); Design.from_string(text, base_dir, fmt='yaml'|'json')
d.parameters / parameter_info / tag_names / set(**values) / with_params(**values) / values() / violations()
 / build_shape() / build() / geometry(resolution) / properties() / outputs(out_dir=None, only=None, rng=None)
 / sample(n, out_dir=None, rng=None, method='sobol', names=None, only=None) / sweep(grid, out_dir=None, only=None)
 / describe() / to_dict() / save(path)
python -m geocore design.yaml [--info] [--set k=v ...] [--out dir] [--only a,b] [--sample N] [--sweep k=v1,v2 ...] [--save-design f]
```
