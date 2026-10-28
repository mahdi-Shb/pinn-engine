# Neural-network outputs: `part.output(...)`

This page documents every neural-network input geocore produces, following
`ModelInput.md`, and the one method you use to get them.

```python
import geocore as gc

gc.set_voxel_size(0.3)
part = (gc.Box((0,0,4),(60,40,8)) | gc.Cylinder((12,0,0),(12,0,16),9)) - gc.Cylinder((12,0,-5),(12,0,25),4.5)
vox = gc.Voxels(part).fillet(4)

part_mg = part.output("MESH_GRAPH")                                   # every MESH_GRAPH array
part_mg = part.output("MESH_GRAPH", coordinates=True, node_features=True)   # only these two
vox_pc  = vox.output("POINT_CLOUD", n_points=4096, mode="mixed")
vox_vg  = vox.output("VOXEL_GRID", shape=64)

stl = gc.load_geometry("PicoGK_Modeler_preview.stl")                 # a 3D file
stl_mg = stl.output("MESH_GRAPH")
```

---

## Contents

1. [Which objects have `.output()`](#1-which-objects-have-output)
2. [Representations and names](#2-representations-and-names)
3. [Choosing arrays](#3-choosing-arrays)
4. [The returned object (`ModelInput`)](#4-the-returned-object-modelinput)
5. [Loading 3D files](#5-loading-3d-files)
6. [Common metadata](#6-common-metadata)
7. [MESH_GRAPH](#7-mesh_graph--meshgraphnet)
8. [POINT_CLOUD](#8-point_cloud--pointnet)
9. [VOXEL_GRID](#9-voxel_grid--fno3d)
10. [PARAMETRIC_SURFACE](#10-parametric_surface--deep-nurbs)
11. [PINN_COLLOCATION](#11-pinn_collocation)
12. [Normalization and units](#12-normalization-and-units)
13. [Worked example: `PicoGK_Modeler_preview.stl`](#13-worked-example-picogk_modeler_previewstl)
14. [Using the outputs in PyTorch](#14-using-the-outputs-in-pytorch)

---

## 1. Which objects have `.output()`

The method is the same however the geometry was made:

| How the part was made | Object | Example |
|---|---|---|
| shapes and CSG | any `Implicit` | `part = gc.Box(...) - gc.Cylinder(...)` |
| PicoGK voxel operations | `Voxels` | `vox = gc.Voxels(part).fillet(4)` |
| a 3D file | `Mesh` | `stl = gc.load_geometry("part.stl")` |
| a beam lattice | `Lattice` | `lat = gc.Lattice.from_cells("bcc", ...)` |
| a parametric design file | `Design` | `d = gc.load_geometry("bracket.yaml"); d.set(length=80)` |
| any of the above, wrapped | `Geometry` | `gc.Geometry(src)` |
| a 2D NURBS domain | `NURBSSurface` | `gc.NURBSSurface.plate_with_hole()` (only `PARAMETRIC_SURFACE`) |

There is also a function form, `gc.output(src, "MESH_GRAPH", ...)`, and
`src.outputs(["MESH_GRAPH", "POINT_CLOUD"])` returns several
representations at once.

Repeated calls on the same object reuse its surface mesh and distance
field, so asking for a second representation is fast. The cache is cleared
automatically when a `Voxels` field is modified or a design parameter
changes.

---

## 2. Representations and names

| Representation | Model | Also accepted (case-insensitive) | Default file |
|---|---|---|---|
| `MESH_GRAPH` | MeshGraphNet (GCN, GAT) | `mesh_graph`, `graph`, `mesh`, `mgn` | `mesh_graph.npz` |
| `POINT_CLOUD` | PointNet (PointNet++, Transolver, GINO, SIREN) | `point_cloud`, `points`, `pointnet`, `pc` | `point_cloud.npz` |
| `VOXEL_GRID` | FNO3D (3D U-Net) | `voxel_grid`, `voxel`, `voxels`, `grid`, `fno`, `fno3d` | `voxel_grid.npz` |
| `PARAMETRIC_SURFACE` | Deep NURBS | `nurbs`, `deep_nurbs`, `parametric` | `parametric_surface.npz` |
| `PINN_COLLOCATION` | PINN training points (geocore extra) | `collocation`, `pinn` | `pinn_collocation.npz` |

---

## 3. Choosing arrays

`output()` accepts two kinds of keyword arguments:

* **Array names** (`coordinates=True`, `batch=False`, …) choose what is
  returned. The array names of each representation are listed in §7–§11.
* **Everything else** is an option of the builder: `n_points`, `mode`,
  `shape`, `channels`, `kind`, `normalization`, ….

| Call | Result |
|---|---|
| `part.output("MESH_GRAPH")` | the 5 metadata strings + every array of the schema |
| `part.output("MESH_GRAPH", coordinates=True, node_features=True)` | only `coordinates` and `node_features` |
| `part.output("MESH_GRAPH", coordinates=True, metadata=True)` | metadata + `coordinates` |
| `part.output("MESH_GRAPH", batch=False)` | everything except `batch` |
| `part.output("MESH_GRAPH", metadata=False)` | every array, no metadata |
| `part.output("MESH_GRAPH", extras=True)` | everything + geocore extras (`faces`, `normalization_center`, `normalization_scale`, `physical_coordinates`) |

Rules:

* Selected arrays come back in schema order.
* A selection flag must be `True` or `False`. Python's own `True` is
  required; a misspelling such as `Ture` is a Python `NameError`.
* Asking for an optional array the builder skips gives a correct value:
  * `point_mask=True` on an unpadded cloud gives an all-`True` mask.
  * `batch=True` gives zeros.
* A misspelled builder option raises an error that lists the valid array
  names and points to the builder's documentation.

---

## 4. The returned object (`ModelInput`)

`output()` returns a `ModelInput`, which is a normal Python `dict` of numpy
arrays with a few additions:

| Member | Meaning |
|---|---|
| `mg["coordinates"]` or `mg.coordinates` | an array |
| `list(mg)`, `mg.keys()` | array names |
| `mg.representation` | `"MESH_GRAPH"` and so on |
| `print(mg)` / `mg.summary()` | table of names, dtypes, shapes and value ranges |
| `mg.complete` | `True` when all required arrays + metadata are present |
| `mg.validate()` | checks every ModelInput.md rule (complete outputs only) |
| `mg.save("out/")` | writes `out/mesh_graph.npz`; complete outputs are validated first |
| `mg.save("my_name.npz")` | any file name; a selection is saved exactly as selected |
| `mg.to_torch("cuda")` | `{name: torch.Tensor}`; strings stay Python `str` / `list` |
| `mg.numpy()` | plain `dict` of numpy arrays |

Files written by `save` load back with `gc.load(path)` (validated) or
`np.load(path)` (no `allow_pickle` needed).

---

## 5. Loading 3D files

```python
part = gc.load_geometry("part.stl")             # also gc.read(...)
part = gc.load_geometry("part.stl", scale=1000)  # file in metres -> mm
part = gc.load_geometry("part.stl", repair=True, resolution=0.2)
d    = gc.load_geometry("bracket.yaml")          # parametric design file
```

| Extension | Becomes |
|---|---|
| `.stl` (binary and ASCII), `.obj` | `Mesh` |
| `.ply`, `.off`, `.glb`, `.3mf`, … | `Mesh` (needs `pip install trimesh`) |
| `.yaml`, `.yml`, `.json` | `Design` (parametric, see the main documentation §15) |

**Repair (`repair="auto"`, the default):**

* **Meshes with holes** (open edges) are re-meshed into a closed surface,
  with a warning. The inside/outside of the hole region is decided by the
  generalized winding number, which closes holes smoothly.
* **Closed meshes with a few non-manifold edges** are kept unchanged, with a
  warning. These are common in voxel-based exporters such as the PicoGK web
  modeler, and their signed distance is still correct.
* **`repair=True`** always re-meshes; **`repair=False`** never does.

Re-meshing keeps the file's level of detail (grid = mean edge length) and
snaps vertices onto the original surface. Sharp edges come out slightly
jagged at that size; pass a smaller `resolution` for crisper edges.

Useful checks: `part.is_watertight()`, `part.manifold_report()`,
`part.volume()`, `part.area()`, `part.bounds`, `part.winding_number(p)`.

You can also turn a file into voxels to apply PicoGK operations before
taking outputs: `gc.Voxels(part).fillet(1).output("VOXEL_GRID")`.

---

## 6. Common metadata

These five strings are in every complete output, as required by
ModelInput.md:

| Key | Type | Value |
|---|---|---|
| `schema_version` | string | `"1.0"` |
| `representation` | string | `MESH_GRAPH`, `POINT_CLOUD`, `VOXEL_GRID` or `PARAMETRIC_SURFACE` |
| `coordinate_system` | string | `cartesian_xyz_right_handed` (Deep NURBS: `cartesian_xy_2d`) |
| `units` | string | `mm`, or `mm/normalized` when coordinates are normalized |
| `normalization_id` | string | the normalization used (§12); `none` when none |

With `extras=True` you also get `normalization_center` (float32 `[3]`) and
`normalization_scale` (float32 scalar), which are enough to undo the
normalization.

---

## 7. MESH_GRAPH → MeshGraphNet

```python
mg = part.output("MESH_GRAPH")                          # surface graph (default)
mg = part.output("MESH_GRAPH", kind="volume", n_interior=20000, k=12)
```

### Arrays

| Array | dtype | Shape | ModelInput.md | Content |
|---|---|---|---|---|
| `coordinates` | float32 | [N, 3] | recommended | node positions x, y, z (normalized, §12) |
| `node_features` | float32 | [N, Fn] | **required** | one row per node; columns named in `node_feature_names` |
| `edge_index` | int64 | [2, E] | **required** | row 0 = sender, row 1 = receiver; every undirected edge appears in **both** directions |
| `edge_features` | float32 | [E, Fe] | **required** | `dx dy dz distance` with `d = x[receiver] − x[sender]` (normalized units) |
| `batch` | int64 | [N] | optional | graph index of each node: all 0 for one part; `gc.batch_graphs` for several |
| `node_feature_names` | string | [Fn] | | column names of `node_features` |
| `edge_feature_names` | string | [Fe] | | `["dx", "dy", "dz", "distance"]` |

### Graph kinds

| `kind=` | Nodes | Edges |
|---|---|---|
| `"surface"` (default) | vertices of the watertight surface mesh | mesh edges |
| `"volume"` | `n_interior` interior points (Sobol) + all surface vertices | symmetric k-nearest-neighbour edges (`k`, or `radius=`) + the surface mesh edges |

### Node feature columns

Choose columns with `node_features=[...]` (a list here, not a flag).
`"tags"` expands to one `tag:<name>` column per named surface.

| Column | Surface graph | Volume graph | Meaning |
|---|---|---|---|
| `nx ny nz` | default | default | outward unit normal (angle-weighted at vertices; SDF gradient for interior nodes) |
| `vertex_area` | default | – | area around the vertex: one third of each touching triangle (normalized units²) |
| `mean_curvature` | default | – | signed mean curvature, positive on convex regions (1 / normalized length). Large values mark sharp edges |
| `sdf` | option | default | signed distance to the wall: 0 on the surface, negative inside |
| `is_boundary` | option | default | 1 for wall nodes, 0 for interior nodes |
| `tag:<name>` | default | default | 1 if the node lies on that named surface. Names come from `.tagged()` or a design file; an untagged leaf is named after its shape (`tag:box`); a mesh has `tag:surface` |

Other options:
* `resolution` (mesh edge length in mm when meshing implicits or voxels);
* `normalization` (`"unit_cube"` default, `"unit_sphere"`, `None` or a
  `gc.Normalizer`);
* `face_tags` and `tag_names` (per-triangle labels for meshes);
* `rng`.

---

## 8. POINT_CLOUD → PointNet

```python
pc = part.output("POINT_CLOUD", n_points=4096, mode="mixed")
pc = gc.output([part_a, part_b], "POINT_CLOUD", n_points=[2048, 1500])   # batch, padded
```

### Arrays

| Array | dtype | Shape | ModelInput.md | Content |
|---|---|---|---|---|
| `coordinates` | float32 | [B, N, 3] | **required** | point positions (B = 1 for one part) |
| `point_features` | float32 | [B, N, F] | optional | columns named in `point_feature_names` |
| `point_mask` | bool | [B, N] | optional | `True` for real points; only produced when clouds of different sizes are padded (or when you ask for it) |
| `point_feature_names` | string | [F] | | feature column names |

The model's `input_dim` must equal `3 + F`. With `features=None` you get
coordinates only (F = 0).

### Options

| Option | Values | Meaning |
|---|---|---|
| `n_points` | int or list | points per cloud |
| `mode` | `surface` (default), `volume`, `mixed` | on the wall (area-uniform) / inside (Sobol) / both |
| `boundary_fraction` | 0.5 | share of wall points in `mixed` mode |
| `features` | default `("nx","ny","nz","sdf","is_boundary")` | also `"tags"`; `None` for none |
| `normalization` | `"unit_sphere"` (default), `"unit_cube"`, `None`, `Normalizer` | per cloud unless `shared_normalization=True` |
| `pad_to` | int | pad every cloud to this length |
| `method` | `sobol`, `halton`, `random`, `grid` | interior sampling |

Feature meanings: `nx ny nz` is the outward normal (SDF gradient for
interior points), `sdf` is the signed distance (normalized), and
`is_boundary` is 1/0. Wall points lie exactly on the surface: implicits and
voxels are projected onto the zero level.

---

## 9. VOXEL_GRID → FNO3D

```python
vg = part.output("VOXEL_GRID", shape=(48, 64, 96))                 # (D, H, W)
vg = part.output("VOXEL_GRID", spacing=0.5, bounds=((-35,-25,-5),(35,25,50)))
vg = part.output("VOXEL_GRID", shape=64, channels=["occupancy", "sdf", "boundary", "tags"])
```

### Arrays

| Array | dtype | Shape | ModelInput.md | Content |
|---|---|---|---|---|
| `voxel_fields` | float32 | [B, C, D, H, W] | **required** | D, H, W = **Z, Y, X** (last axis is x) |
| `channel_names` | string | [C] | | channel order |
| `origin` | float32 | [3] | recommended | **(x, y, z)** centre of voxel [0,0,0] |
| `spacing` | float32 | [3] | recommended | **(dx, dy, dz)** voxel size |
| `bounds` | float32 | [2, 3] | recommended | `[[xmin, ymin, zmin], [xmax, ymax, zmax]]` covered by the cells |

The voxel at index `[b, c, k, j, i]` has its centre at
`origin + (i, j, k) * spacing`. `bounds` equals
`origin − spacing/2 … origin + (W−1, H−1, D−1)·spacing + spacing/2`. This is
the layout of `torch.nn.functional.grid_sample(..., align_corners=False)`
over `bounds`. The model's `input_channels` must equal C.

### Channels (`channels=[...]`)

| Channel | Values | Meaning |
|---|---|---|
| `occupancy` | 0…1 | filled volume, anti-aliased (fraction of the voxel inside) |
| `occupancy_binary` | 0/1 | voxel centre inside |
| `sdf` | normalized length | exact signed distance (negative inside) |
| `tsdf` | −1…1 | `sdf / (truncation · h)` clipped; `truncation=3` by default |
| `boundary` | 0/1 | the surface passes through the voxel |
| `tag:<name>`, or `tags` for all | 0/1 | boundary voxels on that named surface (boundary-condition masks) |
| any name given in `fields={name: f}` | float | your own field `f(xyz [N,3] mm) -> [N]` (material, heat source, …) |

The geometry is rasterized as a filled volume and an SDF, never as
"occupied vertices". The SDF is exact even for `Voxels`: PicoGK's
±1.5-voxel narrow band is redistanced to a full distance field.

Other options: `margin` (0.05 when `bounds` is not given), `normalization`
(`"unit_cube"` by default; maps the grid box to [−1, 1] on its longest side)
and `exact_sdf`.

---

## 10. PARAMETRIC_SURFACE → Deep NURBS

Deep NURBS works on a 2D physical domain, so this representation comes from
a `NURBSSurface`, or from a design file whose `outputs.parametric_surface`
defines the domain. Asking a 3D part for it raises an explanatory error.

```python
s = gc.NURBSSurface.plate_with_hole(R=1, L=4).with_degree(3, 3).refine(12, 10)
ps = s.output(output_dim=2, n_points=4096, dirichlet={"bottom": lambda xy: xy})
ps = part.output("PARAMETRIC_SURFACE", surface=s)       # explicit domain for any object
ps = design.output("PARAMETRIC_SURFACE")                # the domain in the design file
```

| Array | dtype | Shape | Content |
|---|---|---|---|
| `degree` | int64 | [2] | (p_u, p_v) |
| `knot_vector_u`, `knot_vector_v` | float32 | [Ku], [Kv] | clamped, nondecreasing, on [0, 1]; Ku = Nu + p_u + 1 |
| `control_points` | float32 | [Nu, Nv, 2] | physical (x, y); index [i along u, j along v] |
| `weights` | float32 | [Nu, Nv] | positive rational weights |
| `admissible_coefficients` | float32 | [Nu, Nv] | 1 inside, **0 on all four boundaries**, so φ = Σ R_ij a_ij vanishes on the boundary |
| `lifting_coefficients` | float32 | [Nu, Nv, output_dim] | NURBS function equal to the Dirichlet data on the boundary, Coons-blended inside |
| `parametric_coordinates` | float32 | [B, N, 2] | UV samples in [0, 1]² |
| `physical_coordinates` (extra) | float32 | [B, N, 2] | the (x, y) of each UV sample |

Options: `output_dim`, `dirichlet` (a number, a function `g(xy) -> [N]`
or `[N, output_dim]`, or `{bottom|top|left|right: ...}`), `n_points`,
`batch`, `method` (`sobol`, `grid`, `random`, `halton`) and
`include_boundary`.

Sides: bottom is v = 0, top is v = 1, left is u = 0, right is u = 1.

---

## 11. PINN_COLLOCATION

This is not part of ModelInput.md: it gives PINN training points through
the same method.

```python
col = part.output("PINN", n_interior=100_000, n_boundary=20_000, band=(10_000, 0.5))
```

| Array | Shape | Content |
|---|---|---|
| `x_interior`, `sdf_interior` | [Ni, 3], [Ni] | interior points and their (negative) wall distance |
| `x_boundary`, `normal_boundary` | [Nb, 3], [Nb, 3] | wall points and outward normals |
| `tag_boundary`, `tag_names` | [Nb], [T] | named surface of each wall point |
| `x_band`, `sdf_band` | [Nn, 3], [Nn] | near-wall points (with `band=(n, width_mm)`) |

These coordinates are in millimetres unless `normalizer=` is given.
`-sdf_interior` is an exact distance function, so a hard Dirichlet condition
can be written as `u = g + d·NN`.

---

## 12. Normalization and units

* Geometry is in **mm** (PicoGK units); `units` says `mm/normalized` when
  coordinates were normalized.
* The normalization is `x_n = (x − center) / scale`, with one isotropic
  scale, so SDF values, distances and edge lengths are divided by the same
  `scale`.
* `normalization_id` identifies the transform. To reproduce the exact
  normalization used during training, save `norm.to_dict()` once and pass
  `normalization=gc.Normalizer.from_dict(saved)` afterwards.
* Defaults:
  * `MESH_GRAPH` and `VOXEL_GRID` use `unit_cube`: bounds centre → 0,
    largest half-size → 1.
  * `POINT_CLOUD` uses `unit_sphere` per cloud: centroid → 0, farthest
    point → 1.
* Pass `normalization=None` to keep millimetres.

---

## 13. Worked example: `PicoGK_Modeler_preview.stl`

The file (in `examples/stl_example/`) is a binary STL exported by the PicoGK
web modeler: a Ø30 mm sphere with a cylindrical bore along z, flattened at
z = ±12.7 mm. It has 17,984 vertices and 36,064 triangles, it is closed,
and its volume is 8,554.7 mm³. It has 48 non-manifold edges, so it is
loaded unchanged with a warning.

```python
part = gc.load_geometry("examples/stl_example/PicoGK_Modeler_preview.stl")
part_mg = part.output("MESH_GRAPH")
```

```
MESH_GRAPH (12 arrays)
  schema_version             string  []                 1.0
  representation             string  []                 MESH_GRAPH
  coordinate_system          string  []                 cartesian_xyz_right_handed
  units                      string  []                 mm/normalized
  normalization_id           string  []                 unit_cube:cfcb790ae10d
  coordinates                float32 [17984, 3]         [-1, 1]
  node_features              float32 [17984, 6]         [-2.486, 33.87]
  edge_index                 int64   [2, 108096]        [0, 1.798e+04]
  edge_features              float32 [108096, 4]        [-0.05229, 0.05515]
  batch                      int64   [17984]            [0, 0]
  node_feature_names         string  [6]                ['nx', 'ny', 'nz', 'vertex_area', 'mean_curvature', 'tag:surface']
  edge_feature_names         string  [4]                ['dx', 'dy', 'dz', 'distance']
```

```python
part.output("MESH_GRAPH", coordinates=True, node_features=True)
```
```
MESH_GRAPH (2 arrays)
  coordinates                float32 [17984, 3]         [-1, 1]
  node_features              float32 [17984, 6]         [-2.486, 33.87]
```

```python
part.output("POINT_CLOUD", n_points=4096, mode="mixed")   # coordinates [1, 4096, 3], point_features [1, 4096, 5]
part.output("VOXEL_GRID", shape=64)                       # voxel_fields [1, 3, 64, 64, 64]
```

Running `python examples/stl_example/run.py` (optionally with your own
file) writes these files and pictures to `examples/stl_example/output/`.

---

## 14. Using the outputs in PyTorch

```python
t = part.output("MESH_GRAPH").to_torch("cuda")

# MeshGraphNet / PyTorch Geometric
from torch_geometric.data import Data
g = Data(x=t["node_features"], pos=t["coordinates"], edge_index=t["edge_index"],
         edge_attr=t["edge_features"], batch=t["batch"])

# PointNet: [B, N, 3+F] -> usually [B, 3+F, N]
pc = vox.output("POINT_CLOUD", n_points=2048).to_torch("cuda")
x = torch.cat([pc["coordinates"], pc["point_features"]], -1).transpose(1, 2)

# FNO3D: [B, C, D, H, W] directly
vg = vox.output("VOXEL_GRID", shape=64, channels=["occupancy", "sdf"]).to_torch("cuda")
y = fno(vg["voxel_fields"])

# SDF lookup at arbitrary points during PINN training (grid_sample, align_corners=False)
sdf = vox.output("VOXEL_GRID", shape=128, channels=["sdf"], normalization=None)
```

`to_torch` needs a working PyTorch installation.
