# geocore

PicoGK-based geometry core that turns CAD-like geometry into **neural-network
inputs** (one `.npz` per sample, schema of `ModelInput.md`) and **PINN
collocation sets**.

| file | representation | primary model | builder |
|---|---|---|---|
| `mesh_graph.npz` | `MESH_GRAPH` | MeshGraphNet (GCN, GAT) | `gc.mesh_graph` |
| `point_cloud.npz` | `POINT_CLOUD` | PointNet (PointNet++, Transolver, GINO, SIREN) | `gc.point_cloud` |
| `voxel_grid.npz` | `VOXEL_GRID` | FNO3D (3D U-Net) | `gc.voxel_grid` |
| `parametric_surface.npz` | `PARAMETRIC_SURFACE` | Deep NURBS | `gc.parametric_surface` |

**Full documentation:** [docs/geocore_documentation.md](docs/geocore_documentation.md).
**NN outputs reference:** [docs/nn_outputs.md](docs/nn_outputs.md). It covers
`part.output("MESH_GRAPH" | "POINT_CLOUD" | "VOXEL_GRID" | "PARAMETRIC_SURFACE")`,
choosing arrays, loading STL files, and every array explained.

```python
part = gc.load_geometry("part.stl")                    # or any shape, Voxels, Lattice, Design
part_mg = part.output("MESH_GRAPH")                    # all arrays of the schema
part_mg = part.output("MESH_GRAPH", coordinates=True, node_features=True)   # only these
part_mg.save("out/")                                   # out/mesh_graph.npz
```

**Quick result:** `python examples/model_inputs.py` writes all four files for
a demo part, plus a values report (`geometry_values.md`) and photos of the 3D
object. Pass `--geometry part.stl` to use your own model. Example output is
in [examples/output_bracket/](examples/output_bracket/).

## Parametric designs

Describe a part once in a design file (YAML or JSON), with named
parameters. Then change any parameter and get every output again:

```bash
python -m geocore examples/designs/bracket.yaml --info
python -m geocore examples/designs/bracket.yaml --set length=90 mount_holes=4 --out run
python -m geocore examples/designs/bracket.yaml --sample 200 --out dataset   # training set of variants
```

```python
d = gc.Design.load("examples/designs/heatsink.yaml")
d.set(fins=12, fin_height=30)              # checked against min/max and constraints
d.outputs("run/")                          # npz files, photo, parameters.json
```

The file format, expression language, shape types and outputs are
described in [docs/geocore_documentation.md §15](docs/geocore_documentation.md#15-parametric-design-files).
Example designs are in [examples/designs/](examples/designs/): a bracket, a
heat sink, a flange and a graded lattice panel.

## Architecture

```
            your code (import geocore as gc)
                          |
  implicit.py   Sphere Box Cylinder Beam Torus Cone Gyroid ... | & - offset shell
  mesh.py       Mesh (STL/OBJ), exact signed distance (BVH + pseudo normals)
  lattice.py    Lattice (PicoGK beams), unit cells, exact SDF (BVH)
  voxels.py     Voxels  -- PicoGK API: BoolAdd/Subtract/Intersect, Offset,
                           Fillet, Smoothen, Shell, Trim, IntersectImplicit
       |  backend "picogk" ----------------------  backend "dense" ----------
       |  PicoGK C++ runtime (OpenVDB + TBB)       numba kernels on dense
       |  via ctypes; bulk data via numba ->       grids, same voxel lattice
       |  C calls, implicits via a numba cfunc     (Linux / any machine)
  fields.py     exact redistancing, manifold surface nets, trilinear SDF
  sampling.py   Geometry adapter, surface/volume/band samplers, collocation
  representations.py / nurbs.py / io.py   ->   .npz files
  render.py     pictures of the object and of every representation (matplotlib)
```

* **C++ where it pays**: PicoGK's native `picogk.26.2` runtime (shipped in
  `geocore/_bin/`) does the voxel booleans, offsets, fillets, lattice
  rendering and voxel-to-mesh. It is the same engine C# PicoGK projects use,
  so results match them.
* **numba for the rest** (compiled to machine code, parallel): BVH mesh and
  lattice distance queries, redistancing, meshing, interpolation. With no
  C++ compiler needed, everything stays `pip install`-able.
* **Fallback**: PicoGK ships no Linux binary. There `Voxels` uses the
  `"dense"` backend automatically, with the same API and results within
  about 0.3%. To force it, set `GEOCORE_BACKEND=dense`. To use another
  PicoGK build, set `GEOCORE_PICOGK_DIR=<folder with the picogk library>`.

## Install

```bash
pip install -e "E:/study/personal/ai/PINN/Z-lab/geometry - py/geocore[test]"
```

The first import compiles the numba kernels (about 20–40 s, cached on disk
afterwards).

## Use

```python
import numpy as np
import geocore as gc

gc.set_voxel_size(0.3)                         # PicoGK Library(0.3 mm)

# geometry: implicit CSG with named surfaces (-> boundary-condition masks)
# (note: in Python `-` binds tighter than `|`, so bracket the union)
part = ((gc.Box((0, 0, 4), (60, 40, 8)).tagged("base")
         | gc.Box((-26, 0, 24), (8, 40, 40)).tagged("upright")
         | gc.Cylinder((12, 0, 0), (12, 0, 16), 9).tagged("boss"))
        - gc.Cylinder((12, 0, -5), (12, 0, 25), 4.5).tagged("bolt_hole"))

vox = gc.Voxels(part).fillet(4)                # PicoGK voxel ops
vox += gc.Voxels(gc.Lattice.from_cells("bcc", (-30, -20, 8), (30, 20, 20), 6, 0.8))
mesh = gc.Mesh.load("part.stl")                # or any closed mesh

gc.save("out/", gc.mesh_graph(part))                          # surface graph
gc.save("out/", gc.mesh_graph(vox, kind="volume", n_interior=20000))
gc.save("out/", gc.point_cloud(vox, 2048, mode="mixed"))
gc.save("out/", gc.voxel_grid(part, shape=(64, 64, 96),
                              channels=("occupancy", "sdf", "boundary", "tags")))

plate = gc.NURBSSurface.plate_with_hole(R=1, L=4).with_degree(3, 3).refine(12, 10)
gc.save("out/", gc.parametric_surface(plate, output_dim=2, n_points=4096,
                                      dirichlet={"bottom": lambda xy: xy}))

d = gc.load("out/voxel_grid.npz")              # validated on load
x = d["voxel_fields"]                          # float32 [B, C, D, H, W]
```

### PINN collocation

```python
col = gc.collocation(vox, n_interior=100_000, n_boundary=20_000,
                     band=(10_000, 0.5))       # near-wall refinement
# x_interior, sdf_interior, x_boundary, normal_boundary, tag_boundary, x_band
# -sdf_interior is an exact distance function -> hard Dirichlet BCs:
#     u(x) = g(x) + d(x) * NN(x)
```

Any source can also be queried directly: `g = gc.Geometry(src)`, then
`g.sdf(p)`, `g.normal(p)`, `g.tags(p)`, `g.sample_surface(n)` and
`g.sample_volume(n)`.

## Conventions

* **Units** are millimetres (PicoGK) unless you pass `units=`. SDF is
  negative inside.
* **Normalization**: `x_n = (x - center) / scale`, using one isotropic scale,
  so SDF values and edge lengths divide by `scale`.
  * Defaults: `unit_cube` for graphs and voxels, and centroid `unit_sphere`
    for point clouds.
  * `normalization_id` names the transform; `normalization_center` and
    `normalization_scale` are also stored as extras.
  * To reproduce a training-time normalization, pass
    `normalization=gc.Normalizer.from_dict(saved)`.
* **MESH_GRAPH**:
  * `edge_index[0]` are senders and `edge_index[1]` receivers. Every
    undirected edge appears in both directions.
  * Edge features are `dx dy dz distance`, with `d = x[receiver] - x[sender]`.
  * `node_feature_names` lists the columns: normals, `vertex_area`,
    `mean_curvature`, `sdf`, `is_boundary` and `tag:<name>` one-hots.
  * `gc.batch_graphs` merges several graphs into one batched graph.
* **POINT_CLOUD**:
  * `[B, N, 3]`; `features=None` gives coordinates only, so `3 + F` equals
    the model's `input_dim`.
  * Clouds of different sizes are padded, and `point_mask` marks the real
    points.
* **VOXEL_GRID**:
  * `voxel_fields` is `[B, C, D, H, W]` with `D, H, W = Z, Y, X`.
  * `origin` is the centre of voxel `[0, 0, 0]` and `spacing` is
    `(dx, dy, dz)`, both in x, y, z order. `bounds` is the box the cells
    cover; this matches `grid_sample(align_corners=False)`.
  * Geometry is rasterized as filled volume (anti-aliased `occupancy`) and
    an exact SDF, not as occupied vertices.
* **PARAMETRIC_SURFACE**: 2D physical domain `[Nu, Nv, 2]` with clamped
  knots on [0, 1] and positive weights.
  * `admissible_coefficients` are zero on all four boundaries.
  * `lifting_coefficients` interpolate the Dirichlet data on each boundary
    curve and are Coons-blended inside.
  * Extra: `physical_coordinates` gives the physical point of every UV
    sample.

## Accuracy (tested in `tests/`)

| quantity | result |
|---|---|
| primitive SDFs | exact (`\|∇φ\| = 1`) |
| mesh SDF | exact for the mesh (BVH, pseudo-normal sign) |
| `Voxels.sdf` near the surface, sphere at 0.3 mm | ≤ 0.03 mm |
| redistancing (exact input, smooth surface) | ≤ 3% of a voxel near the surface, < 0.25 voxel far away |
| redistancing at sharp edges | rounded to about h/3 (true of any grid level set) |
| volumes (sphere, offset, shell) | ≤ 0.3–0.5% |
| fillet, PicoGK vs dense backend | ≤ 0.3% |
| NURBS constructors and refinement | exact (circles and areas to machine precision) |

Run the tests with `pytest`, and the benchmark with
`python examples/benchmark.py`.

## Speed

Warm timings from `examples/benchmark.py` on the development machine
(16 threads, Windows):

| operation | time |
|---|---|
| CSG implicit SDF, 1M points (compiled stack program) | 16 ms |
| lattice SDF, 5,616 beams, 1M points (BVH) | 0.1 s |
| exact mesh SDF, 94k triangles, 1M points (BVH) | 1.5 s |
| `Voxels(part)` at 0.3 mm (about 4.5M voxels): dense / PicoGK | 0.07 s / 0.19 s |
| `fillet(4)` at 0.3 mm: dense / PicoGK | 3.3 s / 5.0 s |
| redistancing 9.1M voxels | 1.9 s |
| `mesh_graph` (77k nodes) / `point_cloud` (4,096) / `voxel_grid` (64³) | 0.86 s / 0.23 s / 0.06 s |
| `collocation` (100k interior + 20k boundary) | 0.24 s |

## Notes

* PicoGK 26.2's `Voxels_IntersectImplicit` throws on thin features (for
  example a 0.8 mm gyroid at 0.3 mm voxels). geocore renders the implicit
  over the field's extent and uses `BoolIntersect` instead, which gives the
  same result.
* `Voxels.calculate_properties` uses the exact volume of the surface mesh.
  PicoGK's own recipe (mesh → voxels → volume) fills internal cavities.
* The Deep NURBS domain is 2D, as the schema requires; 3D NURBS volumes are
  out of scope.
