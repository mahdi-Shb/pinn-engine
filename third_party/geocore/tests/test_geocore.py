import math
import os
import warnings

import numpy as np
import pytest

import geocore as gc

BACKENDS = ["dense"] + (["picogk"] if gc.PICOGK_AVAILABLE else [])
rng = np.random.default_rng(0)


def bracket():
    """The PicoGK WebModel demo part (without the fillet)."""
    return ((gc.Box((0, 0, 4), (60, 40, 8)).tagged("base")
             | gc.Box((-26, 0, 24), (8, 40, 40)).tagged("upright")
             | gc.Cylinder((12, 0, 0), (12, 0, 16), 9).tagged("boss"))
            - gc.Cylinder((12, 0, -5), (12, 0, 25), 4.5).tagged("bolt_hole"))


# ---------------------------------------------------------------- implicits
def test_primitive_sdfs_are_exact_distances():
    p = rng.uniform(-20, 20, (5000, 3))
    for s in (gc.Sphere((1, 2, 3), 7), gc.Box((0, 0, 0), (10, 6, 4)),
              gc.Cylinder((0, 0, -5), (3, 1, 6), 4), gc.Torus((0, 0, 0), 8, 2),
              gc.Beam((0, 0, 0), (10, 2, 0), 3, 1), gc.Cone((0, 0, 0), (0, 0, 9), 4, 1)):
        g = s.gradient(p)
        # eikonal |grad| = 1 almost everywhere
        assert np.median(np.abs(np.linalg.norm(g, axis=1) - 1)) < 1e-4, type(s).__name__


def test_tags_follow_csg():
    part = bracket()
    assert part.tag_names() == ["base", "upright", "boss", "bolt_hole"]
    d, t = part.sdf_tagged([[12, 4.5, 8], [-26, 0, 44], [19, 0, 16], [0, 0, 0]])
    assert list(t) == [3, 1, 2, 0]   # bolt hole wall, upright top, boss top, base


def test_transform_keeps_distance():
    s = gc.Box((0, 0, 0), (4, 4, 4)).rotate((1, 1, 0), 0.7).translate((3, 0, 0)).scale(2.0)
    p = rng.uniform(-15, 15, (2000, 3))
    assert np.median(np.abs(np.linalg.norm(s.gradient(p), axis=1) - 1)) < 1e-4


# ---------------------------------------------------------------- mesh
def test_mesh_sdf_matches_sphere():
    s = gc.Sphere((0, 0, 0), 10)
    m = gc.Geometry(s, resolution=0.4).surface_mesh()
    assert m.is_watertight()
    assert abs(m.volume() - 4 / 3 * math.pi * 1000) / 4188.8 < 2e-3
    p = rng.uniform(-15, 15, (20000, 3))
    assert np.abs(m.sdf(p) - s.sdf(p)).max() < 0.02


def test_stl_roundtrip(tmp_path):
    m = gc.Mesh.box((0, 0, 0), (1, 2, 3))
    f = tmp_path / "b.stl"
    m.save_stl(str(f))
    m2 = gc.Mesh.load(str(f))
    assert m2.n_vertices == 8 and m2.n_faces == 12
    assert abs(m2.volume() - 6) < 1e-6


def test_manifold_mesher_on_sharp_csg():
    s = gc.Sphere((0.3, 0.1, 0), 10) - gc.Box((0, 0, 10), (8, 8, 8))
    for h in (0.5, 0.2):
        assert gc.Geometry(s, resolution=h).surface_mesh().is_watertight()


# ---------------------------------------------------------------- redistance
def test_redistance_accuracy():
    s = gc.Sphere((0.3, 0.1, 0), 10)
    h = 0.5
    g = gc.GridSpec.lattice(s.bounds, h, pad=4)
    phi = gc.sample_implicit(s, g)
    P = g.points().reshape(g.shape + (3,))
    distorted = phi * (1.5 + np.sin(P[..., 0])).astype(np.float32)
    r = gc.redistance(distorted, g.spacing, g.origin)
    band = np.abs(phi) < 3 * h
    assert np.abs(r - phi)[band].max() < 0.15 * h
    r = gc.redistance(phi, g.spacing, g.origin)
    assert np.abs(r - phi)[band].max() < 0.03 * h
    assert np.abs(r - phi).max() < 0.25 * h


# ---------------------------------------------------------------- voxels
@pytest.mark.parametrize("backend", BACKENDS)
def test_voxels_sphere_volume_and_sdf(backend):
    lib = gc.Library(0.3, backend)
    v = gc.Voxels.sphere((0, 0, 0), 10, lib=lib)
    assert abs(v.volume() / (4 / 3 * math.pi * 1000) - 1) < 3e-3
    p = rng.uniform(-14, 14, (20000, 3))
    exact = gc.Sphere((0, 0, 0), 10).sdf(p)
    near = np.abs(exact) < 1.0
    assert np.abs(v.sdf(p) - exact)[near].max() < 0.03


@pytest.mark.parametrize("backend", BACKENDS)
def test_voxels_offsets_shell(backend):
    lib = gc.Library(0.3, backend)
    s = gc.Voxels.sphere((0, 0, 0), 10, lib=lib)
    assert abs(s.copy().offset(3).volume() / (4 / 3 * math.pi * 13 ** 3) - 1) < 3e-3
    assert abs(s.shell(-2).volume() / (4 / 3 * math.pi * (1000 - 512)) - 1) < 5e-3


@pytest.mark.parametrize("backend", BACKENDS)
def test_fillet_adds_material_in_concave_corners(backend):
    lib = gc.Library(0.4, backend)
    v = gc.Voxels(bracket(), lib=lib)
    v0 = v.volume()
    v.fillet(4)
    gain = v.volume() - v0
    assert 300 < gain < 480          # ~380 mm^3 for this part


def test_backends_agree():
    if not gc.PICOGK_AVAILABLE:
        pytest.skip("PicoGK runtime not available")
    vols = []
    for be in ("picogk", "dense"):
        v = gc.Voxels(bracket(), lib=gc.Library(0.4, be)).fillet(4)
        vols.append(v.volume())
    assert abs(vols[0] / vols[1] - 1) < 3e-3


@pytest.mark.parametrize("backend", BACKENDS)
def test_lattice_and_gyroid(backend):
    lib = gc.Library(0.4, backend)
    lat = gc.Lattice.from_cells("bcc", (0, 0, 0), (20, 20, 20), 10, 1.0)
    lv = gc.Voxels(lat, lib=lib)
    assert lv.volume() > 0
    g = gc.Voxels.sphere((0, 0, 0), 10, lib=lib).intersect_implicit(gc.Gyroid(5, 0.8))
    frac = g.volume() / (4 / 3 * math.pi * 1000)
    assert 0.2 < frac < 0.45


def test_lattice_sdf_matches_brute_force():
    lat = gc.Lattice.from_cells("octet", (0, 0, 0), (30, 30, 30), 10, lambda p: 0.5 + 0.02 * p[:, 2])
    imp = lat.as_implicit()
    p = rng.uniform(-5, 35, (20000, 3))
    ref = np.empty(len(p))
    from geocore import _kernels as K
    K.beams_sdf_brute(p, lat.beams, ref)
    assert np.abs(imp.sdf(p) - ref).max() < 1e-12


# ---------------------------------------------------------------- representations
@pytest.fixture(scope="module")
def part():
    return bracket()


def test_mesh_graph_surface(part, tmp_path):
    g = gc.mesh_graph(part, resolution=1.0)
    gc.validate(g)
    ei = g["edge_index"]
    assert g["edge_features"].shape == (ei.shape[1], 4)
    # normalized: coordinates inside [-1, 1]
    assert np.abs(g["coordinates"]).max() <= 1.0 + 1e-6
    f = gc.save(str(tmp_path), g)
    assert os.path.basename(f) == "mesh_graph.npz"
    gc.load(f)


def test_mesh_graph_volume(part):
    g = gc.mesh_graph(part, kind="volume", resolution=1.5, n_interior=3000)
    gc.validate(g)
    names = list(g["node_feature_names"])
    sdf = g["node_features"][:, names.index("sdf")]
    isb = g["node_features"][:, names.index("is_boundary")]
    assert (sdf[isb == 0] < 0).all() and np.allclose(sdf[isb == 1], 0)


def test_point_cloud(part):
    pc = gc.point_cloud([part, gc.Sphere((0, 0, 0), 5)], [1024, 512], mode="mixed",
                        features=("nx", "ny", "nz", "sdf", "is_boundary"))
    gc.validate(pc, input_dim=8)
    assert pc["coordinates"].shape == (2, 1024, 3)
    assert pc["point_mask"][1].sum() == 512
    s = gc.point_cloud(gc.Sphere((0, 0, 0), 5), 512, features=None)
    assert "point_features" not in s and "point_mask" not in s
    gc.validate(s, input_dim=3)


def test_voxel_grid(part):
    vg = gc.voxel_grid(part, shape=(32, 40, 48), channels=("occupancy", "sdf", "boundary", "tags"),
                       fields={"source": lambda p: np.exp(-np.sum(p ** 2, 1) / 100)})
    with pytest.raises(ValueError):
        gc.voxel_grid(part, shape=16, channels=("nope",))
    vg = gc.voxel_grid(part, shape=(32, 40, 48), channels=("occupancy", "sdf", "source", "tags"),
                       fields={"source": lambda p: np.exp(-np.sum(p ** 2, 1) / 100)})
    gc.validate(vg, input_channels=3 + 4)
    assert vg["voxel_fields"].shape == (1, 7, 32, 40, 48)       # D,H,W = Z,Y,X
    occ = vg["voxel_fields"][0, 0].sum() * np.prod(vg["spacing"]) * float(vg["normalization_scale"]) ** 3
    assert abs(occ / 31740 - 1) < 0.02


def test_voxel_grid_axis_order():
    # a box long in x must be long along W (last axis)
    b = gc.Box((0, 0, 0), (30, 6, 6))
    vg = gc.voxel_grid(b, shape=(16, 16, 16), bounds=((-16, -16, -16), (16, 16, 16)),
                       channels=("occupancy_binary",), normalization=None)
    occ = vg["voxel_fields"][0, 0]
    ext = [np.ptp(np.nonzero(occ.any(axis=tuple(a for a in range(3) if a != k)))[0]) for k in range(3)]
    assert ext[2] > ext[0] and ext[2] > ext[1]


@pytest.mark.parametrize("backend", BACKENDS)
def test_representations_from_voxels(backend):
    v = gc.Voxels(bracket(), lib=gc.Library(0.5, backend))
    gc.validate(gc.mesh_graph(v))
    gc.validate(gc.point_cloud(v, 512))
    vg = gc.voxel_grid(v, shape=32)
    gc.validate(vg)


def test_collocation(part):
    c = gc.collocation(part, 2000, 1000, band=(500, 0.5), rng=1)
    assert (c["sdf_interior"] < 0).all()
    assert np.abs(part.sdf(c["x_boundary"].astype(float))).max() < 1e-3
    assert c["x_interior"].dtype == np.float32


# ---------------------------------------------------------------- NURBS
def test_plate_with_hole_exact():
    s = gc.NURBSSurface.plate_with_hole(1.0, 4.0)
    assert abs(s.area() - (16 - math.pi / 4)) < 1e-9
    u = np.linspace(0, 1, 101)
    hole = s(np.c_[u, np.zeros_like(u)])
    assert np.abs(np.linalg.norm(hole, axis=1) - 1).max() < 1e-12
    r = s.refine(10, 8).with_degree(3, 3)
    uv = rng.random((500, 2))
    assert np.abs(r(uv) - s(uv)).max() < 1e-10          # refinement is exact
    assert r.min_jacobian() > 0


def test_annulus_and_coons():
    a = gc.NURBSSurface.annulus(1, 2, 0, 1.5 * math.pi)
    assert abs(a.area() - 0.75 * math.pi * 3) < 1e-8
    b = gc.NURBSCurve.interpolate(lambda t: np.c_[t, 0.1 * np.sin(math.pi * t)], 8)
    t = gc.NURBSCurve.line((0, 1), (1, 1), 3)
    l = gc.NURBSCurve.line((0, 0), (0, 1), 3)
    r = gc.NURBSCurve.line((1, 0), (1, 1), 3)
    c = gc.NURBSSurface.coons(b, r, t, l)
    tt = np.linspace(0, 1, 50)
    assert np.abs(c(np.c_[tt, np.zeros(50)]) - b(tt)).max() < 1e-10


def test_parametric_surface_dict(tmp_path):
    s = gc.NURBSSurface.quad((0, 0), (2, 0), (2.5, 1.5), (0, 1), degree=(3, 2), n_ctrl=(7, 6))
    g = lambda xy: np.c_[np.sin(xy[:, 0]), xy[:, 1] ** 2]
    d = gc.parametric_surface(s, output_dim=2, dirichlet=g, n_points=256, batch=2)
    gc.validate(d)
    f = gc.save(str(tmp_path), d)
    d2 = gc.load(f)
    assert d2["parametric_coordinates"].shape == (2, 256, 2)
    # lifting reproduces the boundary data (interpolated at Greville points)
    L = d2["lifting_coefficients"].astype(float)
    u = np.linspace(0, 1, 40)
    edge = np.c_[u, np.ones(40)]
    assert np.abs(s.evaluate_field(L, edge) - g(s(edge))).max() < 5e-3
    bad = dict(d)
    a = bad["admissible_coefficients"].copy()
    a[0, 3] = 1
    bad["admissible_coefficients"] = a
    with pytest.raises(gc.SchemaError):
        gc.validate(bad)


def test_normalizer_roundtrip():
    n = gc.Normalizer.from_bounds(gc.BBox((-5, 0, 2), (15, 4, 6)), "unit_cube")
    p = rng.uniform(-5, 15, (100, 3))
    assert np.allclose(n.inverse_points(n.points(p)), p)
    n2 = gc.Normalizer.from_dict(n.to_dict())
    assert n2.id == n.id
