import os
import warnings

import numpy as np
import pytest

import geocore as gc
from geocore.outputs import COMMON, SCHEMA

HERE = os.path.dirname(__file__)


@pytest.fixture(scope="module")
def part():
    return (gc.Box((0, 0, 4), (60, 40, 8)) | gc.Cylinder((12, 0, 0), (12, 0, 16), 9)) \
        - gc.Cylinder((12, 0, -5), (12, 0, 25), 4.5)


def test_mesh_graph_all_arrays(part):
    mg = part.output("MESH_GRAPH", resolution=1.5)
    assert list(mg) == [*COMMON, *SCHEMA["MESH_GRAPH"]]
    assert mg.coordinates.dtype == np.float32 and mg.coordinates.shape[1] == 3
    assert mg.edge_index.dtype == np.int64 and mg.edge_index.shape[0] == 2
    assert mg.edge_features.shape == (mg.edge_index.shape[1], 4)
    assert mg.batch.shape == (len(mg.coordinates),)
    assert len(mg.node_feature_names) == mg.node_features.shape[1]
    assert mg.complete
    mg.validate()


def test_selection(part):
    mg = part.output("MESH_GRAPH", resolution=1.5, coordinates=True, node_features=True)
    assert list(mg) == ["coordinates", "node_features"]
    assert not mg.complete
    mg = part.output("MESH_GRAPH", resolution=1.5, node_features=True, coordinates=True, metadata=True)
    assert list(mg) == [*COMMON, "coordinates", "node_features"]          # schema order
    mg = part.output("MESH_GRAPH", resolution=1.5, batch=False, edge_feature_names=False)
    assert "batch" not in mg and "edge_feature_names" not in mg and "edge_index" in mg
    mg = part.output("MESH_GRAPH", resolution=1.5, extras=True)
    assert "faces" in mg and "normalization_scale" in mg
    with pytest.raises(TypeError, match="True or False"):
        part.output("MESH_GRAPH", coordinates="yes")
    with pytest.raises(TypeError, match="Array names"):
        part.output("MESH_GRAPH", n_pointz=3)


def test_point_cloud_optional_arrays(part):
    pc = part.output("POINT_CLOUD", n_points=300)
    assert list(pc) == [*COMMON, "coordinates", "point_features", "point_feature_names"]
    pc = part.output("POINT_CLOUD", n_points=300, coordinates=True, point_mask=True)
    assert pc.point_mask.shape == (1, 300) and pc.point_mask.all()
    with pytest.raises(ValueError, match="no features"):
        part.output("POINT_CLOUD", n_points=300, features=None, point_features=True)


def test_every_source_type(tmp_path, part):
    gc.set_voxel_size(0.6)
    vox = gc.Voxels(part).fillet(2)
    lat = gc.Lattice.from_cells("bcc", (0, 0, 0), (20, 20, 20), 10, 1.0)
    stl = tmp_path / "part.stl"
    gc.Geometry(part, resolution=1.0).surface_mesh().save_stl(str(stl))
    mesh = gc.load_geometry(str(stl))
    design = gc.Design.from_string("""
parameters: {r: {value: 8, min: 4, max: 12}}
shape: {sphere: {radius: r}, tag: ball}
""")
    for src in (part, vox, lat, mesh, design, gc.Geometry(part)):
        for rep in ("MESH_GRAPH", "POINT_CLOUD", "VOXEL_GRID"):
            out = src.output(rep, **({"shape": 16} if rep == "VOXEL_GRID" else
                                     {"n_points": 128} if rep == "POINT_CLOUD" else {}))
            out.validate()


def test_aliases_and_errors(part):
    assert part.output("voxel", shape=8, voxel_fields=True).representation == "VOXEL_GRID"
    assert part.output("points", n_points=64, coordinates=True).representation == "POINT_CLOUD"
    with pytest.raises(ValueError, match="2D domain"):
        part.output("PARAMETRIC_SURFACE")
    with pytest.raises(ValueError, match="unknown representation"):
        part.output("CUBES")


def test_parametric_surface():
    s = gc.NURBSSurface.plate_with_hole().with_degree(3, 3).refine(10, 8)
    ps = s.output(output_dim=2, n_points=256, dirichlet={"bottom": lambda xy: xy})
    assert list(ps) == [*COMMON, *SCHEMA["PARAMETRIC_SURFACE"]]
    ps.validate()
    sel = s.output("PARAMETRIC_SURFACE", control_points=True, weights=True)
    assert list(sel) == ["control_points", "weights"]
    d = gc.Design.load(os.path.join(HERE, "..", "examples", "designs", "flange.yaml"))
    assert d.output("PARAMETRIC_SURFACE", degree=True)["degree"].tolist() == [3, 3]


def test_design_follows_parameters():
    d = gc.Design.from_string("""
parameters: {r: {value: 8, min: 4, max: 12}}
shape: {sphere: {radius: r}}
""")
    a = d.output("VOXEL_GRID", shape=16, normalization=None, voxel_fields=True, bounds=True)
    d.set(r=11)
    b = d.output("VOXEL_GRID", shape=16, normalization=None, voxel_fields=True, bounds=True)
    assert b.bounds[1, 0] > a.bounds[1, 0]          # bigger part -> bigger grid


def test_save_and_numpy(tmp_path, part):
    mg = part.output("MESH_GRAPH", resolution=2.0)
    f = mg.save(str(tmp_path) + os.sep)
    assert os.path.basename(f) == "mesh_graph.npz"
    gc.load(f)                                       # validated file
    sel = part.output("MESH_GRAPH", resolution=2.0, coordinates=True)
    f2 = sel.save(str(tmp_path / "coords.npz"))
    with np.load(f2) as z:
        assert list(z.files) == ["coordinates"]
    assert set(mg.numpy()) == set(mg)


def test_load_geometry_repairs_holes(tmp_path):
    m = gc.Mesh.box((0, 0, 0), (10, 10, 10))
    holed = gc.Mesh(m.vertices, m.faces[:-1])                    # one triangle missing
    f = tmp_path / "holed.stl"
    holed.save_stl(str(f))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fixed = gc.load_geometry(str(f))
        assert any("open edges" in str(x.message) for x in w)
    assert fixed.is_watertight() and fixed.is_manifold()
    assert abs(fixed.volume() / 1000 - 1) < 0.02
    raw = gc.load_geometry(str(f), repair=False)
    assert raw.manifold_report()["open_edges"] > 0


def test_load_geometry_design():
    d = gc.load_geometry(os.path.join(HERE, "..", "examples", "designs", "bracket.yaml"))
    assert isinstance(d, gc.Design)


def test_batches_and_explicit_surface(part):
    a, b = gc.Sphere((0, 0, 0), 5), gc.Box((0, 0, 0), (8, 8, 8))
    pc = gc.output([a, b], "POINT_CLOUD", n_points=[200, 150])
    assert pc.coordinates.shape == (2, 200, 3) and pc.point_mask[1].sum() == 150
    vg = gc.output([a, b], "VOXEL_GRID", shape=12)
    assert vg.voxel_fields.shape[0] == 2
    mg = gc.output([a, b], "MESH_GRAPH", resolution=1.0)
    assert set(np.unique(mg.batch)) == {0, 1}
    mg.validate()
    s = gc.NURBSSurface.rectangle(0, 2, 0, 1)
    ps = part.output("PARAMETRIC_SURFACE", surface=s, n_points=64)
    ps.validate()
