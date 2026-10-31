"""Real GeoCore integration: sequential designs, layouts, and Solver wiring."""
import pytest
import torch

# The optional geometry dependency must not be required for engine-only tests.
gc = pytest.importorskip("geocore")
from pinn_engine import GeoCoreModel, Model, Quantity, Signature, Solver, Value


@pytest.fixture
def design():
    return gc.Design({
        "name": "box", "parameters": {"length": {"value": 2., "min": 1., "max": 4.}},
        "shape": {"box": {"center": [0, 0, 0], "size": ["length", 1., 1.]}},
    })


def test_parametric_point_cloud_solver_and_seed(design):
    length, xyz, mask, mean = Quantity("length"), Quantity("xyz", (3,)), Quantity("valid"), Quantity("mean", (3,))
    producer = GeoCoreModel(design, "POINT_CLOUD", parameters={"length": length},
                            outputs={"coordinates": xyz, "point_mask": mask},
                            options={"n_points": 16, "features": None, "normalization": None, "resolution": .3}, seed=7)
    class Average(Model):
        signature = Signature({xyz: "point"}, {mean: ()})
        def evaluate(self, values):
            return {mean: values[xyz].mean(0)}
    solver = Solver([Average(), producer], outputs=(xyz, mean, mask))
    values = {length: Value(torch.tensor([2., 3.]), "scenario")}
    a, b = solver(values), solver(values)
    assert a[xyz].axes == ("scenario", "point")
    assert a[mean].axes == ("scenario",)
    assert design.parameters["length"] == 2.
    for index, points in enumerate(a[xyz].tensor):
        assert points.shape == (16, 3) and points.device.type == "cpu"
        torch.testing.assert_close(points, b[xyz].tensor[index])
        torch.testing.assert_close(a[mean].tensor[index], points.mean(0))
        assert a[mask].tensor[index].dtype == torch.bool
        assert a[mask].tensor[index].all()


def test_mesh_variable_sizes_transposed_indices_and_lists(design):
    length, xyz, edges, ef = Quantity("L"), Quantity("vertices", (3,)), Quantity("indices", (2,)), Quantity("ef", (4,))
    model = GeoCoreModel(design, "MESH_GRAPH", parameters={"length": length},
                         outputs={"coordinates": xyz, "edge_index": edges, "edge_features": ef},
                         options={"resolution": .3, "normalization": None}, seed=2)
    result = model({length: Value([torch.tensor(2.), torch.tensor(3.)], "scenario")})
    assert result[edges].axes == ("scenario", "edge")
    assert result[xyz].tensor[0].shape[0] != result[xyz].tensor[1].shape[0]
    for i, indices in enumerate(result[edges].tensor):
        assert indices.dtype == torch.int64 and indices.shape[-1] == 2
        assert indices.max() < len(result[xyz].tensor[i])
        assert indices.shape[0] == result[ef].tensor[i].shape[0]


def test_voxel_channels_last_and_renamed_axes(design):
    field, spacing = Quantity("voxel", (2,)), Quantity("spacing", (3,))
    model = GeoCoreModel(design, "VOXEL_GRID", outputs={"voxel_fields": field, "spacing": spacing},
                         output_axes={"voxel_fields": ("gz", "gy", "gx")},
                         options={"shape": (4, 5, 6), "channels": ["sdf", "occupancy"], "normalization": None},
                         floating_dtype=torch.float64)
    result = model({})
    assert result[field].axes == ("scenario", "gz", "gy", "gx")
    assert result[field].tensor[0].shape == (4, 5, 6, 2)
    assert result[field].tensor[0].dtype == torch.float64
    assert result[spacing].tensor[0].shape == (3,)


def test_nurbs_static_geometry():
    uv, xy, cp = Quantity("uv", (2,)), Quantity("xy", (2,)), Quantity("cp", (2,))
    source = gc.NURBSSurface.plate_with_hole(R=1, L=4)
    model = GeoCoreModel(source, "PARAMETRIC_SURFACE", outputs={"parametric_coordinates": uv,
                         "physical_coordinates": xy, "control_points": cp}, options={"n_points": 8}, seed=1)
    result = model({})
    assert result[uv].tensor[0].shape == result[xy].tensor[0].shape == (8, 2)
    assert result[cp].axes == ("scenario", "control_u", "control_v")


def test_collocation_has_independent_axes(design):
    xi, xb, normal, tags = Quantity("xi", (3,)), Quantity("xb", (3,)), Quantity("normal", (3,)), Quantity("tag")
    model = GeoCoreModel(design, "PINN", outputs={"x_interior": xi, "x_boundary": xb,
                         "normal_boundary": normal, "tag_boundary": tags},
                         options={"n_interior": 8, "n_boundary": 12}, seed=4)
    result = model({})
    assert result[xi].axes == ("scenario", "interior")
    assert result[xb].axes == result[normal].axes == ("scenario", "boundary")
    assert result[xb].tensor[0].shape == (12, 3)
    assert result[tags].tensor[0].dtype == torch.int64


def test_reject_grad_extra_axes_and_invalid_configuration(design):
    length, xyz = Quantity("L"), Quantity("xyz", (3,))
    model = GeoCoreModel(design, "POINT_CLOUD", parameters={"length": length}, outputs={"coordinates": xyz})
    with pytest.raises(ValueError, match="requires_grad"):
        model({length: Value(torch.ones(2, requires_grad=True), "scenario")})
    with pytest.raises(ValueError, match="exactly"):
        model({length: Value(torch.ones(2, 3), ("scenario", "other"))})
    with pytest.raises(ValueError, match="Unknown design parameter"):
        GeoCoreModel(design, "POINT_CLOUD", parameters={"missing": length}, outputs={"coordinates": xyz})
    with pytest.raises(ValueError, match="Unsupported numeric"):
        GeoCoreModel(design, "POINT_CLOUD", outputs={"point_feature_names": xyz})
    with pytest.raises(ValueError, match="controlled"):
        GeoCoreModel(design, "POINT_CLOUD", outputs={"coordinates": xyz}, options={"batch": 2})
    with pytest.raises(ValueError, match="Intrinsic shape"):
        GeoCoreModel(design, "VOXEL_GRID", outputs={"spacing": Quantity("bad", (2,))}, options={"shape": 4})({})
