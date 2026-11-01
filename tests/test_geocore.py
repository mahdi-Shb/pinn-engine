"""Real GeoCore integration: sequential designs, layouts, and Solver wiring."""
import pytest
import torch

# The optional geometry dependency must not be required for engine-only tests.
gc = pytest.importorskip("geocore")
from PINNCore import GeoCoreModel, Model, Quantity, Signature, Solver, Value


@pytest.fixture
def design():
    return gc.Design({
        "name": "box", "parameters": {"length": {"value": 2., "min": 1., "max": 4.}},
        "shape": {"box": {"center": [0, 0, 0], "size": ["length", 1., 1.]}},
    })


@pytest.mark.parametrize("list_parameters", [False, True])
def test_parametric_point_cloud_solver_and_seed(design, list_parameters):
    length, xyz, mask, mean = Quantity("length"), Quantity("xyz", (3,)), Quantity("valid"), Quantity("mean", (3,))
    producer = GeoCoreModel(design, "POINT_CLOUD", parameters={"length": length},
                            outputs={"coordinates": xyz, "point_mask": mask},
                            options={"n_points": 16, "features": None, "normalization": None, "resolution": .3}, seed=7)
    class Average(Model):
        signature = Signature({xyz: "point"}, {mean: ()})
        def evaluate(self, values):
            return {mean: values[xyz].mean(0)}
    solver = Solver([Average(), producer], outputs=(xyz, mean, mask))
    parameters = [torch.tensor(2.), torch.tensor(3.)] if list_parameters else torch.tensor([2., 3.])
    values = {length: Value(parameters, "scenario")}
    a, b = solver(values), solver(values)
    assert isinstance(a[xyz].tensor, torch.Tensor) and a[xyz].tensor.shape == (2, 16, 3)
    assert isinstance(a[mask].tensor, torch.Tensor) and a[mask].tensor.shape == (2, 16)
    assert isinstance(a[mean].tensor, torch.Tensor) and a[mean].tensor.shape == (2, 3)
    assert a[xyz].axes == ("scenario", "point")
    assert a[mean].axes == ("scenario",)
    assert design.parameters["length"] == 2.
    for index, points in enumerate(a[xyz].tensor):
        assert points.shape == (16, 3) and points.device.type == "cpu"
        torch.testing.assert_close(points, b[xyz].tensor[index])
        torch.testing.assert_close(a[mean].tensor[index], points.mean(0))
        assert a[mask].tensor[index].dtype == torch.bool
        assert a[mask].tensor[index].all()


@pytest.mark.parametrize("lengths", [(2., 3.), (2., 2.)])
def test_mesh_variable_sizes_transposed_indices_and_lists(design, lengths):
    length, xyz, edges, ef = Quantity("L"), Quantity("vertices", (3,)), Quantity("indices", (2,)), Quantity("ef", (4,))
    model = GeoCoreModel(design, "MESH_GRAPH", parameters={"length": length},
                         outputs={"coordinates": xyz, "edge_index": edges, "edge_features": ef},
                         options={"resolution": .3, "normalization": None}, seed=2)
    result = model({length: Value([torch.tensor(value) for value in lengths], "scenario")})
    assert result[edges].axes == ("scenario", "edge")
    assert all(isinstance(value.tensor, list) for value in result.values())
    assert (result[xyz].tensor[0].shape[0] == result[xyz].tensor[1].shape[0]) == (lengths[0] == lengths[1])
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
    assert isinstance(result[field].tensor, torch.Tensor)
    assert result[field].tensor.shape == (1, 4, 5, 6, 2)
    assert result[field].tensor[0].dtype == torch.float64
    assert isinstance(result[spacing].tensor, torch.Tensor)
    assert result[spacing].tensor.shape == (1, 3)


@pytest.mark.parametrize("grid_options, dense", [
    ({}, True),
    ({"shape": (3, 4, 5)}, True),
    ({"spacing": .5}, False),
    ({"spacing": .5, "bounds": [[-2., -2., -2.], [2., 2., 2.]]}, True),
])
@pytest.mark.parametrize("lengths", [(2., 3.), (2., 2.)])
def test_voxel_storage_follows_grid_contract_not_current_shapes(design, grid_options, dense, lengths):
    length, field = Quantity("length"), Quantity("voxel", (1,))
    origin, spacing, bounds = Quantity("origin", (3,)), Quantity("spacing", (3,)), Quantity("bounds", (2, 3))
    model = GeoCoreModel(design, "VOXEL_GRID", parameters={"length": length},
                         outputs={"voxel_fields": field, "origin": origin, "spacing": spacing, "bounds": bounds},
                         options={**grid_options, "channels": ["sdf"], "normalization": None})
    result = model({length: Value(torch.tensor(lengths), "scenario")})
    assert isinstance(result[field].tensor, torch.Tensor if dense else list)
    assert result[field].axes == ("scenario", "z", "y", "x")
    for quantity in (origin, spacing, bounds):
        assert isinstance(result[quantity].tensor, torch.Tensor)
        assert result[quantity].tensor.shape == (2,) + quantity.shape
        assert result[quantity].axes == ("scenario",)


@pytest.mark.parametrize("wrapped_design", [False, True])
def test_nurbs_static_geometry(design, wrapped_design):
    uv, xy, cp = Quantity("uv", (2,)), Quantity("xy", (2,)), Quantity("cp", (2,))
    source = gc.NURBSSurface.plate_with_hole(R=1, L=4)
    options = {"n_points": 8}
    if wrapped_design:
        options["surface"] = source
        definition = design.to_dict()
        definition["outputs"] = {"parametric_surface": {}}
        source = gc.Design(definition)
    model = GeoCoreModel(source, "PARAMETRIC_SURFACE", outputs={"parametric_coordinates": uv,
                         "physical_coordinates": xy, "control_points": cp}, options=options, seed=1)
    result = model({})
    assert all(isinstance(value.tensor, torch.Tensor) for value in result.values())
    assert result[uv].tensor[0].shape == result[xy].tensor[0].shape == (8, 2)
    assert result[cp].axes == ("scenario", "control_u", "control_v")


@pytest.mark.parametrize("point_count, dense_points", [(8, True), ("int(4 * length)", False)])
def test_design_nurbs_separates_fixed_samples_from_variable_control_mesh(design, point_count, dense_points):
    data = design.to_dict()
    data["outputs"] = {"parametric_surface": {
        "domain": {"rectangle": {"x0": 0, "x1": "length", "y0": 0, "y1": 1}},
        "n_ctrl": [4, "2 * length"],
        "n_points": point_count,
    }}
    source = gc.Design(data)
    length, uv, xy = Quantity("length"), Quantity("uv", (2,)), Quantity("xy", (2,))
    control, knots, degree = Quantity("control", (2,)), Quantity("knots"), Quantity("degree", (2,))
    model = GeoCoreModel(source, "PARAMETRIC_SURFACE", parameters={"length": length},
                         outputs={"parametric_coordinates": uv, "physical_coordinates": xy,
                                  "control_points": control, "knot_vector_v": knots, "degree": degree}, seed=7)
    result = model({length: Value(torch.tensor([2., 3.]), "scenario")})
    for quantity in (uv, xy):
        assert isinstance(result[quantity].tensor, torch.Tensor if dense_points else list)
        assert result[quantity].axes == ("scenario", "point")
    assert [len(t) for t in result[uv].tensor] == ([8, 8] if dense_points else [8, 12])
    assert isinstance(result[degree].tensor, torch.Tensor) and result[degree].tensor.shape == (2, 2)
    for quantity in (control, knots):
        assert isinstance(result[quantity].tensor, list)
        assert result[quantity].tensor[0].shape != result[quantity].tensor[1].shape


def test_collocation_has_independent_axes(design):
    xi, xb, normal, tags = Quantity("xi", (3,)), Quantity("xb", (3,)), Quantity("normal", (3,)), Quantity("tag")
    model = GeoCoreModel(design, "PINN", outputs={"x_interior": xi, "x_boundary": xb,
                         "normal_boundary": normal, "tag_boundary": tags},
                         options={"n_interior": 8, "n_boundary": 12}, seed=4)
    result = model({})
    assert all(isinstance(value.tensor, torch.Tensor) for value in result.values())
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
