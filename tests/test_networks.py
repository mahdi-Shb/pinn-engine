"""Native neural models: layouts, numerical parity, batching, training, and AD."""

import copy

import pytest
import torch
from torch import nn

from pinn_engine import Derivative, GeoCoreModel, Model, Quantity, Signature, Solver, Value
from pinn_engine import networks
from pinn_engine.networks import ModelInput


@pytest.fixture(autouse=True)
def small_cpu_workloads():
    # Large thread pools make these intentionally tiny networks much slower.
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(previous)


def configuration(kind, **overrides):
    options = {
        "GCN": dict(input_dim=2, hidden_dim=4, num_layers=2),
        "GAT": dict(input_dim=2, hidden_dim=4, num_layers=1, num_heads=2),
        "MeshGraphNet": dict(node_input_dim=2, edge_input_dim=2, latent_dim=4,
                             processor_steps=1, mlp_hidden_dim=8, mlp_layers=2),
        "PointNet": dict(global_dim=8, normalization="none", dropout=0.,
                         input_transform=False, feature_transform=False),
        "PointNet2": dict(sample_counts=(4, 2), radii=(.5, 1.),
                          neighbor_counts=(4, 2), abstraction_channels=((4, 8), (8, 8)),
                          global_dim=8, normalization="none", dropout=0.),
        "Transolver": dict(hidden_dim=8, num_layers=1, num_heads=2, num_slices=2),
        "SIREN": dict(hidden_dim=8, num_hidden_layers=1),
        "GINO": dict(width=4, latent_resolution=(4, 4, 4), modes=(1, 1, 2),
                     num_layers=1, kernel_radius=1.7, kernel_hidden_dim=4,
                     kernel_chunk_size=64),
        "FNO3D": dict(width=4, num_layers=1, modes=(1, 1, 2), projection_channels=8),
        "UNet3D": dict(base_channels=4, levels=2, normalization="none"),
        "DeepNURBS": dict(hidden_dim=8, activation="tanh"),
    }[kind]
    options.update(overrides)
    return getattr(networks, kind + "Config")(**options)


GRAPH = {"GCN", "GAT", "MeshGraphNet"}
GRID = {"FNO3D", "UNet3D"}
ALL_NETWORKS = ("GCN", "GAT", "MeshGraphNet", "PointNet", "PointNet2",
                "Transolver", "SIREN", "GINO", "FNO3D", "UNet3D", "DeepNURBS")


def chain_edges(count):
    # Bidirectional chains exercise message passing without cross-case edges.
    source = torch.arange(count - 1)
    return torch.cat((torch.stack((source, source + 1), -1),
                      torch.stack((source + 1, source), -1)), 0)


def reference_case(kind):
    """Build engine Values and the equivalent upstream numerical request."""
    config = configuration(kind)
    prediction = Quantity("requested_prediction")
    bindings, values = {}, {}
    if kind in GRAPH:
        bindings = {"node_features": Quantity("features", (2,)),
                    "edge_index": Quantity("connectivity", (2,))}
        nodes, edges = torch.randn(6, 2), chain_edges(6)
        values = {bindings["node_features"]: Value(nodes, "vertex"),
                  bindings["edge_index"]: Value(edges, "edge")}
        numerical = ModelInput(node_features=nodes, edge_index=edges.T)
        if kind == "MeshGraphNet":
            bindings["edge_features"] = Quantity("edge_channels", (2,))
            numerical.edge_features = torch.randn(len(edges), 2)
            values[bindings["edge_features"]] = Value(numerical.edge_features, "edge")
        latent_shape, latent_axes = (4,), ("vertex",)
        prediction_axes = ("vertex",)
    elif kind in GRID:
        bindings = {"voxel_fields": Quantity("grid_input")}
        volume = torch.randn(2, 4, 4, 4)
        values = {bindings["voxel_fields"]: Value(volume, ("scenario", "z", "y", "x"))}
        numerical = ModelInput(voxel_fields=volume.unsqueeze(1))
        latent_shape = (4,) if kind == "FNO3D" else (8,)
        latent_axes = (("scenario", "z", "y", "x") if kind == "FNO3D" else
                       ("scenario", "bottleneck_z", "bottleneck_y", "bottleneck_x"))
        prediction_axes = ("scenario", "z", "y", "x")
    else:
        dimensions = 2 if kind == "DeepNURBS" else 3
        bindings = {"coordinates": Quantity("locations", (dimensions,))}
        coordinates = torch.rand(2, 6, dimensions)
        values = {bindings["coordinates"]: Value(coordinates, ("scenario", "point"))}
        numerical = ModelInput(coordinates=coordinates)
        if kind == "GINO":
            bindings["point_features"] = Quantity("source_field")
            numerical.point_features = torch.randn(2, 6, 1)
            values[bindings["point_features"]] = Value(
                numerical.point_features.squeeze(-1), ("scenario", "point"))
            latent_shape, latent_axes = (4,), ("scenario", "latent_z", "latent_y", "latent_x")
        elif kind in {"PointNet", "PointNet2"}:
            latent_shape, latent_axes = (8,), ("scenario",)
        else:
            latent_shape = () if kind == "DeepNURBS" else (8,)
            latent_axes = ("scenario", "point")
        prediction_axes = ("scenario", "point")
    outputs = {"predictions": prediction}
    if kind != "GCN":
        outputs["latent"] = Quantity("requested_latent", latent_shape)
    model = getattr(networks, kind)(config, inputs=bindings, outputs=outputs).eval()
    return model, bindings, outputs, values, numerical, prediction_axes, latent_axes


@pytest.mark.parametrize("kind", ALL_NETWORKS)
def test_all_networks_are_native_models_and_match_numerical_algorithm(kind):
    model, inputs, outputs, values, numerical, pred_axes, latent_axes = reference_case(kind)
    assert isinstance(model, Model) and isinstance(model, nn.Module)
    assert set(model.inputs) == set(inputs.values())
    assert set(model.outputs) == set(outputs.values())
    assert bool(tuple(model.parameters()))
    reference = model.predict(numerical)
    actual = model(values)
    for role, quantity in outputs.items():
        tensor = getattr(reference, role)
        if kind in GRID or (kind == "GINO" and role == "latent"):
            tensor = tensor.movedim(1, -1)
        tensor = tensor.reshape(*tensor.shape[:-1], *quantity.shape)
        assert actual[quantity].axes == (pred_axes if role == "predictions" else latent_axes)
        torch.testing.assert_close(actual[quantity].tensor, tensor, atol=3e-6, rtol=3e-5)
    # Network parameters keep their original names and are registered in Solver.
    solver = Solver([model], outputs=outputs["predictions"])
    result = solver(values)[outputs["predictions"]]
    torch.testing.assert_close(result.tensor, actual[outputs["predictions"]].tensor)
    assert set(solver.state_dict()) == {"part_0." + name for name in model.state_dict()}


@pytest.mark.parametrize("kind", ALL_NETWORKS)
def test_signature_consumption_and_independence(kind):
    model, inputs, outputs, *_ = reference_case(kind)
    signature = model.signature
    assert signature.has_ellipsis == (kind in GRAPH)
    if kind in GRAPH:
        assert signature.inputs[inputs["node_features"]] == ("...", "vertex")
        assert signature.independence["vertex"] is False
        assert signature.independence["..."] is True
    elif kind in GRID:
        assert signature.inputs[inputs["voxel_fields"]] == ("z", "y", "x")
        assert all(signature.independence[axis] is False for axis in ("z", "y", "x"))
    else:
        assert signature.inputs[inputs["coordinates"]] == ("point",)
        assert signature.independence["point"] is (kind in {"SIREN", "DeepNURBS"})
    if kind in {"PointNet", "PointNet2"}:
        assert signature.outputs[outputs["latent"]] == ()


@pytest.mark.parametrize("kind", ("PointNet", "PointNet2"))
def test_classification_reduces_points_to_per_scenario_quantity(kind):
    coordinates, prediction, latent = Quantity("xyz", (3,)), Quantity("class", (2,)), Quantity("embedding", (8,))
    model = getattr(networks, kind)(configuration(kind, task="classification", output_dim=2),
                                  inputs={"coordinates": coordinates},
                                  outputs={"predictions": prediction, "latent": latent}).eval()
    points = torch.rand(2, 6, 3)
    actual = model({coordinates: Value(points, ("case", "point"))})
    reference = model.predict(ModelInput(coordinates=points))
    for q, expected in ((prediction, reference.predictions), (latent, reference.latent)):
        assert actual[q].axes == ("case",)
        torch.testing.assert_close(actual[q].tensor, expected)
    assert model.signature.outputs[prediction] == ()


def test_output_components_can_be_reshaped_to_tensor_quantity_and_inputs_can_be_scalar():
    xyz, feature, stress = Quantity("xyz", (3,)), Quantity("feature"), Quantity("stress", (3, 3))
    model = networks.SIREN(configuration("SIREN", input_dim=1, output_dim=9),
                           inputs={"coordinates": xyz, "point_features": feature},
                           outputs={"predictions": stress})
    coordinates, scalar = torch.rand(2, 3, 6, 3), torch.randn(2, 3, 6)
    axes = ("family", "scenario", "point")
    actual = model({xyz: Value(coordinates, axes), feature: Value(scalar, axes)})[stress]
    assert actual.axes == axes and actual.tensor.shape == (2, 3, 6, 3, 3)
    expected = model.predict(ModelInput(coordinates=coordinates.flatten(0, 1),
                                       point_features=scalar.flatten(0, 1).unsqueeze(-1)))
    torch.testing.assert_close(actual.tensor, expected.predictions.reshape(2, 3, 6, 3, 3))


@pytest.mark.parametrize("kind", ("PointNet", "PointNet2", "Transolver", "SIREN", "GINO"))
def test_no_batchnorm_extra_dense_axes_and_optional_masks(kind):
    xyz, mask, result = Quantity("xyz", (3,)), Quantity("valid"), Quantity("field")
    inputs = {"coordinates": xyz, "point_mask": mask}
    points = torch.rand(2, 2, 6, 3)
    valid = torch.ones(2, 2, 6, dtype=torch.bool)
    valid[..., -1] = False
    values = {xyz: Value(points, ("family", "scenario", "point")),
              mask: Value(valid, ("family", "scenario", "point"))}
    numerical = ModelInput(coordinates=points.flatten(0, 1), point_mask=valid.flatten(0, 1))
    if kind == "GINO":
        inputs["point_features"] = Quantity("feature")
        numerical.point_features = torch.rand(4, 6, 1)
        # Masked padding is not required to contain valid coordinates/features.
        points[..., -1, :] = float("nan")
        numerical.point_features[:, -1, :] = float("nan")
        values[inputs["point_features"]] = Value(numerical.point_features.reshape(2, 2, 6),
                                                 ("family", "scenario", "point"))
    model = getattr(networks, kind)(configuration(kind), inputs=inputs,
                                  outputs={"predictions": result}).eval()
    assert not model.signature.has_ellipsis
    actual = model(values)[result]
    assert actual.axes == ("family", "scenario", "point")
    expected = model.predict(numerical).predictions.reshape(2, 2, 6)
    torch.testing.assert_close(actual.tensor, expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(actual.tensor[..., -1], torch.zeros(2, 2))


@pytest.mark.parametrize("kind", ("PointNet", "PointNet2", "Transolver", "SIREN", "GINO", "DeepNURBS"))
def test_no_batchnorm_ragged_list_mapping_preserves_case_structure(kind):
    dimension = 2 if kind == "DeepNURBS" else 3
    xyz, result = Quantity("xyz", (dimension,)), Quantity("field")
    members = [torch.rand(5, dimension), torch.rand(8, dimension)]
    bindings = {"coordinates": xyz}
    values = {xyz: Value(members, ("case", "point"))}
    if kind == "GINO":
        feature = Quantity("feature")
        bindings["point_features"] = feature
        values[feature] = Value([torch.randn(5), torch.randn(8)], ("case", "point"))
    model = getattr(networks, kind)(configuration(kind), inputs=bindings,
                                  outputs={"predictions": result}).eval()
    actual = model(values)[result]
    assert actual.axes == ("case", "point") and isinstance(actual.tensor, list)
    for index, points in enumerate(members):
        arguments = {"coordinates": points.unsqueeze(0)}
        if kind == "GINO":
            arguments["point_features"] = values[feature].tensor[index][None, :, None]
        expected = model.predict(ModelInput(**arguments)).predictions[0, :, 0]
        torch.testing.assert_close(actual.tensor[index], expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("kind,normalization", (("PointNet", "batch"), ("PointNet2", "batch"),
                                               ("FNO3D", "batch"), ("UNet3D", "batch"),
                                               ("GAT", "batch_norm")))
def test_actual_batchnorm_consumes_entire_batch_and_trains_once(kind, normalization, monkeypatch):
    model, bindings, outputs, values, numerical, *_ = reference_case(kind)
    model = type(model)(configuration(kind, normalization=normalization), inputs=bindings, outputs=outputs).train()
    # Graph inputs use two disjoint copies, while all other cases already have B=2.
    if kind in GRAPH:
        values = {q: Value(torch.stack((v.tensor, v.tensor + (0 if q == bindings["edge_index"] else .3))),
                           ("case",) + v.axes) for q, v in values.items()}
    reference = copy.deepcopy(model)
    assert any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())
    assert model.signature.has_ellipsis
    assert model.signature.independence["..."] is False
    counts = [0]
    original = model._predict
    def counted(request):
        counts[0] += 1
        return original(request)
    monkeypatch.setattr(model, "_predict", counted)
    actual = model(values)[outputs["predictions"]]
    assert counts[0] == 1
    actual.tensor.square().sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            assert module.num_batches_tracked.item() == 1
    if kind not in GRAPH:
        expected = reference.predict(numerical).predictions
        if kind in GRID:
            expected = expected.movedim(1, -1)
        torch.testing.assert_close(actual.tensor, expected.squeeze(-1), atol=3e-6, rtol=3e-5)


def test_batchnorm_is_detected_from_modules_rather_than_configuration_spelling():
    features, edges, field = Quantity("features", (2,)), Quantity("edges", (2,)), Quantity("field")
    common = dict(inputs={"node_features": features, "edge_index": edges}, outputs={"predictions": field})
    # GCN deliberately omits normalization in its last (here only) block.
    no_norm = networks.GCN(configuration("GCN", num_layers=1, normalization="batch_norm"), **common)
    batch_norm = networks.GCN(configuration("GCN", num_layers=2, normalization="batch_norm"), **common)
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in no_norm.modules())
    assert no_norm.signature.independence["..."] is True
    assert batch_norm.signature.independence["..."] is False


def test_batchnorm_independence_tracks_training_frozen_stats_and_untracked_eval_stats():
    features, edges, field = Quantity("features", (2,)), Quantity("edges", (2,)), Quantity("field")
    model = networks.GCN(configuration("GCN", normalization="batch_norm"),
                         inputs={"node_features": features, "edge_index": edges}, outputs={"predictions": field})
    assert model.signature.has_ellipsis and model.signature.independence["..."] is False
    model.eval()
    assert model.signature.has_ellipsis and model.signature.independence["..."] is True
    model.train()
    assert model.signature.independence["..."] is False
    # Even eval() needs batch statistics when BatchNorm has no running buffers.
    model.layers[0].norm = nn.BatchNorm1d(4, track_running_stats=False)
    model.eval()
    assert model.signature.has_ellipsis and model.signature.independence["..."] is False
    values = {features: Value(torch.randn(2, 5, 2), ("case", "vertex")),
              edges: Value(chain_edges(5), "edge")}
    result = model(values)[field]
    assert torch.isfinite(result.tensor).all()
    assert model.signature.dependent_axes(values[features], result, values) == {"case", "vertex"}


def test_equal_sized_batchnorm_lists_and_dense_axes_form_one_real_training_batch(monkeypatch):
    xyz, field = Quantity("xyz", (3,)), Quantity("field")
    model = networks.PointNet(configuration("PointNet", normalization="batch"),
                              inputs={"coordinates": xyz}, outputs={"predictions": field}).train()
    reference = copy.deepcopy(model)
    members = [torch.rand(3, 6, 3), torch.rand(3, 6, 3)]
    count = [0]
    original = model._predict
    def counted(request):
        count[0] += 1
        return original(request)
    monkeypatch.setattr(model, "_predict", counted)
    actual = model({xyz: Value(members, ("family", "scenario", "point"))})[field]
    expected = reference.predict(ModelInput(coordinates=torch.cat(members, 0))).predictions[..., 0]
    assert count[0] == 1 and actual.axes == ("family", "scenario", "point")
    assert isinstance(actual.tensor, list)
    for index in range(2):
        torch.testing.assert_close(actual.tensor[index], expected[index * 3:(index + 1) * 3])
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            assert module.num_batches_tracked.item() == 1


def test_non_batchnorm_training_dropout_works_under_nested_vmap():
    xyz, field = Quantity("xyz", (3,)), Quantity("field")
    model = networks.PointNet(configuration("PointNet", dropout=.5),
                              inputs={"coordinates": xyz}, outputs={"predictions": field}).train()
    # Identical samples should still receive independent dropout draws.
    points = torch.rand(1, 1, 6, 3).expand(2, 3, 6, 3)
    actual = model({xyz: Value(points, ("family", "scenario", "point"))})[field]
    assert not model.signature.has_ellipsis
    assert actual.axes == ("family", "scenario", "point") and actual.tensor.shape == (2, 3, 6)
    assert not torch.equal(actual.tensor[0, 0], actual.tensor[1, 2])
    actual.tensor.square().sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_unequal_batchnorm_lists_reject_training_before_statistics_change_but_support_eval():
    xyz, field = Quantity("xyz", (3,)), Quantity("field")
    model = networks.PointNet(configuration("PointNet", normalization="batch"),
                              inputs={"coordinates": xyz}, outputs={"predictions": field}).train()
    members = [torch.rand(5, 3), torch.rand(8, 3)]
    values = {xyz: Value(members, ("case", "point"))}
    # Padding would alter BatchNorm statistics; silently padding is incorrect.
    with pytest.raises(ValueError, match="Ragged BatchNorm"):
        model(values)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            assert module.num_batches_tracked.item() == 0
    model.eval()
    actual = model(values)[field]
    assert actual.axes == ("case", "point") and isinstance(actual.tensor, list)
    for index, points in enumerate(members):
        reference = model.predict(ModelInput(coordinates=points.unsqueeze(0))).predictions[0, :, 0]
        torch.testing.assert_close(actual.tensor[index], reference)


@pytest.mark.parametrize("kind", ("GCN", "GAT", "MeshGraphNet"))
def test_mesh_disjoint_packing_ragged_batches_solver_and_gradients(kind, monkeypatch):
    features, edges = Quantity("features", (2,)), Quantity("edges", (2,))
    edge_features, field, mean = Quantity("edge_features", (2,)), Quantity("field"), Quantity("mean")
    nodes = [torch.randn(4, 2, requires_grad=True), torch.randn(7, 2, requires_grad=True)]
    indices = [chain_edges(4), chain_edges(7)]
    bindings = {"node_features": features, "edge_index": edges}
    values = {features: Value(nodes, ("geometry", "vertex")),
              edges: Value(indices, ("geometry", "edge"))}
    edge_values = [torch.randn(len(e), 2) for e in indices]
    if kind == "MeshGraphNet":
        bindings["edge_features"] = edge_features
        values[edge_features] = Value(edge_values, ("geometry", "edge"))
    model = getattr(networks, kind)(configuration(kind), inputs=bindings,
                                  outputs={"predictions": field}).eval()
    expected = []
    for index, node in enumerate(nodes):
        kwargs = {"node_features": node, "edge_index": indices[index].T}
        if kind == "MeshGraphNet":
            kwargs["edge_features"] = edge_values[index]
        expected.append(model.predict(ModelInput(**kwargs)).predictions[:, 0])
    count = [0]
    original = model._predict
    def counted(request):
        count[0] += 1
        return original(request)
    monkeypatch.setattr(model, "_predict", counted)
    class Average(Model):
        signature = Signature({field: "vertex"}, {mean: ()})
        def evaluate(self, incoming):
            return {mean: incoming[field].mean()}
    actual = Solver([Average(), model], outputs=(field, mean))(values)
    assert count[0] == 1
    assert actual[field].axes == ("geometry", "vertex")
    assert isinstance(actual[field].tensor, list)
    for index, expected_case in enumerate(expected):
        torch.testing.assert_close(actual[field].tensor[index], expected_case)
        torch.testing.assert_close(actual[mean].tensor[index], expected_case.mean())
    sum(value.square().sum() for value in actual[field].tensor).backward()
    assert all(node.grad is not None and torch.isfinite(node.grad).all() for node in nodes)
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("kind", ("GCN", "GAT", "MeshGraphNet"))
def test_mesh_dense_multiple_batch_axes_and_shared_connectivity(kind):
    features, edges, edge_features = Quantity("features", (2,)), Quantity("edges", (2,)), Quantity("ef", (2,))
    field = Quantity("field")
    node_values, connectivity = torch.randn(2, 3, 5, 2), chain_edges(5)
    inputs = {"node_features": features, "edge_index": edges}
    values = {features: Value(node_values, ("family", "scenario", "vertex")),
              edges: Value(connectivity, "edge")}
    if kind == "MeshGraphNet":
        inputs["edge_features"] = edge_features
        values[edge_features] = Value(torch.randn(len(connectivity), 2), "edge")
    model = getattr(networks, kind)(configuration(kind), inputs=inputs, outputs={"predictions": field}).eval()
    actual = model(values)[field]
    assert actual.axes == ("family", "scenario", "vertex") and actual.tensor.shape == (2, 3, 5)
    for family in range(2):
        for scenario in range(3):
            kwargs = {"node_features": node_values[family, scenario], "edge_index": connectivity.T}
            if kind == "MeshGraphNet":
                kwargs["edge_features"] = values[edge_features].tensor
            reference = model.predict(ModelInput(**kwargs)).predictions[..., 0]
            torch.testing.assert_close(actual.tensor[family, scenario], reference)


def test_mesh_disjoint_batch_axes_follow_declared_input_order_not_dictionary_insertion():
    features, edges, field = Quantity("features", (2,)), Quantity("edges", (2,)), Quantity("field")
    model = networks.GCN(configuration("GCN"), inputs={"node_features": features, "edge_index": edges},
                         outputs={"predictions": field}).eval()
    node_values = torch.randn(2, 5, 2)
    connectivity = torch.stack([(chain_edges(5) + shift) % 5 for shift in range(3)])
    # The values dictionary intentionally starts with the second declared role.
    values = {edges: Value(connectivity, ("scenario", "edge")),
              features: Value(node_values, ("family", "vertex"))}
    actual = model(values)[field]
    assert actual.axes == ("family", "scenario", "vertex") and actual.tensor.shape == (2, 3, 5)
    for family in range(2):
        for scenario in range(3):
            reference = model.predict(ModelInput(node_features=node_values[family],
                                                  edge_index=connectivity[scenario].T)).predictions[:, 0]
            torch.testing.assert_close(actual.tensor[family, scenario], reference)


def test_batchnorm_disjoint_batch_axes_follow_declaration_order_not_dictionary_insertion():
    xyz, feature, field = Quantity("xyz", (3,)), Quantity("feature"), Quantity("field")
    model = networks.PointNet(configuration("PointNet", input_dim=4, normalization="batch"),
                              inputs={"coordinates": xyz, "point_features": feature},
                              outputs={"predictions": field}).eval()
    coordinates, channels = torch.rand(2, 6, 3), torch.randn(3, 6)
    values = {feature: Value(channels, ("scenario", "point")),
              xyz: Value(coordinates, ("family", "point"))}
    actual = model(values)[field]
    assert actual.axes == ("family", "scenario", "point") and actual.tensor.shape == (2, 3, 6)
    reference = model.predict(ModelInput(
        coordinates=coordinates[:, None].expand(2, 3, 6, 3).reshape(6, 6, 3),
        point_features=channels[None, :, :, None].expand(2, 3, 6, 1).reshape(6, 6, 1),
    )).predictions.reshape(2, 3, 6)
    torch.testing.assert_close(actual.tensor, reference)


def test_real_geocore_meshes_feed_native_meshgraphnet_through_solver():
    gc = pytest.importorskip("geocore")
    length, features, edges = Quantity("length"), Quantity("features", (6,)), Quantity("edges", (2,))
    edge_features, field, mean = Quantity("edge_channels", (4,)), Quantity("prediction"), Quantity("mean")
    design = gc.Design({
        "name": "box", "parameters": {"length": {"value": 1., "min": 1., "max": 3.}},
        "shape": {"box": {"center": [0, 0, 0], "size": ["length", 1., 1.]}},
    })
    geometry = GeoCoreModel(design, "MESH_GRAPH", parameters={"length": length}, scenario_axis="case",
                            outputs={"node_features": features, "edge_index": edges,
                                     "edge_features": edge_features},
                            options={"resolution": .5, "normalization": None}, seed=7)
    network = networks.MeshGraphNet(configuration("MeshGraphNet", node_input_dim=6, edge_input_dim=4),
                                    inputs={"node_features": features, "edge_index": edges,
                                            "edge_features": edge_features},
                                    outputs={"predictions": field})
    class Average(Model):
        signature = Signature({field: "vertex"}, {mean: ()})
        def evaluate(self, incoming):
            return {mean: incoming[field].mean()}
    solver = Solver([Average(), network, geometry], outputs=(features, edges, field, mean))
    result = solver({length: Value(torch.tensor([1., 2.]), "case")})
    assert result[field].axes == ("case", "vertex") and isinstance(result[field].tensor, list)
    assert len(result[features].tensor[0]) != len(result[features].tensor[1])
    for i in range(2):
        assert len(result[field].tensor[i]) == len(result[features].tensor[i])
        assert result[edges].tensor[i].max() < len(result[field].tensor[i])
        assert torch.isfinite(result[field].tensor[i]).all()
        torch.testing.assert_close(result[mean].tensor[i], result[field].tensor[i].mean())
    sum(value.square() for value in result[mean].tensor).backward()
    assert any(parameter.grad is not None for parameter in network.parameters())


def test_gino_distinct_queries_are_independent_but_source_predictions_are_not():
    xyz, source, query, output = (Quantity("xyz", (3,)), Quantity("source"),
                                  Quantity("queries", (3,)), Quantity("prediction"))
    model = networks.GINO(configuration("GINO"),
                          inputs={"coordinates": xyz, "point_features": source, "query_coordinates": query},
                          outputs={"predictions": output}).eval()
    assert model.signature.inputs[query] == ("query",)
    assert model.signature.outputs[output] == ("query",)
    assert model.signature.independence["query"] is True
    values = {xyz: Value(torch.rand(2, 5, 3), ("scenario", "point")),
              source: Value(torch.randn(2, 5), ("scenario", "point")),
              query: Value(torch.rand(2, 7, 3), ("scenario", "query"))}
    actual = model(values)[output]
    assert actual.axes == ("scenario", "query") and actual.tensor.shape == (2, 7)
    request = ModelInput(coordinates=values[xyz].tensor, point_features=values[source].tensor[..., None],
                         query_coordinates=values[query].tensor)
    torch.testing.assert_close(actual.tensor, model.predict(request).predictions[..., 0], atol=3e-6, rtol=3e-5)


def test_gino_query_derivative_uses_independent_query_contract():
    xyz, source, query, output = (Quantity("xyz", (3,)), Quantity("source"),
                                  Quantity("queries", (3,)), Quantity("prediction"))
    model = networks.GINO(configuration("GINO"),
                          inputs={"coordinates": xyz, "point_features": source, "query_coordinates": query},
                          outputs={"predictions": output}).eval()
    coordinates, features = torch.rand(1, 5, 3), torch.randn(1, 5, 1)
    queries = torch.rand(1, 7, 3, requires_grad=True)
    reference = model.predict(ModelInput(coordinates=coordinates, point_features=features,
                                         query_coordinates=queries)).predictions
    expected = torch.autograd.grad(reference.sum(), queries)[0]
    derivative = Derivative(output, query)
    actual = Solver([model], outputs=derivative)({
        xyz: Value(coordinates, ("scenario", "point")),
        source: Value(features[..., 0], ("scenario", "point")),
        query: Value(queries.detach(), ("scenario", "query")),
    })[derivative]
    assert actual.axes == ("scenario", "query")
    torch.testing.assert_close(actual.tensor, expected, atol=3e-5, rtol=3e-5)


def test_gino_masked_full_points_match_physically_sliced_sources_and_queries():
    xyz, source, mask, query = (Quantity("xyz", (3,)), Quantity("source"),
                                 Quantity("valid"), Quantity("queries", (3,)))
    prediction, latent = Quantity("prediction"), Quantity("latent", (4,))
    config = configuration("GINO")
    outputs = {"predictions": prediction, "latent": latent}
    inputs = {"coordinates": xyz, "point_features": source, "point_mask": mask}
    queried = networks.GINO(config, inputs={**inputs, "query_coordinates": query}, outputs=outputs).eval()
    default = networks.GINO(config, inputs=inputs, outputs=outputs).eval()
    default.load_state_dict(queried.state_dict())
    coordinates, features, queries = torch.rand(2, 7, 3), torch.randn(2, 7), torch.rand(2, 5, 3)
    valid = torch.tensor([[True, True, False, True, True, False, True],
                          [True, False, True, False, True, False, True]])
    coordinates[~valid], features[~valid] = float("nan"), float("nan")
    values = {xyz: Value(coordinates, ("case", "point")), source: Value(features, ("case", "point")),
              mask: Value(valid, ("case", "point"))}
    actual_queried = queried({**values, query: Value(queries, ("case", "query"))})
    actual_default = default(values)
    for case in range(2):
        # Slicing here exercises the old variable-size mathematical definition,
        # independently of the new fixed-shape masked implementation.
        sliced = dict(coordinates=coordinates[case, valid[case]].unsqueeze(0),
                      point_features=features[case, valid[case]][None, :, None])
        reference_queried = queried.predict(ModelInput(**sliced, query_coordinates=queries[case:case + 1]))
        reference_default = default.predict(ModelInput(**sliced))
        torch.testing.assert_close(actual_queried[prediction].tensor[case],
                                   reference_queried.predictions[0, :, 0], atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(actual_default[prediction].tensor[case, valid[case]],
                                   reference_default.predictions[0, :, 0], atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(actual_default[prediction].tensor[case, ~valid[case]],
                                   torch.zeros((~valid[case]).sum()))
        for actual, reference in ((actual_queried, reference_queried), (actual_default, reference_default)):
            torch.testing.assert_close(actual[latent].tensor[case], reference.latent.movedim(1, -1)[0],
                                       atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("kind", ("PointNet2", "Transolver", "GINO", "FNO3D", "UNet3D", "DeepNURBS", "SIREN"))
def test_no_batchnorm_nested_vmap_preserves_input_and_parameter_gradients(kind):
    field = Quantity("field")
    if kind in GRID:
        input_quantity = Quantity("grid")
        differentiable = torch.randn(2, 2, 4, 4, 4, requires_grad=True)
        inputs = {"voxel_fields": input_quantity}
        values = {input_quantity: Value(differentiable, ("family", "scenario", "z", "y", "x"))}
    else:
        dimension = 2 if kind == "DeepNURBS" else 3
        input_quantity = Quantity("xyz", (dimension,))
        differentiable = torch.rand(2, 2, 6, dimension, requires_grad=True)
        inputs = {"coordinates": input_quantity}
        values = {input_quantity: Value(differentiable, ("family", "scenario", "point"))}
        if kind == "GINO":
            feature = Quantity("source")
            inputs["point_features"] = feature
            values[feature] = Value(torch.randn(2, 2, 6), ("family", "scenario", "point"))
    model = getattr(networks, kind)(configuration(kind), inputs=inputs, outputs={"predictions": field}).train()
    assert not model.signature.has_ellipsis
    actual = model(values)[field]
    assert torch.isfinite(actual.tensor).all()
    actual.tensor.square().mean().backward()
    assert differentiable.grad is not None and torch.isfinite(differentiable.grad).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_siren_engine_derivative_matches_direct_autograd_and_dependent_model_is_rejected():
    xyz, temperature = Quantity("xyz", (3,)), Quantity("temperature")
    model = networks.SIREN(configuration("SIREN"), inputs={"coordinates": xyz},
                           outputs={"predictions": temperature}).eval()
    derivative = Derivative(temperature, xyz)
    points = torch.rand(2, 6, 3, requires_grad=True)
    reference = model.predict(ModelInput(coordinates=points)).predictions[..., 0]
    expected = torch.autograd.grad(reference.sum(), points)[0]
    actual = Solver([model], outputs=derivative)({xyz: Value(points.detach(), ("case", "point"))})[derivative]
    assert actual.axes == ("case", "point")
    torch.testing.assert_close(actual.tensor, expected, atol=3e-5, rtol=3e-5)
    coupled = networks.Transolver(configuration("Transolver"), inputs={"coordinates": xyz},
                                  outputs={"predictions": temperature}).eval()
    with pytest.raises(NotImplementedError, match="point"):
        Solver([coupled], outputs=derivative)({xyz: Value(points.detach(), ("case", "point"))})


def test_input_and_output_bindings_are_validated_at_construction():
    xyz, field = Quantity("xyz", (3,)), Quantity("field")
    model = networks.SIREN(configuration("SIREN"), inputs={"coordinates": xyz}, outputs={"predictions": field})
    assert set(model.inputs) == {xyz}
    for inputs, outputs in (({}, {"predictions": field}),
                            ({"coordinates": xyz, "unknown": field}, {"predictions": field}),
                            ({"coordinates": xyz}, {}),
                            ({"coordinates": xyz}, {"metadata": field}),
                            ({"coordinates": xyz}, {"predictions": field, "latent": field})):
        with pytest.raises((ValueError, TypeError)):
            networks.SIREN(configuration("SIREN"), inputs=inputs, outputs=outputs)
    with pytest.raises(ValueError):
        networks.SIREN(configuration("SIREN", input_dim=1), inputs={"coordinates": xyz},
                       outputs={"predictions": field})
    with pytest.raises(ValueError):
        networks.SIREN(configuration("SIREN"), inputs={"coordinates": xyz, "point_features": field},
                       outputs={"predictions": field})
    with pytest.raises(ValueError):
        networks.SIREN(configuration("SIREN"), inputs={"coordinates": xyz},
                       outputs={"predictions": Quantity("wrong_components", (2,))})
    with pytest.raises(ValueError):
        networks.GCN(configuration("GCN"),
                      inputs={"node_features": Quantity("nodes", (2,)), "edge_index": Quantity("edges", (2,))},
                      outputs={"latent": Quantity("latent", (4,))})
    with pytest.raises(ValueError):
        networks.GCN(configuration("GCN"),
                      inputs={"node_features": Quantity("nodes", (2,)), "edge_index": Quantity("edges", (2,)),
                              "batch": Quantity("batch")}, outputs={"predictions": field})


def test_optional_input_is_required_at_runtime_when_selected():
    xyz, mask, field = Quantity("xyz", (3,)), Quantity("valid"), Quantity("field")
    model = networks.SIREN(configuration("SIREN"), inputs={"coordinates": xyz, "point_mask": mask},
                           outputs={"predictions": field})
    with pytest.raises(ValueError, match="Expected input"):
        model({xyz: Value(torch.rand(4, 3), "point")})


def test_selected_auxiliary_and_metadata_outputs_remain_available_through_predict():
    xyz, field = Quantity("uv", (2,)), Quantity("field")
    model = networks.DeepNURBS(configuration("DeepNURBS"), inputs={"coordinates": xyz},
                               outputs={"predictions": field})
    request = ModelInput(coordinates=torch.rand(1, 5, 2))
    result = model.predict(request)
    assert "physical_coordinates" in result.metadata
    assert result.latent is not None
    assert set(model({xyz: Value(request.coordinates[0], "point")})) == {field}


@pytest.mark.parametrize("layout", ("dense", "ragged", "unbatched"))
def test_gino_outer_checkpoint_matches_third_order_ad_and_training(layout):
    xyz, features, field = Quantity("xyz", (3,)), Quantity("features"), Quantity("field")
    model = networks.GINO(configuration("GINO"),
                          inputs={"coordinates": xyz, "point_features": features},
                          outputs={"predictions": field}).double()
    assert model.checkpoint_batches is True
    reference = copy.deepcopy(model)
    reference.checkpoint_batches = False
    shape = (2, 2, 4) if layout == "dense" else (4,)
    points = torch.rand(*shape, 3, dtype=torch.double)
    source = torch.rand(shape, dtype=torch.double)
    if layout == "ragged":
        points, source = [points, points[:3].clone()], [source, source[:3].clone()]
    axes = {"dense": ("family", "scenario", "point"), "ragged": ("case", "point"),
            "unbatched": ("point",)}[layout]

    def run(network):
        leaves = source if isinstance(source, list) else [source]
        targets = [t.clone().requires_grad_() for t in leaves]
        output = network({xyz: Value(points, axes),
                          features: Value(targets if layout == "ragged" else targets[0], axes)})[field]
        tensors = output.tensor if isinstance(output.tensor, list) else [output.tensor]
        loss = sum(t.square().sum() for t in tensors)
        derivatives = []
        objective = loss
        for _ in range(3):
            derivative = torch.autograd.grad(objective, targets, create_graph=True, retain_graph=True)
            derivatives.append(derivative)
            objective = sum(t.sum() for t in derivative)
        # A loss containing higher derivatives must still train the network.
        (loss + sum(t.square().sum() for t in derivatives[-1])).backward()
        return tensors, derivatives, [p.grad for p in network.parameters()]

    actual, expected = run(model), run(reference)
    for group_a, group_b in zip(actual, expected):
        for a, b in zip(group_a, group_b):
            if isinstance(a, tuple):
                for aa, bb in zip(a, b):
                    assert torch.isfinite(aa).all()
                    torch.testing.assert_close(aa, bb)
            elif a is None:
                assert b is None
            else:
                assert torch.isfinite(a).all()
                torch.testing.assert_close(a, b)


@pytest.mark.parametrize("chunk_size", (1, 64))
def test_gino_outer_checkpoint_solver_query_higher_derivatives(chunk_size):
    xyz, features, query, field = (Quantity("xyz", (3,)), Quantity("features"),
                                     Quantity("query", (3,)), Quantity("field"))
    model = networks.GINO(configuration("GINO"),
                          inputs={"coordinates": xyz, "point_features": features, "query_coordinates": query},
                          outputs={"predictions": field}).double()
    reference = copy.deepcopy(model)
    reference.checkpoint_batches = False
    first = Derivative(field, query)
    second = Derivative(first, query)
    third = Derivative(second, query)
    values = {xyz: Value(torch.rand(2, 4, 3, dtype=torch.double), ("scenario", "point")),
              features: Value(torch.rand(2, 4, dtype=torch.double), ("scenario", "point")),
              query: Value(torch.rand(2, 2, 3, dtype=torch.double), ("scenario", "query"))}
    results = []
    for network in (model, reference):
        results.append(Solver([network], outputs=(first, second, third),
                              derivative_chunk_size=chunk_size)(values))
        results[-1][third].tensor.square().sum().backward()
    for request in (first, second, third):
        assert results[0][request].axes == ("scenario", "query")
        assert torch.isfinite(results[0][request].tensor).all()
        torch.testing.assert_close(results[0][request].tensor, results[1][request].tensor)
    for a, b in zip(model.parameters(), reference.parameters()):
        if a.grad is None:
            assert b.grad is None
        else:
            assert torch.isfinite(a.grad).all()
            torch.testing.assert_close(a.grad, b.grad)


def test_gino_checkpoint_reduces_saved_tensors_without_requiring_input_gradients():
    model, _, outputs, values, *_ = reference_case("GINO")
    saved_bytes = []
    for enabled in (False, True):
        model.checkpoint_batches = enabled
        sizes = []
        def pack(tensor):
            sizes.append(tensor.numel() * tensor.element_size())
            return tensor
        model.zero_grad(set_to_none=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            result = model(values)[outputs["predictions"]].tensor
        result.square().sum().backward()
        assert model.output_projection.weight.grad is not None
        saved_bytes.append(sum(sizes))
    # This measures saved Tensor payload, not peak allocator/GPU memory.
    assert saved_bytes[1] < saved_bytes[0]
    with torch.no_grad():
        result = model(values)[outputs["predictions"]].tensor
        assert not result.requires_grad


def test_only_gino_enables_outer_checkpoint_by_default():
    for kind in ALL_NETWORKS:
        model, *_ = reference_case(kind)
        assert model.checkpoint_batches is (kind == "GINO")
