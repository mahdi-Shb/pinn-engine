"""Execution boundaries, public host behavior, and rendering delegation."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pinn_engine import AutogradBackend, Derivative, DerivativeBackend, Model, Quantity, Signature, Solver, Value
from pinn_engine._graph import ComputationGraph
from pinn_engine.derivatives import DerivativeEngine


def make_models():
    x, z, y = map(Quantity, ("x", "z", "y"))

    class Encode(Model):
        signature = Signature({x: ()}, {z: ()})

        def evaluate(self, values):
            return {z: values[x].square()}

    class Predict(Model):
        signature = Signature({x: (), z: ()}, {y: ()})

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.))

        def evaluate(self, values):
            return {y: self.weight * (values[x] + values[z])}

    return [Predict(), Encode()], (x, z, y)


def test_derivative_engine_runs_without_a_solver_instance():
    models, (x, z, y) = make_models()
    request = Derivative(y, x)
    graph, engine = ComputationGraph(models), DerivativeEngine()
    engine.install(graph, (request,))
    graph.finalize((request,))
    values = {x: Value(torch.tensor([1., 2.]), "point")}
    calls = []

    def run_model(index, call=None):
        calls.append(index)
        model = models[index]
        inputs = {q: values[q] for q in model.inputs}
        values.update(model(inputs) if call is None else call(index, model, inputs))

    engine.execute(graph, graph.needed_nodes((request,)), values, run_model, values.update)
    torch.testing.assert_close(values[request].tensor, torch.tensor([6., 10.]))
    values[request].tensor.sum().backward()
    torch.testing.assert_close(models[0].weight.grad, torch.tensor(8.))
    assert calls == [1, 0]


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_refactored_solver_keeps_fullgraph_capture_for_ordinary_outputs(backend):
    models, (x, _, y) = make_models()
    solver = Solver(models, outputs=y)
    compiled = torch.compile(solver, backend=backend, fullgraph=True)
    points = torch.tensor([1., 2.], requires_grad=True)
    result = compiled({x: Value(points, "point")})[y]
    torch.testing.assert_close(result.tensor, 2 * (points + points.square()))
    result.tensor.sum().backward()
    torch.testing.assert_close(points.grad, 2 + 4 * points)
    assert tuple(solver.state_dict()) == ("part_0.weight",)


def test_chunk_size_and_existing_graph_metadata_delegate_to_their_owners():
    models, (x, _, y) = make_models()
    request = Derivative(y, x)
    solver = Solver(models, outputs=request)
    solver.derivative_chunk_size = 1
    assert solver._derivative_engine.chunk_size == 1
    assert solver._needed_nodes((request,)) == solver._graph.needed_nodes((request,))
    assert solver._ordered is solver._graph.ordered
    assert solver._node_dependencies is solver._graph.dependencies
    assert solver._node_dependencies[("derivative", request)] == (y, x)
    torch.testing.assert_close(solver({x: Value(torch.tensor(2.))})[request].tensor, torch.tensor(10.))
    with pytest.raises(ValueError, match="positive integer"):
        solver.derivative_chunk_size = 0


def test_graphviz_file_rendering_uses_the_same_graph_metadata(monkeypatch, tmp_path):
    models, (x, _, y) = make_models()
    solver = Solver(models, outputs=Derivative(y, x))
    commands = []

    def render(command, **kwargs):
        commands.append((command, kwargs["input"]))
        return SimpleNamespace(returncode=0, stdout=b"<svg/>", stderr=b"")

    monkeypatch.setattr("shutil.which", lambda name: "dot")
    monkeypatch.setattr("subprocess.run", render)
    path = solver.render_graph(tmp_path / "graph.svg")
    assert path.read_bytes() == b"<svg/>"
    assert commands[0][0] == ["dot", "-Tsvg", "-Gdpi=180"]
    assert commands[0][1].decode() == solver.to_dot()
    with pytest.raises(ValueError, match="filename must end"):
        solver.render_graph(tmp_path / "graph.txt")


@pytest.mark.parametrize("selection", [None, "autograd", AutogradBackend()])
def test_solver_backend_selection_keeps_autograd_as_the_default(selection):
    models, (x, _, y) = make_models()
    request = Derivative(y, x)
    options = {} if selection is None else {"derivative_backend": selection}
    solver = Solver(models, outputs=request, **options)
    assert solver.derivative_backend.name == "autograd"
    if isinstance(selection, DerivativeBackend):
        assert solver.derivative_backend is selection
    result = solver({x: Value(torch.tensor(2.))})[request]
    torch.testing.assert_close(result.tensor, torch.tensor(10.))


@pytest.mark.parametrize("name", ["fdm", "spectral", "unknown"])
def test_unimplemented_backend_names_fail_explicitly(name):
    models, _ = make_models()
    with pytest.raises(ValueError, match="Unknown or unimplemented derivative backend"):
        Solver(models, derivative_backend=name)
    with pytest.raises(TypeError, match="DerivativeBackend"):
        Solver(models, derivative_backend=123)


def test_custom_backend_is_not_forced_to_use_autograd_rules_or_a_tape():
    x, y = Quantity("analytic_x"), Quantity("analytic_y")
    request = Derivative(y, x)

    class Square(Model):
        signature = Signature({x: "point"}, {y: "point"})

        def evaluate(self, values):
            return {y: values[x].square()}

    class AnalyticSquare(DerivativeBackend):
        """Exact derivative backend for the known Square model in this test."""

        name = "analytic-square"

        def compute(self, request, values, *, chunk_size):
            assert not torch.is_grad_enabled()
            assert set(values) == {x, y}
            return Value(2 * values[x].tensor, values[x].axes)

    points = torch.tensor([1., 2.])
    solver = Solver([Square()], outputs=request, derivative_backend=AnalyticSquare())
    with torch.no_grad():
        result = solver({x: Value(points, "point")})[request]
    torch.testing.assert_close(result.tensor, 2 * points)
    assert not points.requires_grad and not result.tensor.requires_grad
    assert "analytic-square" in solver.to_dot()
    # Preserved point names now promise independence, even for a full core axis.
    ordinary = Solver([Square()], outputs=request)({x: Value(points, "point")})[request]
    torch.testing.assert_close(ordinary.tensor, result.tensor)
