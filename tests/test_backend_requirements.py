"""Methods declare data/preparation needs without inheriting AD path analysis."""

import pytest
import torch

from pinn_engine import Derivative, DerivativeBackend, Model, Quantity, Signature, Solver, Value
import pinn_engine.derivative_paths as paths


def forbid_path_analysis(*args, **kwargs):
    raise AssertionError("A sampled-data backend must not construct a model-path plan")


def test_sampled_piecewise_linear_method_declares_mesh_without_model_paths(monkeypatch):
    temperature, x, cells = Quantity("T"), Quantity("x"), Quantity("cells", (2,))
    gradient = Derivative(temperature, x)
    loss = Quantity("loss")

    class Loss(Model):
        geometry = None
        signature = Signature({gradient: "cell"}, {loss: ()})

        def evaluate(self, values):
            return {loss: values[gradient].square().mean()}

    class PiecewiseLinear(DerivativeBackend):
        """Exact cell gradients of the 1D piecewise-linear interpolant in this test."""

        name = "piecewise-linear"

        def dependencies(self, graph, request):
            return request.of, request.wrt, cells

        def prepare(self, graph, requests):
            # Method-specific checks need not traverse source models.
            if any(request.holding for request in requests):
                raise NotImplementedError("This sampled method does not implement holding")
            return super().prepare(graph, requests)

        def execute(self, graph, requests, needed, values, run_model, store, chunk_size, *, preparation=None):
            assert preparation == {}
            assert all(isinstance(request, Derivative) for request in requests)
            edges = values[cells].tensor
            for request in requests:
                t, coordinates = values[request.of].tensor, values[request.wrt].tensor
                delta_t = t[edges[:, 1]] - t[edges[:, 0]]
                delta_x = coordinates[edges[:, 1]] - coordinates[edges[:, 0]]
                store({request: Value(delta_t / delta_x, "cell")})
            for kind, index in graph.ordered:
                if kind == "model" and (kind, index) in needed:
                    run_model(index)

    monkeypatch.setattr(paths, "plan_derivative", forbid_path_analysis)
    solver = Solver([Loss()], outputs=(gradient, loss), derivative_backend=PiecewiseLinear())
    assert solver.derivative_plans == {}
    assert set(solver.inputs) == {temperature, x, cells}
    t = torch.tensor([0., 2., 8.], requires_grad=True)
    result = solver({temperature: Value(t, "point"),
                     x: Value(torch.tensor([0., 1., 3.]), "point"),
                     cells: Value(torch.tensor([[0, 1], [1, 2]]), "cell")})
    torch.testing.assert_close(result[gradient].tensor, torch.tensor([2., 3.]))
    torch.testing.assert_close(result[loss].tensor, torch.tensor(6.5))
    result[loss].tensor.backward()
    torch.testing.assert_close(t.grad, torch.tensor([-2., 0.5, 1.5]))
    assert 'input: cells' in solver.to_mermaid()
    assert 'piecewise-linear' in solver.to_dot()

    held = Derivative(temperature, x, holding=(Quantity("unused_held"),))
    with pytest.raises(NotImplementedError, match="does not implement holding"):
        Solver([Loss()], outputs=held, derivative_backend=PiecewiseLinear())


def test_callable_backend_can_require_only_input_values_and_skip_source_forward(monkeypatch):
    x, offset, y = Quantity("x"), Quantity("offset"), Quantity("y")
    request = Derivative(y, x)
    calls = []

    class ShiftedSquare(Model):
        geometry = None
        signature = Signature({x: (), offset: ()}, {y: ()})

        def evaluate(self, values):
            calls.append("source")
            return {y: values[x].square() + values[offset]}

    class AnalyticShiftedSquare(DerivativeBackend):
        name = "analytic-shifted-square"

        def dependencies(self, graph, request):
            return (request.wrt,)

        def execute(self, graph, requests, needed, values, run_model, store, chunk_size, *, preparation=None):
            assert preparation == {}
            assert y not in values
            store({request: Value(2 * values[x].tensor) for request in requests})

    monkeypatch.setattr(paths, "plan_derivative", forbid_path_analysis)
    solver = Solver([ShiftedSquare()], outputs=request, derivative_backend=AnalyticShiftedSquare())
    result = solver({x: Value(torch.tensor(3.)), offset: Value(torch.tensor(5.))})
    torch.testing.assert_close(result[request].tensor, torch.tensor(6.))
    assert calls == []
    assert solver._node_dependencies[("derivative", request)] == (x,)


def test_preparation_metadata_is_not_required_to_contain_path_plans():
    x, y = Quantity("x"), Quantity("y")
    request = Derivative(y, x)

    class Square(Model):
        geometry = None
        signature = Signature({x: ()}, {y: ()})

        def evaluate(self, values):
            return {y: values[x].square()}

    class KnownDerivative(DerivativeBackend):
        name = "known-derivative"

        def dependencies(self, graph, request):
            return (x,)

        def prepare(self, graph, requests):
            return {"coefficient": 2.}

        def execute(self, graph, requests, needed, values, run_model, store, chunk_size, *, preparation=None):
            store({d: Value(preparation["coefficient"] * values[x].tensor) for d in requests})

    solver = Solver([Square()], outputs=request, derivative_backend=KnownDerivative())
    result = solver({x: Value(torch.tensor(4.))})
    torch.testing.assert_close(result[request].tensor, torch.tensor(8.))
    assert solver.derivative_plans == {}
    assert solver.derivative_preparation == {"coefficient": 2.}
    assert "known-derivative" in solver.to_dot()
