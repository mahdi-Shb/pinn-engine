"""Backend-specific values and mesh dependencies, without derivative plans."""

import torch
from pinn_engine import Derivative, DerivativeBackend, Model, Quantity, Signature, Solver, Value


def test_sampled_method_receives_mesh_produced_by_model_without_ad_or_paths():
    temperature, x, cells = Quantity("T"), Quantity("x"), Quantity("cells", (2,))
    gradient, loss = Derivative(temperature, x), Quantity("loss")
    calls = []

    class Connectivity(Model):
        signature = Signature({x: "point"}, {cells: "cell"})

        def evaluate(self, values):
            calls.append("mesh")
            return {cells: torch.tensor([[0, 1], [1, 2]])}

    class Loss(Model):
        signature = Signature({gradient: "cell"}, {loss: ()})

        def evaluate(self, values):
            calls.append("loss")
            return {loss: values[gradient].square().mean()}

    class PiecewiseLinear(DerivativeBackend):
        """Test-only gradient of a 1D piecewise-linear field on cells."""
        name = "piecewise-linear"

        def dependencies(self, graph, request):
            return request.of, request.wrt, cells

        def compute(self, request, values, *, chunk_size):
            assert set(values) == {temperature, x, cells}
            calls.append("gradient")
            edges = values[cells].tensor
            t, coordinates = values[request.of].tensor, values[request.wrt].tensor
            return Value((t[edges[:, 1]] - t[edges[:, 0]]) /
                         (coordinates[edges[:, 1]] - coordinates[edges[:, 0]]), "cell")

    solver = Solver([Loss(), Connectivity()], outputs=(gradient, loss), derivative_backend=PiecewiseLinear())
    assert set(solver.inputs) == {temperature, x}
    t = torch.tensor([0., 2., 8.], requires_grad=True)
    result = solver({temperature: Value(t, "point"), x: Value(torch.tensor([0., 1., 3.]), "point")})
    torch.testing.assert_close(result[gradient].tensor, torch.tensor([2., 3.]))
    torch.testing.assert_close(result[loss].tensor, torch.tensor(6.5))
    result[loss].tensor.backward()
    torch.testing.assert_close(t.grad, torch.tensor([-2., 0.5, 1.5]))
    assert calls == ["mesh", "gradient", "loss"]
    assert "input: cells" in solver.to_mermaid()
    assert "piecewise-linear" in solver.to_dot()
    assert not hasattr(solver, "derivative_plans")


def test_backend_can_omit_primal_value_and_skip_source_forward():
    x, y = Quantity("x"), Quantity("y")
    request = Derivative(y, x)

    class Square(Model):
        signature = Signature({x: ()}, {y: ()})

        def evaluate(self, values):
            raise AssertionError("The analytical derivative does not need primal y")

    class KnownDerivative(DerivativeBackend):
        name = "known-derivative"

        def dependencies(self, graph, request):
            return (x,)

        def compute(self, request, values, *, chunk_size):
            assert set(values) == {x}
            return Value(2 * values[x].tensor)

    solver = Solver([Square()], outputs=request, derivative_backend=KnownDerivative())
    with torch.no_grad():
        result = solver({x: Value(torch.tensor(4.))})
    torch.testing.assert_close(result[request].tensor, torch.tensor(8.))
    assert "known-derivative" in solver.to_dot()
