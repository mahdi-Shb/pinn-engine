"""Direct AD, tensor directions, layout alignment, and training."""

import pytest
import torch
from torch import nn
from torch.func import jacrev

from pinn_engine import Derivative, Model, Quantity, Signature, Solver, Value


class Function(Model):
    geometry = None

    def __init__(self, inputs, outputs, function, name=None, calls=None):
        super().__init__()
        self._signature = Signature(inputs, outputs)
        self.function, self.name, self.calls = function, name, calls

    @property
    def signature(self):
        return self._signature

    def evaluate(self, values):
        if self.calls is not None:
            self.calls[self.name] += 1
        return self.function(values)


@pytest.mark.parametrize("chunk_size", [1, 2, 64])
def test_vector_outputs_of_different_shapes_match_full_jacobians(chunk_size):
    x, a, b = Quantity("x", (3,)), Quantity("a", (2,)), Quantity("b", (2, 2))

    def f(q):
        return torch.stack((q[0] * q[1], q[2].sin()))

    def g(q):
        return q[:2, None] * q[None, 1:]

    model = Function({x: ()}, {a: (), b: ()}, lambda v: {a: f(v[x]), b: g(v[x])})
    da, db = Derivative(a, x), Derivative(b, x)
    point = torch.tensor([0.3, -0.2, 0.7], dtype=torch.float64)
    result = Solver([model], outputs=(da, db), derivative_chunk_size=chunk_size)({x: Value(point)})
    torch.testing.assert_close(result[da].tensor, jacrev(f)(point))
    torch.testing.assert_close(result[db].tensor, jacrev(g)(point))


def test_sample_axes_are_aligned_not_expanded_to_full_sample_jacobians(monkeypatch):
    x, parameter, y = Quantity("x", (2,)), Quantity("parameter"), Quantity("y", (2,))
    model = Function({x: (), parameter: ()}, {y: ()},
                     lambda v: {y: v[parameter] * v[x].square()})
    dx, dp = Derivative(y, x), Derivative(y, parameter)
    solver = Solver([model], outputs=(dx, dp))
    points = torch.arange(24., dtype=torch.float64).reshape(3, 4, 2) / 20
    params = torch.tensor([1., 2., 3.], dtype=torch.float64)
    batch_sizes = []
    original_grad = torch.autograd.grad

    def observe(outputs, inputs, **kwargs):
        batch_sizes.append(kwargs["grad_outputs"][0].shape[0] if kwargs["is_grads_batched"] else 1)
        return original_grad(outputs, inputs, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", observe)
    result = solver({x: Value(points.permute(1, 0, 2), ("point", "scenario")),
                     parameter: Value(params, "scenario")})
    # Input declaration order sets output axes to point, scenario.
    assert result[dx].axes == ("point", "scenario")
    expected_dx = torch.diag_embed(2 * params[:, None, None] * points).permute(1, 0, 2, 3)
    torch.testing.assert_close(result[dx].tensor, expected_dx)
    torch.testing.assert_close(result[dp].tensor, points.square().permute(1, 0, 2))
    # Two component seeds for dx; point x component seeds for dp (scenario shared).
    assert sum(batch_sizes) == 2 + 4 * 2


def test_ragged_values_and_shared_scenario_parameter_keep_training():
    x, parameter, y = Quantity("x"), Quantity("parameter"), Quantity("y")
    model = Function({x: (), parameter: ()}, {y: ()},
                     lambda v: {y: v[x].square() * v[parameter]})
    dx, dp = Derivative(y, x), Derivative(y, parameter)
    points = [torch.tensor([1., 2.], requires_grad=True), torch.tensor([3., 4., 5.], requires_grad=True)]
    parameter_tensor = torch.tensor(2., requires_grad=True)
    result = Solver([model], outputs=(dx, dp))({x: Value(points, ("cases", "point")),
                                              parameter: Value(parameter_tensor)})
    assert result[dx].axes == ("cases", "point")
    assert result[dp].axes == ("cases", "point")
    for index, point in enumerate(points):
        torch.testing.assert_close(result[dx].tensor[index], 4 * point)
        torch.testing.assert_close(result[dp].tensor[index], point.square())
    sum(t.sum() for t in result[dx].tensor).backward()
    torch.testing.assert_close(parameter_tensor.grad, 2 * sum(p.detach().sum() for p in points))


def test_reducing_ragged_list_returns_derivative_on_input_list():
    x, energy = Quantity("x"), Quantity("energy")
    model = Function({x: ("geometry", "point")}, {energy: ()},
                     lambda v: {energy: sum(t.square().sum() for t in v[x])})
    request = Derivative(energy, x)
    points = [torch.tensor([1., 2.]), torch.tensor([3.])]
    result = Solver([model], outputs=request)({x: Value(points, ("geometry", "point"))})[request]
    assert result.axes == ("geometry", "point")
    for actual, point in zip(result.tensor, points):
        torch.testing.assert_close(actual, 2 * point)


def test_identity_derivative_and_distinct_quantities_sharing_storage():
    x, z, y = Quantity("x", (2,)), Quantity("z", (2,)), Quantity("y", (2,))
    model = Function({x: (), z: ()}, {y: ()}, lambda v: {y: v[x] * v[z]})
    request = Derivative(y, x)
    point = torch.tensor([2., 3.])
    result = Solver([model], outputs=(request, Derivative(x, x)))({x: Value(point), z: Value(point)})
    torch.testing.assert_close(result[request].tensor, torch.diag(point))
    torch.testing.assert_close(result[Derivative(x, x)].tensor, torch.eye(2))


def test_derivative_runs_under_no_grad_and_no_persistent_graph_is_reused():
    x, y = Quantity("x"), Quantity("y")
    model = Function({x: ()}, {y: ()}, lambda v: {y: v[x] ** 3})
    request = Derivative(y, x)
    solver = Solver([model], outputs=request)
    with torch.no_grad():
        result = solver({x: Value(torch.tensor(2.))})
    torch.testing.assert_close(result[request].tensor, torch.tensor(12.))
    result = solver({x: Value(torch.tensor(3.))})
    torch.testing.assert_close(result[request].tensor, torch.tensor(27.))
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        solver({x: Value(torch.tensor(2.))})


def test_integer_and_complex_operands_are_rejected_and_chunk_size_is_validated():
    x, y = Quantity("x"), Quantity("y")
    model = Function({x: ()}, {y: ()}, lambda v: {y: v[x] ** 2})
    solver = Solver([model], outputs=Derivative(y, x))
    for tensor in (torch.tensor(2), torch.tensor(2 + 1j)):
        with pytest.raises(TypeError, match="real floating-point"):
            solver({x: Value(tensor)})
    for size in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            Solver([model], derivative_chunk_size=size)


def test_constant_derivative_can_drive_an_outer_loss_with_zero_parameter_gradient():
    x, y = Quantity("x"), Quantity("y")
    model = Function({x: ()}, {y: ()}, lambda v: {y: 2 * v[x] + model.bias})
    model.bias = nn.Parameter(torch.tensor(1.))
    request = Derivative(y, x)
    derivative = Solver([model], outputs=request)({x: Value(torch.tensor(3.))})[request]
    derivative.tensor.square().backward()
    torch.testing.assert_close(model.bias.grad, torch.tensor(0.))


def test_deeponet_shared_trunk_derivatives_train_both_networks():
    torch.manual_seed(7)
    x, condition = Quantity("x"), Quantity("condition")
    branch, trunk, y = Quantity("branch", (4,)), Quantity("trunk", (4,)), Quantity("T")
    branch_model = Function({condition: ()}, {branch: ()},
                            lambda v: {branch: branch_model.net(v[condition].unsqueeze(-1))})
    trunk_model = Function({x: ()}, {trunk: ()},
                           lambda v: {trunk: trunk_model.net(v[x].unsqueeze(-1))})
    branch_model.net = nn.Sequential(nn.Linear(1, 5), nn.Tanh(), nn.Linear(5, 4)).double()
    trunk_model.net = nn.Sequential(nn.Linear(1, 5), nn.Tanh(), nn.Linear(5, 4)).double()
    combine = Function({branch: (), trunk: ()}, {y: ()}, lambda v: {y: (v[branch] * v[trunk]).sum()})
    request = Derivative(y, x)
    solver = Solver([combine, trunk_model, branch_model], outputs=request)
    points = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    conditions = torch.tensor([1., 2.], dtype=torch.float64)
    result = solver({x: Value(points, "point"), condition: Value(conditions, "scenario")})[request]
    assert result.axes == ("scenario", "point")
    expected = torch.stack([torch.stack([
        jacrev(lambda q: (branch_model.net(c.reshape(1)) * trunk_model.net(q.reshape(1))).sum())(point)
        for point in points]) for c in conditions])
    torch.testing.assert_close(result.tensor, expected)
    result.tensor.square().mean().backward()
    for network in (branch_model.net, trunk_model.net):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters())


def test_distinct_sample_axes_expose_a_full_coupled_jacobian_explicitly():
    x, y = Quantity("x"), Quantity("y")
    model = Function({x: "input_point"}, {y: "output_point"},
                     lambda v: {y: v[x] - v[x].mean()})
    request = Derivative(y, x)
    result = Solver([model], outputs=request, derivative_chunk_size=2)(
        {x: Value(torch.tensor([1., 2., 4.]), "input_point")})[request]
    assert result.axes == ("output_point", "input_point")
    torch.testing.assert_close(result.tensor, torch.eye(3) - torch.ones(3, 3) / 3)
