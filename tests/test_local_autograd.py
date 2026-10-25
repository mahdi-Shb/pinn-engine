"""Numerical AD, held paths, shared VJPs, layout alignment, and training."""

from collections import Counter

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


def test_three_paths_and_different_requests_share_forward_and_local_vjps(monkeypatch):
    x, a, b, c, d, t = map(Quantity, ("x", "a", "b", "c", "d", "t"))
    calls = Counter()
    models = [
        Function({x: ()}, {a: ()}, lambda v: {a: v[x] ** 2}, "A", calls),
        Function({x: ()}, {b: ()}, lambda v: {b: v[x].exp()}, "B", calls),
        Function({x: ()}, {c: ()}, lambda v: {c: v[x].sin()}, "C", calls),
        Function({a: (), b: ()}, {d: ()}, lambda v: {d: v[a] * v[b]}, "D", calls),
        Function({d: (), c: ()}, {t: ()}, lambda v: {t: v[d] ** 2 + v[c]}, "T", calls),
    ]
    requests = (Derivative(t, x), Derivative(t, x, holding=(b,)),
                Derivative(t, d), Derivative(t, b), Derivative(d, x, holding=(a,)))
    final_tensor, batch_sizes = [], []
    models[-1].register_forward_hook(lambda m, args, result: final_tensor.append(result[t].tensor))
    original_grad = torch.autograd.grad

    def observe(outputs, inputs, **kwargs):
        if any(output is final_tensor[0] for output in outputs):
            batch_sizes.append(kwargs["grad_outputs"][0].shape[0])
        return original_grad(outputs, inputs, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", observe)
    solver = Solver(models[::-1], outputs=requests)
    point = torch.tensor(0.3, dtype=torch.float64)
    actual = solver({x: Value(point)})
    aa, bb = point.square(), point.exp()
    dd = aa * bb
    expected = (2 * dd * bb * (2 * point + aa) + point.cos(),
                2 * dd * bb * 2 * point + point.cos(), 2 * dd, 2 * dd * aa, aa * bb)
    for request, number in zip(requests, expected):
        torch.testing.assert_close(actual[request].tensor, number)
    assert calls == dict.fromkeys(("A", "B", "C", "D", "T"), 1)
    # Four requests with different wrt/holding share the same root VJP of T.
    assert batch_sizes == [1]
    assert not point.requires_grad  # No mutation of caller-owned input flags.


def test_fork_with_held_and_unheld_children_keeps_the_shared_ancestor():
    x, u, v, z, w, t = map(Quantity, ("x", "u", "v", "z", "w", "t"))
    models = [
        Function({x: ()}, {u: ()}, lambda q: {u: q[x] ** 2}),
        Function({x: ()}, {v: ()}, lambda q: {v: q[x].sin()}),
        Function({u: ()}, {z: ()}, lambda q: {z: q[u].exp()}),
        Function({u: ()}, {w: ()}, lambda q: {w: q[u] ** 3}),
        Function({v: (), z: (), w: ()}, {t: ()}, lambda q: {t: q[v] * q[z] + q[w]}),
    ]
    request = Derivative(t, x, holding=(z,))
    points = torch.tensor([0.2, 0.4], dtype=torch.float64)
    result = Solver(models, outputs=request)({x: Value(points, "point")})[request]
    torch.testing.assert_close(result.tensor, points.cos() * points.square().exp() + 6 * points ** 5)


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


def test_training_through_held_values_and_parameters_matches_functional_ad():
    x, a, b, c, d, t = map(Quantity, ("x", "a", "b", "c", "d", "t"))
    producer = Function({x: ()}, {b: ()}, lambda v: {b: (producer.weight * v[x]).exp()})
    producer.weight = nn.Parameter(torch.tensor(1.3, dtype=torch.float64))
    models = [producer,
              Function({x: ()}, {a: ()}, lambda v: {a: v[x] ** 2}),
              Function({x: ()}, {c: ()}, lambda v: {c: v[x].sin()}),
              Function({a: (), b: ()}, {d: ()}, lambda v: {d: v[a] * v[b]}),
              Function({d: (), c: ()}, {t: ()}, lambda v: {t: v[d] ** 2 + v[c]})]
    request = Derivative(t, x, holding=(b,))
    point = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    result = Solver(models, outputs=(request, t))({x: Value(point)})
    loss = result[request].tensor.square() + result[t].tensor.square()
    actual_grads = torch.autograd.grad(loss, (producer.weight, point))
    theta = producer.weight.detach().clone().requires_grad_()
    xx = point.detach().clone().requires_grad_()
    held = (theta * xx).exp()
    partial = jacrev(lambda q, fixed: (q.square() * fixed).square() + q.sin(), argnums=0)(xx, held)
    reference_loss = partial.square() + ((xx.square() * held).square() + xx.sin()).square()
    expected_grads = torch.autograd.grad(reference_loss, (theta, xx))
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected)


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
        batch_sizes.append(kwargs["grad_outputs"][0].shape[0])
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


def test_all_held_or_unused_inputs_give_differentiable_zeros():
    x, z, y = map(Quantity, ("x", "z", "y"))
    models = [Function({x: ()}, {z: ()}, lambda v: {z: v[x].square()}),
              Function({z: ()}, {y: ()}, lambda v: {y: 3 * v[z]})]
    request = Derivative(y, x, holding=(z,))
    point = torch.tensor([1., 2.], requires_grad=True)
    result = Solver(models, outputs=request)({x: Value(point, "point")})[request]
    torch.testing.assert_close(result.tensor, torch.zeros_like(point))
    result.tensor.sum().backward()
    torch.testing.assert_close(point.grad, torch.zeros_like(point))
    constant = Function({x: ()}, {y: ()}, lambda v: {y: torch.tensor(3.)})
    result = Solver([constant], outputs=Derivative(y, x))({x: Value(torch.tensor(2.))})
    torch.testing.assert_close(result[Derivative(y, x)].tensor, torch.tensor(0.))


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


def test_different_sources_batch_on_their_common_large_intermediate(monkeypatch):
    x, c, a, b = Quantity("x", (3,)), Quantity("c", (16,)), Quantity("a", (2,)), Quantity("b", (3,))
    matrix = torch.arange(48., dtype=torch.float64).reshape(16, 3) / 100
    calls = Counter()
    trunk = Function({x: ()}, {c: ()}, lambda v: {c: (matrix @ v[x]).tanh()}, "C", calls)
    head_a = Function({c: ()}, {a: ()}, lambda v: {a: v[c][:2].square()}, "A", calls)
    head_b = Function({c: ()}, {b: ()}, lambda v: {b: v[c][-3:].sin()}, "B", calls)
    requests = Derivative(a, x), Derivative(b, x), Derivative(b, c)
    shared, batch_sizes = [], []
    trunk.register_forward_hook(lambda m, args, result: shared.append(result[c].tensor))
    original_grad = torch.autograd.grad

    def observe(outputs, inputs, **kwargs):
        if any(output is shared[0] for output in outputs):
            batch_sizes.append(kwargs["grad_outputs"][0].shape[0])
        return original_grad(outputs, inputs, **kwargs)

    point = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64)
    with monkeypatch.context() as patch:
        patch.setattr(torch.autograd, "grad", observe)
        result = Solver([head_a, head_b, trunk], outputs=requests)({x: Value(point)})
    torch.testing.assert_close(result[requests[0]].tensor, jacrev(lambda q: (matrix @ q).tanh()[:2].square())(point))
    torch.testing.assert_close(result[requests[1]].tensor, jacrev(lambda q: (matrix @ q).tanh()[-3:].sin())(point))
    torch.testing.assert_close(result[requests[2]].tensor, jacrev(lambda q: q[-3:].sin())((matrix @ point).tanh()))
    assert calls == {"A": 1, "B": 1, "C": 1}
    assert batch_sizes == [5]  # Two A directions and three B directions together.


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


def test_real_derivative_with_a_complex_intermediate_still_respects_holding():
    x, c, z, y = map(Quantity, ("real_x", "complex_c", "real_z", "real_y"))
    models = [Function({x: ()}, {c: ()}, lambda v: {c: (1 + 2j) * v[x]}),
              Function({c: ()}, {z: ()}, lambda v: {z: v[c].real.square()}),
              Function({c: (), z: ()}, {y: ()}, lambda v: {y: v[c].abs().square() + v[z]})]
    request = Derivative(y, x, holding=(z,))
    points = torch.tensor([0.2, 0.7], dtype=torch.float64)
    result = Solver(models, outputs=request)({x: Value(points, "point")})[request]
    torch.testing.assert_close(result.tensor, 10 * points)
