"""Nested direct AD, shared forward, mixed derivatives, and zero higher orders."""

from collections import Counter
import pytest
import torch
from torch import nn
from torch.func import jacrev
from pinn_engine import AutogradBackend, Derivative, Model, Quantity, Signature, Solver, Value


class Function(Model):
    geometry = None

    def __init__(self, inputs, outputs, fn, calls=None, name="forward"):
        super().__init__()
        self._signature = Signature(inputs, outputs)
        self.fn, self.calls, self.name = fn, calls, name

    @property
    def signature(self):
        return self._signature

    def evaluate(self, values):
        if self.calls is not None:
            self.calls[self.name] += 1
        return self.fn(values)


def test_fifth_order_and_mixed_scenario_derivatives_share_one_forward_and_train():
    x, t, y = Quantity("x"), Quantity("t"), Quantity("y")
    calls = Counter()
    model = Function({x: (), t: ()}, {y: ()},
                     lambda v: {y: model.weight * v[x]**4 + v[t] * v[x]**2}, calls)
    model.weight = nn.Parameter(torch.tensor(2., dtype=torch.float64))
    dx = Derivative(y, x)
    dxx, dxt = Derivative(dx, x), Derivative(dx, t)
    dxxx = Derivative(dxx, x)
    fourth = Derivative(dxxx, x)
    fifth = Derivative(fourth, x)
    solver = Solver([model], outputs=(dx, dxx, dxxx, fourth, fifth, dxt))
    points = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)
    times = torch.tensor([1., 3., 5.], dtype=torch.float64)
    result = solver({x: Value(points, "point"), t: Value(times, "scenario")})
    xx, tt = points[None, :], times[:, None]
    expected = (8*xx**3 + 2*tt*xx, 24*xx**2 + 2*tt,
                (48*xx).expand(3, -1), torch.full((3, 2), 48., dtype=torch.float64),
                torch.zeros((3, 2), dtype=torch.float64), (2*xx).expand(3, -1))
    for request, reference in zip(solver.outputs, expected):
        assert result[request].axes == ("point", "scenario")
        torch.testing.assert_close(result[request].tensor, reference.T)
    result[dxx].tensor.sum().backward()
    torch.testing.assert_close(model.weight.grad, 3 * (12*points.square()).sum())
    torch.testing.assert_close(points.grad, 3 * 48 * points)
    assert calls == {"forward": 1}
    assert not times.requires_grad
    assert set(solver.derivatives) == {dx, dxx, dxxx, fourth, fifth, dxt}


@pytest.mark.parametrize("chunk_size", [1, 2, 64])
def test_vector_hessian_with_vmapped_points_matches_torch_func(chunk_size):
    x, y = Quantity("x", (2,)), Quantity("y", (3,))

    def f(q):
        return torch.stack((q[0]**3 + q[1]**2, (q[0]*q[1]).sin(), q.prod()))

    calls = Counter()
    model = Function({x: ()}, {y: ()}, lambda v: {y: f(v[x])}, calls)
    first = Derivative(y, x)
    second = Derivative(first, x)
    solver = Solver([model], outputs=second, derivative_chunk_size=chunk_size)
    points = torch.tensor([[0.3, -0.2], [0.5, 0.7]], dtype=torch.float64)
    result = solver({x: Value(points, "point")})[second]
    assert result.axes == ("point",)
    assert second.shape == (3, 2, 2)
    torch.testing.assert_close(result.tensor, torch.stack([jacrev(jacrev(f))(p) for p in points]))
    assert calls == {"forward": 1}


def test_model_can_consume_first_derivative_then_be_differentiated_without_replay():
    x, y, energy = Quantity("x"), Quantity("y"), Quantity("energy")
    first = Derivative(y, x)
    request = Derivative(energy, x)
    calls = Counter()
    source = Function({x: ()}, {y: ()}, lambda v: {y: v[x]**3}, calls, "source")
    residual = Function({first: ()}, {energy: ()}, lambda v: {energy: v[first]**2}, calls, "residual")
    result = Solver([residual, source], outputs=request)({x: Value(torch.tensor([1., 2.]), "point")})
    torch.testing.assert_close(result[request].tensor, torch.tensor([36., 288.]))
    assert calls == {"source": 1, "residual": 1}


def test_two_scalar_requests_use_separate_backward_paths_and_one_shared_forward(monkeypatch):
    x, c, a, b = map(Quantity, ("x", "c", "a", "b"))
    calls = Counter()
    models = [Function({x: ()}, {c: ()}, lambda v: {c: v[x]**2}, calls, "C"),
              Function({c: ()}, {a: ()}, lambda v: {a: v[c]**2}, calls, "A"),
              Function({c: ()}, {b: ()}, lambda v: {b: v[c]**3}, calls, "B")]
    da, db = Derivative(a, x), Derivative(b, x)
    seen = []
    original = torch.autograd.grad

    def observe(outputs, inputs, **kwargs):
        seen.append((len(outputs), len(inputs), kwargs["is_grads_batched"]))
        return original(outputs, inputs, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", observe)
    result = Solver(models, outputs=(da, db))({x: Value(torch.tensor(2.))})
    torch.testing.assert_close(result[da].tensor, torch.tensor(32.))
    torch.testing.assert_close(result[db].tensor, torch.tensor(192.))
    assert seen == [(1, 1, False), (1, 1, False)]
    assert calls == {"C": 1, "A": 1, "B": 1}


def test_constant_and_unused_derivatives_remain_valid_at_further_orders():
    x, z, y = Quantity("x"), Quantity("z"), Quantity("y")
    source = Function({x: (), z: ()}, {y: ()}, lambda v: {y: source.bias + 2*v[z]})
    source.bias = nn.Parameter(torch.tensor(3.))
    first = Derivative(y, x)
    third = Derivative(Derivative(first, x), x)
    tensor = torch.tensor([1., 2.], requires_grad=True)
    result = Solver([source], outputs=third)({x: Value(tensor, "point"), z: Value(torch.tensor(4.))})[third]
    torch.testing.assert_close(result.tensor, torch.zeros(2))
    result.tensor.sum().backward()
    torch.testing.assert_close(tensor.grad, torch.zeros(2))
    torch.testing.assert_close(source.bias.grad, torch.tensor(0.))


def test_intermediate_target_with_bypass_branch_is_a_distinct_tensor_boundary():
    x, z, y = map(Quantity, ("x", "z", "y"))
    # Even an identity producer must not turn differentiation wrt z into wrt x.
    models = [Function({x: ()}, {z: ()}, lambda v: {z: v[x]}),
              Function({x: (), z: ()}, {y: ()}, lambda v: {y: v[x]**2 + 3*v[z]})]
    dz, dx = Derivative(y, z), Derivative(y, x)
    result = Solver(models, outputs=(dz, dx))({x: Value(torch.tensor([1., 2.]), "point")})
    torch.testing.assert_close(result[dz].tensor, torch.tensor([3., 3.]))
    torch.testing.assert_close(result[dx].tensor, torch.tensor([5., 7.]))


def test_nonempty_holding_is_explicitly_rejected_but_empty_is_allowed():
    x, y, z = Quantity("x"), Quantity("y"), Quantity("z")
    model = Function({x: ()}, {y: ()}, lambda v: {y: v[x]**2})
    request = Derivative(y, x, holding=(z,))
    with pytest.raises(NotImplementedError, match="nonempty holding"):
        Solver([model], outputs=request)({x: Value(torch.tensor(2.))})
    empty = Derivative(y, x, holding=())
    torch.testing.assert_close(Solver([model], outputs=empty)({x: Value(torch.tensor(2.))})[empty].tensor,
                               torch.tensor(4.))


def test_backend_can_compute_from_only_two_connected_values():
    x, y = Quantity("x"), Quantity("y")
    tensor = torch.tensor(2., requires_grad=True)
    values = {x: Value(tensor), y: Value(tensor**4)}
    backend = AutogradBackend()
    first = Derivative(y, x)
    values[first] = backend.compute(first, values)
    second = backend.compute(Derivative(first, x), {first: values[first], x: values[x]})
    torch.testing.assert_close(second.tensor, torch.tensor(48.))


@pytest.mark.parametrize("batched", [False, True])
def test_intermediate_target_returned_with_its_dependent_sibling_output(batched):
    x, z, y = map(Quantity, ("x", "z", "y"))

    def f(values):
        intermediate = values[x].square()
        return {z: intermediate, y: intermediate**3}

    model = Function({x: ()}, {z: (), y: ()}, f)
    dz, dx = Derivative(y, z), Derivative(y, x)
    tensor = torch.tensor([2., 3.]) if batched else torch.tensor(2.)
    result = Solver([model], outputs=(dz, dx))({x: Value(tensor, "point" if batched else ())})
    torch.testing.assert_close(result[dz].tensor, 3*tensor**4)
    torch.testing.assert_close(result[dx].tensor, 6*tensor**5)


def test_ragged_second_derivative_and_real_outputs_through_complex_operations():
    x, y = Quantity("x"), Quantity("y")
    model = Function({x: ()}, {y: ()}, lambda v: {y: ((1+2j)*v[x]**2).abs().square()})
    second = Derivative(Derivative(y, x), x)
    points = [torch.tensor([0.2, 0.7]), torch.tensor([0.4])]
    result = Solver([model], outputs=second)({x: Value(points, ("geometry", "point"))})[second]
    assert result.axes == ("geometry", "point")
    for actual, point in zip(result.tensor, points):
        torch.testing.assert_close(actual, 60*point.square())
