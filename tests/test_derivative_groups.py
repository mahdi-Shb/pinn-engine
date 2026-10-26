"""Reverse sharing and backend-owned grouping across layouts and frontiers."""

import pytest
import torch
from torch.func import jacrev
from pinn_engine import AutogradBackend, Derivative, DerivativeBackend, Model, Quantity, Signature, Solver, Value


class Function(Model):
    def __init__(self, inputs, outputs, fn):
        super().__init__()
        self._signature, self.fn = Signature(inputs, outputs), fn
        self.calls = 0

    @property
    def signature(self):
        return self._signature

    def evaluate(self, values):
        self.calls += 1
        return self.fn(values)


@pytest.fixture
def reverse_calls(monkeypatch):
    calls = []
    original = torch.autograd.grad

    def observe(outputs, inputs, *args, **kwargs):
        if "is_grads_batched" in kwargs:
            count = kwargs["grad_outputs"][0].shape[0] if kwargs["is_grads_batched"] else 1
            calls.append((len(outputs), len(inputs), count))
        return original(outputs, inputs, *args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", observe)
    return calls


def test_scalar_source_shares_one_reverse_with_different_target_shapes_and_unused_target(reverse_calls):
    x, z, unused, y = Quantity("x"), Quantity("z", (3,)), Quantity("unused"), Quantity("T")
    model = Function({x: (), z: (), unused: ()}, {y: ()}, lambda v: {y: v[x]**2 + v[z].pow(3).sum()})
    requests = tuple(Derivative(y, q) for q in (x, z, unused))
    result = Solver([model], outputs=requests)({x: Value(torch.tensor(2.)),
        z: Value(torch.tensor([1., 2., 3.])), unused: Value(torch.tensor(7.))})
    for request, reference in zip(requests, (torch.tensor(4.), torch.tensor([3., 12., 27.]), torch.tensor(0.))):
        torch.testing.assert_close(result[request].tensor, reference)
    assert reverse_calls == [(1, 3, 1)]
    assert model.calls == 1


@pytest.mark.parametrize("chunk_size", [1, 2, 64])
def test_vector_source_shares_seed_chunks_without_repeating_per_target(chunk_size, reverse_calls):
    x, z, y = Quantity("x", (2,)), Quantity("z"), Quantity("T", (3,))
    model = Function({x: (), z: ()}, {y: ()}, lambda v: {y: torch.stack(
        (v[x][0]*v[z], v[x][1]**2 + v[z]**3, (v[x].sum()*v[z]).sin()))})
    dx, dz = Derivative(y, x), Derivative(y, z)
    xx, zz = torch.tensor([0.3, 0.7], dtype=torch.float64), torch.tensor(0.4, dtype=torch.float64)
    result = Solver([model], outputs=(dx, dz), derivative_chunk_size=chunk_size)({x: Value(xx), z: Value(zz)})
    # Record before functional reference AD adds its own calls.
    seen = list(reverse_calls)
    def f(a, b):
        return torch.stack((a[0]*b, a[1]**2+b**3, (a.sum()*b).sin()))
    expected = jacrev(f, argnums=(0, 1))(xx, zz)
    torch.testing.assert_close(result[dx].tensor, expected[0])
    torch.testing.assert_close(result[dz].tensor, expected[1])
    assert all(outputs == 1 and targets == 2 for outputs, targets, _ in seen)
    assert sum(count for _, _, count in seen) == 3
    assert len(seen) == (3 + chunk_size - 1)//chunk_size


def test_point_and_scenario_seed_layouts_form_distinct_groups(reverse_calls):
    x, z, a, b, y = map(Quantity, ("x", "z", "a", "b", "T"))
    model = Function({x: (), z: (), a: (), b: ()}, {y: ()},
                     lambda v: {y: v[x]*v[z]*(v[a]+v[b])})
    requests = tuple(Derivative(y, q) for q in (x, z, a, b))
    xx, zz = torch.tensor([1., 2., 3.]), torch.tensor([2., 4., 6.])
    aa, bb = torch.tensor([1., 3.]), torch.tensor([0.5, 1.5])
    result = Solver([model], outputs=requests)({x: Value(xx, "point"), z: Value(zz, "point"),
        a: Value(aa, "scenario"), b: Value(bb, "scenario")})
    references = (zz[:, None]*(aa+bb), xx[:, None]*(aa+bb),
                  (xx*zz)[:, None].expand(-1, 2), (xx*zz)[:, None].expand(-1, 2))
    for request, reference in zip(requests, references):
        assert result[request].axes == ("point", "scenario")
        torch.testing.assert_close(result[request].tensor, reference)
    assert sorted(reverse_calls) == [(1, 2, 2), (1, 2, 3)]


def test_shared_axes_can_have_different_target_order(reverse_calls):
    x, z, y = map(Quantity, ("x", "z", "T"))
    xx = torch.arange(6.).reshape(2, 3)
    zz = torch.arange(6., 12.).reshape(3, 2)
    model = Function({x: (), z: ()}, {y: ()}, lambda v: {y: v[x]**2 + v[z]**3})
    dx, dz = Derivative(y, x), Derivative(y, z)
    result = Solver([model], outputs=(dx, dz))({x: Value(xx, ("point", "scenario")),
                                              z: Value(zz, ("scenario", "point"))})
    torch.testing.assert_close(result[dx].tensor, 2*xx)
    torch.testing.assert_close(result[dz].tensor, 3*zz.T.square())
    assert reverse_calls == [(1, 2, 1)]


def test_full_cross_axis_jacobians_share_directions(reverse_calls):
    x, z, y = map(Quantity, ("x", "z", "T"))
    model = Function({x: "input_x", z: "input_z"}, {y: "output"},
                     lambda v: {y: v[x] - v[x].mean() + v[z].square()})
    dx, dz = Derivative(y, x), Derivative(y, z)
    result = Solver([model], outputs=(dx, dz))({x: Value(torch.tensor([1., 2., 3.]), "input_x"),
                                              z: Value(torch.tensor([2., 3., 4.]), "input_z")})
    assert result[dx].axes == ("output", "input_x")
    assert result[dz].axes == ("output", "input_z")
    torch.testing.assert_close(result[dx].tensor, torch.eye(3)-torch.ones(3, 3)/3)
    torch.testing.assert_close(result[dz].tensor, torch.diag(torch.tensor([4., 6., 8.])))
    assert reverse_calls == [(1, 2, 3)]


def test_ragged_groups_select_only_corresponding_target_leaves(reverse_calls):
    x, z, a, y = map(Quantity, ("x", "z", "a", "T"))
    model = Function({x: (), z: (), a: ()}, {y: ()}, lambda v: {y: v[a]*(v[x]**2 + v[z]**3)})
    dx, dz, da = (Derivative(y, q) for q in (x, z, a))
    xx, zz = [torch.tensor([1., 2.]), torch.tensor([3., 4., 5.])], [torch.ones(2), torch.full((3,), 2.)]
    result = Solver([model], outputs=(dx, dz, da))({x: Value(xx, ("case", "point")),
        z: Value(zz, ("case", "point")), a: Value(torch.tensor(2.))})
    for i in range(2):
        torch.testing.assert_close(result[dx].tensor[i], 4*xx[i])
        torch.testing.assert_close(result[dz].tensor[i], 6*zz[i].square())
        torch.testing.assert_close(result[da].tensor[i], xx[i].square()+zz[i].pow(3))
    assert reverse_calls == [(1, 2, 1), (1, 2, 1), (1, 1, 2), (1, 1, 3)]
    assert model.calls == 2  # One evaluate per list member, no replay.


def test_scalar_reduction_shares_reverse_across_all_ragged_members_and_dense_target(reverse_calls):
    x, z, a, energy = map(Quantity, ("x", "z", "a", "energy"))
    model = Function({x: ("case", "point"), z: ("case", "point"), a: ()}, {energy: ()},
        lambda v: {energy: v[a]*sum(t.square().sum() for t in v[x])+sum(t.pow(3).sum() for t in v[z])})
    dx, dz, da = (Derivative(energy, q) for q in (x, z, a))
    xx, zz = [torch.tensor([1., 2.]), torch.tensor([3.])], [torch.tensor([2., 3.]), torch.tensor([4.])]
    result = Solver([model], outputs=(dx, dz, da))({x: Value(xx, ("case", "point")),
        z: Value(zz, ("case", "point")), a: Value(torch.tensor(2.))})
    for i in range(2):
        torch.testing.assert_close(result[dx].tensor[i], 4*xx[i])
        torch.testing.assert_close(result[dz].tensor[i], 3*zz[i].square())
    torch.testing.assert_close(result[da].tensor, torch.tensor(14.))
    assert reverse_calls == [(1, 5, 1)]


def test_higher_orders_group_only_after_their_dependencies_are_ready():
    x, z, y = map(Quantity, ("x", "z", "T"))
    model = Function({x: (), z: ()}, {y: ()}, lambda v: {y: v[x]**4 + v[x]*v[z]**3})
    dx, dz = Derivative(y, x), Derivative(y, z)
    dxx, dxz, dzx, dzz = Derivative(dx, x), Derivative(dx, z), Derivative(dz, x), Derivative(dz, z)
    groups = []
    class Observe(AutogradBackend):
        def compute_group(self, requests, inputs, *, chunk_size, prepared=None):
            groups.append(tuple(requests))
            return super().compute_group(requests, inputs, chunk_size=chunk_size, prepared=prepared)
    result = Solver([model], outputs=(dxx, dxz, dzx, dzz), derivative_backend=Observe())({
        x: Value(torch.tensor(2.)), z: Value(torch.tensor(3.))})
    for request, expected in zip((dxx, dxz, dzx, dzz), (48., 27., 27., 36.)):
        torch.testing.assert_close(result[request].tensor, torch.tensor(expected))
    assert len(groups) == 3
    assert set(groups[0]) == {dx, dz}
    assert {frozenset(g) for g in groups[1:]} == {frozenset((dxx, dxz)), frozenset((dzx, dzz))}
    assert model.calls == 1


def test_sources_are_ready_together_even_when_producers_are_separated_and_unselected_work_is_skipped():
    x, z, u, y = map(Quantity, ("x", "z", "u", "T"))
    d1, d2 = Derivative(y, x), Derivative(y, z)
    dead = Quantity("dead")
    source = Function({x: (), z: ()}, {y: ()}, lambda v: {y: v[x]*v[z]})
    unused = Function({x: ()}, {dead: ()}, lambda v: (_ for _ in ()).throw(AssertionError("Unselected")))
    consume = Function({d1: (), d2: ()}, {u: ()}, lambda v: {u: v[d1]+v[d2]})
    groups = []
    class Observe(AutogradBackend):
        def compute_group(self, requests, inputs, *, chunk_size, prepared=None):
            groups.append(tuple(requests))
            return super().compute_group(requests, inputs, chunk_size=chunk_size, prepared=prepared)
    solver = Solver([consume, unused, source], outputs=u, derivative_backend=Observe())
    torch.testing.assert_close(solver({x: Value(torch.tensor(2.)), z: Value(torch.tensor(3.))})[u].tensor,
                               torch.tensor(5.))
    assert len(groups) == 1 and set(groups[0]) == {d1, d2}
    assert source.calls == consume.calls == 1 and unused.calls == 0


def test_empty_holding_and_total_can_share_the_same_direction(reverse_calls):
    x, y = Quantity("x"), Quantity("T")
    model = Function({x: ()}, {y: ()}, lambda v: {y: v[x]**3})
    total, empty = Derivative(y, x), Derivative(y, x, holding=())
    result = Solver([model], outputs=(total, empty))({x: Value(torch.tensor(2.))})
    torch.testing.assert_close(result[total].tensor, result[empty].tensor)
    assert reverse_calls == [(1, 1, 1)]  # Target tensors are deduplicated too.


def test_backward_training_of_grouped_requests_matches_direct_reference():
    x, z, y = map(Quantity, ("x", "z", "T"))
    model = Function({x: (), z: ()}, {y: ()}, lambda v: {y: (model.weight*v[x]*v[z]).sin()})
    model.weight = torch.nn.Parameter(torch.tensor(1.3, dtype=torch.float64))
    dx, dz = Derivative(y, x), Derivative(y, z)
    xx, zz = torch.tensor(0.2, dtype=torch.float64, requires_grad=True), torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    result = Solver([model], outputs=(dx, dz))({x: Value(xx), z: Value(zz)})
    loss = result[dx].tensor.square() + 2*result[dz].tensor.square()
    actual = torch.autograd.grad(loss, (model.weight, xx, zz))
    w, a, b = (t.detach().clone().requires_grad_() for t in (model.weight, xx, zz))
    reference = (w*b*(w*a*b).cos()).square() + 2*(w*a*(w*a*b).cos()).square()
    expected = torch.autograd.grad(reference, (w, a, b))
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want)


def test_grouped_backend_must_return_exactly_its_requested_keys():
    x, y = Quantity("x"), Quantity("y")
    class Invalid(DerivativeBackend):
        name = "invalid"
        def compute(self, request, values, *, chunk_size):
            return Value(torch.tensor(1.))
        def compute_group(self, requests, inputs, *, chunk_size, prepared=None):
            return {}
    source = Function({x: ()}, {y: ()}, lambda v: {y: v[x]})
    with pytest.raises(ValueError, match="different from its group"):
        Solver([source], outputs=Derivative(y, x), derivative_backend=Invalid())({x: Value(torch.tensor(2.))})


def test_common_reverse_node_receives_one_adjoint_not_one_per_wrt():
    x, z, y = map(Quantity, ("x", "z", "T"))
    backward_visits = []
    def f(values):
        shared = (values[x]+values[z]).sin()
        shared.register_hook(lambda gradient: backward_visits.append(gradient.detach().clone()))
        return {y: shared.square()}
    model = Function({x: (), z: ()}, {y: ()}, f)
    requests = Derivative(y, x), Derivative(y, z)
    result = Solver([model], outputs=requests)({x: Value(torch.tensor(0.2)), z: Value(torch.tensor(0.3))})
    expected = 2*torch.sin(torch.tensor(0.5))*torch.cos(torch.tensor(0.5))
    for value in result.values():
        torch.testing.assert_close(value.tensor, expected)
    assert len(backward_visits) == 1
    torch.testing.assert_close(backward_visits[0], 2*torch.sin(torch.tensor(0.5)))


def test_empty_output_directions_require_no_reverse_sweep_but_keep_zero_training_gradients(reverse_calls):
    x, z, y = Quantity("x", (3,)), Quantity("z"), Quantity("T", (0,))
    model = Function({x: (), z: ()}, {y: ()}, lambda v: {y: v[x][:0]+v[z]})
    dx, dz = Derivative(y, x), Derivative(y, z)
    xx, zz = torch.ones(3, requires_grad=True), torch.tensor(2., requires_grad=True)
    result = Solver([model], outputs=(dx, dz))({x: Value(xx), z: Value(zz)})
    assert result[dx].tensor.shape == (0, 3)
    assert result[dz].tensor.shape == (0,)
    assert reverse_calls == []
    sum(v.tensor.sum() for v in result.values()).backward()
    torch.testing.assert_close(xx.grad, torch.zeros_like(xx))
    torch.testing.assert_close(zz.grad, torch.tensor(0.))
