"""Ragged case batching, retained lists, and shared dense sample axes."""

import pytest
import torch
from torch import nn

from pinn_engine import Model, Quantity, Signature, Solver, Value


class Function(Model):
    signature = None

    def __init__(self, inputs, outputs, function):
        super().__init__()
        self.signature = Signature(inputs, outputs)
        self.function = function

    def evaluate(self, values):
        return self.function(values)


@pytest.mark.parametrize("lengths", [(20, 40), (3,), (4, 4)])
def test_case_loop_preserves_lists_and_gradients(lengths):
    x, y = Quantity("x", (3,)), Quantity("y", (3,))
    model = Function({x: ()}, {y: ()}, lambda v: {y: 2 * v[x]})
    tensors = [torch.randn(n, 3, requires_grad=True) for n in lengths]
    value = Value(tensors, ("case", "point"))
    assert value.tensor is tensors
    result = model({x: value})[y]
    assert isinstance(result.tensor, list)
    assert result.axes == ("case", "point")
    for actual, tensor in zip(result.tensor, tensors):
        torch.testing.assert_close(actual, 2 * tensor)
    sum(t.sum() for t in result.tensor).backward()
    for tensor in tensors:
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, 2))


def test_point_reduction_and_shared_inputs_with_dense_case_axis():
    x, factor, offset, y = map(Quantity, ("x", "factor", "offset", "y"))
    model = Function({x: "point", factor: (), offset: ()}, {y: ()},
                     lambda v: {y: v[x].mean() * v[factor] + v[offset]})
    points = [torch.arange(6.).reshape(2, 3), torch.arange(10.).reshape(2, 5)]
    factors = torch.tensor([[1., 2.], [3., 4.]])
    result = model({x: Value(points, ("case", "scenario", "point")),
                    factor: Value(factors, ("scenario", "case")),
                    offset: Value(torch.tensor(7.))})[y]
    assert result.axes == ("case", "scenario")
    for i, tensor in enumerate(result.tensor):
        torch.testing.assert_close(tensor, points[i].mean(1) * factors[:, i] + 7)


def test_retained_case_list_vmaps_common_scenarios_and_shares_constant_mesh():
    x, cells, weight, score, centered = (Quantity("x"), Quantity("cells", (2,)),
                                        Quantity("weight"), Quantity("score"), Quantity("centered"))
    calls = []

    def evaluate(values):
        # Case and point axes are core. Scenario has been mapped away.
        clouds = values[x]
        calls.append([tuple(t.shape) for t in clouds])
        assert [tuple(t.shape) for t in values[cells]] == [(2, 2), (4, 2)]
        means = torch.stack([t.mean() for t in clouds])
        return {score: (means * values[weight]).sum(),
                centered: [t - t.mean() for t in clouds]}

    model = Function({x: ("case", "point"), cells: ("case", "cell"),
                      weight: "case"},
                     {score: (), centered: ("case", "point")}, evaluate)
    clouds = [torch.randn(3, 2, requires_grad=True), torch.randn(5, 2, requires_grad=True)]
    weights = torch.tensor([[1., 2.], [3., 4.]])
    result = model({x: Value(clouds, ("case", "point", "scenario")),
                    cells: Value([torch.zeros(2, 2), torch.zeros(4, 2)], ("case", "cell")),
                    weight: Value(weights, ("scenario", "case"))})
    assert calls == [[(3,), (5,)]]  # vmap transforms one call, not a scenario loop.
    expected = sum(t.mean(0) * weights[:, i] for i, t in enumerate(clouds))
    torch.testing.assert_close(result[score].tensor, expected)
    assert result[score].axes == ("scenario",)
    assert result[centered].axes == ("case", "scenario", "point")
    for original, actual in zip(clouds, result[centered].tensor):
        torch.testing.assert_close(actual, (original - original.mean(0)).T)
    result[score].tensor.sum().backward()
    assert all(t.grad is not None for t in clouds)


def test_retained_case_and_scenarios_are_delivered_directly():
    x, result = Quantity("x"), Quantity("result")
    model = Function({x: ("case", "scenario", "point")}, {result: ()},
                     lambda v: {result: sum(t.sum() for t in v[x])})
    clouds = [torch.ones(2, 3), torch.ones(4, 5)]
    actual = model({x: Value(clouds, ("case", "scenario", "point"))})
    torch.testing.assert_close(actual[result].tensor, torch.tensor(26.))


def test_ragged_mapped_axis_is_rejected_before_user_code_runs():
    x, y = Quantity("x"), Quantity("y")
    calls = []
    model = Function({x: "case"}, {y: ()}, lambda v: calls.append(v))
    with pytest.raises(ValueError, match="Cannot map ragged axis 'point'"):
        model({x: Value([torch.zeros(20), torch.zeros(40)], ("case", "point"))})
    assert not calls


def test_aligned_points_can_be_mapped_while_case_is_retained():
    x, y = Quantity("x"), Quantity("y")
    model = Function({x: "case"}, {y: ()}, lambda v: {y: sum(v[x])})
    actual = model({x: Value([torch.arange(4.), torch.ones(4)], ("case", "point"))})[y]
    assert actual.axes == ("point",)
    torch.testing.assert_close(actual.tensor, torch.arange(4.) + 1)


def test_solver_shares_encoder_results_and_preserves_hooks_parameters_and_gradients():
    x, embedding, loss = Quantity("x"), Quantity("embedding"), Quantity("loss")

    class Encoder(Model):
        signature = Signature({x: "point"}, {embedding: ()})

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.))

        def evaluate(self, values):
            return {embedding: self.weight * values[x].mean()}

    encoder = Encoder()
    hooks = []
    encoder.register_forward_hook(lambda *args: hooks.append("encoder"))
    reducer = Function({embedding: "case"}, {loss: ()},
                       lambda v: {loss: torch.stack(v[embedding]).square().mean()})
    solver = Solver([reducer, encoder], outputs=(embedding, loss))
    points = [torch.arange(3., requires_grad=True), torch.arange(5., requires_grad=True)]
    actual = solver({x: Value(points, ("case", "point"))})
    torch.testing.assert_close(actual[loss].tensor, torch.tensor(10.))
    actual[loss].tensor.backward()
    assert hooks == ["encoder"]
    assert list(solver.parameters()) == [encoder.weight]
    assert all(t.grad is not None for t in points)
    torch.testing.assert_close(encoder.weight.grad, torch.tensor(10.))


@pytest.mark.parametrize("bad", [[], [[torch.zeros(2)]], [1]])
def test_invalid_list_payloads_fail(bad):
    with pytest.raises((ValueError, TypeError)):
        Value(bad, ("case", "point"))


def test_list_requires_outer_axis_and_matching_intrinsic_shapes():
    with pytest.raises(ValueError, match="outer sample axis"):
        Value([torch.zeros(2)])
    x, y = Quantity("x", (3,)), Quantity("y")
    model = Function({x: "point"}, {y: ()}, lambda v: {y: v[x].sum()})
    with pytest.raises(ValueError, match="Intrinsic shape"):
        model({x: Value([torch.zeros(2, 3), torch.zeros(4, 2)], ("case", "point"))})


@pytest.mark.parametrize("axis, other", [
    ("case", [torch.zeros(2)]),
    ("point", [torch.zeros(2), torch.zeros(7)]),
])
def test_corresponding_input_lists_must_match_per_case(axis, other):
    x, y, z = map(Quantity, ("x", "y", "z"))
    model = Function({x: "point", y: "point"}, {z: ()}, lambda v: {z: v[x].sum()})
    with pytest.raises(ValueError, match=f"Inconsistent size for axis '{axis}'"):
        model({x: Value([torch.zeros(2), torch.zeros(4)], ("case", "point")),
               y: Value(other, ("case", "point"))})


def test_output_cannot_change_corresponding_point_lengths_silently():
    x, y = Quantity("x"), Quantity("y")
    model = Function({x: ("case", "point")}, {y: ("case", "point")},
                     lambda v: {y: [t[:-1] for t in v[x]]})
    with pytest.raises(ValueError, match="Inconsistent size for axis 'point'"):
        model({x: Value([torch.zeros(2), torch.zeros(4)], ("case", "point"))})


def test_list_axis_must_be_first_in_core_signature():
    x, y = Quantity("x"), Quantity("y")
    model = Function({x: ("point", "case")}, {y: ()}, lambda v: {y: torch.tensor(0.)})
    with pytest.raises(ValueError, match="first in the core signature"):
        model({x: Value([torch.zeros(2), torch.zeros(4)], ("case", "point"))})


def test_models_can_produce_a_list_with_a_new_sample_axis():
    x = Quantity("x")
    model = Function({}, {x: ("case", "point")},
                     lambda v: {x: [torch.zeros(2), torch.zeros(4)]})
    result = model({})[x]
    assert result.axes == ("case", "point")
    assert [tuple(t.shape) for t in result.tensor] == [(2,), (4,)]


def test_dense_fullgraph_compile_still_works():
    x, y = Quantity("x"), Quantity("y")
    model = Function({x: ()}, {y: ()}, lambda v: {y: v[x].square()})
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    points = torch.arange(6.).reshape(2, 3).requires_grad_()
    result = compiled({x: Value(points, ("scenario", "point"))})[y]
    torch.testing.assert_close(result.tensor, points.square())
    result.tensor.sum().backward()
    torch.testing.assert_close(points.grad, 2 * points)


@pytest.mark.parametrize("retained", [False, True])
def test_list_fullgraph_capture_and_backward(retained):
    x, y = Quantity("compiled_x"), Quantity("compiled_y")
    if retained:
        model = Function({x: ("case", "point")}, {y: ("case", "point")},
                         lambda v: {y: [t.square() for t in v[x]]})
    else:
        model = Function({x: ()}, {y: ()}, lambda v: {y: v[x].square()})
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    points = [torch.randn(2, 3, requires_grad=True), torch.randn(2, 5, requires_grad=True)]
    result = compiled({x: Value(points, ("case", "scenario", "point"))})[y]
    assert result.axes == ("case", "scenario", "point")
    for actual, original in zip(result.tensor, points):
        torch.testing.assert_close(actual, original.square())
    sum(t.sum() for t in result.tensor).backward()
    for tensor in points:
        torch.testing.assert_close(tensor.grad, 2 * tensor)


def test_multiple_mapped_axes_inside_retained_list():
    x, y = Quantity("multi_x"), Quantity("multi_y")
    model = Function({x: ("case", "point")}, {y: ()},
                     lambda v: {y: sum(t.mean() for t in v[x])})
    points = [torch.randn(2, 3, 4), torch.randn(2, 3, 7)]
    actual = model({x: Value(points, ("case", "scenario", "time", "point"))})[y]
    assert actual.axes == ("scenario", "time")
    torch.testing.assert_close(actual.tensor, sum(t.mean(-1) for t in points))


def test_generator_and_encoder_share_one_solver_graph():
    sizes, x, embedding = Quantity("sizes"), Quantity("generated_x"), Quantity("generated_embedding")
    generator = Function({sizes: "case"}, {x: ("case", "point")},
                         lambda v: {x: [torch.arange(int(n), dtype=torch.float32) for n in v[sizes]]})
    encoder = Function({x: "point"}, {embedding: ()}, lambda v: {embedding: v[x].mean()})
    solver = Solver([encoder, generator], outputs=embedding)
    actual = solver({sizes: Value(torch.tensor([20, 40]), "case")})[embedding]
    assert actual.axes == ("case",)
    torch.testing.assert_close(torch.stack(actual.tensor), torch.tensor([9.5, 19.5]))


@pytest.mark.parametrize("axis, as_list", [("case", False), ("cases", True)])
def test_representation_not_axis_name_selects_execution(axis, as_list):
    x, y = Quantity("named_x"), Quantity("named_y")
    calls = []

    def evaluate(values):
        calls.append(tuple(values[x].shape))
        return {y: values[x].square()}

    model = Function({x: "point"}, {y: "point"}, evaluate)
    tensor = ([torch.arange(3.), torch.arange(5.)] if as_list
              else torch.arange(10.).reshape(2, 5))
    result = model({x: Value(tensor, (axis, "point"))})[y]
    assert result.axes == (axis, "point")
    assert isinstance(result.tensor, list) == as_list
    if as_list:
        assert calls == [(3,), (5,)]
        for actual, original in zip(result.tensor, tensor):
            torch.testing.assert_close(actual, original.square())
    else:
        assert calls == [(5,)]  # A dense axis named case still uses vmap.
        torch.testing.assert_close(result.tensor, tensor.square())
