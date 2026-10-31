"""Wildcard Values and explicit promises of independence on preserved axes."""
import pytest
import torch
from pinn_engine import Derivative, Model, Quantity, Signature, Solver, Value


class Function(Model):
    def __init__(self, inputs, outputs, function):
        super().__init__()
        self._signature = Signature(inputs, outputs)
        self.function = function

    @property
    def signature(self):
        return self._signature

    def evaluate(self, values):
        return self.function(values)


@pytest.mark.parametrize("tensor,axes", [
    (torch.tensor(2.), ()),
    (torch.ones(3, 2), ("point", "scenario")),
    ([torch.ones(20), torch.ones(40)], ("case", "point")),
])
def test_wildcard_receives_original_value_objects_once(monkeypatch, tensor, axes):
    x, y = Quantity("x"), Quantity("y")
    inputs = {x: Value(tensor, axes)}
    calls = []
    def evaluate(values):
        assert values is inputs and values[x].tensor is tensor
        calls.append("called")
        return {y: values[x]}
    def forbidden(*args, **kwargs):
        raise AssertionError("Wildcard execution must not vmap")
    monkeypatch.setattr(torch.func, "vmap", forbidden)
    result = Function({x: "..."}, {y: "..."}, evaluate)(inputs)
    assert result[y] is inputs[x]
    assert calls == ["called"]


def test_ellipsis_point_reduction_preserves_scenarios_and_training():
    x, y, loss = Quantity("x"), Quantity("y"), Quantity("loss")
    model = Function({x: ("...", "point")}, {y: "..."}, lambda v: {
        y: Value(v[x].tensor.square().mean(-1), v[x].axes[:-1])
    })
    reduce = Function({y: "scenario"}, {loss: ()}, lambda v: {loss: v[y].sum()})
    points = torch.arange(6.).reshape(2, 3).requires_grad_()
    solver = Solver([reduce, model], outputs=(y, loss))
    result = solver({x: Value(points, ("scenario", "point"))})
    assert result[y].axes == ("scenario",)
    result[loss].tensor.backward()
    torch.testing.assert_close(points.grad, 2 * points / 3)


def test_omitting_ellipsis_reduces_all_captured_axes():
    x, y = Quantity("x"), Quantity("y")
    model = Function({x: "..."}, {y: ()}, lambda v: {
        y: Value(sum(t.sum() for t in v[x].tensor))
    })
    result = model({x: Value([torch.ones(2), torch.ones(4)], ("case", "point"))})[y]
    assert result.axes == ()
    torch.testing.assert_close(result.tensor, torch.tensor(6.))


def test_wildcard_list_point_reduction_and_embedding():
    x, y = Quantity("x", (3,)), Quantity("embedding", (3,))
    model = Function({x: ("...", "point")}, {y: "..."}, lambda v: {
        y: Value([t.mean(0) for t in v[x].tensor], ("case",))
    })
    result = model({x: Value([torch.ones(20, 3), torch.ones(40, 3)], ("case", "point"))})[y]
    assert result.axes == ("case",)
    assert [tuple(t.shape) for t in result.tensor] == [(3,), (3,)]


def test_wildcards_accept_input_specific_layouts_and_shared_scalar():
    x, z, bias, y = map(Quantity, ("x", "z", "bias", "y"))
    def evaluate(v):
        assert v[z].axes == ("point", "scenario")
        return {y: Value(v[x].tensor + v[z].tensor.T + v[bias].tensor, v[x].axes)}
    model = Function({x: "...", z: "...", bias: ()}, {y: "..."}, evaluate)
    result = model({x: Value(torch.ones(2, 3), ("scenario", "point")),
                    z: Value(torch.ones(3, 2), ("point", "scenario")), bias: Value(torch.tensor(1.))})[y]
    torch.testing.assert_close(result.tensor, torch.full((2, 3), 3.))


@pytest.mark.parametrize("wildcard", [False, True])
def test_preserved_core_axis_allows_aligned_first_and_second_derivatives(wildcard):
    x, y = Quantity("x"), Quantity("y")
    if wildcard:
        model = Function({x: "..."}, {y: "..."}, lambda v: {
            y: Value(v[x].tensor.pow(3), v[x].axes)
        })
    else:
        model = Function({x: "point"}, {y: "point"}, lambda v: {y: v[x].pow(3)})
    first = Derivative(y, x)
    second = Derivative(first, x)
    points = torch.tensor([2., 3.], requires_grad=True)
    result = Solver([model], outputs=(first, second))({x: Value(points, "point")})
    torch.testing.assert_close(result[first].tensor, 3 * points.square())
    torch.testing.assert_close(result[second].tensor, 6 * points)
    result[second].tensor.sum().backward()
    torch.testing.assert_close(points.grad, torch.full_like(points, 6.))


def test_renamed_axis_requests_full_coupled_jacobian():
    x, y = Quantity("x"), Quantity("y")
    model = Function({x: ("...", "point")}, {y: ("...", "query")}, lambda v: {
        y: Value(v[x].tensor - v[x].tensor.mean(-1, keepdim=True), ("scenario", "query"))
    })
    request = Derivative(y, x)
    result = Solver([model], outputs=request)({x: Value(torch.ones(2, 3), ("scenario", "point"))})[request]
    torch.testing.assert_close(result.tensor, (torch.eye(3) - torch.ones(3, 3)/3).expand(2, 3, 3))


def test_reduction_and_rebroadcasting_do_not_erase_coupling():
    x, average, y = map(Quantity, ("x", "average", "y"))
    reducer = Function({x: "..."}, {average: ()}, lambda v: {average: Value(v[x].tensor.mean())})
    broadcast = Function({x: (), average: ()}, {y: ()}, lambda v: {y: v[x] - v[average]})
    with pytest.raises(NotImplementedError, match="non-independent"):
        Solver([reducer, broadcast], outputs=Derivative(y, x))({x: Value(torch.arange(3.), "point")})


def test_value_and_wildcard_output_contracts_are_validated():
    x, y = Quantity("x"), Quantity("y")
    with pytest.raises(ValueError, match="requires an input"):
        Signature({x: ()}, {y: "..."})
    with pytest.raises(ValueError, match="repeat"):
        Signature({x: ("...", "...")}, {y: ()})
    with pytest.raises(ValueError, match="reserved"):
        Value(torch.ones(2), "...")
    wrong = Function({x: "..."}, {y: ()}, lambda v: {y: v[x]})
    with pytest.raises(ValueError, match="Output axes"):
        wrong({x: Value(torch.ones(2), "point")})
    raw = Function({x: "..."}, {y: "..."}, lambda v: {y: v[x].tensor})
    with pytest.raises(TypeError, match="Value object"):
        raw({x: Value(torch.ones(2), "point")})


def test_dense_wildcard_fullgraph_capture():
    x, y = Quantity("compile_x"), Quantity("compile_y")
    model = Function({x: "..."}, {y: "..."}, lambda v: {y: Value(v[x].tensor.square(), v[x].axes)})
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    points = torch.arange(6.).reshape(2, 3).requires_grad_()
    output = compiled({x: Value(points, ("point", "scenario"))})[y]
    output.tensor.sum().backward()
    torch.testing.assert_close(points.grad, 2 * points)
