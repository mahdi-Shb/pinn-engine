"""Independence metadata affects derivative paths, not normal evaluation."""
import pytest
import torch
from pinn_engine import Derivative, Model, Quantity, Signature, Solver, Value


class Function(Model):
    def __init__(self, inputs, outputs, function, independence=None):
        super().__init__()
        self._signature = Signature(inputs, outputs, independence)
        self.function = function

    @property
    def signature(self):
        return self._signature

    def evaluate(self, values):
        return self.function(values)


def test_defaults_overrides_and_input_dictionary_copy():
    x, y = Quantity("x"), Quantity("y")
    assert Signature({x: "point"}, {y: "point"}).independence == {"point": True}
    assert Signature({x: ("...", "point")}, {y: ("...", "point")}).independence == {"...": True, "point": True}
    rules = {"point": False}
    signature = Signature({x: "point"}, {y: "point"}, rules)
    rules["point"] = True
    assert signature.independence == {"point": False}


@pytest.mark.parametrize("rules,error", [
    ({"point": 1}, TypeError), ({1: True}, TypeError),
    ({"other": False}, ValueError), ({"...": False}, ValueError), ([], TypeError),
])
def test_invalid_rules_are_rejected(rules, error):
    x, y = Quantity("x"), Quantity("y")
    with pytest.raises(error):
        Signature({x: "point"}, {y: "point"}, rules)


def test_normal_forward_and_parameter_training_are_allowed_but_ad_rejects(monkeypatch):
    x, y = Quantity("x"), Quantity("y")
    weight = torch.tensor(2., requires_grad=True)
    model = Function({x: "point"}, {y: "point"},
                     lambda v: {y: weight * (v[x] - v[x].mean())}, {"point": False})
    request = Derivative(y, x)
    solver = Solver([model], outputs=(y, request))
    inputs = {x: Value(torch.tensor([1., 2., 4.]), "point")}
    forward = solver(inputs, outputs=y)[y].tensor
    torch.testing.assert_close(forward, weight * (inputs[x].tensor - inputs[x].tensor.mean()))
    forward.square().sum().backward()
    assert weight.grad is not None
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid AD must fail before numerical backward")
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    with pytest.raises(NotImplementedError, match="point"):
        solver(inputs, outputs=request)


def test_one_dependent_branch_blocks_join_and_later_true_does_not_restore_it():
    x, a, b, z = map(Quantity, ("x", "a", "b", "z"))
    mixed = Function({x: "point"}, {a: "point"}, lambda v: {a: v[x] - v[x].mean()}, {"point": False})
    independent = Function({x: ()}, {b: ()}, lambda v: {b: v[x].square()})
    join = Function({a: (), b: ()}, {z: ()}, lambda v: {z: v[a] + v[b]})
    dz, db = Derivative(z, x), Derivative(b, x)
    solver = Solver([mixed, independent, join], outputs=(z, dz, db))
    inputs = {x: Value(torch.tensor([1., 2., 3.]), "point")}
    assert solver(inputs, outputs=z)[z].tensor.shape == (3,)
    torch.testing.assert_close(solver(inputs, outputs=db)[db].tensor, 2 * inputs[x].tensor)
    with pytest.raises(NotImplementedError, match="point"):
        solver(inputs, outputs=dz)


def test_unrelated_input_branch_is_not_blocked():
    x, other, a, b, z = map(Quantity, ("x", "other", "a", "b", "z"))
    mixed = Function({other: "point"}, {a: "point"}, lambda v: {a: v[other] - v[other].mean()}, {"point": False})
    independent = Function({x: ()}, {b: ()}, lambda v: {b: v[x].square()})
    join = Function({a: (), b: ()}, {z: ()}, lambda v: {z: v[a] + v[b]})
    request = Derivative(z, x)
    inputs = {x: Value(torch.tensor([1., 2.]), "point"), other: Value(torch.ones(2), "point")}
    result = Solver([mixed, independent, join], outputs=request)(inputs)[request]
    torch.testing.assert_close(result.tensor, 2 * inputs[x].tensor)


def test_false_rule_also_blocks_if_later_output_is_scalar():
    x, y, loss = map(Quantity, ("x", "y", "loss"))
    mixed = Function({x: "point"}, {y: "point"}, lambda v: {y: v[x] - v[x].mean()}, {"point": False})
    reduce = Function({y: "point"}, {loss: ()}, lambda v: {loss: v[y].square().sum()})
    with pytest.raises(NotImplementedError, match="point"):
        Solver([mixed, reduce], outputs=Derivative(loss, x))({x: Value(torch.arange(3.), "point")})


@pytest.mark.parametrize("axes", [("scenario", "point"), ("family", "time", "point")])
def test_wildcard_false_expands_runtime_axis_names(axes):
    x, y = Quantity("x"), Quantity("y")
    model = Function({x: "..."}, {y: "..."}, lambda v: {y: Value(v[x].tensor.square(), v[x].axes)}, {"...": False})
    inputs = {x: Value(torch.ones((2,) * len(axes)), axes)}
    with pytest.raises(NotImplementedError) as error:
        Solver([model], outputs=Derivative(y, x))(inputs)
    assert all(axis in str(error.value) for axis in axes)
    assert model.signature.independence == {"...": False}


def test_named_wildcard_override_and_target_specific_paths():
    x, scale, y = map(Quantity, ("x", "scale", "y"))
    model = Function({x: "...", scale: "..."}, {y: "..."}, lambda v: {
        y: Value(v[scale].tensor[:, None] * (v[x].tensor - v[x].tensor.mean(-1, keepdim=True)), v[x].axes)
    }, {"...": False, "scenario": True})
    dx, ds = Derivative(y, x), Derivative(y, scale)
    solver = Solver([model], outputs=(dx, ds))
    inputs = {x: Value(torch.arange(6.).reshape(2, 3), ("scenario", "point")),
              scale: Value(torch.tensor([2., 3.]), "scenario")}
    actual = solver(inputs, outputs=ds)[ds]
    torch.testing.assert_close(actual.tensor, inputs[x].tensor - inputs[x].tensor.mean(-1, keepdim=True))
    with pytest.raises(NotImplementedError, match="point"):
        solver(inputs, outputs=dx)


def test_wildcard_false_does_not_override_explicit_point_default():
    x, y = Quantity("x"), Quantity("y")
    model = Function({x: ("...", "point")}, {y: ("...", "point")}, lambda v: {y: v[x]}, {"...": False})
    assert model.signature.independence == {"...": False, "point": True}
    # The wildcard group is empty; point still defaults to independent.
    request = Derivative(y, x)
    result = Solver([model], outputs=request)({x: Value(torch.ones(3), "point")})[request]
    torch.testing.assert_close(result.tensor, torch.ones(3))


def test_invalid_wildcard_named_rule_fails_at_runtime():
    x, y = Quantity("x"), Quantity("y")
    model = Function({x: "..."}, {y: "..."}, lambda v: {y: v[x]}, {"missing": False})
    with pytest.raises(ValueError, match="not shared at runtime"):
        model({x: Value(torch.ones(3), "point")})
