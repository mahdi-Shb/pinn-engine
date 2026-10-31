"""Declarations, nested derivatives, AD restrictions, and numerical execution."""

import pytest
import torch

from contracts import (
    Derivative, Model, PartialDerivative, Quantity, Signature, Solver,
    TotalDerivative, Value,
)


def test_derivative_identity_is_symbolic_and_holding_is_unordered():
    a, b, c, d = Quantity("a", (3,)), Quantity("b", (2,)), Quantity("c"), Quantity("d")
    total = TotalDerivative(a, b)
    assert type(total) is Derivative
    assert total.shape == (3, 2)
    assert total == Derivative(Quantity("a", (3,)), Quantity("b", (2,)))
    assert {total: 7}[Derivative(a, b)] == 7
    assert total != Quantity(total.name, total.shape)
    assert total != PartialDerivative(a, b, holding=())
    assert PartialDerivative(a, b, holding=(c, d)) == Derivative(a, b, holding=(d, c))
    assert PartialDerivative(a, b, holding=(c,)) != PartialDerivative(a, b, holding=(d,))
    with pytest.raises(ValueError, match="fixed"):
        PartialDerivative(a, b, holding=(b,))
    with pytest.raises(TypeError, match="Quantity"):
        Derivative(a, 0)
    assert Derivative(total, b).shape == (3, 2, 2)


def test_derivative_detects_operand_shape_conflicts():
    temperature = Quantity("T")
    x = Quantity("x", (2,))
    gradient = Derivative(temperature, x)
    assert gradient.shape == (2,)
    with pytest.raises(ValueError, match="Conflicting shapes"):
        Signature({Quantity("x", (3,)): (), gradient: ()}, {Quantity("r"): ()})


@pytest.mark.parametrize("output_axis", ["point_normalized"])
def test_coupled_model_preserves_outer_scenarios_and_training_gradients(output_axis):
    x, centered = Quantity("x"), Quantity("centered")

    class Center(Model):
        signature = Signature({x: "point"}, {centered: output_axis})

        def evaluate(self, values):
            points = values[x]
            return {centered: points - points.mean()}

    # Deliberately permute runtime axes; evaluate still sees a point vector.
    tensor = torch.arange(12.).reshape(3, 4).requires_grad_()
    result = Center()({x: Value(tensor, ("point", "scenario"))})[centered]
    assert result.axes == ("scenario", output_axis)
    torch.testing.assert_close(result.tensor, (tensor - tensor.mean(dim=0)).T)
    result.tensor.square().sum().backward()
    assert tensor.grad is not None


def test_reusing_a_core_axis_still_requires_its_original_size():
    x, y = Quantity("x"), Quantity("y")

    class DropLast(Model):
        signature = Signature({x: "point"}, {y: "point"})

        def evaluate(self, values):
            return {y: values[x][:-1]}

    with pytest.raises(ValueError, match="Inconsistent size for axis 'point'"):
        DropLast()({x: Value(torch.arange(3.), "point")})


def test_reduction_is_legal_but_rebroadcasting_it_does_not_restore_independence():
    x, mean, temperature = map(Quantity, ("x", "mean", "T"))
    total = TotalDerivative(temperature, x)
    reduced = TotalDerivative(mean, x)

    class Mean(Model):
        signature = Signature({x: "point"}, {mean: ()})

        def evaluate(self, values):
            return {mean: values[x].mean()}

    class Predict(Model):
        signature = Signature({x: (), mean: ()}, {temperature: ()})

        def evaluate(self, values):
            return {temperature: values[x] - values[mean]}

    solver = Solver([Predict(), Mean()], outputs=(total, reduced))
    values = {x: Value(torch.tensor([1., 2., 4.]), "point")}
    with pytest.raises(NotImplementedError, match="non-independent sample axes"):
        solver(values, outputs=total)
    torch.testing.assert_close(solver(values, outputs=reduced)[reduced].tensor, torch.full((3,), 1/3))


def test_point_coupling_does_not_block_ad_of_an_independent_scenario_parameter():
    x, parameter, temperature = map(Quantity, ("x", "parameter", "T"))
    spatial = TotalDerivative(temperature, x)
    scenario = TotalDerivative(temperature, parameter)

    class MixPoints(Model):
        signature = Signature({x: "point", parameter: ()}, {temperature: "query"})

        def evaluate(self, values):
            return {temperature: values[parameter] * (values[x] - values[x].mean())}

    solver = Solver([MixPoints()], outputs=(spatial, scenario))
    values = {
        x: Value(torch.arange(6.).reshape(2, 3), ("scenario", "point")),
        parameter: Value(torch.tensor([2., 3.]), "scenario"),
    }
    # A renamed output axis requests the full point-to-query Jacobian.
    spatial_result = solver(values, outputs=spatial)[spatial]
    expected = values[parameter].tensor[:, None, None] * (torch.eye(3) - torch.ones(3, 3) / 3)
    torch.testing.assert_close(spatial_result.tensor, expected)
    result = solver(values, outputs=scenario)[scenario]
    torch.testing.assert_close(result.tensor, values[x].tensor - values[x].tensor.mean(dim=1, keepdim=True))


def test_models_cannot_provide_symbolic_derivatives_and_cycles_are_rejected():
    x, temperature = Quantity("x"), Quantity("T")
    request = Derivative(temperature, x)

    class Inject(Model):
        signature = Signature({x: ()}, {request: ()})

        def evaluate(self, values):
            raise AssertionError("Must never execute")

    with pytest.raises(ValueError, match="produced by Solver"):
        Solver([Inject()])

    class Cyclic(Model):
        signature = Signature({request: ()}, {temperature: ()})

        def evaluate(self, values):
            raise AssertionError("Must never execute")

    with pytest.raises(ValueError, match="Cyclic"):
        Solver([Cyclic()])
