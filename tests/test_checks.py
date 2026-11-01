"""Data-dependent checks inside ordinary and automatically mapped models."""

import pytest
import torch
from torch import nn

from pinncore import Derivative, Model, Quantity, Signature, Solver, Value, check


def test_check_accepts_true_conditions_and_reports_the_callers_message():
    assert check(torch.tensor(True), "scalar") is None
    assert check(torch.tensor([[True, True], [True, True]]), "array") is None
    with pytest.raises(ValueError, match="Geometry has no valid points"):
        check(torch.tensor([True, False]), "Geometry has no valid points")


@pytest.mark.parametrize("condition", [True, [True], torch.tensor(1), torch.tensor(1.)])
def test_check_requires_a_boolean_tensor(condition):
    with pytest.raises(TypeError, match="boolean Tensor"):
        check(condition, "invalid condition")


def test_check_requires_a_string_message():
    with pytest.raises(TypeError, match="message"):
        check(torch.tensor(True), 7)


@pytest.mark.parametrize("nested", [False, True])
def test_check_inside_vmap_rejects_any_invalid_member_without_changing_outputs(nested):
    def evaluate(x):
        check((x >= 0).all(), "negative sample")
        return x.square()

    # A non-leading mapped axis must work as well as nested automatic batches.
    mapped = torch.vmap(evaluate, in_dims=1, out_dims=1)
    if nested:
        mapped = torch.vmap(mapped)
    values = torch.arange(1., 13.).reshape((2, 2, 3) if nested else (4, 3))
    torch.testing.assert_close(mapped(values), values.square())
    invalid = values.clone()
    invalid.flatten()[-1] = -1
    with pytest.raises(ValueError, match="negative sample"):
        mapped(invalid)


def test_check_inside_point_reduction_model_keeps_its_core_signature():
    points, mask, center = Quantity("points", (2,)), Quantity("valid"), Quantity("center", (2,))

    class Center(Model):
        signature = Signature({points: "point", mask: "point"}, {center: ()})

        def evaluate(self, values):
            check(values[mask].any(), "Geometry has no valid points")
            weights = values[mask].unsqueeze(-1)
            return {center: (values[points] * weights).sum(0) / weights.sum()}

    xyz = torch.arange(1., 25.).reshape(2, 2, 3, 2).requires_grad_()
    valid = torch.tensor([[[True, True, False], [True, False, False]],
                          [[True, True, True], [False, True, True]]])
    axes = ("family", "scenario", "point")
    solver = Solver([Center()], outputs=center)
    actual = solver({points: Value(xyz, axes), mask: Value(valid, axes)})[center]
    expected = (xyz * valid.unsqueeze(-1)).sum(-2) / valid.sum(-1, keepdim=True)
    assert actual.axes == ("family", "scenario")
    torch.testing.assert_close(actual.tensor, expected)
    actual.tensor.sum().backward()
    gradient = valid.unsqueeze(-1).expand_as(xyz) / valid.sum(-1)[..., None, None]
    torch.testing.assert_close(xyz.grad, gradient)
    invalid = valid.clone()
    invalid[-1, -1] = False
    with pytest.raises(ValueError, match="Geometry has no valid points"):
        solver({points: Value(xyz, axes), mask: Value(invalid, axes)})


def test_check_preserves_forward_reverse_and_second_order_functional_ad():
    def cubic(x):
        check((x >= 0).all(), "negative input")
        return x ** 3

    values = torch.tensor([.5, 2.], dtype=torch.double)
    jacobian = torch.diag(3 * values.square())
    torch.testing.assert_close(torch.func.jacrev(cubic)(values), jacobian)
    torch.testing.assert_close(torch.func.jacfwd(cubic)(values), jacobian)
    torch.testing.assert_close(torch.func.hessian(lambda x: cubic(x).sum())(values), torch.diag(6 * values))


def test_checked_model_supports_solver_second_derivatives_and_parameter_training():
    x, y = Quantity("x"), Quantity("y")

    class Cubic(Model):
        signature = Signature({x: ()}, {y: ()})

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2., dtype=torch.double))

        def evaluate(self, values):
            check(values[x] >= 0, "negative input")
            return {y: self.weight * values[x] ** 3}

    model = Cubic()
    second = Derivative(Derivative(y, x), x)
    values = torch.tensor([[.5, 1.], [1.5, 2.]], dtype=torch.double, requires_grad=True)
    result = Solver([model], outputs=(y, second))({x: Value(values, ("scenario", "point"))})
    torch.testing.assert_close(result[y].tensor, 2 * values ** 3)
    torch.testing.assert_close(result[second].tensor, 12 * values)
    result[second].tensor.sum().backward()
    torch.testing.assert_close(model.weight.grad, 6 * values.detach().sum())
    torch.testing.assert_close(values.grad, torch.full_like(values, 12.))
