"""Declarations, derivative planning, AD restrictions, and numerical execution."""

import pytest
import torch
from torch import nn

from contracts import (
    Derivative, Field, Model, PartialDerivative, Quantity, Signature, Solver,
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
    with pytest.raises(NotImplementedError, match="first-order"):
        Derivative(total, b)


def test_derivative_retains_domain_and_detects_operand_shape_conflicts():
    domain = object()
    temperature = Field("T", geometry=domain)
    x = Quantity("x", (2,))
    gradient = Derivative(temperature, x)
    assert gradient._domain is domain
    assert gradient.shape == (2,)
    with pytest.raises(ValueError, match="Conflicting shapes"):
        Signature({Quantity("x", (3,)): (), gradient: ()}, {Quantity("r"): ()})


@pytest.mark.parametrize("output_axis", ["point", "point_normalized"])
def test_coupled_model_preserves_outer_scenarios_and_training_gradients(output_axis):
    x, centered = Quantity("x"), Quantity("centered")

    class Center(Model):
        geometry = None
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
        geometry = None
        signature = Signature({x: "point"}, {y: "point"})

        def evaluate(self, values):
            return {y: values[x][:-1]}

    with pytest.raises(ValueError, match="Inconsistent size for axis 'point'"):
        DropLast()({x: Value(torch.arange(3.), "point")})


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("independent_source", [False, True])
def test_ad_checks_only_wrt_dependent_paths_and_respects_holding(held, independent_source):
    x, z, temperature = map(Quantity, ("x", "z", "T"))
    source = Quantity("geometry_points") if independent_source else x
    request = (PartialDerivative(temperature, x, holding=(z,)) if held
               else TotalDerivative(temperature, x))

    class Center(Model):
        geometry = None
        signature = Signature({source: "point"}, {z: "point"})

        def evaluate(self, values):
            return {z: values[source] - values[source].mean()}

    class Predict(Model):
        geometry = None
        signature = Signature({x: (), z: ()}, {temperature: ()})

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.))

        def evaluate(self, values):
            return {temperature: self.weight * (values[x] + values[z])}

    predictor = Predict()
    # Graph construction and inference are legal even with a pending AD request.
    solver = Solver([predictor, Center()], outputs=request)
    tensor = torch.arange(1., 4., requires_grad=True)
    values = {x: Value(tensor, "point")}
    if independent_source:
        values[source] = Value(torch.tensor([2., 4., 8.]), "point")
    result = solver(values, outputs=temperature)[temperature]
    result.tensor.square().sum().backward()
    assert tensor.grad is not None
    assert predictor.weight.grad is not None

    # A downstream pointwise model cannot undo upstream coordinate coupling.
    # Holding z or sourcing z independently of x removes that AD path.
    if not held and not independent_source:
        with pytest.raises(NotImplementedError, match="non-independent sample axes"):
            solver(values)
    else:
        torch.testing.assert_close(solver(values)[request].tensor, torch.full_like(tensor, 2.))


def test_reduction_is_legal_but_rebroadcasting_it_does_not_restore_independence():
    x, mean, temperature = map(Quantity, ("x", "mean", "T"))
    total = TotalDerivative(temperature, x)
    partial = PartialDerivative(temperature, x, holding=(mean,))
    reduced = TotalDerivative(mean, x)

    class Mean(Model):
        geometry = None
        signature = Signature({x: "point"}, {mean: ()})

        def evaluate(self, values):
            return {mean: values[x].mean()}

    class Predict(Model):
        geometry = None
        signature = Signature({x: (), mean: ()}, {temperature: ()})

        def evaluate(self, values):
            return {temperature: values[x] - values[mean]}

    solver = Solver([Predict(), Mean()], outputs=(total, partial, reduced))
    values = {x: Value(torch.tensor([1., 2., 4.]), "point")}
    with pytest.raises(NotImplementedError, match="non-independent sample axes"):
        solver(values, outputs=total)
    torch.testing.assert_close(solver(values, outputs=partial)[partial].tensor, torch.ones(3))
    torch.testing.assert_close(solver(values, outputs=reduced)[reduced].tensor, torch.full((3,), 1/3))


def test_point_coupling_does_not_block_ad_of_an_independent_scenario_parameter():
    x, parameter, temperature = map(Quantity, ("x", "parameter", "T"))
    spatial = TotalDerivative(temperature, x)
    scenario = TotalDerivative(temperature, parameter)

    class MixPoints(Model):
        geometry = None
        signature = Signature({x: "point", parameter: ()}, {temperature: "point"})

        def evaluate(self, values):
            return {temperature: values[parameter] * (values[x] - values[x].mean())}

    solver = Solver([MixPoints()], outputs=(spatial, scenario))
    values = {
        x: Value(torch.arange(6.).reshape(2, 3), ("scenario", "point")),
        parameter: Value(torch.tensor([2., 3.]), "scenario"),
    }
    with pytest.raises(NotImplementedError, match="non-independent sample axes"):
        solver(values, outputs=spatial)
    result = solver(values, outputs=scenario)[scenario]
    torch.testing.assert_close(result.tensor, values[x].tensor - values[x].tensor.mean(dim=1, keepdim=True))


def make_paths():
    x, z, temperature, residual = map(Quantity, ("x", "z", "T", "residual"))
    total = TotalDerivative(temperature, x)
    partial = PartialDerivative(temperature, x, holding=(z,))

    class Encode(Model):
        geometry = None
        signature = Signature({x: ()}, {z: ()})

        def evaluate(self, values):
            return {z: values[x].square()}

    class Predict(Model):
        geometry = None
        signature = Signature({x: (), z: ()}, {temperature: ()})

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.))

        def evaluate(self, values):
            return {temperature: self.weight * (values[x] + values[z])}

    class Residual(Model):
        geometry = None
        signature = Signature({total: (), partial: ()}, {residual: ()})

        def evaluate(self, values):
            return {residual: values[total] - values[partial]}

    # Reverse the natural order to exercise real dependency planning.
    solver = Solver([Residual(), Predict(), Encode()], outputs=residual)
    return solver, (x, z, temperature, residual, total, partial)


def test_solver_plans_total_and_partial_paths_and_keeps_derivatives_internal():
    solver, (x, z, temperature, residual, total, partial) = make_paths()
    assert solver.inputs == (x,)
    assert solver.derivative_plans[total].producer_index == 1
    assert solver.derivative_plans[total].model_indices == (2, 1)
    assert solver.derivative_plans[total].function_inputs == (x,)
    assert solver.derivative_plans[partial].model_indices == (1,)
    assert solver.derivative_plans[partial].function_inputs == (x, z)
    assert solver.derivative_plans[partial].depends_on_wrt
    assert total not in solver.inputs and partial not in solver.inputs

    tensor = torch.tensor([1., 2.], requires_grad=True)
    # Same Solver, same graph, derivative-free output selection for inference.
    result = solver({x: Value(tensor, "point")}, outputs=temperature)[temperature]
    torch.testing.assert_close(result.tensor, 2 * (tensor + tensor.square()))
    result.tensor.sum().backward()
    torch.testing.assert_close(tensor.grad, 2 + 4 * tensor)
    assert solver.models[1].weight.grad is not None
    # The residual uses real total and partial derivatives.
    torch.testing.assert_close(solver({x: Value(tensor, "point")})[residual].tensor, 4 * tensor)
    with pytest.raises(ValueError, match="Expected input"):
        solver({x: Value(tensor, "point"), total: Value(tensor, "point")})


def test_derivative_graph_contains_source_wrt_hold_and_residual_edges():
    solver, (_, _, _, _, total, partial) = make_paths()
    mermaid = solver.to_mermaid()
    assert 'm1 -->|"of: T"| d0' in mermaid
    assert 'i0 -->|"wrt: x"| d0' in mermaid
    assert 'm2 -->|"holding: z"| d1' in mermaid
    assert 'd0 -->|"total' in mermaid and '| m0' in mermaid
    dot = solver.to_dot()
    assert 'm1 -> d0' in dot and 'i0 -> d0' in dot
    assert 'm2 -> d1' in dot and 'd0 -> m0' in dot and 'd1 -> m0' in dot
    assert 'local reverse-mode AD' in dot.replace('<BR/>', ' ')


def test_direct_derivative_output_and_intermediate_wrt_are_planned():
    solver, (x, z, temperature, _, _, _) = make_paths()
    request = Derivative(temperature, z)
    graph = Solver(solver.models[1:], outputs=request)
    plan = graph.derivative_plans[request]
    assert plan.producer_index == 0
    assert plan.model_indices == (0,)
    assert plan.function_inputs == (x, z)
    assert graph.inputs == (x,)
    assert 'd0 --> o0' in graph.to_mermaid()
    torch.testing.assert_close(graph({x: Value(torch.tensor(2.))})[request].tensor, torch.tensor(2.))


def test_derivative_only_hold_path_is_structurally_independent():
    solver, (x, z, temperature, _, _, _) = make_paths()
    # Of=z and wrt=x but hold a genuinely intermediate quantity in a new graph.
    middle = Quantity("middle")

    class Final(Model):
        geometry = None
        signature = Signature({z: ()}, {middle: ()})

        def evaluate(self, values):
            return {middle: 3 * values[z]}

    request = PartialDerivative(middle, x, holding=(z,))
    graph = Solver([solver.models[2], Final()], outputs=request)
    assert graph.derivative_plans[request].model_indices == (1,)
    assert graph.derivative_plans[request].function_inputs == (z,)
    assert not graph.derivative_plans[request].depends_on_wrt


def test_models_cannot_provide_symbolic_derivatives_and_cycles_are_rejected():
    x, temperature = Quantity("x"), Quantity("T")
    request = Derivative(temperature, x)

    class Inject(Model):
        geometry = None
        signature = Signature({x: ()}, {request: ()})

        def evaluate(self, values):
            raise AssertionError("Must never execute")

    with pytest.raises(ValueError, match="produced by Solver"):
        Solver([Inject()])

    class Cyclic(Model):
        geometry = None
        signature = Signature({request: ()}, {temperature: ()})

        def evaluate(self, values):
            raise AssertionError("Must never execute")

    with pytest.raises(ValueError, match="Cyclic"):
        Solver([Cyclic()])
