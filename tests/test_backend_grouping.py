"""Executable grouping contracts for sampled, forward, and Taylor-style methods.

These small test backends are not new built-in numerical methods.
"""

import torch
from torch.func import jvp
from pinn_engine import Derivative, DerivativeBackend, Model, Quantity, Signature, Solver, Value


class Function(Model):
    def __init__(self, inputs, outputs, fn):
        super().__init__()
        self._signature, self.fn = Signature(inputs, outputs), fn
    @property
    def signature(self):
        return self._signature
    def evaluate(self, values):
        return self.fn(values)


class TriangleGradients(DerivativeBackend):
    """Test-only P1 triangle gradients, sharing one geometric inverse per group."""
    name = "test-p1-triangles"
    def __init__(self, x, y, mesh_for):
        self.x, self.y, self.mesh_for = x, y, mesh_for
        self.operator_builds, self.fields_solved = 0, []

    def dependencies(self, graph, request):
        return request.of, self.x, self.y, self.mesh_for[request.of]

    def group_key(self, request, values):
        mesh = self.mesh_for[request.of]
        return (id(values[self.x].tensor), id(values[self.y].tensor),
                id(values[mesh].tensor), values[request.of].axes)

    def compute(self, request, values, *, chunk_size):
        return self.compute_group((request,), {request: values}, chunk_size=chunk_size)[request]

    def prepare_group(self, requests, inputs):
        first = requests[0]
        values = inputs[first]
        mesh = values[self.mesh_for[first.of]].tensor
        x, y = values[self.x].tensor, values[self.y].tensor
        matrix = torch.stack((torch.ones_like(x), x, y), dim=-1)[mesh]
        self.operator_builds += 1
        return mesh, torch.linalg.inv(matrix)

    def compute_group(self, requests, inputs, *, chunk_size, prepared=None):
        mesh, inverse = self.prepare_group(requests, inputs) if prepared is None else prepared
        fields = tuple(dict.fromkeys(r.of for r in requests))
        self.fields_solved.append(fields)
        field_values = {r.of: inputs[r][r.of].tensor for r in requests}
        rhs = torch.stack([field_values[q] for q in fields], dim=-1)[mesh]
        coefficients = inverse @ rhs
        return {r: Value(coefficients[:, 1 if r.wrt == self.x else 2, fields.index(r.of)], "cell")
                for r in requests}


def test_mesh_operator_is_shared_across_fields_and_coordinate_directions_and_remains_trainable():
    x, y, t, u = map(Quantity, ("x", "y", "T", "U"))
    mesh = Quantity("cells", (3,))
    mesh_calls = []
    def cells(values):
        mesh_calls.append("mesh")
        return {mesh: torch.tensor([[0, 1, 2], [1, 3, 2]])}
    generator = Function({x: "point"}, {mesh: "cell"}, cells)
    backend = TriangleGradients(x, y, {t: mesh, u: mesh})
    requests = tuple(Derivative(q, coordinate) for q in (t, u) for coordinate in (x, y))
    solver = Solver([generator], outputs=requests, derivative_backend=backend)
    xx, yy = torch.tensor([0., 1., 0., 1.]), torch.tensor([0., 0., 1., 1.])
    theta, phi = torch.tensor(2., requires_grad=True), torch.tensor(4., requires_grad=True)
    def inputs():
        return {x: Value(xx, "point"), y: Value(yy, "point"),
                t: Value(theta*xx + 3*yy, "point"), u: Value(-xx + phi*yy, "point")}
    result = solver(inputs())
    for r, expected in zip(requests, (2., 3., -1., 4.)):
        torch.testing.assert_close(result[r].tensor, torch.full((2,), expected))
    sum(v.tensor.square().mean() for v in result.values()).backward()
    torch.testing.assert_close(theta.grad, torch.tensor(4.))
    torch.testing.assert_close(phi.grad, torch.tensor(8.))
    assert backend.operator_builds == 1
    assert backend.fields_solved == [(t, u)]
    assert mesh_calls == ["mesh"]
    # A later solver call receives new field graphs and rebuilds its operator;
    # grouping must not introduce a stale persistent cache.
    solver(inputs())
    assert backend.operator_builds == 2
    assert mesh_calls == ["mesh", "mesh"]


def test_different_meshes_and_unselected_fields_do_not_share_or_execute_extra_work():
    x, y, t, u = map(Quantity, ("x", "y", "T", "U"))
    ma, mb = Quantity("cells_a", (3,)), Quantity("cells_b", (3,))
    mesh_calls = []
    def generate(q):
        def f(values):
            mesh_calls.append(q)
            return {q: torch.tensor([[0, 1, 2]])}
        return Function({x: "point"}, {q: "cell"}, f)
    backend = TriangleGradients(x, y, {t: ma, u: mb})
    dt, du = Derivative(t, x), Derivative(u, x)
    solver = Solver([generate(ma), generate(mb)], outputs=(dt, du), derivative_backend=backend)
    xx, yy = torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])
    inputs = {x: Value(xx, "point"), y: Value(yy, "point"), t: Value(2*xx+yy, "point"),
              u: Value(3*xx-yy, "point")}
    solver(inputs, outputs=dt)
    assert backend.operator_builds == 1 and mesh_calls == [ma]
    assert backend.fields_solved == [(t,)]
    solver(inputs)
    assert backend.operator_builds == 3  # Two distinct mesh identities.
    assert backend.fields_solved[1:] == [(t,), (u,)]


def test_forward_style_groups_by_input_and_shares_one_jvp_across_different_output_shapes():
    x, z = Quantity("x"), Quantity("z")
    a, b, c = Quantity("A", (2,)), Quantity("B", (2, 2)), Quantity("C")
    functions = {a: lambda v: torch.stack((v**2, v**3)),
                 b: lambda v: v.sin()*torch.ones((2, 2))+v*torch.eye(2), c: lambda v: v**3}
    class ForwardExample(DerivativeBackend):
        name = "test-forward-jvp"
        def __init__(self):
            self.calls = []
        def dependencies(self, graph, request):
            return (request.wrt,)
        def group_key(self, request, values):
            return request.wrt, values[request.wrt].axes
        def compute(self, request, values, *, chunk_size):
            return self.compute_group((request,), {request: values}, chunk_size=chunk_size)[request]
        def compute_group(self, requests, inputs, *, chunk_size, prepared=None):
            target = requests[0].wrt
            value = inputs[requests[0]][target]
            self.calls.append((target, tuple(r.of for r in requests)))
            # This small example has a scalar input, so one tangent yields the
            # full derivative of arbitrarily shaped outputs with respect to it.
            _, derivatives = jvp(lambda v: tuple(functions[r.of](v) for r in requests),
                                 (value.tensor,), (torch.ones_like(value.tensor),))
            return {r: Value(d, value.axes) for r, d in zip(requests, derivatives)}
    def forbidden(values):
        raise AssertionError("A callable-based method does not need the primal Model forward")
    source = Function({x: (), z: ()}, {a: (), b: (), c: ()}, forbidden)
    backend = ForwardExample()
    requests = Derivative(a, x), Derivative(b, x), Derivative(c, z)
    result = Solver([source], outputs=requests, derivative_backend=backend)({
        x: Value(torch.tensor(2.)), z: Value(torch.tensor(3.))})
    torch.testing.assert_close(result[requests[0]].tensor, torch.tensor([4., 12.]))
    torch.testing.assert_close(result[requests[1]].tensor, torch.cos(torch.tensor(2.))*torch.ones(2, 2)+torch.eye(2))
    torch.testing.assert_close(result[requests[2]].tensor, torch.tensor(27.))
    assert backend.calls == [(x, (a, b)), (z, (c,))]


def test_taylor_style_backend_can_fuse_orders_by_declaring_primitive_dependencies():
    x, y = Quantity("x"), Quantity("T")
    first = Derivative(y, x)
    second, third = Derivative(first, x), Derivative(Derivative(first, x), x)
    class PolynomialSeries(DerivativeBackend):
        """Test-only derivatives of x**4, reusing powers across requested orders."""
        name = "test-polynomial-series"
        def __init__(self):
            self.groups = []
        def dependencies(self, graph, request):
            return (x,)  # No dependency on an inner derivative VALUE.
        def group_key(self, request, values):
            base = request
            while isinstance(base, Derivative):
                assert base.wrt == x
                base = base.of
            return base, x
        def compute(self, request, values, *, chunk_size):
            return self.compute_group((request,), {request: values}, chunk_size=chunk_size)[request]
        def compute_group(self, requests, inputs, *, chunk_size, prepared=None):
            self.groups.append(tuple(requests))
            value = inputs[requests[0]][x]
            xx = value.tensor
            squared = xx.square()  # Reused for both first and second orders.
            formula = {first: lambda: 4*squared*xx, second: lambda: 12*squared, third: lambda: 24*xx}
            return {r: Value(formula[r](), value.axes) for r in requests}
    source = Function({x: ()}, {y: ()}, lambda v: (_ for _ in ()).throw(AssertionError("Unneeded primal")))
    backend = PolynomialSeries()
    solver = Solver([source], outputs=(first, second, third), derivative_backend=backend)
    xx = torch.tensor(2., requires_grad=True)
    result = solver({x: Value(xx)})
    for r, expected in zip((first, second, third), (32., 48., 48.)):
        torch.testing.assert_close(result[r].tensor, torch.tensor(expected))
    sum(v.tensor for v in result.values()).backward()
    torch.testing.assert_close(xx.grad, torch.tensor(120.))
    assert len(backend.groups) == 1 and set(backend.groups[0]) == {first, second, third}
    assert all(solver._node_dependencies[("derivative", r)] == (x,) for r in (first, second, third))
    backend.groups.clear()
    solver({x: Value(torch.tensor(3.))}, outputs=third)
    assert backend.groups == [(third,)]  # No forced execution of unselected inner nodes.


def test_default_grouping_is_singleton_and_each_request_gets_only_its_dependencies():
    x, z, y = map(Quantity, ("x", "z", "T"))
    class Default(DerivativeBackend):
        name = "test-default"
        def __init__(self):
            self.groups = []
        def compute_group(self, requests, inputs, *, chunk_size, prepared=None):
            self.groups.append(tuple(requests))
            return super().compute_group(requests, inputs, chunk_size=chunk_size, prepared=prepared)
        def compute(self, request, values, *, chunk_size):
            assert set(values) == {request.of, request.wrt}
            return Value(torch.ones_like(values[request.wrt].tensor))
    source = Function({x: (), z: ()}, {y: ()}, lambda v: {y: v[x]+v[z]})
    backend = Default()
    requests = Derivative(y, x), Derivative(y, z)
    result = Solver([source], outputs=requests, derivative_backend=backend)({
        x: Value(torch.tensor(2.)), z: Value(torch.tensor(3.))})
    assert backend.groups == [(requests[0],), (requests[1],)]
    assert set(result) == set(requests)


def test_shared_mesh_preparation_survives_later_frontier_but_not_next_solver_call():
    x, y, t, u = map(Quantity, ("x", "y", "T", "U"))
    mesh = Quantity("cells", (3,))
    dt, du = Derivative(t, x), Derivative(u, x)
    # U needs dT/dx before it can be formed, so these requests cannot execute
    # together. Their geometric inverse still must be constructed only once.
    generate = Function({x: "point"}, {mesh: "cell"}, lambda v: {mesh: torch.tensor([[0, 1, 2]])})
    produce_u = Function({x: "point", dt: "cell"}, {u: "point"},
                         lambda v: {u: (v[dt].mean()+1)*v[x]})
    backend = TriangleGradients(x, y, {t: mesh, u: mesh})
    solver = Solver([produce_u, generate], outputs=(dt, du), derivative_backend=backend)
    xx, yy = torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])
    values = {x: Value(xx, "point"), y: Value(yy, "point"), t: Value(2*xx+3*yy, "point")}
    result = solver(values)
    torch.testing.assert_close(result[dt].tensor, torch.tensor([2.]))
    torch.testing.assert_close(result[du].tensor, torch.tensor([3.]))
    assert backend.fields_solved == [(t,), (u,)]
    assert backend.operator_builds == 1
    solver(values)
    assert backend.operator_builds == 2
