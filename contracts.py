"""Standalone quantity, model, and derivative contracts for PINN Engine.

This module declares quantities, named Tensor values, and model contracts.
evaluate consumes raw Tensors; forward handles named sample axes with
vmap. Derivatives are symbolic requests: Solver plans their source paths but
does NOT implement numerical differentiation. Memory policy remains open.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from html import escape
from typing import Mapping, Sequence

import torch
from torch import nn

# An axis is just its semantic name, independent of size or Tensor position.
Axis = str


class Geometry(ABC):
    """Abstract domain operations for the new design, without an implementation.

    Concrete geometry implementations are supplied by the application.
    Geometry identity remains object identity, even when shapes are equivalent.
    """

    @abstractmethod
    def boundary(self) -> Geometry:
        """Return the boundary as a geometry."""
        raise NotImplementedError

    @abstractmethod
    def union(self, other: Geometry) -> Geometry:
        """Return the union of this domain and other."""
        raise NotImplementedError

    @abstractmethod
    def intersection(self, other: Geometry) -> Geometry:
        """Return the intersection, which may be an empty domain."""
        raise NotImplementedError

    @abstractmethod
    def difference(self, other: Geometry) -> Geometry:
        """Return the portion of this domain outside other."""
        raise NotImplementedError

    @abstractmethod
    def is_subset_of(self, other: Geometry) -> bool:
        """Test inclusion, including equality; raise if it cannot be determined.

        An unsupported or unknown relationship must not silently return False.
        """
        raise NotImplementedError

    def is_superset_of(self, other: Geometry) -> bool:
        """Test the reverse inclusion using the same subset contract."""
        return other.is_subset_of(self)

    def __or__(self, other: Geometry) -> Geometry:
        """Support a | b as shorthand for a.union(b)."""
        return self.union(other)

    def __and__(self, other: Geometry) -> Geometry:
        """Support a & b as shorthand for a.intersection(b)."""
        return self.intersection(other)

    def __sub__(self, other: Geometry) -> Geometry:
        """Support a - b as shorthand for a.difference(b)."""
        return self.difference(other)


@dataclass(frozen=True, eq=False)
class Quantity:
    """Declare a nonspatial quantity without a runtime value, version, or cache.

    shape is intrinsic: scalar (), vector (3,), stress tensor (3, 3).
    Scenario and point counts are excluded from shape.
    Definitions with the same name and domain identify the same dictionary key.
    Construct new instances freely; there is no global registry or get factory.
    Signature checks shape consistency when declarations are used together.
    """

    name: str
    shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Quantity.name must be a non-empty string")

        shape = tuple(self.shape)
        if any(type(size) is not int or size < 0 for size in shape):
            raise ValueError("Quantity.shape must contain non-negative integers")
        object.__setattr__(self, "shape", shape)

    @property
    def _domain(self) -> Geometry | None:
        # Nonspatial quantities have no domain; Field overrides this property.
        return None

    @property
    def _identity(self) -> tuple:
        # Geometry identity currently means the same object, regardless of shape
        # equality or whether two geometries belong to the same class.
        domain = self._domain
        return (None if domain is None else id(domain), self.name)

    def __hash__(self) -> int:
        return hash(self._identity)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return NotImplemented
        return self._identity == other._identity


@dataclass(frozen=True, eq=False)
class Field(Quantity):
    """A spatial quantity whose values are defined at points of a geometry.

    Example: Field("T", shape=(), geometry=solid).
    Reuse the same geometry object when redeclaring the same field.
    """

    geometry: Geometry = field(kw_only=True)

    def __post_init__(self) -> None:
        if self.geometry is None:
            raise ValueError("Field.geometry is required")
        super().__post_init__()

    @property
    def _domain(self) -> Geometry:
        return self.geometry


@dataclass(frozen=True, eq=False, init=False)
class Derivative(Quantity):
    """A first-order, full-Quantity derivative, independent of any AD backend.

    Derivative(a, b) is total along the hosting Solver's computation graph.
    Derivative(a, b, holding=(z,)) is partial: hold z's current value fixed
    while varying b. holding=None and holding=() deliberately differ in kind.
    Holding concerns this derivative only, NOT freezing or detaching training.

    Intrinsic shape is a.shape + b.shape, output components first. Runtime
    sample axes are NOT guessed here. Independent mapped samples can align;
    Solver validates that assumption before the default AD execution boundary.
    Full coupled Jacobians and numerical derivative backends remain pending.
    No component indexing or higher-order derivatives are supported yet.
    """

    of: Quantity
    wrt: Quantity
    holding: tuple[Quantity, ...] | None

    def __init__(
        self, of: Quantity, wrt: Quantity, *,
        holding: Sequence[Quantity] | None = None,
    ) -> None:
        if not isinstance(of, Quantity) or not isinstance(wrt, Quantity):
            raise TypeError("Derivative operands must be complete Quantity objects")
        fixed = None if holding is None else tuple(holding)
        operands = (of, wrt, *(fixed or ()))
        if any(not isinstance(q, Quantity) for q in operands):
            raise TypeError("holding must contain Quantity objects")
        if any(isinstance(q, Derivative) for q in operands):
            raise NotImplementedError("Only first-order derivative declarations are supported")
        if fixed is not None:
            if len(set(fixed)) != len(fixed):
                raise ValueError("holding cannot repeat quantities")
            if wrt in fixed or of in fixed:
                raise ValueError("Do not hold the differentiated input or output fixed")
        object.__setattr__(self, "of", of)
        object.__setattr__(self, "wrt", wrt)
        object.__setattr__(self, "holding", fixed)
        suffix = "" if fixed is None else "; holding=" + ",".join(q.name for q in fixed)
        name = f"{self.kind}({of.name}, {wrt.name}{suffix})"
        Quantity.__init__(self, name, of.shape + wrt.shape)

    @property
    def kind(self) -> str:
        return "total" if self.holding is None else "partial"

    @property
    def _domain(self) -> Geometry | None:
        return self.of._domain

    @property
    def _identity(self) -> tuple:
        # Symbolic operands, not generated display text, define identity.
        # The order of held quantities is mathematically irrelevant.
        return ("derivative", self.kind, self.of, self.wrt,
                frozenset(self.holding or ()))


def TotalDerivative(of: Quantity, wrt: Quantity) -> Derivative:
    """Readable constructor for a total derivative; there is only one class."""
    return Derivative(of, wrt)


def PartialDerivative(
    of: Quantity, wrt: Quantity, *, holding: Sequence[Quantity],
) -> Derivative:
    """Require an explicit declaration of quantities held fixed."""
    return Derivative(of, wrt, holding=holding)


def _related_quantities(quantity: Quantity):
    """Include symbolic operands when checking repeated shape declarations."""
    yield quantity
    if isinstance(quantity, Derivative):
        yield quantity.of
        yield quantity.wrt
        yield from quantity.holding or ()


def _axes(names: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize axis declarations without splitting a bare string into letters."""
    if isinstance(names, str):
        names = (names,)
    result = tuple(names)
    if any(not isinstance(axis, str) for axis in result):
        raise TypeError("Axis names must be strings")
    if any(not axis.strip() for axis in result):
        raise ValueError("Axis names must be non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError("An axis declaration cannot repeat the same sample axis")
    return result


@dataclass(frozen=True, eq=False)
class Value:
    """A Tensor together with its actual, ordered sample-axis names.

    Example: Value(temperatures, axes=("scenario", "point")) for scalar data
    shaped (S, P). Intrinsic Quantity dimensions follow these axes and are not
    named here: a vector value can have shape (S, P, 3) with the same axes.

    The Tensor is kept by reference, with no copying or detaching. Frozen fields
    protect the metadata but do not make Tensor storage immutable. Versions,
    caching, and other execution metadata may be considered later.
    """

    tensor: torch.Tensor
    axes: str | Sequence[str] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("Value.tensor must be a Tensor")
        axes = _axes(self.axes)
        if len(axes) > self.tensor.ndim:
            raise ValueError("Value cannot have more sample axes than Tensor dimensions")
        object.__setattr__(self, "axes", axes)


def _check_value(quantity: Quantity, value: Value, sizes: dict[str, int]) -> None:
    """Validate a value against its Quantity and shared sample-axis sizes."""
    if not isinstance(value, Value):
        raise TypeError(f"Value for {quantity.name!r} must be a Value object")
    if value.tensor.ndim != len(value.axes) + len(quantity.shape):
        raise ValueError(f"Rank mismatch for {quantity.name!r}; check its sample axes")
    if tuple(value.tensor.shape[len(value.axes):]) != quantity.shape:
        raise ValueError(f"Intrinsic shape mismatch for {quantity.name!r}")
    for axis, size in zip(value.axes, value.tensor.shape):
        if axis in sizes and sizes[axis] != size:
            raise ValueError(f"Inconsistent size for axis {axis!r}")
        sizes[axis] = size


@dataclass(frozen=True)
class Signature:
    """Axes consumed or produced in full by one call to the user function.

    Each Quantity maps to the axes required by the function.
    Accept axis names either alone or in a sequence, and normalize them to
    tuples of strings. A bare string names one axis, not its letters.
    () passes no sample axes; the Quantity's intrinsic shape still applies.
    (point,) passes the full point axis, for example for attention or integration.
    Core axes may also appear in outputs, for example for a graph network that
    predicts at its input points. Reusing a name preserves sample identity and
    length, NOT independence: consuming a whole axis can mix its samples.
    Rename an output axis when its sample identity, ordering, or length changes.
    AD compatibility belongs to the requested derivative path, not this class.
    Additional runtime batch axes are not included in this signature.

    This structured representation is an initial notation for the skeleton.
    Text parsing is not implemented. Model.forward uses the structured
    signature to construct vmap calls directly.
    """

    inputs: Mapping[Quantity, str | Sequence[str]]
    outputs: Mapping[Quantity, str | Sequence[str]]

    def __post_init__(self) -> None:
        # Validate declarations locally instead of retaining a global registry.
        known_quantities: dict[Quantity, Quantity] = {}
        for side in ("inputs", "outputs"):
            declarations = {}
            for quantity, axes in getattr(self, side).items():
                if not isinstance(quantity, Quantity):
                    raise TypeError("Signature keys must be Quantity objects")
                for related in _related_quantities(quantity):
                    previous = known_quantities.get(related)
                    if previous is not None and previous.shape != related.shape:
                        raise ValueError(f"Conflicting shapes for quantity {related.name!r}")
                    known_quantities[related] = related

                declarations[quantity] = _axes(axes)

            # Copy declarations so changes to the original dictionary cannot
            # change the model's signature.
            # A plain copied dict also lets torch.compile trace Quantity keys.
            object.__setattr__(self, side, declarations)

    @classmethod
    def from_string(cls, text: str) -> Signature:
        """Parse a text signature; syntax and name resolution are still pending."""
        raise NotImplementedError("Signature text parsing is not implemented yet")


def _batch_axes(signature: Signature, layouts) -> tuple[str, ...]:
    """Derive mapped axes from layouts, checking whole-axis requirements."""
    core_axes = set(
        axis for side in (signature.inputs, signature.outputs)
        for axes in side.values() for axis in axes
    )
    batch = tuple(dict.fromkeys(
        axis for quantity in signature.inputs for axis in layouts[quantity]
        if axis not in core_axes
    ))
    for quantity, core in signature.inputs.items():
        actual = set(layouts[quantity])
        if not set(core) <= actual:
            raise ValueError(f"Missing core axes for {quantity.name!r}")
        if (actual - set(core)) & core_axes:
            raise ValueError(
                f"An axis of {quantity.name!r} is core elsewhere in this model; "
                "declare it in this input's signature too"
            )
    return batch


class Model(nn.Module, ABC):
    """An abstract computation, not necessarily a neural or trainable model.

    evaluate follows the core signature and receives raw Tensors.
    forward accepts Value objects and handles extra axes automatically.
    Call model(values) to preserve PyTorch hooks. Subclass constructors must call
    super().__init__() before assigning parameters or child modules.
    Subclasses provide signature and geometry through properties or class
    attributes. Their constructors are free to accept whatever they need.
    """

    @property
    @abstractmethod
    def signature(self) -> Signature:
        """Declare the inputs, outputs, and core axes of one function call."""
        raise NotImplementedError

    @property
    @abstractmethod
    def geometry(self) -> Geometry | None:
        """Declare the execution domain, or None for a nonspatial model.

        Compatibility checks with Field domains will use geometry relationships;
        that validation is not implemented by this declaration-only skeleton.
        """
        raise NotImplementedError

    @property
    def inputs(self) -> tuple[Quantity, ...]:
        """Derive input quantities from signature instead of declaring them twice."""
        return tuple(self.signature.inputs)

    @property
    def outputs(self) -> tuple[Quantity, ...]:
        """Derive output quantities from signature instead of declaring them twice."""
        return tuple(self.signature.outputs)

    @abstractmethod
    def evaluate(
        self, values: Mapping[Quantity, torch.Tensor]
    ) -> dict[Quantity, torch.Tensor]:
        """Evaluate one call according to signature using the subclass algorithm.

        For a pointwise call, each Field value has only its intrinsic shape.
        For a full-point call, the point axis precedes the Field's intrinsic shape.
        Returned Quantity keys must match outputs.
        """
        raise NotImplementedError

    def forward(self, values: Mapping[Quantity, Value]) -> dict[Quantity, Value]:
        """Apply evaluate to named batches, returning Tensors with their axes.

        Axes declared in signature are passed to evaluate in full and in the
        declared order. Other axes are mapped independently; an input missing
        a mapped axis is shared along it. Equal axis names declare matched
        samples and require equal lengths, rather than inferring matches by size.
        """
        signature = self.signature
        if set(values) != set(signature.inputs):
            raise ValueError(f"Expected input quantities: {[q.name for q in signature.inputs]}")
        declarations = {quantity: quantity for quantity in signature.inputs}
        sizes = {}
        for quantity, value in values.items():
            if quantity.shape != declarations[quantity].shape:
                raise ValueError(f"Conflicting shapes for {quantity.name!r}")
            _check_value(quantity, value, sizes)

        batch = _batch_axes(signature, {q: value.axes for q, value in values.items()})
        arguments = []
        for quantity, core in signature.inputs.items():
            value = values[quantity]
            actual = value.axes
            ordered = tuple(axis for axis in batch if axis in actual) + tuple(core)
            permutation = tuple(actual.index(axis) for axis in ordered)
            permutation += tuple(range(len(actual), value.tensor.ndim))
            arguments.append(value.tensor.permute(permutation))

        def call(*tensors):
            # User code sees only the core Tensor shapes, with no Value wrappers.
            result = self.evaluate(dict(zip(signature.inputs, tensors)))
            if set(result) != set(signature.outputs):
                raise ValueError("Model returned quantities different from its signature")
            for tensor in result.values():
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError("Model.evaluate must return raw Tensor values")
            return tuple(result[quantity] for quantity in signature.outputs)

        # Inputs share batch-axis order after permutation. Wrapping inside-out
        # makes each mapped axis leading when its corresponding vmap sees it.
        mapped = call
        for axis in reversed(batch):
            in_dims = tuple(0 if axis in values[q].axes else None for q in signature.inputs)
            mapped = torch.func.vmap(mapped, in_dims=in_dims, out_dims=0)

        outputs = {}
        for quantity, tensor in zip(signature.outputs, mapped(*arguments)):
            value = Value(tensor, batch + tuple(signature.outputs[quantity]))
            _check_value(quantity, value, sizes)
            outputs[quantity] = value
        return outputs


@dataclass(frozen=True)
class DerivativePlan:
    """Describe the function to differentiate, without implementing its backend.

    model_indices are dependency-ordered indices into Solver.models. Traversal
    stops at wrt and held quantities; their current values become function
    inputs. Other external inputs also remain fixed during this derivative.
    This is structural reachability, not proof of a nonzero numerical derivative.
    A future backend must preserve the outer parameter-training graph.
    """

    derivative: Derivative
    producer_index: int | None
    model_indices: tuple[int, ...]
    function_inputs: tuple[Quantity, ...]
    depends_on_wrt: bool

    @property
    def dependencies(self) -> tuple[Quantity, ...]:
        # These values must exist before the derivative execution placeholder.
        return tuple(dict.fromkeys((self.derivative.of, self.derivative.wrt,
                                    *(self.derivative.holding or ()))))


class Solver(nn.Module):
    """Wire Models and symbolic first derivatives in one hosting graph.

    Derivatives are virtual producer nodes, never external Tensor inputs and
    never user Model outputs. Plans identify source models and held cut points.
    Numerical evaluation of every derivative deliberately raises NotImplemented.

    Selected outputs determine the executed ancestors. A forward outputs=
    override permits inference and unfinished physics in the same Solver.
    There is no numerical derivative backend, grouping, persistent cache, or
    scheduling optimization here. The default derivative boundary checks AD
    compatibility. Ordinary outputs retain the v2 named-axis execution.
    """

    def __init__(
        self, models: Sequence[Model],
        outputs: Quantity | Sequence[Quantity] | None = None,
    ) -> None:
        super().__init__()
        self.models = tuple(models)
        if not self.models or any(not isinstance(model, Model) for model in self.models):
            raise ValueError("Provide at least one Model")

        self._quantities = {}
        produced, required = {}, {}
        derivatives = {}
        self.geometry = None
        for index, model in enumerate(self.models):
            signature = model.signature
            if not signature.outputs:
                raise ValueError("Each connected model must declare an output")
            for side in (signature.inputs, signature.outputs):
                for quantity in side:
                    self._register_quantity(quantity)
                    if isinstance(quantity, Derivative):
                        derivatives[quantity] = None
            for quantity in signature.outputs:
                if isinstance(quantity, Derivative):
                    raise ValueError("Derivative quantities are produced by Solver, not by Models")
                if quantity in produced:
                    raise ValueError(f"Multiple producers for {quantity.name!r}")
                produced[quantity] = index
            required.update(dict.fromkeys(signature.inputs))

            # Keep the existing same-domain rule; geometric subset policy is open.
            if model.geometry is not None:
                if self.geometry is not None and model.geometry is not self.geometry:
                    raise ValueError("Connected models must share one geometry object")
                self.geometry = model.geometry
            self.add_module(f"part_{index}", model)

        # A derivative can be requested directly, without a residual consumer.
        if outputs is not None:
            outputs = (outputs,) if isinstance(outputs, Quantity) else tuple(outputs)
            for quantity in outputs:
                if not isinstance(quantity, Quantity):
                    raise TypeError("Solver outputs must be Quantity objects")
                self._register_quantity(quantity)
                if isinstance(quantity, Derivative):
                    derivatives[quantity] = None
        self.derivatives = tuple(derivatives)
        for derivative in self.derivatives:
            required.update(dict.fromkeys((derivative.of, derivative.wrt,
                                           *(derivative.holding or ()))))
        provided = set(produced) | set(self.derivatives)
        self.inputs = tuple(q for q in required if q not in provided)
        selected = tuple(produced) if outputs is None else (
            (outputs,) if isinstance(outputs, Quantity) else tuple(outputs)
        )
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Select at least one distinct forward output")
        for quantity in selected:
            if quantity not in provided:
                raise ValueError(f"Unknown forward output: {quantity!r}")
            if quantity.shape != self._quantities[quantity].shape:
                raise ValueError(f"Conflicting shapes for {quantity.name!r}")
        self.outputs = selected
        self._producers = produced

        # Both ordinary models and derivative requests participate in ordering.
        # The latter must precede residual consumers, but follow source values.
        self._node_dependencies = {
            ("model", index): tuple(model.inputs)
            for index, model in enumerate(self.models)
        }
        for derivative in self.derivatives:
            self._node_dependencies[("derivative", derivative)] = tuple(dict.fromkeys(
                (derivative.of, derivative.wrt, *(derivative.holding or ()))
            ))
        available, pending = set(self.inputs), dict(self._node_dependencies)
        self._ordered = []
        while pending:
            ready = [node for node, dependencies in pending.items()
                     if set(dependencies) <= available]
            if not ready:
                raise ValueError("Cyclic model dependencies; use distinct input/output quantities")
            for node in ready:
                self._ordered.append(node)
                kind, key = node
                available.update(self.models[key].outputs if kind == "model" else (key,))
                del pending[node]

        # Planning is independent of actual Tensor sizes and the AD/FD backend.
        self.derivative_plans = {
            derivative: self._plan_derivative(derivative)
            for derivative in self.derivatives
        }

    def _register_quantity(self, quantity: Quantity) -> None:
        if not isinstance(quantity, Quantity):
            raise TypeError("Expected Quantity objects")
        for related in _related_quantities(quantity):
            previous = self._quantities.get(related)
            if previous is not None and previous.shape != related.shape:
                raise ValueError(f"Conflicting shapes for {related.name!r}")
            self._quantities[related] = related

    def _plan_derivative(self, derivative: Derivative) -> DerivativePlan:
        stops = {derivative.wrt, *(derivative.holding or ())}
        leaves, indices, visited = {}, [], set()

        def visit(quantity):
            if quantity in visited:
                return
            visited.add(quantity)
            if isinstance(quantity, Derivative):
                raise NotImplementedError("Differentiating paths containing derivatives is not implemented")
            if quantity in stops or quantity not in self._producers:
                leaves[quantity] = None
                return
            index = self._producers[quantity]
            for required in self.models[index].inputs:
                visit(required)
            if index not in indices:
                indices.append(index)

        visit(derivative.of)
        # Preserve global dependency order, including models with multiple outputs.
        ordered = tuple(key for kind, key in self._ordered
                        if kind == "model" and key in indices)
        return DerivativePlan(derivative, self._producers.get(derivative.of),
                              ordered, tuple(leaves), derivative.wrt in leaves)

    def _producer_node(self, quantity: Quantity):
        if isinstance(quantity, Derivative):
            return ("derivative", quantity)
        if quantity in self._producers:
            return ("model", self._producers[quantity])
        return None

    def _needed_nodes(self, outputs):
        needed = set()

        def visit(quantity):
            node = self._producer_node(quantity)
            if node is None or node in needed:
                return
            needed.add(node)
            for dependency in self._node_dependencies[node]:
                visit(dependency)

        for quantity in outputs:
            visit(quantity)
        return needed

    def _evaluate_derivative(
        self, plan: DerivativePlan, values: Mapping[Quantity, Value],
    ) -> Value:
        """Default AD boundary; numerical differentiation is still unimplemented.

        Re-evaluating paths at perturbed wrt values and preserving held values,
        parameter gradients, and mapped/core-axis semantics are NOT implemented.
        Do not return a zero, detached Tensor, or externally supplied substitute.
        A future non-AD backend can override this method and apply its own
        capability checks instead of the AD-only validation below.
        """
        self.validate_autograd(plan, values)
        raise NotImplementedError(
            f"Derivative execution is not implemented: {plan.derivative.name}; "
            f"source model indices={plan.model_indices}"
        )

    def validate_autograd(
        self, plan: DerivativePlan, values: Mapping[Quantity, Value],
    ) -> None:
        """Reject unsupported alignment of dependent samples for this AD request.

        values contains the current source values, including of and wrt. Their
        layouts tell us which axes an independent-sample AD backend would align.
        Consuming any such axis on a wrt-dependent path removes that guarantee.
        This is conservative: evaluate's internals are not inspected.

        A reduction to a scalar has no shared sample axis and remains legal.
        A reduction followed by broadcasting back onto that axis is not legal.
        Held and wrt-independent branches do not count as differentiated paths.
        This check does not detach values or restrict ordinary loss.backward().
        """
        derivative = plan.derivative
        shared = set(values[derivative.of].axes) & set(values[derivative.wrt].axes)
        if not shared or not plan.depends_on_wrt:
            return

        # model_indices also contains branches needed to evaluate constants.
        # Only propagate coordinate dependence from wrt, stopping at holding.
        dependent = {derivative.wrt}
        held = set(derivative.holding or ())
        for index in plan.model_indices:
            model = self.models[index]
            signature = model.signature
            active_inputs = [q for q in signature.inputs if q in dependent]
            if not active_inputs:
                continue
            consumed = {axis for q in active_inputs for axis in signature.inputs[q]}
            unsupported = shared & consumed
            if unsupported:
                raise NotImplementedError(
                    f"Autograd across non-independent sample axes is not supported: "
                    f"{sorted(unsupported)}; model {index} ({type(model).__name__}) "
                    f"consumes these axes on the path for {derivative.name}. "
                    "For independent computation, leave mapped axes out of the "
                    "signature; otherwise use a suitable non-AD backend or an "
                    "appropriate partial derivative with holding."
                )
            dependent.update(q for q in signature.outputs if q not in held)

    def forward(
        self, values: Mapping[Quantity, Value], *,
        outputs: Quantity | Sequence[Quantity] | None = None,
    ) -> dict[Quantity, Value]:
        """Execute selected ancestors; unfinished derivatives fail explicitly."""
        if set(values) != set(self.inputs):
            raise ValueError(f"Expected input quantities: {[q.name for q in self.inputs]}")
        selected = self.outputs if outputs is None else (
            (outputs,) if isinstance(outputs, Quantity) else tuple(outputs)
        )
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Select at least one distinct forward output")
        for quantity in selected:
            if quantity not in self._producers and quantity not in self.derivative_plans:
                raise ValueError(f"Unknown forward output: {quantity!r}")
            if quantity.shape != self._quantities[quantity].shape:
                raise ValueError(f"Conflicting shapes for {quantity.name!r}")
        pool = dict(values)
        sizes = {}
        for quantity, value in values.items():
            if quantity.shape != self._quantities[quantity].shape:
                raise ValueError(f"Conflicting shapes for {quantity.name!r}")
            _check_value(quantity, value, sizes)

        needed = self._needed_nodes(selected)
        for node in self._ordered:
            if node not in needed:
                continue
            kind, key = node
            if kind == "derivative":
                value = self._evaluate_derivative(self.derivative_plans[key], pool)
                _check_value(key, value, sizes)
                pool[key] = value
                continue
            model = self.models[key]
            signature = model.signature
            inputs = {quantity: pool[quantity] for quantity in signature.inputs}
            # Calling the module preserves hooks and each model's own vmap.
            result = model(inputs)
            if set(result) != set(signature.outputs):
                raise ValueError("Model returned quantities different from its signature")
            for quantity, value in result.items():
                _check_value(quantity, value, sizes)
                pool[quantity] = value
        return {quantity: pool[quantity] for quantity in selected}

    def _source_node(self, quantity: Quantity, external) -> str:
        if quantity in external:
            return external[quantity]
        if isinstance(quantity, Derivative):
            return f"d{self.derivatives.index(quantity)}"
        return f"m{self._producers[quantity]}"

    def _graph_quantity_label(self, quantity: Quantity) -> str:
        """Disambiguate equal names on different domains without object ids."""
        label = quantity.name
        matches = [q for q in self._quantities if q.name == quantity.name]
        if len(matches) > 1:
            domains = []
            for q in matches:
                if all(q._domain is not d for d in domains):
                    domains.append(q._domain)
            if quantity._domain is None:
                label += " [nonspatial]"
            else:
                number = next(i for i, d in enumerate(domains, 1)
                              if d is quantity._domain)
                label += f" [domain {number}]"
        return label

    def to_mermaid(self, direction: str = "LR") -> str:
        """Return a styled Mermaid diagram. Runtime/sample sizes are not inferred.

        Paste the returned text into a Mermaid code block to display it.
        Model numbers refer to their original position in self.models.
        """
        if direction not in {"LR", "TB"}:
            raise ValueError("direction must be 'LR' or 'TB'")

        def label(text):
            # Mermaid's numeric entities prevent user labels being parsed as syntax.
            return "".join(f"#{ord(c)};" if c in '&<>"|#[]{}()' else c
                           for c in str(text).replace("\r", " ").replace("\n", " "))

        lines = [f"flowchart {direction}"]
        external = {q: f"i{i}" for i, q in enumerate(self.inputs)}
        for q, node in external.items():
            lines.append(f'  {node}(["Input: {label(self._graph_quantity_label(q))}"]):::input')
        for index, model in enumerate(self.models):
            name = label(type(model).__name__)
            lines.append(f'  m{index}["{index + 1}. {name}"]:::model')
        for index, derivative in enumerate(self.derivatives):
            lines.append(f'  d{index}["{label(derivative.name)}"]:::derivative')
            for quantity, role in self._derivative_graph_inputs(derivative):
                source = self._source_node(quantity, external)
                caption = label(f"{role}: {self._graph_quantity_label(quantity)}")
                lines.append(f'  {source} -->|"{caption}"| d{index}')
        for index, model in enumerate(self.models):
            for q in model.inputs:
                source = self._source_node(q, external)
                lines.append(f'  {source} -->|"{label(self._graph_quantity_label(q))}"| m{index}')
        for index, q in enumerate(self.outputs):
            lines.append(f'  o{index}(["Output: {label(self._graph_quantity_label(q))}"]):::output')
            lines.append(f'  {self._source_node(q, external)} --> o{index}')
        lines += [
            "  classDef input fill:#EFF6FF,stroke:#93C5FD,color:#1E40AF,stroke-width:1.5px;",
            "  classDef model fill:#F5F3FF,stroke:#A78BFA,color:#4C1D95,stroke-width:1.5px;",
            "  classDef output fill:#ECFDF5,stroke:#6EE7B7,color:#065F46,stroke-width:1.5px;",
            "  classDef derivative fill:#FFF7ED,stroke:#FB923C,color:#9A3412,stroke-width:1.5px;",
            "  linkStyle default stroke:#94A3B8,stroke-width:1.5px;",
        ]
        return "\n".join(lines)

    @staticmethod
    def _derivative_graph_inputs(derivative: Derivative):
        yield derivative.of, "of"
        yield derivative.wrt, "wrt"
        for quantity in derivative.holding or ():
            yield quantity, "holding"

    def to_dot(self, direction: str = "LR", title: str = "Solver computation graph") -> str:
        """Return Graphviz DOT with automatic layout and vector-quality labels.

        Blue: external inputs. Purple: models. Green: selected returned outputs.
        Core axes are signature axes, not runtime batch axes. No model executes.
        """
        import json
        from textwrap import wrap

        if direction not in {"LR", "TB"}:
            raise ValueError("direction must be 'LR' or 'TB'")

        def quote(text):
            return json.dumps(str(text), ensure_ascii=False)

        def html(text):
            return escape(str(text), quote=True)

        def wrapped(text, width=30):
            return '<BR/>'.join(html(line) for line in
                               wrap(str(text), width, break_long_words=True) or [""])

        def card(kicker, heading, detail, color):
            return ('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3">'
                    f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="{color}">{html(kicker)}</FONT></TD></TR>'
                    f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="15"><B>{wrapped(heading)}</B></FONT></TD></TR>'
                    f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#64748B">{wrapped(detail, 42)}</FONT></TD></TR>'
                    '</TABLE>>')

        def shape(q):
            return "scalar" if not q.shape else "shape: " + " x ".join(map(str, q.shape))

        lines = [
            "digraph Solver {",
            f"  graph [rankdir={direction}, bgcolor=\"#FFFFFF\", pad=0.45, nodesep=0.45,",
            "         ranksep=0.85, splines=spline, outputorder=edgesfirst,",
            "         fontname=\"DejaVu Sans\", fontsize=22, fontcolor=\"#0F172A\",",
            f"         labelloc=t, labeljust=l, label={quote(title)}];",
            "  node [shape=box, style=\"rounded,filled\", fontname=\"DejaVu Sans\",",
            "        fontcolor=\"#0F172A\", penwidth=1.3, margin=\"0.20,0.14\"];",
            "  edge [color=\"#94A3B8\", penwidth=1.5, arrowsize=0.7,",
            "        fontname=\"DejaVu Sans\", fontsize=10, fontcolor=\"#475569\"];",
        ]
        external = {q: f"i{i}" for i, q in enumerate(self.inputs)}
        for q, node in external.items():
            caption = card("EXTERNAL INPUT", self._graph_quantity_label(q), shape(q), "#2563EB")
            lines.append(f'  {node} [label={caption}, fillcolor="#EFF6FF", color="#93C5FD"];')
        # IDs stay tied to self.models; layout is computed from dependencies.
        for index, model in enumerate(self.models):
            sig = model.signature
            core = tuple(dict.fromkeys(a for side in (sig.inputs, sig.outputs)
                                       for axes in side.values() for a in axes))
            detail = "core axes: " + ", ".join(core) if core else "pointwise / no core sample axes"
            caption = card(f"MODEL {index + 1:02d}", type(model).__name__, detail, "#7C3AED")
            lines.append(f'  m{index} [label={caption}, fillcolor="#F5F3FF", color="#C4B5FD"];')
        for index, derivative in enumerate(self.derivatives):
            plan = self.derivative_plans[derivative]
            detail = f"planned source models: {plan.model_indices}; execution not implemented"
            caption = card("DERIVATIVE REQUEST", derivative.name, detail, "#C2410C")
            lines.append(f'  d{index} [label={caption}, fillcolor="#FFF7ED", color="#FB923C"];')
            for quantity, role in self._derivative_graph_inputs(derivative):
                source = self._source_node(quantity, external)
                caption = quote(f"{role}: {self._graph_quantity_label(quantity)}")
                lines.append(f"  {source} -> d{index} [label={caption}, style=dashed];")
        for index, model in enumerate(self.models):
            for q in model.inputs:
                source = self._source_node(q, external)
                # Input nodes already name the quantity; only internal arrows need it.
                attr = "" if q in external else f" [label={quote(self._graph_quantity_label(q))}]"
                lines.append(f"  {source} -> m{index}{attr};")
        for index, q in enumerate(self.outputs):
            caption = card("RETURNED OUTPUT", self._graph_quantity_label(q), shape(q), "#059669")
            lines.append(f'  o{index} [label={caption}, fillcolor="#ECFDF5", color="#6EE7B7"];')
            lines.append(f'  {self._source_node(q, external)} -> o{index} [color="#34A88A"];')
        if external:
            lines.append("  { rank=same; " + "; ".join(external.values()) + "; }")
        lines.append("  { rank=same; " + "; ".join(f"o{i}" for i in range(len(self.outputs))) + "; }")
        lines.append("}")
        return "\n".join(lines)

    def render_graph(
        self, filename="solver_graph.svg", *, direction: str = "LR",
        title: str = "Solver computation graph",
    ):
        """Save .svg/.png/.pdf using the Graphviz 'dot' executable; return Path.

        Install the Graphviz system application first (not just its Python package).
        SVG is recommended for zooming; PNG is rendered at 180 dpi.
        Example: solver.render_graph("solver_graph.svg")
        """
        from pathlib import Path
        import shutil
        import subprocess

        path = Path(filename).expanduser()
        fmt = path.suffix.lower().lstrip(".")
        if fmt not in {"svg", "png", "pdf"}:
            raise ValueError("filename must end in .svg, .png, or .pdf")
        source = self.to_dot(direction=direction, title=title)
        executable = shutil.which("dot")
        if executable is None:
            raise RuntimeError(
                "Graphviz 'dot' was not found. Install Graphviz and add its bin folder "
                "to PATH. On Colab/Ubuntu: !apt-get -qq install graphviz. "
                "You can still use solver.to_mermaid() without Graphviz."
            )
        result = subprocess.run(
            [executable, f"-T{fmt}", "-Gdpi=180"], input=source.encode("utf-8"),
            capture_output=True, check=False,
        )
        if result.returncode:
            raise RuntimeError("Graphviz could not render the graph: "
                               + result.stderr.decode("utf-8", errors="replace"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(result.stdout)
        return path

def _example() -> None:
    """A trainable pointwise model followed by a loss over all sample axes."""
    x, prediction, loss = map(Quantity, ("x", "prediction", "loss"))

    class Predict(Model):
        geometry = None
        signature = Signature({x: ()}, {prediction: ()})

        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))

        def evaluate(self, values):
            return {prediction: self.network(values[x].unsqueeze(-1)).squeeze(-1)}

    class Loss(Model):
        geometry = None
        signature = Signature({prediction: ("scenario", "point")}, {loss: ()})

        def evaluate(self, values):
            return {loss: values[prediction].square().mean()}

    solver = Solver([Loss(), Predict()], outputs=loss)
    result = solver({x: Value(torch.rand(3, 2), ("scenario", "point"))})
    result[loss].tensor.backward()
    print("Loss shape:", tuple(result[loss].tensor.shape))
    print(solver.to_mermaid())
    print("Saved graph:", solver.render_graph("solver_graph.svg"))
    print("Saved graph:", solver.render_graph("solver_graph.png"))


if __name__ == "__main__":
    _example()
