"""Dependency metadata shared by execution, derivative planning, and rendering.

This module knows model nodes and generic virtual producers. It does not import
derivatives or autograd, own runtime Tensors, or register PyTorch parameters.
"""

from .model import Model
from .quantities import Quantity


class ComputationGraph:
    """Build and query the declared DAG without executing any model."""

    def __init__(self, models):
        self.models = tuple(models)
        if not self.models or any(not isinstance(model, Model) for model in self.models):
            raise ValueError("Provide at least one Model")
        self.quantities = {}
        self.producers = {}
        self.model_producers = {}
        self.dependencies = {}
        self.node_outputs = {}
        self.geometry = None
        for index, model in enumerate(self.models):
            if not model.outputs:
                raise ValueError("Each connected model must declare an output")
            self.add_node(("model", index), model.inputs, model.outputs)
            self.model_producers.update(dict.fromkeys(model.outputs, index))
            # Geometry subset policy is still open; preserve object identity.
            if model.geometry is not None:
                if self.geometry is not None and model.geometry is not self.geometry:
                    raise ValueError("Connected models must share one geometry object")
                self.geometry = model.geometry
        self.default_outputs = tuple(self.producers)

    def register_quantity(self, quantity):
        if not isinstance(quantity, Quantity):
            raise TypeError("Expected Quantity objects")
        previous = self.quantities.get(quantity)
        if previous is not None and previous.shape != quantity.shape:
            raise ValueError(f"Conflicting shapes for {quantity.name!r}")
        self.quantities[quantity] = quantity

    def add_node(self, node, inputs, outputs):
        """Register a producer; virtual-node policy belongs to its provider."""
        inputs, outputs = tuple(inputs), tuple(outputs)
        for quantity in (*inputs, *outputs):
            self.register_quantity(quantity)
        for quantity in outputs:
            if quantity in self.producers:
                raise ValueError(f"Multiple producers for {quantity.name!r}")
            self.producers[quantity] = node
        self.dependencies[node] = inputs
        self.node_outputs[node] = outputs

    def finalize(self, outputs):
        required = dict.fromkeys(q for inputs in self.dependencies.values() for q in inputs)
        self.inputs = tuple(q for q in required if q not in self.producers)
        self.outputs = self.select_outputs(self.default_outputs if outputs is None else outputs)
        available, pending = set(self.inputs), dict(self.dependencies)
        ordered = []
        while pending:
            ready = [node for node, inputs in pending.items() if set(inputs) <= available]
            if not ready:
                raise ValueError("Cyclic model dependencies; use distinct input/output quantities")
            for node in ready:
                ordered.append(node)
                available.update(self.node_outputs[node])
                del pending[node]
        self.ordered = tuple(ordered)

    def select_outputs(self, outputs=None):
        selected = self.outputs if outputs is None else (
            (outputs,) if isinstance(outputs, Quantity) else tuple(outputs)
        )
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Select at least one distinct forward output")
        for quantity in selected:
            if not isinstance(quantity, Quantity):
                raise TypeError("Solver outputs must be Quantity objects")
            if quantity not in self.producers:
                raise ValueError(f"Unknown forward output: {quantity!r}")
            if quantity.shape != self.quantities[quantity].shape:
                raise ValueError(f"Conflicting shapes for {quantity.name!r}")
        return tuple(selected)

    def producer_node(self, quantity):
        return self.producers.get(quantity)

    def needed_nodes(self, outputs):
        """Find ancestors of outputs; node-specific semantics are not required."""
        needed = set()

        def visit(quantity):
            node = self.producer_node(quantity)
            if node is None or node in needed:
                return
            needed.add(node)
            for dependency in self.dependencies[node]:
                visit(dependency)

        for quantity in outputs:
            visit(quantity)
        return needed
