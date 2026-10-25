"""PyTorch model host with graph validation and delegated derivative execution."""

from typing import Mapping, Sequence
from torch import nn

from .quantities import Quantity
from .model import Model
from .values import Value, _check_value
from ._graph import ComputationGraph
from .derivatives import DerivativeEngine
from .derivative_backends import DerivativeBackend
from . import _visualization


class Solver(nn.Module):
    """Register models and execute selected outputs from their declared graph.

    ComputationGraph owns dependency metadata. DerivativeEngine owns virtual
    producers and derivative paths; the selected backend owns numerical execution.
    Solver only checks
    runtime values, invokes modules through their normal hooks, and stores results.
    """

    def __init__(
        self, models: Sequence[Model],
        outputs: Quantity | Sequence[Quantity] | None = None,
        *, derivative_backend: str | DerivativeBackend = "autograd",
        derivative_chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self._derivative_engine = DerivativeEngine(derivative_chunk_size, backend=derivative_backend)
        self._graph = ComputationGraph(models)
        requested = None if outputs is None else (
            (outputs,) if isinstance(outputs, Quantity) else tuple(outputs)
        )
        self._derivative_engine.install(self._graph, requested)
        self._graph.finalize(requested)
        self._derivative_engine.build_plans(self._graph)
        # Keep existing state_dict names and ordinary nn.Module behavior.
        self.models = self._graph.models
        for index, model in enumerate(self.models):
            self.add_module(f"part_{index}", model)

    @property
    def inputs(self):
        return self._graph.inputs

    @property
    def outputs(self):
        return self._graph.outputs

    @property
    def geometry(self):
        return self._graph.geometry

    @property
    def derivatives(self):
        return self._derivative_engine.requests

    @property
    def derivative_plans(self):
        return self._derivative_engine.plans

    @property
    def derivative_backend(self):
        return self._derivative_engine.backend

    @property
    def derivative_chunk_size(self):
        return self._derivative_engine.chunk_size

    @derivative_chunk_size.setter
    def derivative_chunk_size(self, size):
        self._derivative_engine.chunk_size = size

    def validate_autograd(self, plan, values):
        """Compatibility alias for the selected backend capability check."""
        self._derivative_engine.validate(self._graph, plan, values)

    def forward(
        self, values: Mapping[Quantity, Value], *,
        outputs: Quantity | Sequence[Quantity] | None = None,
    ) -> dict[Quantity, Value]:
        if set(values) != set(self.inputs):
            raise ValueError(f"Expected input quantities: {[q.name for q in self.inputs]}")
        selected = self._graph.select_outputs(outputs)
        pool, sizes = {}, {}
        self._store(values, pool, sizes)
        needed = self._graph.needed_nodes(selected)

        def run_model(index, call=None):
            model = self.models[index]
            inputs = {q: pool[q] for q in model.inputs}
            result = model(inputs) if call is None else call(index, model, inputs)
            if set(result) != set(model.outputs):
                raise ValueError("Model returned quantities different from its signature")
            self._store(result, pool, sizes)

        handled = self._derivative_engine.execute(
            self._graph, needed, pool, run_model,
            lambda result: self._store(result, pool, sizes),
        )
        if not handled:
            for node in self._graph.ordered:
                if node in needed:
                    run_model(node[1])
        return {quantity: pool[quantity] for quantity in selected}

    def _store(self, values, pool, sizes):
        """One validation path for external, model, and virtual-node values."""
        for quantity, value in values.items():
            if quantity.shape != self._graph.quantities[quantity].shape:
                raise ValueError(f"Conflicting shapes for {quantity.name!r}")
            _check_value(quantity, value, sizes)
            pool[quantity] = value

    def to_mermaid(self, direction="LR"):
        return _visualization.to_mermaid(self._graph, self._derivative_engine, direction)

    def to_dot(self, direction="LR", title="Solver computation graph"):
        return _visualization.to_dot(self._graph, self._derivative_engine, direction, title)

    def render_graph(self, filename="solver_graph.svg", *, direction="LR", title="Solver computation graph"):
        return _visualization.render_graph(
            self._graph, self._derivative_engine, filename, direction=direction, title=title,
        )

    # Existing notebooks inspect these metadata views. They remain aliases of
    # the shared graph rather than duplicated state or derivative logic.
    @property
    def _ordered(self):
        return self._graph.ordered

    @property
    def _node_dependencies(self):
        return self._graph.dependencies

    @property
    def _producers(self):
        return self._graph.model_producers

    @property
    def _quantities(self):
        return self._graph.quantities

    def _producer_node(self, quantity):
        return self._graph.producer_node(quantity)

    def _needed_nodes(self, outputs):
        return self._graph.needed_nodes(outputs)

    def _source_node(self, quantity, external):
        return _visualization.source_node(self._graph, self._derivative_engine, quantity, external)

    def _graph_quantity_label(self, quantity):
        return _visualization.quantity_label(self._graph, quantity)

    @staticmethod
    def _derivative_graph_inputs(derivative):
        return DerivativeEngine.graph_inputs(derivative)
