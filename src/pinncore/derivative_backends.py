"""Value-based numerical methods; no model-path plans or forward tapes."""

from abc import ABC, abstractmethod
from contextlib import nullcontext


class DerivativeBackend(ABC):
    """Compute requests from declared values, independently of Solver.

    Grouping is method-specific: reverse directions, forward tangents, and mesh
    operators have different reuse criteria. The default keeps requests separate.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    def description(self) -> str:
        return self.name

    def dependencies(self, graph, request):
        """Add mesh/connectivity/grid quantities here, including model outputs.

        An analytical method may also omit an unnecessary primal output.
        """
        return tuple(dict.fromkeys((request.of, request.wrt)))

    def execution_context(self):
        """Cover model and derivative operations; preserve the current mode by default."""
        return nullcontext()

    def prepare_target(self, value, *, separate=False):
        """Optionally prepare a wrt value before any consumer uses it."""
        return value

    def validate(self, graph, request, values):
        """Optional capability/layout checks, without constructing a plan."""

    def group_key(self, request, values):
        """Return a hashable reuse key, or None to keep this request separate.

        values contains only this request's declared dependencies. Equal keys
        mean compute_group can handle these ready requests together. Include
        every relevant layout, operator, direction, or mesh distinction.
        """
        return None

    def prepare_group(self, requests, inputs):
        """Optionally build reusable numeric data for this key, e.g. a mesh operator.

        The engine retains it only for this Solver call, including later ready
        frontiers with the same key. Include all preparation dependencies in the
        key; do not assume every field or derivative order is already available.
        This is runtime method data, never a model-path derivative plan.
        """
        return None

    def compute_group(self, requests, inputs, *, chunk_size, prepared=None):
        """Return a request-to-Value mapping; default to separate computation.

        inputs maps EACH request to its own dependency-value mapping. A method
        opting into reuse overrides this together with group_key. prepared is
        its own optional prepare_group result. Only selected
        requests and their required ancestors are supplied, never unrelated work.
        """
        return {request: self.compute(request, inputs[request], chunk_size=chunk_size)
                for request in requests}

    @abstractmethod
    def compute(self, request, values, *, chunk_size):
        """Return one Value, receiving only the declared dependency values.

        AD needs connected of/wrt tensors, not just numerical data. A mesh
        method receives the extra values it declared. The engine handles all
        model execution, ordering, and storing this result.
        """
        raise NotImplementedError


def resolve_backend(backend):
    if isinstance(backend, DerivativeBackend):
        return backend
    if not isinstance(backend, str):
        raise TypeError("derivative_backend must be a method name or a DerivativeBackend")
    if backend == "autograd":
        from ._autograd import AutogradBackend
        return AutogradBackend()
    raise ValueError(f"Unknown or unimplemented derivative backend {backend!r}; available: 'autograd'")
