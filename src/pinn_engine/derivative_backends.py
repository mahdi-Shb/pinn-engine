"""Value-based numerical methods; no model-path plans or forward tapes."""

from abc import ABC, abstractmethod
from contextlib import nullcontext


class DerivativeBackend(ABC):
    """Compute one request from declared values, independently of Solver."""

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
        """Cover this call's model and derivative operations; no grad by default."""
        return nullcontext()

    def prepare_target(self, value, *, separate=False):
        """Optionally prepare a wrt value before any consumer uses it."""
        return value

    def validate(self, graph, request, values):
        """Optional capability/layout checks, without constructing a plan."""

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
