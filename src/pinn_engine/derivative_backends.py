"""Small execution interface for numerical derivative methods.

Declarations and graph plans do not choose an algorithm. A backend owns its
runtime requirements, capability checks, and execution against that graph.
"""

from abc import ABC, abstractmethod


class DerivativeBackend(ABC):
    """Numerical execution, without requiring an autograd tape or grad mode."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable method name, also used in graph descriptions."""
        raise NotImplementedError

    @property
    def description(self) -> str:
        return self.name

    def validate(self, graph, plan, values):
        """Optionally reject requests this backend cannot interpret correctly."""

    @abstractmethod
    def execute(self, graph, plans, needed, values, run_model, store, chunk_size):
        """Compute requests and their consumers through the host callbacks.

        run_model(index, call=None) executes and validates a model; an optional
        call adapter can record its inputs. store(mapping) validates and stores
        computed Values. Neither callback exposes the hosting Solver.
        """
        raise NotImplementedError


def resolve_backend(backend):
    """Resolve the implemented method or an explicitly supplied backend object."""
    if isinstance(backend, DerivativeBackend):
        return backend
    if not isinstance(backend, str):
        raise TypeError("derivative_backend must be a method name or a DerivativeBackend")
    if backend == "autograd":
        from ._autograd import AutogradBackend
        return AutogradBackend()
    raise ValueError(f"Unknown or unimplemented derivative backend {backend!r}; available: 'autograd'")
