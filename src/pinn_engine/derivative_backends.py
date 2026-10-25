"""Small execution interface for numerical derivative methods.

A backend declares the values it needs and optionally prepares method-specific
metadata. Model-path plans, tapes, grids, and meshes are never mandatory here.
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

    def dependencies(self, graph, request):
        """Default to sampled output/input values; add mesh/grid data as needed.

        Callable-based methods may need different values, including held inputs.
        Requirements are registered as graph edges before ordering the graph.
        """
        return tuple(dict.fromkeys((request.of, request.wrt)))

    def prepare(self, graph, requests):
        """Optional static preparation, owned by this method; no paths by default.

        Return a metadata mapping. Values need not be DerivativePlan objects.
        This is called after graph ordering, without evaluating models.
        """
        return {}

    def path_plans(self, preparation):
        """Optional inspection view for path-aware methods; empty otherwise."""
        return {}

    def validate(self, graph, request, values, *, preparation=None):
        """Optionally reject requests this backend cannot interpret correctly."""

    def describe_request(self, request, *, preparation):
        """Supply diagram details without requiring model-path metadata."""
        return self.description

    @abstractmethod
    def execute(self, graph, requests, needed, values, run_model, store, chunk_size, *, preparation=None):
        """Compute requests and their consumers through the host callbacks.

        requests contains raw Derivative declarations. preparation contains only
        what this backend's optional prepare returned, possibly an empty mapping.
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
