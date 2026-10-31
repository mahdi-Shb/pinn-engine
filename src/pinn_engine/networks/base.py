from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import torch

from pinn_engine.model import Model, Signature
from pinn_engine.quantities import Quantity
from pinn_engine.values import Value

from .configs import BaseModelConfig
from .types import ModelCapabilities, ModelInput, ModelOutput


class BaseNetwork(Model, ABC):
    """Native named-quantity model with the original numeric prediction API."""

    config_class = BaseModelConfig
    # Dropout draws independently for automatically mapped batch members.
    vmap_randomness = "different"

    def __init__(
        self,
        config: BaseModelConfig | None = None,
        *,
        inputs: Mapping[str, Quantity],
        outputs: Mapping[str, Quantity],
    ):
        super().__init__()
        self.config = config or self.config_class()
        from ._contract import initialize_contract

        initialize_contract(self, inputs, outputs)

    @property
    def signature(self) -> Signature:
        """Derive axes and independence from the actual network modules."""
        from ._contract import contract_signature

        return contract_signature(self)

    def evaluate(
        self, values: Mapping[Quantity, torch.Tensor | list[torch.Tensor] | Value]
    ) -> dict[Quantity, torch.Tensor | list[torch.Tensor] | Value]:
        from ._contract import evaluate_contract

        return evaluate_contract(self, values)

    def forward(self, values: Mapping[Quantity, Value]) -> dict[Quantity, Value]:
        """Validate tensor data before entering automatic vectorization."""
        from ._contract import validate_contract

        validate_contract(self, values)
        return super().forward(values)

    @property
    @abstractmethod
    def required_inputs(self) -> set[str]:
        raise NotImplementedError

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            representation=self.config.representation,
            requires=set(self.required_inputs),
            supports=set(),
            variable_geometry=False,
            variable_node_count=False,
        )

    @property
    def supported_representations(self) -> set[str]:
        rep = self.capabilities.representation
        return {rep} if rep is not None else set()

    def validate_inputs(self, inputs: ModelInput) -> bool:
        if not isinstance(inputs, ModelInput):
            raise TypeError(f"Expected ModelInput, got {type(inputs).__name__}.")

        required = set(self.required_inputs)
        available = inputs.available_fields()
        missing = required - available
        if missing:
            missing_fields = ", ".join(sorted(missing))
            raise ValueError(
                f"{self.__class__.__name__} requires inputs: {missing_fields}. "
                f"Available: {sorted(available)}."
            )

        rep = self.capabilities.representation
        if rep is not None and inputs.representation is not None and inputs.representation != rep:
            raise ValueError(
                f"Model expects representation '{rep}' but received '{inputs.representation}'."
            )

        return True

    @abstractmethod
    def _predict(self, inputs: ModelInput) -> ModelOutput:
        raise NotImplementedError

    def predict(self, inputs: ModelInput) -> ModelOutput:
        """Run the numeric API, retaining auxiliary and structured outputs."""
        self.validate_inputs(inputs)
        return self._predict(inputs)

    @staticmethod
    def _ensure_tensor(value: Any, *, name: str = "tensor") -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value, dtype=torch.float32)
