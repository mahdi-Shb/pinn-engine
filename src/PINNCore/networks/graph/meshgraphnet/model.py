from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from PINNCore.quantities import Quantity
from PINNCore.model import Model

import torch

from ...configs import BaseModelConfig
from ..._contract import initialize_contract, contract_signature, prepare_inputs, finish_outputs

from .decoder import MeshGraphNetDecoder
from .encoder import MeshGraphNetEncoder
from .processor import MeshGraphNetProcessor


@dataclass
class MeshGraphNetConfig(BaseModelConfig):
    node_input_dim: int = 1
    edge_input_dim: int = 1
    output_dim: int = 1
    latent_dim: int = 128
    processor_steps: int = 15
    mlp_hidden_dim: int = 128
    mlp_layers: int = 2
    activation: str = "relu"
    normalization: str = "layer_norm"
    residual: bool = True
    representation: str = "GRAPH"

    def __post_init__(self) -> None:
        self.name = "meshgraphnet"
        self.model_type = "meshgraphnet"
        self.representation = "GRAPH"
        self._validate()

    def _validate(self) -> None:
        for name in (
            "node_input_dim",
            "edge_input_dim",
            "output_dim",
            "latent_dim",
            "mlp_hidden_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.processor_steps < 1:
            raise ValueError("processor_steps must be at least 1.")
        if self.mlp_layers < 1:
            raise ValueError("mlp_layers must be at least 1.")
        valid_activations = {
            "relu", "gelu", "silu", "elu", "tanh", "identity", "leaky_relu"
        }
        if self.activation.lower() not in valid_activations:
            raise ValueError(
                f"Unknown activation '{self.activation}'. "
                f"Available: {sorted(valid_activations)}"
            )
        valid_normalizations = {"layer_norm", "none"}
        if self.normalization.lower() not in valid_normalizations:
            raise ValueError(
                f"Unknown normalization '{self.normalization}'. "
                f"Available: {sorted(valid_normalizations)}"
            )


class MeshGraphNet(Model):
    """Encode-process-decode graph network for per-node physical predictions.

    Architectural decisions follow Pfaff et al. (ICLR 2021), DeepMind's
    ``meshgraphnets/core_model.py``, and NVIDIA PhysicsNeMo's maintained MGN:
    separate node/edge encoders; independent edge-first interaction blocks;
    receiver-wise sum aggregation; node and edge residual updates; and an
    unnormalized node decoder. Data normalization, graph construction, target
    semantics, training noise, physics losses, and rollout remain external.
    """

    config_class = MeshGraphNetConfig
    vmap_randomness = "different"

    @property
    def signature(self):
        return contract_signature(self)

    def __init__(
        self,
        config: MeshGraphNetConfig | None = None,
        *,
        inputs: Mapping[str, Quantity],
        outputs: Mapping[str, Quantity],
    ) -> None:
        super().__init__()
        self.config = config or MeshGraphNetConfig()
        initialize_contract(self, inputs, outputs)
        common = {
            "mlp_hidden_dim": self.config.mlp_hidden_dim,
            "mlp_layers": self.config.mlp_layers,
            "activation": self.config.activation,
        }
        self.encoder = MeshGraphNetEncoder(
            self.config.node_input_dim,
            self.config.edge_input_dim,
            self.config.latent_dim,
            normalization=self.config.normalization,
            **common,
        )
        self.processor = MeshGraphNetProcessor(
            self.config.latent_dim,
            self.config.processor_steps,
            normalization=self.config.normalization,
            residual=self.config.residual,
            **common,
        )
        self.decoder = MeshGraphNetDecoder(
            self.config.latent_dim,
            self.config.output_dim,
            **common,
        )

    def evaluate(self, values):
        requests, layout = prepare_inputs(self, values)
        results = []
        for inputs in requests:
            node_features = inputs["node_features"]
            edge_features = inputs["edge_features"]
            edge_index = inputs["edge_index"]
            if not node_features.is_floating_point():
                raise TypeError("node_features must use a floating-point dtype.")
            if node_features.shape[1] != self.config.node_input_dim:
                raise ValueError(
                    "MeshGraphNet expects node feature dimension "
                    f"{self.config.node_input_dim}, got {node_features.shape[1]}."
                )
            if edge_features.dim() != 2:
                raise ValueError(
                    f"edge_features must have shape [E, F_edge], got {tuple(edge_features.shape)}."
                )
            if not edge_features.is_floating_point():
                raise TypeError("edge_features must use a floating-point dtype.")
            if edge_features.shape[0] != edge_index.shape[1]:
                raise ValueError(
                    "edge_features and edge_index must describe the same number of edges: "
                    f"got {edge_features.shape[0]} and {edge_index.shape[1]}."
                )
            if edge_features.shape[1] != self.config.edge_input_dim:
                raise ValueError(
                    "MeshGraphNet expects edge feature dimension "
                    f"{self.config.edge_input_dim}, got {edge_features.shape[1]}."
                )
            parameter = next(self.parameters())
            node_features = node_features.to(device=parameter.device, dtype=parameter.dtype)
            edge_features = edge_features.to(device=parameter.device, dtype=parameter.dtype)
            edge_index = edge_index.to(device=parameter.device, dtype=torch.long)

            node_latent, edge_latent = self.encoder(node_features, edge_features)
            node_latent, edge_latent = self.processor(
                node_latent, edge_latent, edge_index
            )
            predictions = self.decoder(node_latent)
            metadata = {"batch": inputs["batch"]} if inputs.get("batch") is not None else {}
            results.append({
                "predictions": predictions,
                "latent": node_latent,
                "auxiliary": {"edge_latent": edge_latent},
                "metadata": metadata,
            })
        return finish_outputs(self, results, layout)
