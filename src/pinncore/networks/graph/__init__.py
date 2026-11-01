from __future__ import annotations

from ..registry import ModelRegistry

from .gat import GAT, GATConfig
from .gcn import GCN, GCNConfig
from .meshgraphnet import MeshGraphNet, MeshGraphNetConfig

ModelRegistry.register("gcn", GCN, GCNConfig)
ModelRegistry.register("gat", GAT, GATConfig)
ModelRegistry.register("meshgraphnet", MeshGraphNet, MeshGraphNetConfig)

__all__ = [
    "GAT",
    "GATConfig",
    "GCN",
    "GCNConfig",
    "MeshGraphNet",
    "MeshGraphNetConfig",
]
