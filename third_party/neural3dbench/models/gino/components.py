from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class KernelIntegralTransform(nn.Module):
    """Compact-support learned kernel integral with bounded pairwise workspace."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int,
        radius: float,
        chunk_size: int,
        max_pairs: int,
    ) -> None:
        super().__init__()
        self.source_projection = nn.Linear(input_channels, output_channels)
        self.kernel = nn.Sequential(
            nn.Linear(3, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )
        self.radius = radius
        self.chunk_size = chunk_size
        self.max_pairs = max_pairs

    def forward(
        self,
        source_coordinates: torch.Tensor,
        target_coordinates: torch.Tensor,
        source_features: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        *,
        use_checkpoint: bool = True,
    ) -> torch.Tensor:
        """Integrate valid sources without data-dependent Tensor indexing.

        A mask keeps the source shape fixed so independently batched calls can
        be transformed with ``vmap``. Invalid source coordinates and features
        may contain padding values, including NaNs, and do not contribute.
        Callers using functional transforms disable checkpointing explicitly;
        PyTorch checkpoint's saved-tensor hooks do not support those transforms.
        """
        if source_mask is not None:
            if source_mask.shape != source_coordinates.shape[:1]:
                raise ValueError("source_mask must have shape [N].")
            if source_mask.dtype != torch.bool:
                raise TypeError("source_mask must use a boolean dtype.")
            source_coordinates = torch.where(
                source_mask.unsqueeze(-1), source_coordinates, 0.0
            )
            source_features = torch.where(
                source_mask.unsqueeze(-1), source_features, 0.0
            )

        if source_coordinates.shape[0] == 0 or target_coordinates.shape[0] == 0:
            return source_features.new_zeros(
                (target_coordinates.shape[0], self.source_projection.out_features)
            )

        projected = self.source_projection(source_features)
        source_count = source_coordinates.shape[0]
        query_chunk_size = min(
            self.chunk_size,
            max(1, int(math.sqrt(self.max_pairs))),
            max(1, self.max_pairs // source_count),
        )
        source_chunk_size = max(1, self.max_pairs // query_chunk_size)
        outputs = []

        for query_start in range(0, target_coordinates.shape[0], query_chunk_size):
            query = target_coordinates[
                query_start : query_start + query_chunk_size
            ]

            def integrate(query_chunk, sources, values, mask):
                return self._integrate_query_chunk(
                    query_chunk, sources, values, source_chunk_size, mask
                )

            if use_checkpoint and torch.is_grad_enabled():
                output = checkpoint(
                    integrate, query, source_coordinates, projected, source_mask,
                    use_reentrant=False,
                )
            else:
                output = integrate(query, source_coordinates, projected, source_mask)
            outputs.append(output)

        return torch.cat(outputs, dim=0)

    def _integrate_query_chunk(
        self,
        query: torch.Tensor,
        source_coordinates: torch.Tensor,
        projected: torch.Tensor,
        source_chunk_size: int,
        source_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        numerator = projected.new_zeros((query.shape[0], projected.shape[-1]))
        denominator = projected.new_zeros((query.shape[0], 1))
        for source_start in range(0, source_coordinates.shape[0], source_chunk_size):
            source_end = source_start + source_chunk_size
            source = source_coordinates[source_start:source_end]
            delta = query[:, None, :] - source[None, :, :]
            distance = torch.linalg.vector_norm(delta, dim=-1)
            cutoff = (distance < self.radius).to(distance.dtype)
            cutoff = cutoff * 0.5 * (1.0 + torch.cos(math.pi * distance / self.radius))
            learned_kernel = torch.nn.functional.softplus(
                self.kernel(delta.reshape(-1, 3)).reshape_as(distance)
            )
            weights = cutoff * learned_kernel
            if source_mask is not None:
                weights = weights * source_mask[source_start:source_end].unsqueeze(0)
            numerator = numerator + weights @ projected[source_start:source_end]
            denominator = denominator + weights.sum(dim=1, keepdim=True)
        return numerator / denominator.clamp_min(1e-8)
