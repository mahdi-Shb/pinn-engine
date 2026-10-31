"""Normalization of dense, ragged, and explicitly masked feature batches.

Inputs use PyTorch's channel-first layout: (B, C, *spatial). A list contains
pieces of ONE batch, not independent BatchNorm calls. Each real spatial sample
has equal weight. Dense calls without a mask retain PyTorch's implementation.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn


_mask = ContextVar("normalization_mask", default=None)


@contextmanager
def normalization_mask(mask, *, enabled=True):
    """Scope a spatial mask without threading it through every pointwise layer.

    Dense post-pooling (B, C) layers deliberately do not inherit a spatial mask.
    Disabled scopes preserve the caller's mask. ContextVars isolate concurrent
    calls; no module hooks, monkey patches, or persistent per-batch state exist.
    """
    if not enabled:
        yield
        return
    token = _mask.set(mask)
    try:
        yield
    finally:
        _mask.reset(token)


def has_normalization_mask():
    return _mask.get() is not None


def batch_norms(module):
    """Inspect actual layers rather than configuration spellings."""
    return [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def uses_batch_statistics(module):
    return any(m.training or (m.running_mean is None and m.running_var is None)
               for m in batch_norms(module))


def masked_point_inputs(points, features, mask):
    """Keep invalid inputs out of pointwise convolutions and their gradients."""
    if mask is None:
        return points, features
    valid = mask.unsqueeze(-1)
    return (torch.where(valid, points, 0.),
            None if features is None else torch.where(valid, features, 0.))


def require_mask_aware_batch_norm(module):
    """Do not silently pad a network containing an unadapted native BN."""
    if any(not isinstance(m, _BatchNorm) for m in batch_norms(module)):
        raise TypeError("Ragged point batches require pinn_engine.normalization BatchNorm layers")


def _checked_mask(tensor, mask):
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("Normalization masks must be boolean Tensors")
    if mask.shape != tensor.shape[:1] + tensor.shape[2:]:
        raise ValueError("Mask shape must be (B, *spatial), without the channel axis")
    if mask.device != tensor.device:
        raise ValueError("Normalization mask and features must be on the same device")
    return mask


class _BatchNorm:
    def forward(self, inputs, mask=None):
        if isinstance(inputs, torch.Tensor):
            # The ambient mask belongs to point/spatial layers, not pooled FCs.
            if mask is None and inputs.ndim > 2:
                mask = _mask.get()
            if mask is None:
                return super().forward(inputs)
            return self.forward([inputs], [mask])[0]
        if not isinstance(inputs, list) or not inputs:
            raise TypeError("BatchNorm expects a Tensor or a nonempty flat Tensor list")
        masks = [None] * len(inputs) if mask is None else mask
        if not isinstance(masks, list) or len(masks) != len(inputs):
            raise ValueError("List input requires a matching list of masks")
        rows, valid_masks, lengths = [], [], []
        first = inputs[0]
        for tensor, valid in zip(inputs, masks):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("BatchNorm list members must be Tensors")
            self._check_input_dim(tensor)
            if (tensor.ndim != first.ndim or tensor.shape[1] != first.shape[1]
                    or tensor.dtype != first.dtype or tensor.device != first.device):
                raise ValueError("BatchNorm members must share rank, channels, dtype, and device")
            valid = _checked_mask(tensor, valid)
            flat = tensor.movedim(1, -1).reshape(-1, tensor.shape[1])
            selected = flat if valid is None else flat[valid.reshape(-1)]
            rows.append(selected)
            valid_masks.append(valid)
            lengths.append(selected.shape[0])
        if not sum(lengths):
            raise ValueError("BatchNorm needs at least one real sample")
        # Native BatchNorm owns affine parameters, momentum, biased training
        # variance, unbiased running variance, and ONE buffer update per call.
        packed = torch.cat(rows).T.reshape(1, first.shape[1], -1)
        if first.ndim == 2:
            packed = packed.squeeze(0).T
        else:
            packed = packed.reshape(packed.shape + (1,) * (first.ndim - 3))
        normalized = super().forward(packed).movedim(1, -1).reshape(-1, first.shape[1])
        outputs = []
        for tensor, valid, values in zip(inputs, valid_masks, normalized.split(lengths)):
            shape = tensor.shape[:1] + tensor.shape[2:] + tensor.shape[1:2]
            if valid is not None:
                restored = values.new_zeros((valid.numel(), tensor.shape[1]))
                restored[valid.reshape(-1)] = values
                values = restored
            outputs.append(values.reshape(shape).movedim(-1, 1))
        return outputs


class BatchNorm1d(_BatchNorm, nn.BatchNorm1d):
    """BatchNorm1d with joint list statistics and optional spatial masks."""


class BatchNorm2d(_BatchNorm, nn.BatchNorm2d):
    """BatchNorm2d with joint list statistics and optional spatial masks."""


class BatchNorm3d(_BatchNorm, nn.BatchNorm3d):
    """BatchNorm3d with joint list statistics and optional spatial masks."""


class InstanceNorm1d(nn.InstanceNorm1d):
    """Per-case, per-channel point normalization; supports masks and lists."""

    def forward(self, inputs, mask=None):
        if isinstance(inputs, list):
            if not inputs:
                raise ValueError("InstanceNorm needs a nonempty Tensor list")
            masks = [None] * len(inputs) if mask is None else mask
            if not isinstance(masks, list) or len(masks) != len(inputs):
                raise ValueError("List input requires a matching list of masks")
            return [self(t, m) for t, m in zip(inputs, masks)]
        if mask is None and inputs.ndim == 3:
            mask = _mask.get()
        if mask is None:
            return super().forward(inputs)
        self._check_input_dim(inputs)
        mask = _checked_mask(inputs, mask)
        if self.track_running_stats:
            raise ValueError("Masked InstanceNorm requires track_running_stats=False")
        # Reductions keep their dimensions, so this also works under vmap.
        valid = mask.unsqueeze(1)
        work = inputs.float() if inputs.dtype in (torch.float16, torch.bfloat16) else inputs
        clean = torch.where(valid, work, 0.)
        count = valid.sum(-1, keepdim=True).clamp_min(1)
        mean = clean.sum(-1, keepdim=True) / count
        centered = torch.where(valid, work - mean, 0.)
        variance = centered.square().sum(-1, keepdim=True) / count
        output = centered * torch.rsqrt(variance + self.eps)
        if self.affine:
            output = output * self.weight[None, :, None] + self.bias[None, :, None]
        return output.masked_fill(~valid, 0.).to(inputs.dtype)


def point_normalization(kind, channels, *, pooled=False):
    """PointNet policy: instance normalization ends when point pooling ends."""
    if kind == "batch":
        return BatchNorm1d(channels)
    if kind == "instance" and not pooled:
        return InstanceNorm1d(channels, affine=True, track_running_stats=False)
    if kind in {"none", "instance"}:
        return nn.Identity()
    raise ValueError("normalization must be batch, instance, or none")
