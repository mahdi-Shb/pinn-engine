"""Adapt ragged point batches to the existing dense PointNet algorithms.

Padding is an internal representation only: original pooling masks are used
and the shared normalization module excludes invalid spatial samples from BN.
No encoder, decoder, T-Net, or neighborhood algorithm is duplicated here.
"""

from math import prod

import torch
from torch.nn import functional as F

from ..normalization import normalization_mask, require_mask_aware_batch_norm
from ..values import Value
from ._contract import _model_input, _outputs, _specs


def evaluate_ragged_points(model, batch, groups):
    require_mask_aware_batch_norm(model)
    # Each group may contain additional dense scenario axes. Flatten them into
    # B as usual and pad only the point axis, never intrinsic components.
    maximum = max(raw["coordinates"].shape[1] for _, raw in groups)
    padded, masks = {}, []
    for _, raw in groups:
        count, points = raw["coordinates"].shape[:2]
        mask = raw.get("point_mask")
        mask = (torch.ones((count, points), dtype=torch.bool, device=raw["coordinates"].device)
                if mask is None else mask.squeeze(-1))
        masks.append(F.pad(mask, (0, maximum - points), value=False))
        for role, tensor in raw.items():
            if role == "point_mask":
                continue
            padded.setdefault(role, []).append(F.pad(tensor, (0, 0, 0, maximum - points)))
    raw = {role: torch.cat(tensors) for role, tensors in padded.items()}
    mask = torch.cat(masks)
    raw["point_mask"] = mask.unsqueeze(-1)
    # The network still runs once. Post-pooling BN sees all real B embeddings.
    with normalization_mask(mask.to(next(model.parameters()).device)):
        result = model._predict(_model_input(model, raw))
    flat = _outputs(model, result, (mask.shape[0],))
    specs = _specs(model)[1]
    outputs = {}
    for role, quantity in model.output_quantities.items():
        members, cursor = [], 0
        for shape, original in groups:
            count = prod(shape)
            tensor = flat[quantity][cursor:cursor + count]
            if "point" in specs[role][0]:
                tensor = tensor[:, :original["coordinates"].shape[1]]
            members.append(tensor.reshape(shape + tensor.shape[1:]))
            cursor += count
        outputs[quantity] = Value(members, batch + specs[role][0])
    return outputs
