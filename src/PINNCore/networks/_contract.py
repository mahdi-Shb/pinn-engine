"""Named quantities and batch layouts for the native Neural3DBench models.

The numerical networks retain their modules and state_dict names. This module
only translates their channel-first/component conventions to engine Values.
"""

from __future__ import annotations

from math import prod
from typing import Mapping

import torch

from ..model import Signature
from ..quantities import Quantity
from ..values import Value
from ..checks import check
from .normalization import batch_norms, uses_batch_statistics


_GRAPHS = {"gcn", "gat", "meshgraphnet"}
_GRIDS = {"fno3d", "unet3d"}
_SPATIAL = ("z", "y", "x")


def _specs(model):
    """Return role -> (core sample axes, component count), independently of B."""
    c = model.config
    name = c.name
    if name in _GRAPHS:
        width = c.node_input_dim if name == "meshgraphnet" else c.input_dim
        inputs = {"node_features": (("vertex",), width), "edge_index": (("edge",), 2)}
        if name == "meshgraphnet":
            inputs["edge_features"] = (("edge",), c.edge_input_dim)
        outputs = {"predictions": (("vertex",), c.output_dim)}
        if name != "gcn":
            outputs["latent"] = (("vertex",), c.latent_dim if name == "meshgraphnet" else c.hidden_dim)
        return inputs, outputs, set(inputs)
    if name in _GRIDS:
        inputs = {"voxel_fields": (_SPATIAL, c.input_channels)}
        latent_axes = _SPATIAL if name == "fno3d" else tuple("bottleneck_" + a for a in _SPATIAL)
        width = c.width if name == "fno3d" else c.base_channels * c.channel_multiplier ** (c.levels - 1)
        return inputs, {"predictions": (_SPATIAL, c.output_channels), "latent": (latent_axes, width)}, set(inputs)

    spatial = 2 if name == "deep_nurbs" else getattr(c, "spatial_dim", 3)
    inputs = {"coordinates": (("point",), spatial)}
    required = {"coordinates"}
    if name != "deep_nurbs":
        features = c.feature_dim if name in {"pointnet", "pointnet2"} else c.input_dim
        if features:
            inputs["point_features"] = (("point",), features)
            required.add("point_features")
        inputs["point_mask"] = (("point",), 1)
    point_axes = ("point",)
    if name in {"pointnet", "pointnet2"}:
        prediction_axes = () if c.task == "classification" else point_axes
        outputs = {"predictions": (prediction_axes, c.output_dim), "latent": ((), c.global_dim)}
    elif name == "gino":
        inputs["query_coordinates"] = (("query",), 3)
        prediction_axes = ("query",) if "query_coordinates" in model.input_quantities else point_axes
        outputs = {"predictions": (prediction_axes, c.output_dim),
                   "latent": (("latent_z", "latent_y", "latent_x"), c.width)}
    else:
        width = c.output_dim if name == "deep_nurbs" else c.hidden_dim
        outputs = {"predictions": (point_axes, c.output_dim), "latent": (point_axes, width)}
    return inputs, outputs, required


def initialize_contract(model, inputs, outputs):
    """Fix optional roles and check intrinsic component counts at construction."""
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise TypeError("inputs and outputs must map role names to Quantities")
    model.input_quantities = dict(inputs)
    model.output_quantities = dict(outputs)
    input_specs, output_specs, required = _specs(model)
    missing = required - set(inputs)
    if missing:
        raise ValueError(f"Missing required inputs: {sorted(missing)}")
    if not outputs:
        raise ValueError("Select at least one numeric output")
    for side, bindings, specs in (("inputs", inputs, input_specs), ("outputs", outputs, output_specs)):
        unknown = set(bindings) - set(specs)
        if unknown:
            raise ValueError(f"Unsupported {side}: {sorted(unknown)}")
        if any(not isinstance(q, Quantity) for q in bindings.values()):
            raise TypeError(f"{side} must contain Quantity objects")
        if len(set(bindings.values())) != len(bindings):
            raise ValueError(f"Each {side} role needs a distinct Quantity")
        for role, q in bindings.items():
            if prod(q.shape) != specs[role][1]:
                raise ValueError(f"{side} {role!r}: Quantity shape {q.shape} needs {specs[role][1]} components")
    if set(inputs.values()) & set(outputs.values()):
        raise ValueError("Input and output Quantities must be distinct")
    # Also reject different intrinsic shapes under one semantic Quantity name.
    Signature({q: input_specs[r][0] for r, q in inputs.items()},
              {q: output_specs[r][0] for r, q in outputs.items()})


def contract_signature(model):
    inputs, outputs, _ = _specs(model)
    whole_batch = model.config.name in _GRAPHS or bool(batch_norms(model))
    prefix = ("...",) if whole_batch else ()
    incoming = {q: prefix + inputs[r][0] for r, q in model.input_quantities.items()}
    outgoing = {q: prefix + outputs[r][0] for r, q in model.output_quantities.items()}
    shared = {a for axes in incoming.values() for a in axes} & {a for axes in outgoing.values() for a in axes}
    independent = model.config.name in {"siren", "deep_nurbs"}
    rules = {a: independent for a in shared if a != "..."}
    if "query" in shared:
        # GINO queries do not feed its source encoder or one another.
        rules["query"] = True
    if "..." in shared:
        # Frozen running statistics do not couple distinct batch members.
        rules["..."] = not uses_batch_statistics(model)
    return Signature(incoming, outgoing, rules)


def _prepared_batches(model, values):
    """Broadcast by axis name, flatten dense batches, retain outer ragged groups.

    Each returned group has one upstream B dimension. No padding is introduced.
    Missing named batch axes are shared, never matched by numerical size alone.
    """
    if model.config.name in {"pointnet", "pointnet2"} and batch_norms(model):
        if any(isinstance(value.tensor, list) for value in values.values()):
            raise ValueError(
                f"{type(model).__name__} with BatchNorm requires Tensor inputs; "
                "list inputs are not supported and are not padded automatically"
            )
    specs, output_specs, _ = _specs(model)
    core = {a for roles, definitions in ((model.input_quantities, specs), (model.output_quantities, output_specs))
            for role in roles for a in definitions[role][0]}
    batch = tuple(dict.fromkeys(a for q in model.input_quantities.values()
                               for a in values[q].axes if a not in core))
    list_axes = {v.axes[0] for v in values.values() if isinstance(v.tensor, list)}
    if len(list_axes) > 1:
        raise ValueError("Neural3DBench supports one outer list batch axis per call")
    list_axis = next(iter(list_axes), None)
    if list_axis is not None and (not batch or batch[0] != list_axis):
        raise ValueError("The list axis must be the first batch axis")
    count = next(len(v.tensor) for v in values.values() if isinstance(v.tensor, list)) if list_axis else 1
    groups = []
    for index in range(count):
        selected = {}
        for q, value in values.items():
            axes, tensor = value.axes, value.tensor
            if list_axis in axes:
                position = axes.index(list_axis)
                tensor = tensor[index] if isinstance(tensor, list) else tensor.select(position, index)
                axes = tuple(a for a in axes if a != list_axis)
            selected[q] = (tensor, axes)
        dense_axes = tuple(a for a in batch if a != list_axis)
        sizes = {a: t.shape[axes.index(a)] for t, axes in selected.values() for a in axes}
        shape = tuple(sizes[a] for a in dense_axes)
        if any(n == 0 for n in shape):
            raise ValueError("Batch axes must be nonempty")
        raw = {}
        for role, q in model.input_quantities.items():
            tensor, axes = selected[q]
            wanted = specs[role][0]
            if any(a not in axes for a in wanted):
                raise ValueError(f"Missing core axes for {role!r}: {wanted}")
            if set(axes) - set(dense_axes) != set(wanted):
                raise ValueError(f"Unexpected core axes for {role!r}")
            present = tuple(a for a in dense_axes if a in axes)
            order = present + wanted
            permutation = tuple(axes.index(a) for a in order) + tuple(range(len(axes), tensor.ndim))
            tensor = tensor.permute(permutation)
            for dim, axis in enumerate(dense_axes):
                if axis not in present:
                    tensor = tensor.unsqueeze(dim)
            spatial_shape = tuple(sizes[a] for a in wanted)
            tensor = tensor.expand(shape + spatial_shape + q.shape)
            raw[role] = tensor.reshape((prod(shape),) + spatial_shape + (specs[role][1],))
        groups.append((shape, raw))
    return batch, list_axis, groups



def _native_input(model, raw):
    """Translate engine component layouts to the original numeric layouts."""
    raw = dict(raw)
    if model.config.name in _GRIDS:
        raw["voxel_fields"] = raw["voxel_fields"].movedim(-1, 1)
    if "point_mask" in raw:
        raw["point_mask"] = raw["point_mask"].squeeze(-1)
    return raw


def _outputs(model, result, batch_shape):
    """Restore intrinsic Quantity shapes without folding sample axes into them."""
    converted = {}
    for role, q in model.output_quantities.items():
        tensor = result[role]
        if model.config.name in _GRIDS or (model.config.name == "gino" and role == "latent"):
            tensor = tensor.movedim(1, -1)
        converted[q] = tensor.reshape(batch_shape + tensor.shape[1:-1] + q.shape)
    return converted


def _pack_graphs(groups):
    """Validate local indices before offsetting and packing disjoint graphs."""
    nodes, edges, features, lengths = [], [], [], []
    offset = 0
    for _, raw in groups:
        for i in range(raw["node_features"].shape[0]):
            node = raw["node_features"][i]
            edge = raw["edge_index"][i]
            if node.ndim != 2 or node.shape[0] == 0:
                raise ValueError("node_features must have shape [N, F] with N > 0")
            if edge.ndim != 2 or edge.shape[-1] != 2:
                raise ValueError("edge_index must have shape [E, 2]")
            if edge.dtype not in (torch.int32, torch.int64):
                raise TypeError("edge_index must have an integer dtype")
            check(((edge >= 0) & (edge < node.shape[0])).all(),
                  "edge_index entries must be in the range [0, N) for each graph")
            lengths.append(node.shape[0])
            nodes.append(node)
            edges.append(edge + offset)
            if "edge_features" in raw:
                features.append(raw["edge_features"][i])
            offset += node.shape[0]
    packed = {"node_features": torch.cat(nodes), "edge_index": torch.cat(edges).T.long()}
    if features:
        packed["edge_features"] = torch.cat(features)
    return packed, lengths


def prepare_inputs(model, values):
    """Prepare numeric requests; the calling model owns all network computation."""
    if not model.signature.has_ellipsis:
        specs, _, _ = _specs(model)
        raw = {role: values[q].reshape((1,) + values[q].shape[:len(specs[role][0])] + (specs[role][1],))
               for role, q in model.input_quantities.items()}
        return [_native_input(model, raw)], None

    batch, list_axis, groups = _prepared_batches(model, values)
    layout = {"batch": batch, "list_axis": list_axis, "groups": groups}
    if model.config.name in _GRAPHS:
        packed, lengths = _pack_graphs(groups)
        layout["lengths"] = lengths
        return [packed], layout

    shapes = {tuple(t.shape[1:] for t in raw.values()) for _, raw in groups}
    if len(shapes) > 1 and uses_batch_statistics(model):
        raise ValueError("Ragged grid execution requires network-specific handling before BatchNorm")
    layout["combined"] = len(shapes) == 1
    if layout["combined"]:
        raw = {role: torch.cat([g[role] for _, g in groups], dim=0) for role in groups[0][1]}
        return [_native_input(model, raw)], layout
    return [_native_input(model, raw) for _, raw in groups], layout


def finish_outputs(model, results, layout):
    """Restore named outputs after the model has evaluated its numeric requests."""
    if layout is None:
        return _outputs(model, results[0], ())
    groups, batch, list_axis = layout["groups"], layout["batch"], layout["list_axis"]
    if "lengths" in layout:
        outputs = {}
        for role, q in model.output_quantities.items():
            split = results[0][role].split(layout["lengths"])
            collected, cursor = [], 0
            for shape, raw in groups:
                count = raw["node_features"].shape[0]
                tensors = split[cursor:cursor + count]
                cursor += count
                collected.append(torch.stack(tensors).reshape(shape + (tensors[0].shape[0],) + q.shape))
            outputs[q] = Value(collected if list_axis else collected[0], batch + ("vertex",))
        return outputs

    outputs = {q: [] for q in model.outputs}
    if layout["combined"]:
        flat = _outputs(model, results[0], (sum(prod(shape) for shape, _ in groups),))
        cursor = 0
        for shape, _ in groups:
            count = prod(shape)
            for q, tensor in flat.items():
                outputs[q].append(tensor[cursor:cursor + count].reshape(shape + tensor.shape[1:]))
            cursor += count
    else:
        for (shape, _), result in zip(groups, results):
            for q, tensor in _outputs(model, result, shape).items():
                outputs[q].append(tensor)
    specs = _specs(model)[1]
    return {q: Value(outputs[q] if list_axis else outputs[q][0], batch + specs[role][0])
            for role, q in model.output_quantities.items()}
