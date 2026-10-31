"""CPU GeoCore representation generation as an ordinary engine Model.

GeoCore lives in src/geocore and is imported only when constructing this model.
Install its optional dependencies with pip install -e ".[geometry]".
Outputs stay on CPU for explicit transfer.
"""

from __future__ import annotations

from typing import Mapping
import torch

from .model import Model, Signature
from .quantities import Quantity
from .values import Value, _axes


# Array name -> sample axes inside one geometry, optional permutation, B=1 axis.
# Feature/component dimensions always trail sample axes in the engine.
_LAYOUTS = {
    "MESH_GRAPH": {
        "coordinates": (("vertex",), None, False),
        "node_features": (("vertex",), None, False),
        "edge_index": (("edge",), (1, 0), False),
        "edge_features": (("edge",), None, False),
        "batch": (("vertex",), None, False),
        "faces": (("face",), None, False),
    },
    "POINT_CLOUD": {
        "coordinates": (("point",), None, True),
        "point_features": (("point",), None, True),
        "point_mask": (("point",), None, True),
    },
    "VOXEL_GRID": {
        "voxel_fields": (("z", "y", "x"), (1, 2, 3, 0), True),
        "origin": ((), None, False),
        "spacing": ((), None, False),
        "bounds": ((), None, False),
    },
    "PARAMETRIC_SURFACE": {
        "degree": ((), None, False),
        "knot_vector_u": (("knot_u",), None, False),
        "knot_vector_v": (("knot_v",), None, False),
        "control_points": (("control_u", "control_v"), None, False),
        "weights": (("control_u", "control_v"), None, False),
        "admissible_coefficients": (("control_u", "control_v"), None, False),
        "lifting_coefficients": (("control_u", "control_v"), None, False),
        "parametric_coordinates": (("point",), None, True),
        "physical_coordinates": (("point",), None, True),
    },
    "PINN_COLLOCATION": {
        "x_interior": (("interior",), None, False),
        "sdf_interior": (("interior",), None, False),
        "x_boundary": (("boundary",), None, False),
        "normal_boundary": (("boundary",), None, False),
        "tag_boundary": (("boundary",), None, False),
        "x_band": (("band",), None, False),
        "sdf_band": (("band",), None, False),
    },
}


class GeoCoreModel(Model):
    """Generate selected numeric arrays sequentially for a batch of designs.

    source is a GeoCore Design or a fixed geometry (Mesh, Implicit, Voxels,
    NURBSSurface, etc.). parameters maps Design parameter names to scalar
    Quantities. Each incoming value contains the full `scenario_axis`; parameters
    may be dense vectors or lists of scalar Tensors. Unspecified Design
    parameters keep their template values, and the template is never mutated.

    outputs maps GeoCore array names to caller-owned Quantities. output_axes
    optionally renames the default per-geometry sample axes for each array.
    Every output is a flat Tensor list with scenario_axis first, including
    equal-sized members. A fixed source without parameters produces one member.

    options is forwarded to GeoCore.output. Its internal B is always one;
    scenarios are generated individually without padding/graph concatenation.
    seed replays sampling on each call; None draws fresh samples. Integer and
    boolean arrays keep their dtype. floating_dtype optionally casts floats.
    Geometry generation is CPU-only, non-differentiable, and not vmapped or
    compiled by this wrapper. Reject gradient-requiring parameters explicitly.
    """

    def __init__(
        self, source, representation: str, *, outputs: Mapping[str, Quantity],
        parameters: Mapping[str, Quantity] | None = None,
        options: Mapping | None = None, scenario_axis: str = "scenario",
        output_axes: Mapping[str, str | tuple[str, ...]] | None = None,
        seed: int | None = None, floating_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        try:
            import geocore as gc
        except ImportError as error:
            raise ImportError("GeoCoreModel requires geometry dependencies; install .[geometry]") from error
        from geocore.outputs import representation_name

        self.source = source
        self.representation = representation_name(representation)
        if not isinstance(scenario_axis, str):
            raise TypeError("scenario_axis must be a string")
        self.scenario_axis = _axes(scenario_axis)[0]
        self.parameters_by_name = dict(parameters or {})
        self.outputs_by_name = dict(outputs)
        self.options = dict(options or {})
        self.seed = seed
        self.floating_dtype = floating_dtype
        if not self.outputs_by_name:
            raise ValueError("Select at least one numeric GeoCore output")
        if floating_dtype is not None and not floating_dtype.is_floating_point:
            raise TypeError("floating_dtype must be a floating-point torch dtype")
        if self.parameters_by_name and not isinstance(source, gc.Design):
            raise TypeError("Runtime design parameters require a GeoCore Design source")
        for name, quantity in self.parameters_by_name.items():
            if name not in source.parameters:
                raise ValueError(f"Unknown design parameter: {name!r}")
            if not isinstance(quantity, Quantity) or quantity.shape != ():
                raise ValueError("Design parameters must be scalar Quantities")
        if any(not isinstance(q, Quantity) for q in self.outputs_by_name.values()):
            raise TypeError("Outputs must be Quantity objects")
        if len(set(self.outputs_by_name.values())) != len(self.outputs_by_name):
            raise ValueError("Each output must map to a distinct Quantity")
        if set(self.parameters_by_name.values()) & set(self.outputs_by_name.values()):
            raise ValueError("Input and output Quantities must be distinct")
        unknown = set(self.outputs_by_name) - set(_LAYOUTS[self.representation])
        if unknown:
            raise ValueError(f"Unsupported numeric outputs for {self.representation}: {sorted(unknown)}")
        overrides = dict(output_axes or {})
        if set(overrides) - set(self.outputs_by_name):
            raise ValueError("output_axes must refer to selected outputs")
        layouts = {}
        for name, q in self.outputs_by_name.items():
            default = _LAYOUTS[self.representation][name][0]
            axes = _axes(overrides.get(name, default))
            if len(axes) != len(default) or self.scenario_axis in axes:
                raise ValueError(f"Invalid sample axes for {name!r}")
            layouts[q] = (self.scenario_axis,) + axes
        self._signature = Signature(
            {q: (self.scenario_axis,) for q in self.parameters_by_name.values()}, layouts,
        )
        if {"metadata", "extras", "rng", "batch"} & self.options.keys():
            raise ValueError("metadata, extras, rng and batch are controlled by GeoCoreModel")

    @property
    def signature(self):
        return self._signature

    def forward(self, values: Mapping[Quantity, Value]) -> dict[Quantity, Value]:
        # Do not let Model.forward hide an undeclared CPU iteration in vmap.
        for value in values.values():
            if not isinstance(value, Value) or value.axes != (self.scenario_axis,):
                raise ValueError("GeoCore inputs must contain exactly the declared scenario axis")
        return super().forward(values)

    def evaluate(self, values):
        import numpy as np
        import geocore as gc

        # .item() is intentional: geometry parameters cross a CPU/Python boundary.
        for value in values.values():
            members = value if isinstance(value, list) else (value,)
            if any(t.device.type != "cpu" or t.requires_grad for t in members):
                raise ValueError("GeoCore parameters must be CPU Tensors without requires_grad")
        count = len(next(iter(values.values()))) if values else 1
        if count == 0:
            raise ValueError("Provide at least one scenario")
        rng = np.random.default_rng(self.seed)
        result = {q: [] for q in self.outputs}
        for index in range(count):
            parameters = {name: values[q][index].item() for name, q in self.parameters_by_name.items()}
            source = self.source.with_params(**parameters) if parameters else self.source
            options = dict(self.options)
            # Voxel rasterization is deterministic and has no rng argument.
            if self.representation != "VOXEL_GRID":
                options["rng"] = rng
            if self.representation == "PARAMETRIC_SURFACE":
                options["batch"] = 1
            data = gc.output(source, self.representation, metadata=False, extras=True, **options)
            for name, q in self.outputs_by_name.items():
                if name == "point_mask" and name not in data:
                    data[name] = np.ones(data["coordinates"].shape[:2], dtype=bool)
                if name not in data:
                    raise ValueError(f"GeoCore did not produce {name!r}; check builder options")
                array = np.asarray(data[name])
                axes, permutation, squeeze_batch = _LAYOUTS[self.representation][name]
                if squeeze_batch:
                    if array.shape[0] != 1:
                        raise ValueError("GeoCore must return exactly one internal batch member")
                    array = array[0]
                if permutation is not None:
                    array = array.transpose(permutation)
                tensor = torch.from_numpy(np.array(array, copy=True, order="C"))
                if tensor.is_floating_point() and self.floating_dtype is not None:
                    tensor = tensor.to(self.floating_dtype)
                result[q].append(tensor)
        return result
