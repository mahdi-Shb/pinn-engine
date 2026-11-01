"""Run after installing the engine with the geometry extra."""
import torch
import geocore as gc
from pinncore import GeoCoreModel, Model, Quantity, Signature, Solver, Value

# GeoCore creates two variants without mutating this template.
design = gc.Design({
    "name": "wing", "units": "mm",
    "parameters": {"span": {"value": 2., "min": 1., "max": 4.}},
    "shape": {"box": {"center": [0, 0, 0], "size": ["span", 1, .2]}},
})
span = Quantity("span")
coordinates = Quantity("coordinates", (3,))
centroid = Quantity("centroid", (3,))
producer = GeoCoreModel(
    design, "POINT_CLOUD", parameters={"span": span},
    outputs={"coordinates": coordinates}, scenario_axis="case",
    options={"n_points": 32, "features": None, "normalization": None, "resolution": .15},
    seed=7,
)


class Centroid(Model):
    # Model vmaps over the dense case axis; this function sees one point set.
    signature = Signature({coordinates: "point"}, {centroid: ()})

    def evaluate(self, values):
        return {centroid: values[coordinates].mean(0)}


if __name__ == "__main__":
    solver = Solver([Centroid(), producer], outputs=(coordinates, centroid))
    result = solver({span: Value(torch.tensor([2., 3.]), "case")})
    print("Point batch shape:", tuple(result[coordinates].tensor.shape))  # (2, 32, 3)
    print("Centroids:", result[centroid].tensor)  # (2, 3)
    # Transfer the dense batch explicitly if the next stage uses CUDA:
    # gpu_points = Value(result[coordinates].tensor.to("cuda"), ("case", "point"))
