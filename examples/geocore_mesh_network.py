"""Generate two ragged meshes, predict a vertex field, and train its network.

Run after installing the engine and optional geometry dependency:
    pip install -e ".[geometry]"
    python examples/geocore_mesh_network.py
"""

import torch

from pinn_engine import GeoCoreModel, Quantity, Solver, Value
from pinn_engine.networks import MeshGraphNet, MeshGraphNetConfig
import geocore as gc


def main():
    # A shared design becomes one mesh per sampled length.
    design = gc.Design({
        "name": "box",
        "parameters": {"length": {"value": 2., "min": 1., "max": 4.}},
        "shape": {"box": {"center": [0, 0, 0], "size": ["length", 1., 1.]}},
    })
    length = Quantity("length")
    nodes = Quantity("surface_features", (6,))
    edges = Quantity("connectivity", (2,))
    edge_features = Quantity("edge_features", (4,))
    temperature = Quantity("temperature")
    geometry = GeoCoreModel(
        design, "MESH_GRAPH", parameters={"length": length},
        outputs={"node_features": nodes, "edge_index": edges, "edge_features": edge_features},
        options={"resolution": .4, "normalization": None},
        floating_dtype=torch.float32, seed=7,
    )
    network = MeshGraphNet(
        MeshGraphNetConfig(node_input_dim=6, edge_input_dim=4, latent_dim=8,
                           processor_steps=1, mlp_hidden_dim=16),
        inputs={"node_features": nodes, "edge_index": edges, "edge_features": edge_features},
        outputs={"predictions": temperature},
    )
    # Solver infers geometry -> network from the matching Quantity definitions.
    solver = Solver([network, geometry], outputs=temperature)
    prediction = solver({length: Value(torch.tensor([2., 3.]), "scenario")})[temperature]
    print("axes:", prediction.axes)
    print("per-mesh shapes:", [tuple(t.shape) for t in prediction.tensor])
    # This demonstrates gradient flow to the network; it is not a physical loss.
    loss = sum(t.square().mean() for t in prediction.tensor)
    loss.backward()
    print("loss:", loss.item())


if __name__ == "__main__":
    main()
