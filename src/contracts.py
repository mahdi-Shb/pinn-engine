"""Compatibility imports for the modular pinn_engine package.

Existing notebooks can keep using `from contracts import ...`.
Implementation lives in pinn_engine/; no class definitions are duplicated here.
"""

from pinn_engine import (
    Axis, Geometry, Quantity, Field, Value, Signature, Model,
    Derivative, TotalDerivative, PartialDerivative, DerivativePlan,
    DerivativeBackend, AutogradBackend, Solver,
)
from pinn_engine import __all__

def _example() -> None:
    """A trainable pointwise model followed by a loss over all sample axes."""
    import torch
    from torch import nn

    x, prediction, loss = map(Quantity, ("x", "prediction", "loss"))

    class Predict(Model):
        geometry = None
        signature = Signature({x: ()}, {prediction: ()})

        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))

        def evaluate(self, values):
            return {prediction: self.network(values[x].unsqueeze(-1)).squeeze(-1)}

    class Loss(Model):
        geometry = None
        signature = Signature({prediction: ("scenario", "point")}, {loss: ()})

        def evaluate(self, values):
            return {loss: values[prediction].square().mean()}

    solver = Solver([Loss(), Predict()], outputs=loss, derivative_backend="autograd")
    result = solver({x: Value(torch.rand(3, 2), ("scenario", "point"))})
    result[loss].tensor.backward()
    print("Loss shape:", tuple(result[loss].tensor.shape))
    print(solver.to_mermaid())
    print("Saved graph:", solver.render_graph("solver_graph.svg"))
    print("Saved graph:", solver.render_graph("solver_graph.png"))



if __name__ == "__main__":
    _example()
