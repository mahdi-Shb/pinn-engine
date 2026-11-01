"""Compatibility imports for the modular pinncore package.

Existing notebooks can keep using `from contracts import ...`.
Implementation lives in pinncore/; no class definitions are duplicated here.
"""

from pinncore import (
    Axis, Quantity, Value, Signature, Model,
    Derivative, TotalDerivative, PartialDerivative,
    DerivativeBackend, AutogradBackend, Solver, GeoCoreModel, check,
)
from pinncore import __all__

def _example() -> None:
    """A trainable pointwise model followed by a loss over all sample axes."""
    import torch
    from torch import nn

    x, prediction, loss = map(Quantity, ("x", "prediction", "loss"))

    class Predict(Model):
        signature = Signature({x: ()}, {prediction: ()})

        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))

        def evaluate(self, values):
            return {prediction: self.network(values[x].unsqueeze(-1)).squeeze(-1)}

    class Loss(Model):
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
