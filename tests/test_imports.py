"""The legacy facade and modular API must share the same class objects."""

import contracts
import pinn_engine


def test_compatibility_exports_are_identical():
    assert set(contracts.__all__) == set(pinn_engine.__all__)
    for name in pinn_engine.__all__:
        assert getattr(contracts, name) is getattr(pinn_engine, name)
