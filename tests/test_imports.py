"""The legacy facade and modular API must share the same class objects."""

import contracts
import pinn_engine


def test_compatibility_exports_are_identical():
    assert set(contracts.__all__) == set(pinn_engine.__all__)
    for name in pinn_engine.__all__:
        assert getattr(contracts, name) is getattr(pinn_engine, name)


def test_public_api_has_no_geometry_or_field_contract():
    for module in (contracts, pinn_engine):
        assert not hasattr(module, "Geometry")
        assert not hasattr(module, "Field")
    assert not hasattr(pinn_engine.Model, "geometry")
    assert not hasattr(pinn_engine.Solver, "geometry")
    assert not hasattr(pinn_engine.Quantity("T"), "_domain")
