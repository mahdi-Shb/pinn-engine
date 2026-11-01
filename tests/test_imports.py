"""The legacy facade and modular API must share the same class objects."""

import contracts
import PINNCore


def test_compatibility_exports_are_identical():
    assert set(contracts.__all__) == set(PINNCore.__all__)
    for name in PINNCore.__all__:
        assert getattr(contracts, name) is getattr(PINNCore, name)


def test_public_api_has_no_geometry_or_field_contract():
    for module in (contracts, PINNCore):
        assert not hasattr(module, "Geometry")
        assert not hasattr(module, "Field")
    assert not hasattr(PINNCore.Model, "geometry")
    assert not hasattr(PINNCore.Solver, "geometry")
    assert not hasattr(PINNCore.Quantity("T"), "_domain")
