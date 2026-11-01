"""The legacy facade and modular API must share the same class objects."""

import contracts
import pinncore


def test_compatibility_exports_are_identical():
    assert set(contracts.__all__) == set(pinncore.__all__)
    for name in pinncore.__all__:
        assert getattr(contracts, name) is getattr(pinncore, name)


def test_public_api_has_no_geometry_or_field_contract():
    for module in (contracts, pinncore):
        assert not hasattr(module, "Geometry")
        assert not hasattr(module, "Field")
    assert not hasattr(pinncore.Model, "geometry")
    assert not hasattr(pinncore.Solver, "geometry")
    assert not hasattr(pinncore.Quantity("T"), "_domain")
