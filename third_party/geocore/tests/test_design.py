import json
import math
import os

import numpy as np
import pytest

import geocore as gc
from geocore.design import evaluate, DesignError

HERE = os.path.dirname(__file__)
DESIGNS = os.path.join(HERE, "..", "examples", "designs")

PLATE = """
name: plate
parameters:
  L: {value: 40, min: 20, max: 80}
  W: {value: 20, min: 10, max: 40}
  t: {value: 4, min: 2, max: 8}
  r: {value: 3, min: 1, max: 6}
  n: {value: 3, min: 1, max: 6, type: int}
  holes_on: {value: true}
derived:
  pitch: L / (n + 1)
constraints:
  - 2 * r < pitch - 1
  - 2 * r < W - 2
shape:
  difference:
    - box: {center: [0, 0, t / 2], size: [L, W, t]}
      tag: plate
    - linear_array:
        count: n
        step: [pitch, 0, 0]
        shape: {cylinder: {center: [0, 0, t / 2], axis: z, height: t + 2, radius: r}}
      translate: [-L / 2 + pitch, 0, 0]
      enabled: holes_on
      tag: holes
outputs:
  point_cloud: {n_points: 256}
  voxel_grid: {shape: 16, channels: [occupancy, sdf, tags]}
"""


def plate():
    return gc.Design.from_string(PLATE)


# ---------------------------------------------------------------- expressions
def test_expressions():
    env = {"a": 2.0, "b": 3.0}
    assert evaluate("a * b + 1", env) == 7
    assert evaluate("max(a, b) if a < b else 0", env) == 3
    assert abs(evaluate("sqrt(a**2 + b**2)", env) - math.sqrt(13)) < 1e-12
    assert evaluate(["a", "b / 2", 5], env) == [2.0, 1.5, 5]
    assert evaluate("$a", env) == 2.0
    x = np.array([0.0, 1.0])
    assert np.allclose(evaluate("a + x * b", dict(env, x=x)), [2, 5])


@pytest.mark.parametrize("bad", ["__import__('os')", "a.__class__", "open('x')", "[c for c in 'ab']",
                                 "lambda: 1", "a if", "unknown + 1"])
def test_expressions_are_sandboxed(bad):
    with pytest.raises(DesignError):
        evaluate(bad, {"a": 1})


# ---------------------------------------------------------------- parameters
def test_parameters_and_constraints():
    d = plate()
    assert d.parameters["n"] == 3 and isinstance(d.parameters["n"], int)
    with pytest.raises(DesignError, match="maximum"):
        d.set(L=500)
    with pytest.raises(DesignError, match="integer"):
        d.set(n=2.5)
    with pytest.raises(DesignError, match="unknown parameter"):
        d.set(nope=1)
    with pytest.raises(DesignError, match="constraint"):
        d.set(r=6, n=6)                       # holes would overlap
    assert d.parameters["r"] == 3 and d.parameters["n"] == 3      # all-or-nothing
    d2 = d.with_params(L=60)
    assert d.parameters["L"] == 40 and d2.parameters["L"] == 60


def test_shape_follows_parameters():
    d = plate()
    v0 = gc.Geometry(d.build(), resolution=0.4).surface_mesh().volume()
    exact = 40 * 20 * 4 - 3 * math.pi * 9 * 4
    assert abs(v0 / exact - 1) < 0.01
    d.set(n=5, r=2)
    v1 = gc.Geometry(d.build(), resolution=0.4).surface_mesh().volume()
    assert abs(v1 / (40 * 20 * 4 - 5 * math.pi * 4 * 4) - 1) < 0.01
    d.set(holes_on=False)
    assert abs(gc.Geometry(d.build(), resolution=0.4).surface_mesh().volume() / 3200 - 1) < 0.005


def test_stable_tag_channels_across_variants():
    d = plate()
    a = d.outputs(only=["voxel_grid"])["voxel_grid"]
    b = d.with_params(holes_on=False).outputs(only=["voxel_grid"])["voxel_grid"]
    assert list(a["channel_names"]) == list(b["channel_names"]) == \
        ["occupancy", "sdf", "tag:plate", "tag:holes"]
    assert a["voxel_fields"][0, 3].sum() > 0 and b["voxel_fields"][0, 3].sum() == 0


def test_json_and_roundtrip(tmp_path):
    d = plate()
    d.set(L=50)
    f = d.save(str(tmp_path / "p.json"))
    d2 = gc.Design.load(f)
    assert d2.parameters == d.parameters
    y = d.save(str(tmp_path / "p.yaml"))
    assert gc.Design.load(y).parameters["L"] == 50


def test_errors_point_to_the_problem():
    with pytest.raises(DesignError, match="shape.difference\\[0\\].box"):
        gc.Design.from_string("shape: {difference: [{box: {centre: [0,0,0], size: [1,1,1]}}]}").build()
    with pytest.raises(DesignError, match="unknown shape"):
        gc.Design.from_string("shape: {ref: nothing}").build()
    with pytest.raises(DesignError, match="unknown section"):
        gc.Design.from_string("shape: {sphere: {radius: 1}}\nparams: {}")


# ---------------------------------------------------------------- outputs
def test_outputs_files(tmp_path):
    d = plate()
    res = d.outputs(str(tmp_path))
    assert set(res) == {"point_cloud", "voxel_grid"}
    for f in ("point_cloud.npz", "voxel_grid.npz", "parameters.json", "design.yaml"):
        assert (tmp_path / f).exists()
    gc.load(str(tmp_path / "voxel_grid.npz"))
    meta = json.load(open(tmp_path / "parameters.json"))
    assert meta["parameters"]["L"] == 40 and abs(meta["derived"]["pitch"] - 10) < 1e-12


def test_sample_and_sweep(tmp_path):
    d = plate()
    rows = d.sample(4, str(tmp_path / "ds"), rng=0, only=["point_cloud"])
    assert len(rows) == 4
    for r in rows:
        dd = d.with_params(**r)        # every sample satisfies the constraints
        assert not dd.violations()
    assert len(open(tmp_path / "ds" / "parameters.csv").read().strip().splitlines()) == 5
    assert (tmp_path / "ds" / "sample_0003" / "point_cloud.npz").exists()
    rows = d.sweep({"n": [1, 2, 6], "r": [2, 5]})
    assert {"n": 6, "r": 5} not in rows and len(rows) >= 3


def test_voxel_operations():
    src = """
parameters: {R: {value: 8}, f: {value: 2}}
shape:
  union:
    - box: {center: [0, 0, 2], size: [30, 30, 4]}
      tag: base
    - cylinder: {a: [0, 0, 0], b: [0, 0, 15], radius: R}
      tag: post
voxels:
  voxel_size: 0.5
  backend: dense
  operations: [{fillet: f}]
"""
    d = gc.Design.from_string(src)
    v = d.build()
    assert isinstance(v, gc.Voxels)
    g = d.geometry()
    assert g.tag_names == ["base", "post"]
    p, n, t = g.sample_surface(500, rng=0)
    assert set(np.unique(t)) == {0, 1}
    v0 = gc.Geometry(d.build_shape(), resolution=0.5).surface_mesh().volume()
    assert v.volume() > v0                 # the fillet added material


@pytest.mark.parametrize("name", ["bracket", "heatsink", "flange", "lattice_panel"])
def test_example_designs_build(name):
    d = gc.Design.load(os.path.join(DESIGNS, f"{name}.yaml"))
    shape = d.build_shape()
    assert shape.bounds is not None
    assert d.tag_names
    assert isinstance(d.describe(), str)


def test_cli(tmp_path):
    from geocore.__main__ import main
    f = tmp_path / "plate.yaml"
    f.write_text(PLATE)
    assert main([str(f), "--set", "L=60", "n=4", "--out", str(tmp_path / "o"),
                 "--only", "point_cloud"]) == 0
    meta = json.load(open(tmp_path / "o" / "parameters.json"))
    assert meta["parameters"]["L"] == 60 and meta["parameters"]["n"] == 4
    assert main([str(f), "--set", "L=999"]) == 2
