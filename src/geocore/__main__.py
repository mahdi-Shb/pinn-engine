"""Command line: run a parametric design file.

    python -m geocore design.yaml --info
    python -m geocore design.yaml --out run1
    python -m geocore design.yaml --set length=80 fillet=2 --out run2
    python -m geocore design.yaml --sample 50 --out dataset
    python -m geocore design.yaml --sweep fillet=0,2,4 holes=2,3 --out sweep
    python -m geocore design.yaml --set length=90 --save-design bracket_long.yaml
"""
from __future__ import annotations

import argparse
import sys
import time


def _value(s: str):
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    return s


def _pairs(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"expected name=value, got '{it}'")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m geocore", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design", help="design file (.yaml / .yml / .json)")
    ap.add_argument("--set", nargs="*", metavar="NAME=VALUE", help="change parameters")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--only", help="comma separated subset of outputs, e.g. voxel_grid,photo")
    ap.add_argument("--sample", type=int, metavar="N", help="N random variants (min/max ranges)")
    ap.add_argument("--sweep", nargs="*", metavar="NAME=V1,V2", help="factorial sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--info", action="store_true", help="print parameters and exit")
    ap.add_argument("--save-design", metavar="FILE", help="write the (changed) design file")
    a = ap.parse_args(argv)

    from .design import Design, DesignError
    try:
        d = Design.load(a.design)
        if a.set:
            d.set(**{k: _value(v) for k, v in _pairs(a.set).items()})
        if a.info:
            print(d.describe())
            return 0
        if a.save_design:
            print("saved", d.save(a.save_design))
        only = a.only.split(",") if a.only else None
        t = time.time()
        if a.sample:
            if not a.out:
                raise SystemExit("--sample needs --out")
            rows = d.sample(a.sample, a.out, rng=a.seed, only=only)
            print(f"{len(rows)} variants -> {a.out} (parameters.csv)  {time.time() - t:.1f}s")
        elif a.sweep:
            if not a.out:
                raise SystemExit("--sweep needs --out")
            grid = {k: [_value(x) for x in v.split(",")] for k, v in _pairs(a.sweep).items()}
            rows = d.sweep(grid, a.out, only=only)
            print(f"{len(rows)} variants -> {a.out} (parameters.csv)  {time.time() - t:.1f}s")
        elif a.out:
            res = d.outputs(a.out, only=only, rng=a.seed)
            print(f"{d.name}: {', '.join(res)} -> {a.out}  {time.time() - t:.1f}s")
        elif not a.save_design:
            print(d.describe())
    except DesignError as e:
        print(f"design error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
