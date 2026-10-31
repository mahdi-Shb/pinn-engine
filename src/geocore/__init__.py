"""geocore - PicoGK-based geometry core that turns CAD-like geometry into
neural-network inputs (MeshGraphNet, PointNet, FNO3D, Deep NURBS) and PINN
collocation sets.

Quick start::

    import geocore as gc

    gc.set_voxel_size(0.3)                         # like PicoGK.Library(0.3)
    part = gc.Box((0, 0, 4), (60, 40, 8)).tagged("wall") \\
         | gc.Cylinder((12, 0, 0), (12, 0, 16), 9).tagged("boss")
    vox = gc.Voxels(part)                          # PicoGK C++ (OpenVDB)
    vox.fillet(4)

    gc.save("out/", gc.voxel_grid(vox, shape=64))  # -> out/voxel_grid.npz
    gc.save("out/", gc.point_cloud(vox, 2048))
    gc.save("out/", gc.mesh_graph(vox))
"""
from . import _native
from .implicit import (BBox, Implicit, Sphere, Box, Cylinder, Beam, Capsule, Torus, Plane,
                       Cone, Gyroid, SchwarzP, Diamond, CubicStrutLattice, FunctionImplicit,
                       Union, Intersection, Difference, Complement, SmoothUnion,
                       SmoothDifference, SmoothIntersection, Offset, Shell, Transform,
                       union, intersection, difference)
from .mesh import Mesh, MeshImplicit
from .lattice import Lattice, LatticeImplicit
from .fields import GridSpec, GridImplicit, redistance, sample_implicit, mesh_from_grid
from .voxels import Library, Voxels, set_voxel_size, default_library
from .normalize import Normalizer
from .sampling import Geometry, collocation
from .representations import (mesh_graph, point_cloud, voxel_grid, batch_graphs,
                              SCHEMA_VERSION)
from .nurbs import NURBSCurve, NURBSSurface, make_compatible, parametric_surface, sample_uv
from .io import save, load, validate, SchemaError
from .design import Design, DesignError
from .outputs import ModelInput, output, load_geometry, SCHEMA
read = load_geometry

__version__ = "0.1.0"

PICOGK_AVAILABLE = _native.AVAILABLE
PICOGK_VERSION = _native.VERSION


def info() -> dict:
    """Backend information."""
    return {"geocore": __version__, "picogk_available": _native.AVAILABLE,
            "picogk_version": _native.VERSION, "picogk_error": _native.LOAD_ERROR}
