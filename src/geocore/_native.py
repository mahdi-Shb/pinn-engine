"""ctypes binding to the PicoGK C++ runtime (OpenVDB based voxel kernel).

The runtime is looked up in this order:
  1. the directory in the environment variable ``GEOCORE_PICOGK_DIR``
  2. ``geocore/_bin/<platform>`` (shipped copies of the PicoGK natives)

When it is not found (e.g. on Linux, where PicoGK ships no binary) the
``AVAILABLE`` flag is False and geocore uses its numba "dense" backend.

Bulk data (mesh vertices/triangles, implicit rendering) never goes through
per-element Python calls: numba jitted loops call the C functions directly, and
implicit rendering uses a numba ``cfunc`` as the PicoGK distance callback.
"""
from __future__ import annotations

import ctypes as C
import os
import platform
import sys

import numba as nb
import numpy as np
from numba import types
from numba.extending import intrinsic

V3 = C.c_float * 3


class BBox3(C.Structure):
    _fields_ = [("min", V3), ("max", V3)]


H = C.c_int64  # every PicoGK handle is a 64 bit value

AVAILABLE = False
LOAD_ERROR: str | None = None
VERSION: str | None = None
_dll = None


def _candidate_dirs():
    env = os.environ.get("GEOCORE_PICOGK_DIR")
    if env:
        yield env
    here = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == "win32":
        yield os.path.join(here, "_bin", "win-x64")
    elif sys.platform == "darwin" and platform.machine() == "arm64":
        yield os.path.join(here, "_bin", "osx-arm64")
    else:
        yield os.path.join(here, "_bin", "linux-x64")


def _find_lib(d):
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):
        if f.startswith(("picogk", "libpicogk")) and f.endswith((".dll", ".dylib", ".so")):
            return os.path.join(d, f)
    return None


def _load():
    global _dll, AVAILABLE, LOAD_ERROR, VERSION
    if os.environ.get("GEOCORE_BACKEND", "").lower() == "dense":
        LOAD_ERROR = "disabled by GEOCORE_BACKEND=dense"
        return
    errors = []
    for d in _candidate_dirs():
        path = _find_lib(d)
        if path is None:
            errors.append(f"no PicoGK runtime in {d}")
            continue
        try:
            if sys.platform == "win32":
                os.add_dll_directory(d)
            _dll = C.CDLL(path)
            break
        except OSError as e:  # pragma: no cover - platform specific
            errors.append(f"{path}: {e}")
    if _dll is None:
        LOAD_ERROR = "; ".join(errors)
        return
    _prototypes(_dll)
    buf = C.create_string_buffer(256)
    _dll.Library_GetVersion(buf)
    VERSION = buf.value.decode()
    AVAILABLE = True


def _p(f, res, *args):
    f.restype = res
    f.argtypes = list(args)
    return f


def _prototypes(L):
    P3 = C.POINTER(V3)
    PI = C.POINTER(C.c_int)
    PF = C.POINTER(C.c_float)
    VP = C.c_void_p
    _p(L.Library_GetVersion, None, C.c_char_p)
    _p(L.Library_hCreateInstance, H, C.c_float)
    _p(L.Library_DestroyInstance, None, H)
    _p(L.Library_VoxelsToMm, None, H, P3, P3)
    _p(L.Library_MmToVoxels, None, H, P3, P3)
    # mesh (pointer args as void* so numba can call them)
    _p(L.Mesh_hCreate, H, H)
    _p(L.Mesh_hCreateFromVoxels, H, H, H)
    _p(L.Mesh_Destroy, None, H, H)
    _p(L.Mesh_nAddVertex, C.c_int, H, H, VP)
    _p(L.Mesh_nVertexCount, C.c_int, H, H)
    _p(L.Mesh_GetVertex, None, H, H, C.c_int, VP)
    _p(L.Mesh_nAddTriangle, C.c_int, H, H, VP)
    _p(L.Mesh_nTriangleCount, C.c_int, H, H)
    _p(L.Mesh_GetTriangle, None, H, H, C.c_int, VP)
    # lattice
    _p(L.Lattice_hCreate, H, H)
    _p(L.Lattice_Destroy, None, H, H)
    _p(L.Lattice_AddSphere, None, H, H, VP, C.c_float)
    _p(L.Lattice_AddBeam, None, H, H, VP, VP, C.c_float, C.c_float, C.c_bool)
    # voxels
    _p(L.Voxels_hCreate, H, H)
    _p(L.Voxels_hCreateCopy, H, H, H)
    _p(L.Voxels_hCreateSphere, H, H, P3, C.c_float)
    _p(L.Voxels_hCreateCapsule, H, H, P3, P3, C.c_float, C.c_float)
    _p(L.Voxels_hCreateMeshShell, H, H, H, C.c_float)
    _p(L.Voxels_bIsEmpty, C.c_bool, H, H)
    _p(L.Voxels_Destroy, None, H, H)
    _p(L.Voxels_BoolAdd, None, H, H, H)
    _p(L.Voxels_BoolSubtract, None, H, H, H)
    _p(L.Voxels_BoolIntersect, None, H, H, H)
    _p(L.Voxels_Offset, None, H, H, C.c_float)
    _p(L.Voxels_DoubleOffset, None, H, H, C.c_float, C.c_float)
    _p(L.Voxels_TripleOffset, None, H, H, C.c_float)
    _p(L.Voxels_RenderMesh, None, H, H, H)
    _p(L.Voxels_RenderImplicit, None, H, H, C.POINTER(BBox3), VP)
    _p(L.Voxels_IntersectImplicit, None, H, H, VP)
    _p(L.Voxels_RenderLattice, None, H, H, H)
    _p(L.Voxels_bIsEqual, C.c_bool, H, H, H)
    _p(L.Voxels_fCalculateVolume, C.c_float, H, H)
    _p(L.Voxels_bIsInside, C.c_bool, H, H, P3)
    _p(L.Voxels_GetSurfaceNormal, None, H, H, P3, P3)
    _p(L.Voxels_bClosestPointOnSurface, C.c_bool, H, H, P3, P3)
    _p(L.Voxels_bRayCastToSurface, C.c_bool, H, H, P3, P3, P3)
    _p(L.Voxels_GetVoxelDimensions, None, H, H, PI, PI, PI, PI, PI, PI)
    _p(L.Voxels_GetZSlice, None, H, H, C.c_int, VP, PF)


_load()


def dll():
    if not AVAILABLE:
        raise RuntimeError("PicoGK runtime not available: " + str(LOAD_ERROR))
    return _dll


def v3(x) -> V3:
    return V3(float(x[0]), float(x[1]), float(x[2]))


# --------------------------------------------------------------------------
# numba helpers: integer address -> pointer
# --------------------------------------------------------------------------
@intrinsic
def _voidptr(typingctx, addr):
    sig = types.voidptr(types.int64)

    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], context.get_value_type(types.voidptr))
    return sig, codegen


@intrinsic
def _f64ptr(typingctx, addr):
    sig = types.CPointer(types.float64)(types.int64)

    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], context.get_value_type(signature.return_type))
    return sig, codegen


@intrinsic
def _f32ptr(typingctx, addr):
    sig = types.CPointer(types.float32)(types.int64)

    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], context.get_value_type(signature.return_type))
    return sig, codegen


# --------------------------------------------------------------------------
# bulk mesh transfer
# --------------------------------------------------------------------------
_jit_mesh = None


def _build_mesh_jit():
    L = _dll
    get_v = L.Mesh_GetVertex
    get_t = L.Mesh_GetTriangle
    add_v = L.Mesh_nAddVertex
    add_t = L.Mesh_nAddTriangle

    @nb.njit(cache=False)
    def pull(lib, msh, V, T):
        base_v = V.ctypes.data
        base_t = T.ctypes.data
        for i in range(V.shape[0]):
            get_v(lib, msh, i, _voidptr(base_v + 12 * i))
        for i in range(T.shape[0]):
            get_t(lib, msh, i, _voidptr(base_t + 12 * i))

    @nb.njit(cache=False)
    def push(lib, msh, V, T):
        base_v = V.ctypes.data
        base_t = T.ctypes.data
        for i in range(V.shape[0]):
            add_v(lib, msh, _voidptr(base_v + 12 * i))
        for i in range(T.shape[0]):
            add_t(lib, msh, _voidptr(base_t + 12 * i))

    return pull, push


def mesh_pull(lib: int, msh: int):
    global _jit_mesh
    if _jit_mesh is None:
        _jit_mesh = _build_mesh_jit()
    L = dll()
    nv = L.Mesh_nVertexCount(lib, msh)
    nt = L.Mesh_nTriangleCount(lib, msh)
    V = np.empty((nv, 3), np.float32)
    T = np.empty((nt, 3), np.int32)
    if nv or nt:
        _jit_mesh[0](lib, msh, V, T)
    return V, T


def mesh_push(lib: int, V: np.ndarray, T: np.ndarray) -> int:
    global _jit_mesh
    if _jit_mesh is None:
        _jit_mesh = _build_mesh_jit()
    L = dll()
    msh = L.Mesh_hCreate(lib)
    V = np.ascontiguousarray(V, np.float32)
    T = np.ascontiguousarray(T, np.int32)
    if len(V) or len(T):
        _jit_mesh[1](lib, msh, V, T)
    return msh


def lattice_push(lib: int, beams: np.ndarray, round_cap: np.ndarray) -> int:
    """beams [M,8] = a(3) b(3) ra rb."""
    L = dll()
    lat = L.Lattice_hCreate(lib)
    b = np.ascontiguousarray(beams, np.float32)
    pa = b[:, 0:3].copy()
    pb = b[:, 3:6].copy()
    for k in range(len(b)):
        L.Lattice_AddBeam(lib, lat, pa[k].ctypes.data, pb[k].ctypes.data,
                          float(b[k, 6]), float(b[k, 7]), bool(round_cap[k]))
    return lat


# --------------------------------------------------------------------------
# implicit callback: nearest lookup in a dense float32 grid sampled on the
# PicoGK voxel lattice, so values are exact at the voxel centres PicoGK asks
# for.  The grid description lives in a static state block.
# --------------------------------------------------------------------------
_STATE = np.zeros(16, np.float64)   # ptr, nx, ny, nz, ox, oy, oz, 1/h, outside
_STATE_ADDR = int(_STATE.ctypes.data)
_cb = None


def _build_callback():
    addr = _STATE_ADDR

    @nb.cfunc(types.float32(types.CPointer(types.float32)))
    def lookup(p):
        st = nb.carray(_f64ptr(addr), 16)
        nx = int(st[1]); ny = int(st[2]); nz = int(st[3])
        data = nb.carray(_f32ptr(np.int64(st[0])), nx * ny * nz)
        i = int(np.floor((p[0] - st[4]) * st[7] + 0.5))
        j = int(np.floor((p[1] - st[5]) * st[7] + 0.5))
        k = int(np.floor((p[2] - st[6]) * st[7] + 0.5))
        if i < 0 or j < 0 or k < 0 or i >= nx or j >= ny or k >= nz:
            return np.float32(st[8])
        return data[(i * ny + j) * nz + k]
    return lookup


def grid_callback(phi: np.ndarray, origin, h: float, outside: float):
    """Point the shared callback at a dense [X,Y,Z] float32 grid.

    Returns (address, keepalive).  Not thread safe (PicoGK calls are serial
    from Python's point of view)."""
    global _cb
    if _cb is None:
        _cb = _build_callback()
    phi = np.ascontiguousarray(phi, np.float32)
    nx, ny, nz = phi.shape
    _STATE[:9] = [phi.ctypes.data, nx, ny, nz, origin[0], origin[1], origin[2],
                  1.0 / h, outside]
    return _cb.address, phi
