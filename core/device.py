"""
core/device.py
==============
CPU / GPU device abstraction for the Numerical Ax = b Lab.

This module is the single place where the choice between NumPy (CPU) and
CuPy (GPU) is made.  Every other module imports `get_array_module` and
uses the returned namespace (called `xp` by convention) instead of
importing numpy directly.

Usage
-----
    from core.device import get_array_module, to_numpy, DeviceError

    xp = get_array_module(use_gpu=True)   # returns cupy if available, else raises
    A  = xp.array(...)                    # works for both numpy and cupy
    A_cpu = to_numpy(A)                   # always returns a numpy array

GPU detection
-------------
CuPy is imported lazily so the app starts even when CuPy is not installed.
`gpu_available()` returns False in that case and the UI disables the GPU option.

Supported CUDA operations
--------------------------
CuPy covers almost all NumPy + SciPy operations used here:
  - cupy.linalg  : svd, qr, norm, cond, cholesky
  - cupyx.scipy.linalg   : lu_factor/lu_solve, cho_factor/cho_solve,
                            solve_triangular
  - cupyx.scipy.sparse         : csc_matrix
  - cupyx.scipy.sparse.linalg  : splu, gmres, cg

Limitations noted inline where fall-back to CPU is necessary:
  - mpmath always runs on CPU (arbitrary-precision library)
  - float128 is not available on GPU; high-prec residual falls back to
    float64 on GPU side, then uses mpmath on the CPU copy for small m.
  - SuperLU (splu) in CuPy uses a different internal path than SciPy;
    behaviour is equivalent.
"""

from __future__ import annotations

import numpy as np
from typing import Any

# ── Lazy CuPy import ──────────────────────────────────────────────────────────

_cupy = None          # module reference once imported
_cupy_checked = False # whether we have already tried to import


def _try_import_cupy() -> bool:
    """Attempt to import cupy.  Returns True on success."""
    global _cupy, _cupy_checked
    if _cupy_checked:
        return _cupy is not None
    _cupy_checked = True
    try:
        import cupy as cp          # noqa: F401
        # Quick sanity check — allocate a tiny array to verify CUDA is alive
        cp.array([1.0])
        _cupy = cp
        return True
    except Exception:
        _cupy = None
        return False


def gpu_available() -> bool:
    """Return True if CuPy is installed and at least one CUDA device is visible."""
    return _try_import_cupy()


def gpu_info() -> dict:
    """
    Return a dict with GPU device information, or a 'not available' dict.

    Keys: available, device_count, devices (list of name strings)
    """
    if not _try_import_cupy():
        return {"available": False, "device_count": 0, "devices": []}
    try:
        cp = _cupy
        n  = cp.cuda.runtime.getDeviceCount()
        names = []
        for i in range(n):
            props = cp.cuda.runtime.getDeviceProperties(i)
            names.append(props["name"].decode() if isinstance(props["name"], bytes)
                         else str(props["name"]))
        return {"available": True, "device_count": n, "devices": names}
    except Exception as exc:
        return {"available": True, "device_count": 1,
                "devices": [f"Device 0 (details unavailable: {exc})"]}


# ── Array module accessor ─────────────────────────────────────────────────────

class DeviceError(RuntimeError):
    """Raised when GPU is requested but CuPy / CUDA is unavailable."""


def get_array_module(use_gpu: bool = False):
    """
    Return the array namespace to use.

    Parameters
    ----------
    use_gpu : bool
        If True, return cupy.  If False (default), return numpy.

    Raises
    ------
    DeviceError
        If use_gpu=True but CuPy is not installed or no CUDA device is found.
    """
    if not use_gpu:
        return np
    if not _try_import_cupy():
        raise DeviceError(
            "GPU requested but CuPy is not installed or no CUDA device was found.  "
            "Install CuPy with:  pip install cupy-cuda12x  "
            "(replace 12x with your CUDA version)."
        )
    return _cupy


def to_numpy(arr: Any) -> np.ndarray:
    """
    Convert arr to a NumPy array, regardless of whether it lives on CPU or GPU.

    - numpy arrays are returned as-is (no copy).
    - cupy arrays are transferred to host memory via .get().
    - Python scalars / lists are converted via np.asarray().
    """
    if isinstance(arr, np.ndarray):
        return arr
    # Check for cupy array without importing cupy at top level
    if _cupy is not None and isinstance(arr, _cupy.ndarray):
        return arr.get()
    return np.asarray(arr)


def is_gpu_array(arr: Any) -> bool:
    """Return True if arr is a CuPy (GPU) array."""
    if _cupy is None:
        return False
    return isinstance(arr, _cupy.ndarray)


# ── Scipy / Cupyx linalg accessor ─────────────────────────────────────────────

def get_linalg(use_gpu: bool = False):
    """
    Return the dense linear algebra module.

    CPU: scipy.linalg
    GPU: cupyx.scipy.linalg
    """
    if not use_gpu:
        import scipy.linalg as sla
        return sla
    if not _try_import_cupy():
        raise DeviceError("GPU linalg requested but CuPy is not available.")
    import cupyx.scipy.linalg as cpla
    return cpla


def get_sparse(use_gpu: bool = False):
    """
    Return the sparse matrix module.

    CPU: scipy.sparse
    GPU: cupyx.scipy.sparse
    """
    if not use_gpu:
        import scipy.sparse as sp
        return sp
    if not _try_import_cupy():
        raise DeviceError("GPU sparse requested but CuPy is not available.")
    import cupyx.scipy.sparse as cpsp
    return cpsp


def get_sparse_linalg(use_gpu: bool = False):
    """
    Return the sparse linear algebra module.

    CPU: scipy.sparse.linalg
    GPU: cupyx.scipy.sparse.linalg
    """
    if not use_gpu:
        import scipy.sparse.linalg as spla
        return spla
    if not _try_import_cupy():
        raise DeviceError("GPU sparse linalg requested but CuPy is not available.")
    import cupyx.scipy.sparse.linalg as cpspla
    return cpspla
