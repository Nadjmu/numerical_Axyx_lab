"""
core/problem_creation.py
========================
Generates the matrix A for the eigenvalue problem  Ax = λx.

GPU support
-----------
All public functions accept an optional ``use_gpu`` boolean (default False).
When True, arrays are returned as CuPy arrays on the current CUDA device.
Generation always happens on the CPU (NumPy) and is then transferred via
``cupy.asarray()``.  The performance-critical solver operations run on the GPU.

Import support
--------------
``load_npy``      — load a .npy file from a Streamlit file uploader.
``sparsity_mask`` — return the non-zero boolean mask of an array, used to
                    inherit sparsity when perturbing an imported matrix.

Public API
----------
create_matrix(...)          -> np.ndarray  or  cp.ndarray
apply_perturbation(...)     -> np.ndarray  or  cp.ndarray
matrix_info(...)            -> dict  (always CPU scalars)
compatible_structures(type) -> list[str]
load_npy(...)               -> np.ndarray  or  cp.ndarray
sparsity_mask(A)            -> np.ndarray (bool, always CPU)
"""

from __future__ import annotations

import numpy as np

from core.device import get_array_module, to_numpy


# ──────────────────────────────────────────────────────────────────────────────
# Compatibility table
# ──────────────────────────────────────────────────────────────────────────────

_SPARSE_STRUCTURES = [
    "Dense",
    "Sparse Tridiagonal",
    "Sparse Block-Tridiagonal",
    "Sparse Banded",
]

COMPATIBILITY: dict[str, list[str]] = {
    "Random Gaussian":      _SPARSE_STRUCTURES,
    "Symmetric Random":     _SPARSE_STRUCTURES,
    "Random SPD":           ["Dense"],
    "Diagonal":             ["Dense"],
    "Hilbert":              _SPARSE_STRUCTURES,
    "Toeplitz":             _SPARSE_STRUCTURES,
    "Circulant":            _SPARSE_STRUCTURES,
    "Tridiagonal Symm.":    ["Dense"],
}


def compatible_structures(matrix_type: str) -> list[str]:
    return COMPATIBILITY.get(matrix_type, _SPARSE_STRUCTURES)


# ──────────────────────────────────────────────────────────────────────────────
# Base matrix generators  (always CPU NumPy)
# ──────────────────────────────────────────────────────────────────────────────

def _random_gaussian(m: int, seed: int = 42, **_) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((m, m))


def _symmetric_random(m: int, seed: int = 42, **_) -> np.ndarray:
    G = np.random.default_rng(seed).standard_normal((m, m))
    return (G + G.T) / 2.0


def _random_spd(m: int, seed: int = 42, type_param: int = 6, **_) -> np.ndarray:
    rng  = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((m, m)))
    lam  = np.logspace(0, type_param, m)
    return (Q * lam) @ Q.T


def _diagonal(m: int, seed: int = 42, type_param: int = 4, **_) -> np.ndarray:
    rng   = np.random.default_rng(seed)
    half  = type_param / 2.0
    lam   = np.logspace(-half, half, m)
    signs = rng.choice([-1.0, 1.0], size=m)
    lam   = rng.permutation(lam * signs)
    return np.diag(lam)


def _hilbert(m: int, **_) -> np.ndarray:
    i = np.arange(m, dtype=np.float64).reshape(-1, 1)
    j = np.arange(m, dtype=np.float64).reshape(1, -1)
    return 1.0 / (i + j + 1.0)


def _toeplitz(m: int, seed: int = 42, **_) -> np.ndarray:
    rng   = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi)
    k     = np.arange(m, dtype=np.float64)
    f     = np.cos(k + phase) * np.exp(-k / 4.0)
    A     = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            A[i, j] = f[abs(j - i)]
    return A


def _circulant(m: int, seed: int = 42, **_) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c   = rng.standard_normal(m) * np.exp(-np.arange(m, dtype=float) / (m / 4.0))
    A   = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        A[i] = np.roll(c, i)
    return A


def _tridiagonal_symm(m: int, seed: int = 42, type_param: int = 1, **_) -> np.ndarray:
    rng  = np.random.default_rng(seed)
    diag = rng.standard_normal(m)
    off  = rng.standard_normal(m - 1) * float(type_param)
    return np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)


_BASE_GENERATORS: dict[str, callable] = {
    "Random Gaussian":   _random_gaussian,
    "Symmetric Random":  _symmetric_random,
    "Random SPD":        _random_spd,
    "Diagonal":          _diagonal,
    "Hilbert":           _hilbert,
    "Toeplitz":          _toeplitz,
    "Circulant":         _circulant,
    "Tridiagonal Symm.": _tridiagonal_symm,
}

MATRIX_TYPE_NOTES: dict[str, str] = {
    "Random Gaussian": (
        "Entries i.i.d. N(0,1).  Eigenvalues are generally complex, "
        "distributed approximately on a disk of radius √m (circular law)."
    ),
    "Symmetric Random": (
        "A = (G + Gᵀ)/2.  Real eigenvalues guaranteed by symmetry.  "
        "Eigenvalue distribution follows the Wigner semicircle law."
    ),
    "Random SPD": (
        "A = QΛQᵀ with κ(A) = 10^k exactly (set k with the slider).  "
        "All eigenvalues positive.  Dense only — sparsity destroys SPD."
    ),
    "Diagonal": (
        "D = diag(λ), λ from logspace with prescribed spread.  "
        "Eigenvalues are exactly known — ideal for ground-truth benchmarking."
    ),
    "Hilbert": (
        "H[i,j] = 1/(i+j+1).  Symmetric positive definite.  "
        "Eigenvalues cluster exponentially near zero — severely ill-conditioned."
    ),
    "Toeplitz": (
        "Constant along each diagonal: T[i,j] = f(|i−j|) with a decaying cosine f.  "
        "Symmetric; eigenvalues lie on an interval (not closed form in general)."
    ),
    "Circulant": (
        "Each row is a cyclic shift of the first row.  "
        "Eigenvalues = DFT of the first row — known analytically."
    ),
    "Tridiagonal Symm.": (
        "Symmetric tridiagonal with random diagonal and off-diagonal entries.  "
        "Models the 1-D FDM Laplacian.  Natural test case for Lanczos."
    ),
}

MATRIX_TYPE_PARAM_NOTES: dict[str, str] = {
    "Random SPD":        "log₁₀(κ_target) — controls condition number and eigenvalue spread.",
    "Diagonal":          "log₁₀(spread) — eigenvalues range over 10^k.",
    "Tridiagonal Symm.": "Off-diagonal scale — multiplier for the off-diagonal entries.",
}


# ──────────────────────────────────────────────────────────────────────────────
# Sparsity structure masks  (always CPU NumPy)
# ──────────────────────────────────────────────────────────────────────────────

def _tridiagonal_mask(m: int) -> np.ndarray:
    mask = np.zeros((m, m), dtype=bool)
    for d in (-1, 0, 1):
        rows = np.arange(max(0, -d), min(m, m - d))
        if rows.size:
            mask[rows, rows + d] = True
    return mask


def _block_tridiagonal_mask(m: int, block_size: int) -> np.ndarray:
    bs   = max(1, block_size)
    mask = np.zeros((m, m), dtype=bool)
    nblk = (m + bs - 1) // bs
    for br in range(nblk):
        for bc in range(nblk):
            if abs(br - bc) <= 1:
                r0, r1 = br * bs, min((br + 1) * bs, m)
                c0, c1 = bc * bs, min((bc + 1) * bs, m)
                mask[r0:r1, c0:c1] = True
    return mask


def _banded_mask(m: int, num_diags: int) -> np.ndarray:
    half = num_diags // 2
    mask = np.zeros((m, m), dtype=bool)
    for d in range(-half, half + 1):
        rows = np.arange(max(0, -d), min(m, m - d))
        if rows.size:
            mask[rows, rows + d] = True
    return mask


def _apply_structure_numpy(A: np.ndarray, structure: str,
                            struct_param: int) -> np.ndarray:
    if structure == "Dense":
        return A.copy()
    m = A.shape[0]
    if structure == "Sparse Tridiagonal":
        return A * _tridiagonal_mask(m)
    if structure == "Sparse Block-Tridiagonal":
        return A * _block_tridiagonal_mask(m, struct_param)
    if structure == "Sparse Banded":
        return A * _banded_mask(m, struct_param)
    raise ValueError(f"Unknown structure: {structure!r}")


STRUCTURE_NOTES: dict[str, str] = {
    "Dense":                    "",
    "Sparse Tridiagonal":       "Non-zeros on diagonals −1, 0, +1 only.",
    "Sparse Block-Tridiagonal": "Main block-diagonal + two neighbouring blocks.  param = block size.",
    "Sparse Banded":            "param diagonals centred on the main diagonal.",
}


# ──────────────────────────────────────────────────────────────────────────────
# Matrix modifications  (CPU NumPy)
# ──────────────────────────────────────────────────────────────────────────────

def _symmetrize(A: np.ndarray) -> np.ndarray:
    return (A + A.T) / 2.0


def _make_spd(A: np.ndarray) -> np.ndarray:
    n     = A.shape[0]
    B     = A.T @ A
    alpha = n * np.finfo(np.float64).eps * np.linalg.norm(B, 1) + 1e-10
    B    += alpha * np.eye(n)
    return B


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def create_matrix(
    matrix_type:            str,
    m:                      int,
    structure:              str,
    struct_param:           int,
    make_hermitian:         bool,
    make_positive_definite: bool,
    dtype:                  np.dtype,
    seed:                   int  = 42,
    type_param:             int  = 4,
    use_gpu:                bool = False,
):
    """
    Construct the matrix A for Ax = λx.

    When ``use_gpu=True`` the returned array is a CuPy array.
    Generation always happens on the CPU; result is transferred via cupy.asarray().
    """
    generator = _BASE_GENERATORS.get(matrix_type)
    if generator is None:
        raise ValueError(f"Unknown matrix type: {matrix_type!r}")

    allowed = compatible_structures(matrix_type)
    if structure not in allowed:
        raise ValueError(
            f"Structure {structure!r} is not compatible with {matrix_type!r}.  "
            f"Allowed: {allowed}"
        )

    A = generator(m, seed=seed, type_param=type_param)
    A = _apply_structure_numpy(A, structure, struct_param)

    if make_hermitian:
        A = _symmetrize(A)
    if make_positive_definite:
        A = _make_spd(A)

    A = A.astype(dtype)

    if use_gpu:
        xp = get_array_module(use_gpu=True)
        return xp.asarray(A)
    return A


def apply_perturbation(
    arr,
    order:                  int,
    seed:                   int  = 13,
    structure:              str  = "Dense",
    struct_param:           int  = 1,
    make_hermitian:         bool = False,
    make_positive_definite: bool = False,
    use_gpu:                bool = False,
    custom_mask:            np.ndarray | None = None,
):
    """
    Add a structure-aware random perturbation of magnitude 10^order × ‖arr‖.

    Works whether ``arr`` is a NumPy or CuPy array.

    custom_mask: when provided (for imported matrices), noise is zeroed outside
    the mask so the perturbation inherits the imported array's sparsity pattern.
    """
    arr_cpu = to_numpy(arr)
    rng     = np.random.default_rng(seed)
    noise   = rng.standard_normal(arr_cpu.shape)

    if arr_cpu.ndim == 2:
        if custom_mask is not None:
            noise = noise * custom_mask.astype(float)
        else:
            noise = _apply_structure_numpy(noise, structure, struct_param)
        if make_hermitian:
            noise = _symmetrize(noise)
        if make_positive_definite:
            noise = _symmetrize(noise)

    noise_norm = np.linalg.norm(noise)
    arr_norm   = np.linalg.norm(arr_cpu)
    if noise_norm > 0 and arr_norm > 0:
        noise = noise * (arr_norm * (10.0 ** order) / noise_norm)
    else:
        noise = noise * (10.0 ** order)

    result_cpu = (arr_cpu + noise).astype(arr_cpu.dtype)

    if use_gpu:
        xp = get_array_module(use_gpu=True)
        return xp.asarray(result_cpu)
    return result_cpu


def matrix_info(A) -> dict:
    """Basic descriptive statistics. Works for both NumPy and CuPy arrays."""
    A_cpu = to_numpy(A)
    m, n  = A_cpu.shape
    nnz   = int(np.count_nonzero(A_cpu))
    total = m * n
    return {
        "shape":        (m, n),
        "dtype":        str(A_cpu.dtype),
        "nnz":          nnz,
        "density":      nnz / total if total > 0 else 0.0,
        "memory_bytes": A_cpu.nbytes,
    }


def load_npz(file_obj, use_gpu: bool = False):
    """
    Load a .npz file saved with scipy.sparse.save_npz() and return a dense array.

    The sparse matrix is converted to dense via .toarray() before any further
    processing, so the rest of the app sees a plain numpy array.

    Raises ValueError for non-square, non-finite, or wrong-format files.
    """
    import io
    import scipy.sparse as sp
    raw = io.BytesIO(file_obj.read())
    try:
        A_sp = sp.load_npz(raw)
    except Exception as exc:
        raise ValueError(f"Could not read sparse .npz file: {exc}") from exc

    arr = A_sp.toarray()

    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D matrix but got shape {arr.shape}.")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"Imported matrix must be square (got {arr.shape[0]} × {arr.shape[1]})."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Imported array contains NaN or Inf values.")
    if not (np.issubdtype(arr.dtype, np.floating) or
            np.issubdtype(arr.dtype, np.complexfloating)):
        arr = arr.astype(np.float64)

    if use_gpu:
        xp = get_array_module(use_gpu=True)
        return xp.asarray(arr)
    return arr


def load_npy(file_obj, expected_ndim: int = 2, use_gpu: bool = False):
    """
    Load a .npy file uploaded via Streamlit file_uploader.

    Validates: correct number of dimensions, square shape (for matrices),
    all-finite values. Casts non-float dtypes to float64.
    """
    import io
    arr = np.load(io.BytesIO(file_obj.read()))

    if arr.ndim != expected_ndim:
        raise ValueError(
            f"Expected a {expected_ndim}-D array but got shape {arr.shape}."
        )
    if arr.ndim == 2 and arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"Imported matrix must be square (got {arr.shape[0]} × {arr.shape[1]})."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Imported array contains NaN or Inf values.")

    # Preserve float and complex dtypes; cast only integer/bool inputs
    if not (np.issubdtype(arr.dtype, np.floating) or
            np.issubdtype(arr.dtype, np.complexfloating)):
        arr = arr.astype(np.float64)

    if use_gpu:
        xp = get_array_module(use_gpu=True)
        return xp.asarray(arr)
    return arr


def sparsity_mask(A) -> np.ndarray:
    """
    Return a boolean mask of the non-zero entries of A (always CPU numpy).
    Used to inherit sparsity pattern of an imported matrix during perturbation.
    """
    A_cpu = to_numpy(A)
    return A_cpu != 0.0
