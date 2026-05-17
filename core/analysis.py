"""
core/analysis.py
================
Stability and sensitivity analysis for the eigenvalue problem  Ax = λx.

Public API
----------
eigenpair_analysis(A, eigenvalues, eigenvectors, params) -> dict
spectral_sensitivity(A, eigenvalues, eigenvectors)       -> dict
high_precision_residual(A, lam, v)                       -> tuple

Metrics
-------

Section 3 — Problem-specific sensitivity metrics
-------------------------------------------------
spectral_radius      ρ(A) = max|λᵢ|
spectral_gap         |λ₁| − |λ₂| (absolute) and |λ₁ − λ₂| / |λ₁| (relative)
                     computed on magnitude-sorted eigenvalues
eigenvalue_cond      κ(λᵢ) = 1 / |yᵢᴴxᵢ|  (condition number of eigenvalue λᵢ)
                     requires left eigenvectors — available only from NumPy eig.
                     Set to None for iterative/partial solvers.

Section 4 — Solution quality metrics
--------------------------------------
eigenvector_residual   ‖Avᵢ − λ̃ᵢvᵢ‖₂ / (‖A‖₂ · ‖vᵢ‖₂)   backward error per pair
rayleigh_accuracy      |λ̃ᵢ − ρ(vᵢ)| / |λ̃ᵢ|               deviation from RQ
orthogonality_error    ‖VᴴV − I‖_F                         eigenvector orthogonality

High-precision residual
-----------------------
Same strategy as the Ax=b lab:
  mpmath (50 dps)  for m ≤ MP_SIZE_THRESHOLD
  float128         for m > threshold  (x86 Linux only)
  float64          fallback
"""

from __future__ import annotations

import numpy as np

# ── mpmath ────────────────────────────────────────────────────────────────────
MP_DPS             = 50
MP_SIZE_THRESHOLD  = 200

try:
    import mpmath as _mpmath
    _MPMATH_AVAILABLE = True
except ImportError:
    _MPMATH_AVAILABLE = False

_F128_EPS      = float(np.finfo(np.float128).eps)
_F64_EPS       = float(np.finfo(np.float64).eps)
_FLOAT128_WIDER = _F128_EPS < _F64_EPS


# ──────────────────────────────────────────────────────────────────────────────
# High-precision eigenvector residual
# ──────────────────────────────────────────────────────────────────────────────

def _residual_mpmath(A: np.ndarray, lam: complex,
                     v: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Compute r = Av − λv and ‖r‖₂ using mpmath at MP_DPS decimal places."""
    _mpmath.mp.dps = MP_DPS
    m    = len(v)
    A_mp = _mpmath.matrix(A.tolist())
    v_mp = _mpmath.matrix([[complex(x)] for x in v])
    lam_mp = _mpmath.mpc(lam.real, lam.imag)
    r_mp = A_mp * v_mp - lam_mp * v_mp
    norm_r = float(_mpmath.sqrt(
        sum(abs(r_mp[i, 0]) ** 2 for i in range(m))
    ))
    r = np.array([complex(r_mp[i, 0]) for i in range(m)], dtype=complex)
    return r, norm_r, f"mpmath ({MP_DPS} dps)"


def _residual_float128(A: np.ndarray, lam: complex,
                       v: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Compute r = Av − λv using float128 (x86 Linux only)."""
    A_r = A.real.astype(np.float128)
    A_i = A.imag.astype(np.float128)
    v_r = v.real.astype(np.float128)
    v_i = v.imag.astype(np.float128)
    lam_r = np.float128(lam.real)
    lam_i = np.float128(lam.imag)

    # (A_r + iA_i)(v_r + iv_i) − (lam_r + i lam_i)(v_r + iv_i)
    res_r = (A_r @ v_r - A_i @ v_i) - (lam_r * v_r - lam_i * v_i)
    res_i = (A_r @ v_i + A_i @ v_r) - (lam_r * v_i + lam_i * v_r)

    norm_r = float(np.sqrt(np.sum(res_r ** 2 + res_i ** 2)))
    r      = (res_r + 1j * res_i).astype(complex)
    return r, norm_r, "float128 (80-bit extended)"


def _residual_float64(A: np.ndarray, lam: complex,
                      v: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Standard float64 eigenvector residual."""
    r      = A.astype(complex) @ v - lam * v
    norm_r = float(np.linalg.norm(r))
    return r, norm_r, "float64 (standard)"


def high_precision_residual(
    A: np.ndarray, lam: complex, v: np.ndarray
) -> tuple[np.ndarray, float, str]:
    """
    Compute r = Av − λv and ‖r‖₂ in the highest precision available.

    Returns
    -------
    r      : np.ndarray (complex128) — residual vector
    norm_r : float                   — ‖r‖₂ at high precision
    label  : str                     — which precision was used
    """
    m = A.shape[0]
    if _MPMATH_AVAILABLE and m <= MP_SIZE_THRESHOLD:
        try:
            return _residual_mpmath(A, lam, v)
        except Exception:
            pass
    if _FLOAT128_WIDER:
        try:
            return _residual_float128(A, lam, v)
        except Exception:
            pass
    return _residual_float64(A, lam, v)


# ──────────────────────────────────────────────────────────────────────────────
# Per-eigenpair metrics
# ──────────────────────────────────────────────────────────────────────────────

def _eigenvector_residual_norm(
    A: np.ndarray, lam: complex, v: np.ndarray, norm_A: float
) -> tuple[float, str]:
    """
    Normalised eigenvector residual:
        ‖Av − λv‖₂ / (‖A‖₂ · ‖v‖₂)

    A backward-stable solver gives this ≈ ε_mach.
    """
    _, norm_r, prec = high_precision_residual(A, lam, v)
    norm_v          = float(np.linalg.norm(v))
    denom           = norm_A * norm_v
    res             = norm_r / denom if denom > 0 else float("inf")
    return res, prec


def _rayleigh_quotient(A: np.ndarray, v: np.ndarray) -> complex:
    """
    ρ(v) = vᴴAv / vᴴv

    The Rayleigh quotient is the best scalar approximation to the eigenvalue
    for a given vector v.  For an exact eigenvector, ρ(v) = λ exactly.
    """
    v_c = v.astype(complex)
    return complex(v_c.conj() @ A.astype(complex) @ v_c / (v_c.conj() @ v_c))


def _rayleigh_accuracy(A: np.ndarray, lam: complex, v: np.ndarray) -> float:
    """
    |λ̃ − ρ(v)| / max(|λ̃|, 1)

    Measures how far the computed eigenvalue is from the Rayleigh quotient
    of its own eigenvector.  Zero for exact eigenpairs.
    """
    rq    = _rayleigh_quotient(A, v)
    denom = max(abs(lam), 1.0)
    return float(abs(lam - rq) / denom)


# ──────────────────────────────────────────────────────────────────────────────
# Eigenvalue condition numbers  (requires left eigenvectors)
# ──────────────────────────────────────────────────────────────────────────────

def eigenvalue_condition_numbers(A: np.ndarray) -> np.ndarray:
    """
    κ(λᵢ) = 1 / |yᵢᴴxᵢ|

    where xᵢ is the right eigenvector and yᵢ is the left eigenvector
    (normalised so ‖xᵢ‖₂ = ‖yᵢ‖₂ = 1).

    For a normal matrix (A*A = AA*) all κ(λᵢ) = 1.
    Large κ(λᵢ) indicates the eigenvalue is sensitive to perturbations.

    Uses NumPy eig which also returns left eigenvectors via solve.
    Returns array of condition numbers aligned with the eigenvalue ordering
    from the provided right eigenvectors.
    """
    A_c = A.astype(complex)
    # Get both right and left eigenvectors
    vals_r, vecs_r = np.linalg.eig(A_c)
    # Left eigenvectors = right eigenvectors of Aᴴ, re-sorted to match
    vals_l, vecs_l = np.linalg.eig(A_c.conj().T)

    # Match left to right by closest eigenvalue
    m      = A_c.shape[0]
    kappas = np.full(m, np.inf)

    for i, lam_r in enumerate(vals_r):
        # Find the matching left eigenvector
        diffs = np.abs(vals_l - lam_r)
        j     = int(np.argmin(diffs))
        x     = vecs_r[:, i]
        y     = vecs_l[:, j]
        x    /= np.linalg.norm(x)
        y    /= np.linalg.norm(y)
        dot   = abs(y.conj() @ x)
        kappas[i] = 1.0 / dot if dot > 1e-14 else np.inf

    return kappas


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def spectral_sensitivity(
    A:           np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray | None,
) -> dict:
    """
    Compute problem-specific sensitivity metrics (Section 3).

    Parameters
    ----------
    A            : coefficient matrix
    eigenvalues  : computed eigenvalues (complex array, magnitude-sorted)
    eigenvectors : computed eigenvectors (columns), or None

    Returns
    -------
    dict with keys:
        spectral_radius     float
        spectral_gap_abs    float   |λ₁| − |λ₂|
        spectral_gap_rel    float   (|λ₁| − |λ₂|) / |λ₁|
        eigenvalue_conds    np.ndarray or None   κ(λᵢ) for each i
        norm_A              float   ‖A‖₂
    """
    mags = np.abs(eigenvalues)
    idx  = np.argsort(mags)[::-1]   # magnitude descending
    mags_sorted = mags[idx]

    spectral_radius = float(mags_sorted[0]) if len(mags_sorted) > 0 else 0.0

    if len(mags_sorted) >= 2:
        gap_abs = float(mags_sorted[0] - mags_sorted[1])
        gap_rel = gap_abs / mags_sorted[0] if mags_sorted[0] > 0 else 0.0
    else:
        gap_abs = 0.0
        gap_rel = 0.0

    norm_A = float(np.linalg.norm(A.astype(complex), ord=2))

    # Eigenvalue condition numbers — only for full solvers with both eigenvector sets
    try:
        eig_conds = eigenvalue_condition_numbers(A)
        # Re-sort to match the input eigenvalue ordering
        eig_conds = eig_conds[idx]
    except Exception:
        eig_conds = None

    return {
        "spectral_radius":  spectral_radius,
        "spectral_gap_abs": gap_abs,
        "spectral_gap_rel": gap_rel,
        "eigenvalue_conds": eig_conds,
        "norm_A":           norm_A,
    }


def eigenpair_analysis(
    A:            np.ndarray,
    eigenvalues:  np.ndarray,
    eigenvectors: np.ndarray | None,
    params:       dict,
) -> dict:
    """
    Compute per-eigenpair quality metrics (Section 4).

    Parameters
    ----------
    A, eigenvalues, eigenvectors : from solver output
    params : dict with key 'sort_by' ('magnitude' | 'algebraic')

    Returns
    -------
    dict with keys:
        residual_norms      list[float]   ‖Avᵢ − λᵢvᵢ‖ / (‖A‖·‖vᵢ‖)
        rayleigh_accuracies list[float]   |λᵢ − ρ(vᵢ)| / max(|λᵢ|,1)
        orthogonality_error float         ‖VᴴV − I‖_F
        residual_prec       str           precision used for residuals
        norm_A              float
    """
    norm_A = float(np.linalg.norm(A.astype(complex), ord=2))

    if eigenvectors is None or eigenvectors.shape[1] == 0:
        return {
            "residual_norms":      [],
            "rayleigh_accuracies": [],
            "orthogonality_error": None,
            "residual_prec":       "N/A",
            "norm_A":              norm_A,
        }

    n_pairs = eigenvectors.shape[1]
    res_norms  = []
    rq_accs    = []
    prec_label = "float64 (standard)"

    for i in range(n_pairs):
        lam = complex(eigenvalues[i])
        v   = eigenvectors[:, i]

        res, prec = _eigenvector_residual_norm(A, lam, v, norm_A)
        res_norms.append(res)
        if i == 0:
            prec_label = prec

        rqa = _rayleigh_accuracy(A, lam, v)
        rq_accs.append(rqa)

    # Orthogonality of the eigenvector matrix
    V   = eigenvectors.astype(complex)
    k   = V.shape[1]
    orth_err = float(np.linalg.norm(V.conj().T @ V - np.eye(k), ord="fro"))

    return {
        "residual_norms":      res_norms,
        "rayleigh_accuracies": rq_accs,
        "orthogonality_error": orth_err,
        "residual_prec":       prec_label,
        "norm_A":              norm_A,
    }
