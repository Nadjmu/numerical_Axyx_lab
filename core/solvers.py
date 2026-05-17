"""
core/solvers.py
===============
Eigenvalue solver implementations for  Ax = λx.

GPU support
-----------
Every solver detects whether the input A is a CuPy array and dispatches to
the appropriate GPU routine.  All returned eigenvalue/eigenvector arrays are
always CPU numpy — the solvers call ``to_numpy`` internally before returning,
so display and analysis code never needs to handle CuPy arrays.

GPU solver mapping
------------------
| CPU                            | GPU                                     |
|--------------------------------|-----------------------------------------|
| numpy.linalg.eig               | cupy.linalg.eig                         |
| numpy.linalg.eigh              | cupy.linalg.eigh                        |
| numpy.linalg.solve  (RQI)      | cupy.linalg.solve                       |
| numpy.linalg.qr (Pure/Prac QR) | cupy.linalg.qr                          |
| Arnoldi loop (A @ v)           | cupy matmul — works natively on GPU     |
| Lanczos loop (A @ v)           | cupy matmul — works natively on GPU     |

Notes
-----
- Hessenberg reduction in Practical QR uses scipy.linalg.hessenberg (no CuPy
  equivalent).  This is a one-time O(m³) CPU cost; the iterative QR steps
  that dominate for large m then run on the GPU.
- The Arnoldi and Lanczos loops are device-agnostic: CuPy supports @, .conj(),
  linalg.norm, etc.  The small k×k Hessenberg/tridiagonal eigen-problem is
  solved on CPU (negligible cost).
- High-precision residuals in analysis.py always run on the CPU — no accuracy
  regression on the GPU path.
"""

from __future__ import annotations

import numpy as np

from core.device import get_array_module, is_gpu_array, to_numpy


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _xp(A):
    return get_array_module(is_gpu_array(A))


def _sort_cpu(vals_cpu: np.ndarray, vecs_cpu: np.ndarray | None,
              sort_by: str):
    """Sort CPU numpy eigenvalue/vector arrays."""
    if sort_by == "algebraic":
        idx = np.argsort(vals_cpu.real)[::-1]
    else:
        idx = np.argsort(np.abs(vals_cpu))[::-1]
    vals_sorted = vals_cpu[idx]
    vecs_sorted = vecs_cpu[:, idx] if vecs_cpu is not None else None
    return vals_sorted, vecs_sorted


# ──────────────────────────────────────────────────────────────────────────────
# Direct solvers
# ──────────────────────────────────────────────────────────────────────────────

def solve_eig(A, params: dict) -> dict:
    """
    All eigenvalues and right eigenvectors via eig.

    CPU: numpy.linalg.eig  (LAPACK dgeev)
    GPU: cupy.linalg.eig
    """
    gpu = is_gpu_array(A)
    xp  = _xp(A)
    vals, vecs = xp.linalg.eig(A.astype(complex))
    vals_cpu = to_numpy(vals).astype(complex)
    vecs_cpu = to_numpy(vecs).astype(complex)
    vals_cpu, vecs_cpu = _sort_cpu(vals_cpu, vecs_cpu, params.get("sort_by", "magnitude"))
    return {
        "eigenvalues":  vals_cpu,
        "eigenvectors": vecs_cpu,
        "method":       f"NumPy eig (LAPACK dgeev) {'[GPU]' if gpu else '[CPU]'}",
        "success":      True,
        "message":      f"All {len(vals_cpu)} eigenvalues computed.  Reference solver.",
        "converged_at": None,
        "history":      None,
    }


def solve_eigh(A, params: dict) -> dict:
    """
    All eigenvalues for a symmetric matrix.

    CPU: numpy.linalg.eigh  (LAPACK dsyev)
    GPU: cupy.linalg.eigh
    """
    gpu   = is_gpu_array(A)
    A_cpu = to_numpy(A).astype(float)
    if not np.allclose(A_cpu, A_cpu.T, atol=1e-8):
        raise ValueError(
            "'eigh' requires a symmetric matrix.  "
            "Enable the Hermitian option or choose a symmetric matrix type."
        )
    xp = _xp(A)
    vals, vecs = xp.linalg.eigh(A.astype(float))
    vals_cpu = to_numpy(vals).astype(complex)
    vecs_cpu = to_numpy(vecs).astype(complex)
    vals_cpu, vecs_cpu = _sort_cpu(vals_cpu, vecs_cpu, params.get("sort_by", "magnitude"))
    return {
        "eigenvalues":  vals_cpu,
        "eigenvectors": vecs_cpu,
        "method":       f"NumPy eigh (LAPACK dsyev) {'[GPU]' if gpu else '[CPU]'}",
        "success":      True,
        "message":      (
            f"All {len(vals_cpu)} eigenvalues computed (symmetric solver).  "
            "Real eigenvalues guaranteed."
        ),
        "converged_at": None,
        "history":      None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Rayleigh Quotient Iteration
# ──────────────────────────────────────────────────────────────────────────────

def solve_rqi(A, params: dict) -> dict:
    """
    Rayleigh Quotient Iteration — finds ONE eigenpair near the initial shift.

    The shifted solve uses cupy.linalg.solve on GPU.
    The loop is device-agnostic.
    """
    gpu      = is_gpu_array(A)
    xp       = _xp(A)
    m        = A.shape[0]
    shift    = float(params.get("rqi_shift", 0.0))
    tol      = float(params.get("rqi_tol", 1e-12))
    max_iter = int(params.get("rqi_max_iter", 100))
    seed     = int(params.get("seed", 42))

    rng    = np.random.default_rng(seed)
    v_cpu  = rng.standard_normal(m).astype(complex)
    v_cpu /= np.linalg.norm(v_cpu)
    v      = xp.asarray(v_cpu)
    sigma  = xp.array(shift + 0j, dtype=complex)
    eye    = xp.eye(m, dtype=complex)

    history   = []
    converged = False

    for k in range(max_iter):
        try:
            w = xp.linalg.solve(A.astype(complex) - sigma * eye, v)
        except Exception:
            w = xp.linalg.solve(A.astype(complex) - (sigma + 1e-14) * eye, v)

        v     = w / xp.linalg.norm(w)
        sigma = (v.conj() @ A.astype(complex) @ v).real + 0j

        res = float(xp.linalg.norm(A.astype(complex) @ v - sigma * v))
        history.append(res)

        if res < tol:
            converged = True
            break

    vals_cpu = to_numpy(xp.array([sigma])).astype(complex)
    vecs_cpu = to_numpy(v).astype(complex).reshape(-1, 1)

    return {
        "eigenvalues":  vals_cpu,
        "eigenvectors": vecs_cpu,
        "method":       f"Rayleigh Quotient Iteration {'[GPU]' if gpu else '[CPU]'}",
        "success":      converged,
        "message":      (
            f"Converged in {k + 1} iterations.  Final residual = {history[-1]:.2e}."
        ) if converged else (
            f"Did not converge in {max_iter} iterations.  Final residual = {history[-1]:.2e}."
        ),
        "converged_at": k + 1 if converged else None,
        "history":      history,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pure QR Algorithm
# ──────────────────────────────────────────────────────────────────────────────

def solve_pure_qr(A, params: dict) -> dict:
    """
    Pure QR Algorithm without shifts.  Uses cupy.linalg.qr on GPU.
    """
    gpu      = is_gpu_array(A)
    xp       = _xp(A)
    max_iter = int(params.get("qr_max_iter", 200))
    tol      = float(params.get("qr_tol", 1e-10))

    T       = A.astype(complex).copy()
    m       = T.shape[0]
    Q_accum = xp.eye(m, dtype=complex)
    history = []
    converged = False

    for k in range(max_iter):
        Q, R    = xp.linalg.qr(T)
        T       = R @ Q
        Q_accum = Q_accum @ Q

        if m >= 2:
            off_diag = float(xp.abs(T[1:, 0]).max())
            history.append(off_diag)
            if off_diag < tol:
                converged = True
                break

    vals_cpu = to_numpy(xp.diag(T)).astype(complex)
    vecs_cpu = to_numpy(Q_accum).astype(complex)
    vals_cpu, vecs_cpu = _sort_cpu(vals_cpu, vecs_cpu, params.get("sort_by", "magnitude"))

    return {
        "eigenvalues":  vals_cpu,
        "eigenvectors": vecs_cpu,
        "method":       f"Pure QR Algorithm {'[GPU]' if gpu else '[CPU]'}",
        "success":      converged,
        "message":      (
            f"Converged in {k + 1} QR steps.  Final off-diagonal = {history[-1]:.2e}."
        ) if (converged and history) else (
            f"Did not converge in {max_iter} iterations."
            + (f"  Final off-diagonal = {history[-1]:.2e}." if history else "")
        ),
        "converged_at": k + 1 if converged else None,
        "history":      history,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Practical QR Algorithm
# ──────────────────────────────────────────────────────────────────────────────

def solve_practical_qr(A, params: dict) -> dict:
    """
    Practical QR with Hessenberg reduction and Wilkinson shifts.

    Hessenberg reduction is done on CPU (scipy.linalg — no CuPy equivalent).
    The iterative QR steps that dominate for large m run on the GPU.
    """
    gpu      = is_gpu_array(A)
    xp       = _xp(A)
    max_iter = int(params.get("qr_max_iter", 200))
    tol      = float(params.get("qr_tol", 1e-10))

    import scipy.linalg as la
    A_cpu        = to_numpy(A).astype(complex)
    H_cpu, Z_cpu = la.hessenberg(A_cpu, calc_q=True)

    H       = xp.asarray(H_cpu)
    Q_accum = xp.asarray(Z_cpu)
    m       = H.shape[0]

    history     = []
    converged   = False
    total_iters = 0

    for outer in range(max_iter):
        converged_this = True
        for i in range(m - 1, 0, -1):
            if float(xp.abs(H[i, i - 1])) < tol:
                continue
            converged_this = False

            a = H[i - 1, i - 1]; b = H[i - 1, i]
            c = H[i, i - 1];      d = H[i, i]
            tr   = a + d
            det  = a * d - b * c
            disc = (tr ** 2 - 4 * det) ** 0.5
            lam1 = (tr + disc) / 2
            lam2 = (tr - disc) / 2
            shift = lam1 if abs(lam1 - d) < abs(lam2 - d) else lam2

            eye     = xp.eye(m, dtype=complex)
            Q, R    = xp.linalg.qr(H - shift * eye)
            H       = R @ Q + shift * eye
            Q_accum = Q_accum @ Q

            sub = float(xp.abs(H[i, i - 1]))
            history.append(sub)
            total_iters += 1

            if sub < tol:
                break

        if converged_this:
            converged = True
            break

    vals_cpu = to_numpy(xp.diag(H)).astype(complex)
    vecs_cpu = to_numpy(Q_accum).astype(complex)
    vals_cpu, vecs_cpu = _sort_cpu(vals_cpu, vecs_cpu, params.get("sort_by", "magnitude"))

    return {
        "eigenvalues":  vals_cpu,
        "eigenvectors": vecs_cpu,
        "method":       f"Practical QR (Hessenberg + Wilkinson shifts) {'[GPU]' if gpu else '[CPU]'}",
        "success":      converged,
        "message":      (
            f"Converged in {total_iters} QR steps.  "
            f"Final sub-diagonal entry = {history[-1]:.2e}."
        ) if (converged and history) else (
            f"Did not fully converge in {max_iter} iterations."
            + (f"  Final sub-diagonal = {history[-1]:.2e}." if history else "")
        ),
        "converged_at": total_iters if converged else None,
        "history":      history,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Arnoldi Iteration
# ──────────────────────────────────────────────────────────────────────────────

def solve_arnoldi(A, params: dict) -> dict:
    """
    Arnoldi Iteration — k dominant eigenpairs for any square matrix.

    The matvec A @ v runs on-device.
    The small k×k Hessenberg eigen-problem is solved on CPU (negligible).
    """
    gpu      = is_gpu_array(A)
    xp       = _xp(A)
    m        = A.shape[0]
    k        = min(int(params.get("krylov_k", min(6, m))), m)
    tol      = float(params.get("krylov_tol", 1e-10))
    max_iter = int(params.get("krylov_max_iter", 300))
    seed     = int(params.get("seed", 42))

    rng    = np.random.default_rng(seed)
    b_cpu  = rng.standard_normal(m).astype(complex)
    b_cpu /= np.linalg.norm(b_cpu)

    A_c = A.astype(complex)
    V   = xp.zeros((m, k + 1), dtype=complex)
    H   = xp.zeros((k + 1, k), dtype=complex)
    V[:, 0] = xp.asarray(b_cpu)

    breakdown = False
    j_final   = k
    history   = []

    for j in range(k):
        w = A_c @ V[:, j]
        for i in range(j + 1):
            H[i, j] = V[:, i].conj() @ w
            w        = w - H[i, j] * V[:, i]
        H[j + 1, j] = xp.linalg.norm(w)
        history.append(float(abs(H[j + 1, j])))

        if abs(H[j + 1, j]) < tol:
            j_final   = j + 1
            breakdown = True
            break
        V[:, j + 1] = w / H[j + 1, j]

        if j + 1 >= max_iter:
            j_final = j + 1
            break

    j_final = j_final if breakdown else k

    # Small eigen-problem on CPU
    H_k_cpu             = to_numpy(H[:j_final, :j_final])
    ritz_v_cpu, ritz_y_cpu = np.linalg.eig(H_k_cpu)

    # Ritz vectors on device then pull to CPU
    ritz_y      = xp.asarray(ritz_y_cpu)
    ritz_vec    = V[:, :j_final] @ ritz_y
    norms       = xp.linalg.norm(ritz_vec, axis=0, keepdims=True)
    norms       = xp.where(xp.abs(norms) < 1e-14, xp.ones_like(norms), norms)
    ritz_vec   /= norms
    ritz_v_cpu  = ritz_v_cpu.astype(complex)
    ritz_vec_cpu = to_numpy(ritz_vec).astype(complex)

    ritz_v_cpu, ritz_vec_cpu = _sort_cpu(ritz_v_cpu, ritz_vec_cpu,
                                          params.get("sort_by", "magnitude"))

    return {
        "eigenvalues":  ritz_v_cpu,
        "eigenvectors": ritz_vec_cpu,
        "method":       f"Arnoldi Iteration (k={j_final}) {'[GPU]' if gpu else '[CPU]'}",
        "success":      True,
        "message":      (
            f"Arnoldi factorisation with k={j_final} Krylov vectors.  "
            + ("Invariant subspace reached (lucky breakdown)." if breakdown
               else f"Final |h_{{k+1,k}}| = {history[-1]:.2e}.")
        ),
        "converged_at": j_final,
        "history":      history,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Lanczos Iteration
# ──────────────────────────────────────────────────────────────────────────────

def solve_lanczos(A, params: dict) -> dict:
    """
    Lanczos Iteration — k extreme eigenpairs for symmetric matrices.

    The three-term recurrence matvec runs on-device.
    The k×k tridiagonal eigen-problem is solved on CPU with numpy.linalg.eigh.
    """
    gpu = is_gpu_array(A)
    xp  = _xp(A)
    m   = A.shape[0]
    k   = min(int(params.get("krylov_k", min(6, m))), m)
    tol = float(params.get("krylov_tol", 1e-10))
    seed = int(params.get("seed", 42))

    A_cpu = to_numpy(A).astype(float)
    if not np.allclose(A_cpu, A_cpu.T, atol=1e-6):
        raise ValueError(
            "'Lanczos' requires a symmetric matrix.  "
            "Enable the Hermitian option or choose a symmetric matrix type."
        )

    A_f   = A.astype(float)
    rng   = np.random.default_rng(seed)
    b_cpu = rng.standard_normal(m)
    b_cpu /= np.linalg.norm(b_cpu)

    V     = xp.zeros((m, k + 1), dtype=float)
    alpha = xp.zeros(k, dtype=float)
    beta  = xp.zeros(k, dtype=float)
    V[:, 0] = xp.asarray(b_cpu)

    history   = []
    breakdown = False
    j_final   = k

    for j in range(k):
        w = A_f @ V[:, j]
        if j > 0:
            w -= beta[j - 1] * V[:, j - 1]
        alpha[j] = float(V[:, j] @ w)
        w        = w - alpha[j] * V[:, j]

        # Full reorthogonalisation
        for i in range(j + 1):
            w -= float(V[:, i] @ w) * V[:, i]

        beta_j = float(xp.linalg.norm(w))
        history.append(beta_j)

        if j < k - 1:
            beta[j] = beta_j
            if beta_j < tol:
                j_final   = j + 1
                breakdown = True
                break
            V[:, j + 1] = w / beta_j

    j_final = j_final if breakdown else k

    # Tridiagonal eigen-problem on CPU
    alpha_cpu = to_numpy(alpha[:j_final])
    beta_cpu  = to_numpy(beta[:j_final - 1])
    T_cpu     = (np.diag(alpha_cpu)
                 + np.diag(beta_cpu, 1)
                 + np.diag(beta_cpu, -1))
    ritz_vals_cpu, ritz_y_cpu = np.linalg.eigh(T_cpu)
    ritz_vals_cpu = ritz_vals_cpu.astype(complex)

    # Ritz vectors on device then pull to CPU
    ritz_y      = xp.asarray(ritz_y_cpu)
    ritz_vec    = V[:, :j_final] @ ritz_y
    norms       = xp.linalg.norm(ritz_vec, axis=0, keepdims=True)
    norms       = xp.where(xp.abs(norms) < 1e-14, xp.ones_like(norms), norms)
    ritz_vec   /= norms
    ritz_vec_cpu = to_numpy(ritz_vec).astype(complex)

    ritz_vals_cpu, ritz_vec_cpu = _sort_cpu(ritz_vals_cpu, ritz_vec_cpu,
                                             params.get("sort_by", "magnitude"))

    return {
        "eigenvalues":  ritz_vals_cpu,
        "eigenvectors": ritz_vec_cpu,
        "method":       f"Lanczos Iteration (k={j_final}) {'[GPU]' if gpu else '[CPU]'}",
        "success":      True,
        "message":      (
            f"Lanczos factorisation with k={j_final} Krylov vectors.  "
            "Full reorthogonalisation applied.  "
            + ("Invariant subspace reached." if breakdown
               else f"Final β = {history[-1]:.2e}.")
        ),
        "converged_at": j_final,
        "history":      history,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Solver registry
# ──────────────────────────────────────────────────────────────────────────────

SOLVERS: dict[str, callable] = {
    "NumPy eig":                          solve_eig,
    "NumPy eigh (symmetric)":             solve_eigh,
    "Rayleigh Quotient Iteration":        solve_rqi,
    "Pure QR":                            solve_pure_qr,
    "Practical QR (Wilkinson shifts)":    solve_practical_qr,
    "Arnoldi Iteration":                  solve_arnoldi,
    "Lanczos Iteration":                  solve_lanczos,
}

PARTIAL_SOLVERS = {"Rayleigh Quotient Iteration", "Arnoldi Iteration",
                   "Lanczos Iteration"}

DIRECT_SOLVERS = {"NumPy eig", "NumPy eigh (symmetric)"}
