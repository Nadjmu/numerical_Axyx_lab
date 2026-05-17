"""
ui/solver_ui.py
===============
Streamlit sidebar widgets for eigensolver selection and parameters.

Returns
-------
render_solver_ui() -> dict with keys:
    solver_name    : str
    sort_by        : 'magnitude' | 'algebraic'
    rqi_shift      : float
    rqi_tol        : float
    rqi_max_iter   : int
    qr_max_iter    : int
    qr_tol         : float
    krylov_k       : int         — current Krylov dimension (or list if swept)
    krylov_k_values: list[int]
    krylov_tol     : float
    krylov_max_iter: int
    compare: {
        axis         : 'solver' | None
        solver_values: list[str]
    }
"""

from __future__ import annotations

import streamlit as st
from core.solvers import SOLVERS, PARTIAL_SOLVERS, DIRECT_SOLVERS

SOLVER_NOTES: dict[str, str] = {
    "NumPy eig": (
        "LAPACK dgeev — computes all eigenvalues and right eigenvectors.  "
        "Works for any square matrix; eigenvalues may be complex.  "
        "Reference solver — use for ground-truth comparison."
    ),
    "NumPy eigh (symmetric)": (
        "LAPACK dsyev — computes all eigenvalues for a symmetric matrix.  "
        "Returns real eigenvalues in ascending order.  Faster than eig.  "
        "Requires Hermitian matrix."
    ),
    "Rayleigh Quotient Iteration": (
        "Finds ONE eigenpair near the initial shift σ₀.  "
        "Cubic convergence near a simple eigenvalue.  "
        "Very sensitive to the choice of shift — set σ near the target eigenvalue."
    ),
    "Pure QR": (
        "Full QR iteration without shifts: Aₖ₊₁ = Rₖ Qₖ.  "
        "Eigenvalues appear on the diagonal of the limit.  "
        "Slow convergence — pedagogical baseline.  "
        "Shift-free: convergence rate governed by |λ₂/λ₁|."
    ),
    "Practical QR (Wilkinson shifts)": (
        "Hessenberg reduction + QR steps with Wilkinson shifts.  "
        "Industry-standard; typically quadratic to cubic convergence.  "
        "Mirrors LAPACK's implicit QR (dgehrd + dlahqr)."
    ),
    "Arnoldi Iteration": (
        "Krylov subspace method for any square matrix.  "
        "Builds an orthonormal basis and extracts Ritz values.  "
        "Best for finding a few dominant eigenvalues of a large matrix.  "
        "k = Krylov dimension (sweepable)."
    ),
    "Lanczos Iteration": (
        "Arnoldi specialised to symmetric matrices: uses three-term recurrence.  "
        "Ritz values from a symmetric tridiagonal — always real.  "
        "Full reorthogonalisation applied to suppress ghost eigenvalues.  "
        "k = Krylov dimension (sweepable)."
    ),
}

ALL_SOLVERS = list(SOLVERS.keys())

_KRYLOV_SOLVERS = {"Arnoldi Iteration", "Lanczos Iteration"}
_RQI_SOLVERS    = {"Rayleigh Quotient Iteration"}
_QR_SOLVERS     = {"Pure QR", "Practical QR (Wilkinson shifts)"}


def _parse_int_list(raw: str, lo: int, hi: int) -> list[int] | None:
    try:
        vals = [int(v.strip()) for v in raw.split(",") if v.strip()]
        vals = sorted(set(v for v in vals if lo <= v <= hi))
        return vals if vals else None
    except ValueError:
        return None


def render_solver_ui() -> dict:
    st.markdown("---")
    st.header("Solver")

    solver_name = st.selectbox("Method", ALL_SOLVERS)

    with st.expander("Solver info", expanded=False):
        st.caption(SOLVER_NOTES.get(solver_name, ""))

    # ── Eigenvalue sort order ─────────────────────────────────────────────────
    st.markdown("---")
    st.header("Eigenvalue ordering")
    sort_by = st.radio(
        "Sort eigenvalues by",
        ["magnitude", "algebraic"],
        horizontal=True,
        help=(
            "magnitude: descending |λ| (works for complex spectra).  "
            "algebraic: descending Re(λ) (meaningful for real/symmetric spectra)."
        ),
    )

    # ── Solver-specific parameters ────────────────────────────────────────────
    st.markdown("---")
    st.header("Solver parameters")

    # Defaults
    rqi_shift       = 0.0
    rqi_tol         = 1e-12
    rqi_max_iter    = 100
    qr_max_iter     = 300
    qr_tol          = 1e-10
    krylov_k        = 6
    krylov_k_values = [6]
    krylov_tol      = 1e-10
    krylov_max_iter = 300
    sweep_k         = False

    if solver_name in _RQI_SOLVERS:
        rqi_shift    = st.number_input(
            "Initial shift σ₀",
            value=0.0, format="%.4f",
            help="Starting shift for RQI.  Set near the target eigenvalue for faster convergence.",
        )
        rqi_tol      = float(st.select_slider(
            "Convergence tolerance",
            options=[1e-6, 1e-8, 1e-10, 1e-12, 1e-14],
            value=1e-12,
            key="rqi_tol",
        ))
        rqi_max_iter = int(st.number_input(
            "Max iterations", min_value=10, max_value=10_000, value=100, step=10,
            key="rqi_max_iter",
        ))

    if solver_name in _QR_SOLVERS:
        qr_max_iter = int(st.number_input(
            "Max iterations", min_value=10, max_value=10_000, value=300, step=10,
            key="qr_max_iter",
        ))
        qr_tol = float(st.select_slider(
            "Convergence tolerance",
            options=[1e-6, 1e-8, 1e-10, 1e-12, 1e-14],
            value=1e-10,
            key="qr_tol",
        ))

    if solver_name in _KRYLOV_SOLVERS:
        col_k, col_sw_k = st.columns([3, 1])
        with col_k:
            krylov_k = int(st.number_input(
                "Krylov dimension k",
                min_value=1, max_value=2_000, value=6, step=1,
                help="Number of Krylov vectors.  k ≤ m.",
                key="krylov_k_input",
            ))
        with col_sw_k:
            st.markdown("<br>", unsafe_allow_html=True)
            sweep_k = st.checkbox("Sweep", key="sweep_k",
                                  help="Run for multiple values of k.")

        krylov_k_values = [krylov_k]
        if sweep_k:
            raw_k    = st.text_input(
                "k values (comma-separated)", value=str(krylov_k),
                key="k_sweep_input", help="e.g. 2, 4, 8, 16",
            )
            parsed_k = _parse_int_list(raw_k, 1, 2_000)
            if parsed_k is None:
                st.warning("Enter valid integers ≥ 1.")
            else:
                krylov_k_values = parsed_k
            st.caption("✓ " + ",  ".join(str(v) for v in krylov_k_values))

        krylov_tol = float(st.select_slider(
            "Convergence tolerance",
            options=[1e-6, 1e-8, 1e-10, 1e-12, 1e-14],
            value=1e-10,
            key="krylov_tol",
        ))
        krylov_max_iter = int(st.number_input(
            "Max iterations", min_value=10, max_value=10_000, value=300, step=10,
            key="krylov_max_iter",
        ))

    # ── Solver compare ────────────────────────────────────────────────────────
    st.markdown("---")
    compare_solvers     = st.checkbox(
        "Compare solvers",
        key="compare_solvers_checkbox",
        help="Select multiple solvers — each becomes a separate series.",
    )
    solver_values       = [solver_name]
    solver_compare_axis = None

    if compare_solvers:
        solver_values = st.multiselect(
            "Solvers to compare",
            options = ALL_SOLVERS,
            default = [solver_name],
            key     = "compare_solver_values",
        )
        if not solver_values:
            st.warning("Select at least one solver.")
            solver_values = [solver_name]
        if solver_name in solver_values:
            solver_values = [solver_name] + [s for s in solver_values
                                             if s != solver_name]
        solver_compare_axis = "solver"
        st.caption("✓ " + ",  ".join(solver_values))

    # ── k sweep contributes to the global sweep axis ──────────────────────────
    # Reported via a flag so app.py can build the combined sweep product
    return {
        "solver_name":      solver_name,
        "sort_by":          sort_by,
        "rqi_shift":        rqi_shift,
        "rqi_tol":          rqi_tol,
        "rqi_max_iter":     rqi_max_iter,
        "qr_max_iter":      qr_max_iter,
        "qr_tol":           qr_tol,
        "krylov_k":         krylov_k,
        "krylov_k_values":  krylov_k_values,
        "krylov_tol":       krylov_tol,
        "krylov_max_iter":  krylov_max_iter,
        "sweep_k":          sweep_k,
        "compare": {
            "axis":          solver_compare_axis,
            "solver_values": solver_values,
        },
    }
