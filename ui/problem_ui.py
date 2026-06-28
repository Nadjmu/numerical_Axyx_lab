"""
ui/problem_ui.py
================
Streamlit sidebar widgets for matrix A and experiment axes.
Includes CPU/GPU device selector and optional A import from .npy file.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from core.problem_creation import (
    MATRIX_TYPE_NOTES,
    MATRIX_TYPE_PARAM_NOTES,
    STRUCTURE_NOTES,
    _BASE_GENERATORS,
    compatible_structures,
    load_npy,
    load_npz,
)
from core.device import gpu_available, gpu_info

MATRIX_TYPES   = list(_BASE_GENERATORS.keys())
ALL_STRUCTURES = list(STRUCTURE_NOTES.keys())


def _memory_estimate(m: int, dtype_label: str) -> str:
    bpe   = 8 if "64" in dtype_label else 4
    total = m * m * bpe
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1_000:
            return f"{total:.0f} {unit}"
        total /= 1_000
    return f"{total:.2f} TB"


def _parse_int_list(raw: str, lo: int, hi: int) -> list[int] | None:
    try:
        vals = [int(v.strip()) for v in raw.split(",") if v.strip()]
        vals = sorted(set(v for v in vals if lo <= v <= hi))
        return vals if vals else None
    except ValueError:
        return None


def _parse_order_list(raw: str) -> list[int] | None:
    return _parse_int_list(raw, -16, 0)


def structure_label(name: str, param: int) -> str:
    if name == "Sparse Block-Tridiagonal":
        return f"Block-Tridiag (bs={param})"
    if name == "Sparse Banded":
        return f"Banded ({param} diags)"
    return name


def _render_struct_param(s_name: str, m_ref: int, key_suffix: str = "") -> int:
    if s_name == "Sparse Block-Tridiagonal":
        max_bs = max(1, m_ref)
        return st.slider(
            "Block size" if not key_suffix else f"Block size — {s_name}",
            min_value=1, max_value=max_bs,
            value=min(4, max_bs),
            help="1 = near-tridiagonal, m = one dense block.",
            key=f"sbs{key_suffix}",
        )
    if s_name == "Sparse Banded":
        max_diags = max(1, 2 * m_ref - 1)
        return st.slider(
            "Non-zero diagonals" if not key_suffix else f"Non-zero diags — {s_name}",
            min_value=1, max_value=min(max_diags, 101),
            value=3, step=2,
            help="3 = tridiagonal, 5 = pentadiagonal, …",
            key=f"sbd{key_suffix}",
        )
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Device selector
# ─────────────────────────────────────────────────────────────────────────────

def render_device_selector() -> bool:
    """
    Render the CPU / GPU device selector at the top of the sidebar.
    Returns use_gpu : bool.
    """
    st.header("⚙️ Device")

    _gpu_avail = gpu_available()
    _info      = gpu_info() if _gpu_avail else {}

    if not _gpu_avail:
        st.radio(
            "Compute device", ["CPU"], index=0, horizontal=True,
            help="CuPy not found or no CUDA device available.  "
                 "Install with:  pip install cupy-cuda12x",
            disabled=True,
        )
        st.caption(
            "🟡 GPU not available.  Install CuPy:  `pip install cupy-cuda12x`"
        )
        return False

    device_choice = st.radio(
        "Compute device", ["CPU", "GPU"], index=0, horizontal=True,
        help="GPU uses CuPy.  All solvers are supported.",
    )
    use_gpu = (device_choice == "GPU")

    if use_gpu:
        devices = _info.get("devices", [])
        n       = _info.get("device_count", 0)
        label   = devices[0] if devices else "Unknown GPU"
        st.success(f"🟢 GPU  ·  {label}  ·  {n} device(s)")
    else:
        st.info("🔵 CPU  ·  NumPy / SciPy / LAPACK")

    st.markdown("---")
    return use_gpu


def render_problem_ui() -> dict:
    """Render all problem-creation sidebar widgets and return parameter dict."""

    # ── Device selector ───────────────────────────────────────────────────────
    use_gpu = render_device_selector()

    # ═════════════════════════════════════════════════════════════════════════
    # Matrix A
    # ═════════════════════════════════════════════════════════════════════════
    st.header("Matrix A")

    # ── Import toggle ─────────────────────────────────────────────────────────
    import_A = st.checkbox("Import A from .npy / .npz file", value=False, key="import_A_cb")

    imported_A_array = None
    matrix_type  = "Custom (imported)" if import_A else None
    seed         = 42
    type_param   = 4
    m            = 6
    structure    = "Dense"
    struct_param = 1
    make_hermitian = False
    make_pd        = False
    sweep_m        = False

    if import_A:
        uploaded_A = st.file_uploader(
            "Upload A  (.npy  or  .npz CSC sparse)",
            type=["npy", "npz"], key="upload_A",
            help=(
                "Dense:   np.save('A.npy', A)\n"
                "Sparse:  scipy.sparse.save_npz('A.npz', A_csc)"
            ),
        )
        if uploaded_A is not None:
            try:
                if uploaded_A.name.endswith(".npz"):
                    imported_A_array = load_npz(uploaded_A, use_gpu=False)
                else:
                    imported_A_array = load_npy(uploaded_A, expected_ndim=2,
                                                use_gpu=False)
                m_imp, _ = imported_A_array.shape
                fmt = ".npz (sparse→dense)" if uploaded_A.name.endswith(".npz") else ".npy"
                st.success(
                    f"Loaded A ({fmt}): shape {m_imp} × {m_imp},  "
                    f"dtype {imported_A_array.dtype},  "
                    f"nnz {int((imported_A_array != 0).sum()):,}"
                )
                m = m_imp
            except Exception as e:
                st.error(f"Could not load A: {e}")
                imported_A_array = None
        else:
            st.info("Upload a .npy or .npz file to use a custom matrix.")
        m_values = [m]
        st.caption("ℹ️ Structure, size, and symmetry controls are disabled for imported matrices.")

    else:
        # ── Standard controls ─────────────────────────────────────────────────
        matrix_type = st.selectbox("Type", MATRIX_TYPES, index=0)

        note = MATRIX_TYPE_NOTES.get(matrix_type, "")
        if note:
            st.caption(note)

        if matrix_type != "Hilbert":
            seed = st.slider("Seed", min_value=0, max_value=100, value=42, step=1)

        if matrix_type in MATRIX_TYPE_PARAM_NOTES:
            param_note = MATRIX_TYPE_PARAM_NOTES[matrix_type]
            if matrix_type == "Random SPD":
                type_param = st.slider(
                    "log₁₀(κ_target)", min_value=1, max_value=16, value=6,
                    help=param_note,
                )
                st.caption(f"Target κ(A) = 10^{type_param} = {10**type_param:.0e}")
            elif matrix_type == "Diagonal":
                type_param = st.slider(
                    "log₁₀(spread)", min_value=1, max_value=14, value=4,
                    help=param_note,
                )
                st.caption(f"Eigenvalue spread ≈ 10^{type_param}")
            elif matrix_type == "Tridiagonal Symm.":
                type_param = st.slider(
                    "Off-diagonal scale", min_value=1, max_value=10, value=1,
                    help=param_note,
                )

        col_m, col_sw_m = st.columns([3, 1])
        with col_m:
            m = int(st.number_input("Size (m)", min_value=2, max_value=5_000,
                                    value=6, step=1))
        with col_sw_m:
            st.markdown("<br>", unsafe_allow_html=True)
            sweep_m = st.checkbox("Sweep", key="sweep_m",
                                  help="Run for multiple sizes of m.")

        m_values = [m]
        if sweep_m:
            raw_m    = st.text_input("Sizes (comma-separated)", value=str(m),
                                     key="m_sweep_input", help="e.g. 4, 8, 16, 32")
            parsed_m = _parse_int_list(raw_m, 2, 5_000)
            if parsed_m is None:
                st.warning("Enter valid integers between 2 and 5000.")
            else:
                m_values = parsed_m
            st.caption("✓ " + ",  ".join(str(v) for v in m_values))

        allowed_structs = compatible_structures(matrix_type)
        structure = st.selectbox("Structure", allowed_structs)

        struct_note = STRUCTURE_NOTES.get(structure, "")
        if struct_note:
            st.caption(struct_note)

        m_ref        = min(m_values)
        struct_param = _render_struct_param(structure, m_ref)

        make_hermitian = st.checkbox("Hermitian  (A ← (A + Aᵀ) / 2)", value=False)
        make_pd        = st.checkbox("Positive Definite  (A ← AᵀA + αI)", value=False)

    # ── dtype (always shown) ──────────────────────────────────────────────────
    dtype_A_label = st.radio("A dtype",
                             ["float64  (double)", "float32  (single)"],
                             horizontal=True)
    dtype_A = np.float64 if "64" in dtype_A_label else np.float32

    if import_A and imported_A_array is not None:
        dtype_A = imported_A_array.dtype
        if imported_A_array.dtype not in (np.dtype("float64"), np.dtype("float32")):
            st.info(
                f"Detected dtype **{imported_A_array.dtype}** — preserved as-is.  "
                "Dtype selector applies only to generated matrices."
            )
        else:
            st.caption(f"ℹ️ Imported dtype **{imported_A_array.dtype}** preserved (selector ignored).")
    elif not import_A:
        mem = _memory_estimate(m_values[0], dtype_A_label)
        if m_values[0] ** 2 * (8 if "64" in dtype_A_label else 4) > 400_000_000:
            st.warning(f"Estimated dense allocation: **{mem}** — may be slow or OOM.")
        else:
            st.caption(f"Estimated dense memory: {mem}")

    # ═════════════════════════════════════════════════════════════════════════
    # Perturbation
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("Perturbation")

    perturb_A       = st.checkbox("Perturb A", value=False)
    perturb_A_order = -6
    order_A_values  = [-6]
    sweep_order_A   = False

    if perturb_A:
        col_oA, col_sw_oA = st.columns([3, 1])
        with col_oA:
            perturb_A_order = st.slider(
                "Order (A)  [10^k · ‖A‖]",
                min_value=-16, max_value=0, value=-6,
                help="Adds noise of magnitude 10^k × ‖A‖.",
            )
        with col_sw_oA:
            st.markdown("<br>", unsafe_allow_html=True)
            sweep_order_A = st.checkbox("Sweep", key="sweep_oA")

        order_A_values = [int(perturb_A_order)]
        if sweep_order_A:
            raw_oA    = st.text_input("Orders (comma-separated, −16 to 0)",
                                      value=str(int(perturb_A_order)),
                                      key="oA_sweep_input")
            parsed_oA = _parse_order_list(raw_oA)
            if parsed_oA is None:
                st.warning("Enter integers between −16 and 0.")
            else:
                order_A_values = parsed_oA
            st.caption("✓ " + ",  ".join(f"10^{v}" for v in order_A_values))
        else:
            st.caption(f"Perturbation magnitude ≈ 10^{perturb_A_order} × ‖A‖")

    # ═════════════════════════════════════════════════════════════════════════
    # Compare axis
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("Compare")
    st.caption(
        "Fix two of {matrix type, structure, solver} and compare the third.  "
        "Each compared value becomes a separate series (legend) in the plots."
    )

    compare_axis_label = st.radio(
        "Compare axis",
        ["None", "Matrix type", "Structure"],
        horizontal=True,
        key="compare_axis_radio",
    )

    matrix_type_values = [matrix_type]
    if compare_axis_label == "Matrix type":
        matrix_type_values = st.multiselect(
            "Matrix types to compare",
            options = MATRIX_TYPES,
            default = MATRIX_TYPES[:2],
            key     = "compare_matrix_types",
        )
        if not matrix_type_values:
            st.warning("Select at least one matrix type.")
            matrix_type_values = [matrix_type]
        incompat = [mt for mt in matrix_type_values
                    if structure not in compatible_structures(mt)]
        if incompat:
            st.warning(
                f"Structure '{structure}' is not compatible with: "
                f"{', '.join(incompat)}.  Those will use Dense instead."
            )
        st.caption("✓ " + ",  ".join(matrix_type_values))

    structure_values = [{"name": structure, "param": struct_param}]
    if compare_axis_label == "Structure":
        struct_options   = compatible_structures(matrix_type)
        selected_structs = st.multiselect(
            "Structures to compare",
            options = struct_options,
            default = [structure] if structure in struct_options else [struct_options[0]],
            key     = "compare_structures",
        )
        if not selected_structs:
            st.warning("Select at least one structure.")
            selected_structs = [structure]

        structure_values = []
        m_ref = min(m_values)
        for s_name in selected_structs:
            s_param = _render_struct_param(s_name, m_ref, key_suffix=f"_{s_name}")
            structure_values.append({"name": s_name, "param": s_param})

        st.caption("✓ " + ",  ".join(
            structure_label(sv["name"], sv["param"]) for sv in structure_values
        ))

    _axis_map        = {"None": None, "Matrix type": "matrix_type", "Structure": "structure"}
    compare_axis_key = _axis_map[compare_axis_label]

    active_sweeps = []
    if sweep_m       and len(m_values)       > 1: active_sweeps.append("m")
    if sweep_order_A and len(order_A_values) > 1: active_sweeps.append("perturb_A_order")

    if len(active_sweeps) > 1:
        st.warning("Multiple sweep axes — first used as x-axis for plots.")

    sweep_param = active_sweeps[0] if active_sweeps else None

    return {
        "matrix_type":       matrix_type,
        "seed":              seed,
        "type_param":        int(type_param),
        "m":                 m_values[0],
        "structure":         structure,
        "struct_param":      int(struct_param),
        "make_hermitian":    make_hermitian,
        "make_pd":           make_pd,
        "dtype_A":           dtype_A,
        "perturb_A":         perturb_A,
        "perturb_A_order":   int(perturb_A_order),
        "use_gpu":           use_gpu,
        "import_A":          import_A,
        "imported_A_array":  imported_A_array,
        "sweep": {
            "param":          sweep_param,
            "m_values":       m_values,
            "order_A_values": order_A_values,
        },
        "compare": {
            "axis":               compare_axis_key,
            "matrix_type_values": matrix_type_values,
            "structure_values":   structure_values,
        },
    }
