"""
app.py
======
Numerical Eigenvalue Lab — Streamlit entry point.

Run with:
    streamlit run app.py

GPU support
-----------
All matrices are optionally transferred to the GPU after generation/import.
Solvers dispatch to CuPy routines when the input array is a CuPy array.
All results (eigenvalues, eigenvectors, A) are converted back to CPU numpy
before being stored in the instance dict, so display and analysis code
is unchanged.

Import support
--------------
A can be imported from an external .npy file via a checkbox in the sidebar.
The imported array bypasses type/structure/size controls.
Perturbation inherits the imported array's sparsity pattern automatically.
"""

from __future__ import annotations

import itertools
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from core.problem_creation import (
    create_matrix, apply_perturbation, matrix_info,
    compatible_structures, sparsity_mask,
)
from core.solvers import SOLVERS, PARTIAL_SOLVERS, DIRECT_SOLVERS
from core.analysis import spectral_sensitivity, eigenpair_analysis
from core.device import to_numpy, gpu_available, gpu_info
from ui.problem_ui import render_problem_ui, structure_label
from ui.solver_ui import render_solver_ui
from ui.analysis_ui import (
    render_array,
    render_heatmap,
    render_vector_heatmap,
    render_spectrum_plot,
    render_eigenvalue_table,
    render_spectral_radius_plot,
    render_spectral_gap_plot,
    render_eig_cond_plot,
    render_convergence_plot,
    _make_quality_plot,
    _sweep_axis_label,
    _sweep_x_values,
    _fmt,
    _bytes_to_human,
    _SERIES_COLORS,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Eigenvalue Lab", layout="wide")

st.title("Numerical Ax = λx Lab")
st.caption(
    "Explore eigenvalue algorithms, spectral sensitivity, and numerical stability.  "
    "Configure the problem and solver in the sidebar, then click **Run Experiment**."
)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    prob_params   = render_problem_ui()   # includes use_gpu, import_A
    solver_params = render_solver_ui()

    st.markdown("---")

    p_compare = prob_params["compare"]
    s_compare = solver_params["compare"]

    if p_compare["axis"] and s_compare["axis"]:
        st.warning(
            "Both a matrix/structure and a solver compare axis are active.  "
            "Using the matrix/structure axis."
        )
        s_compare = {"axis": None, "solver_values": [solver_params["solver_name"]]}

    if p_compare["axis"]:
        compare = p_compare
        compare["solver_values"] = [solver_params["solver_name"]]
    elif s_compare["axis"]:
        compare = {
            "axis":               "solver",
            "matrix_type_values": [prob_params["matrix_type"]],
            "structure_values":   [{"name": prob_params["structure"],
                                    "param": prob_params["struct_param"]}],
            "solver_values":      s_compare["solver_values"],
        }
    else:
        compare = {
            "axis":               None,
            "matrix_type_values": [prob_params["matrix_type"]],
            "structure_values":   [{"name": prob_params["structure"],
                                    "param": prob_params["struct_param"]}],
            "solver_values":      [solver_params["solver_name"]],
        }

    sw      = prob_params["sweep"]
    k_vals  = solver_params["krylov_k_values"]
    sweep_k = solver_params["sweep_k"]

    is_krylov = solver_params["solver_name"] in {"Arnoldi Iteration", "Lanczos Iteration"}
    if not is_krylov:
        k_vals  = [solver_params["krylov_k"]]
        sweep_k = False

    n_m    = len(sw["m_values"])
    n_oA   = len(sw["order_A_values"])
    n_k    = len(k_vals) if sweep_k else 1
    n_inst = n_m * n_oA * n_k

    if compare["axis"] == "matrix_type":
        n_series = len(compare["matrix_type_values"])
    elif compare["axis"] == "structure":
        n_series = len(compare["structure_values"])
    elif compare["axis"] == "solver":
        n_series = len(compare["solver_values"])
    else:
        n_series = 1

    n_total = n_series * n_inst
    if n_total > 1:
        st.caption(
            f"Experiment: **{n_series} series** × **{n_inst} instances** "
            f"= **{n_total} runs**."
        )

    # Device badge
    use_gpu = prob_params.get("use_gpu", False)
    if use_gpu:
        _info = gpu_info()
        _dev  = _info["devices"][0] if _info["devices"] else "GPU"
        st.caption(f"🟢 Running on GPU: {_dev}")
    else:
        st.caption("🔵 Running on CPU")

    run = st.button(
        "Run Experiment",
        type="primary",
        use_container_width=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────

if "series_list" not in st.session_state:
    st.session_state.series_list = None


# ──────────────────────────────────────────────────────────────────────────────
# Helper: run one instance
# ──────────────────────────────────────────────────────────────────────────────

def _run_instance(
    p:              dict,
    s:              dict,
    solver_name:    str,
    matrix_type_i:  str,
    structure_i:    str,
    struct_param_i: int,
    m_i:            int,
    order_A_i:      int,
    krylov_k_i:     int,
) -> dict:
    use_gpu = p.get("use_gpu", False)
    try:
        # ── Build A ───────────────────────────────────────────────────────────
        if p.get("import_A") and p.get("imported_A_array") is not None:
            # Preserve original dtype — do not cast imported matrices
            A_cpu = p["imported_A_array"]
            if use_gpu:
                import cupy as cp
                A = cp.asarray(A_cpu)
            else:
                A = A_cpu
            A_mask = sparsity_mask(A_cpu)   # inherit sparsity for perturbation
        else:
            A = create_matrix(
                matrix_type            = matrix_type_i,
                m                      = m_i,
                structure              = structure_i,
                struct_param           = struct_param_i,
                make_hermitian         = p["make_hermitian"],
                make_positive_definite = p["make_pd"],
                dtype                  = p["dtype_A"],
                seed                   = p["seed"],
                type_param             = p.get("type_param", 4),
                use_gpu                = use_gpu,
            )
            A_mask = None

        A_original = A.copy()

        if p["perturb_A"]:
            A = apply_perturbation(
                A,
                order                  = order_A_i,
                structure              = structure_i,
                struct_param           = struct_param_i,
                make_hermitian         = p["make_hermitian"],
                make_positive_definite = p["make_pd"],
                use_gpu                = use_gpu,
                custom_mask            = A_mask,
            )

        delta_A = (A - A_original) if p["perturb_A"] else None

        solver_p = {
            "sort_by":          s["sort_by"],
            "rqi_shift":        s["rqi_shift"],
            "rqi_tol":          s["rqi_tol"],
            "rqi_max_iter":     s["rqi_max_iter"],
            "qr_max_iter":      s["qr_max_iter"],
            "qr_tol":           s["qr_tol"],
            "krylov_k":         krylov_k_i,
            "krylov_tol":       s["krylov_tol"],
            "krylov_max_iter":  s["krylov_max_iter"],
            "seed":             p["seed"],
        }

        solver_fn = SOLVERS[solver_name]
        result    = solver_fn(A, solver_p)
        # result eigenvalues/vecs are already CPU numpy (solvers call to_numpy internally)

        # analysis always needs CPU numpy
        A_cpu_analysis = to_numpy(A)
        sensitivity = spectral_sensitivity(
            A_cpu_analysis, result["eigenvalues"], result["eigenvectors"]
        )
        quality = eigenpair_analysis(
            A_cpu_analysis, result["eigenvalues"], result["eigenvectors"], solver_p
        )

        return {
            "error":         None,
            "matrix_type":   matrix_type_i,
            "structure":     structure_i,
            "struct_param":  struct_param_i,
            "solver_name":   solver_name,
            "m":             m_i,
            "order_A":       order_A_i,
            "krylov_k":      krylov_k_i,
            "use_gpu":       use_gpu,
            "imported_A":    p.get("import_A", False),
            # All arrays stored as CPU numpy for display
            "A":             to_numpy(A),
            "A_original":    to_numpy(A_original),
            "delta_A":       to_numpy(delta_A) if delta_A is not None else None,
            "result":        result,
            "sensitivity":   sensitivity,
            "quality":       quality,
            "prob_params":   p,
            "solver_params": s,
        }

    except (ValueError, np.linalg.LinAlgError) as exc:
        return {"error": str(exc), "m": m_i, "order_A": order_A_i,
                "krylov_k": krylov_k_i, "solver_name": solver_name}
    except MemoryError:
        return {"error": "Out of memory — reduce m or use a sparser structure.",
                "m": m_i, "order_A": order_A_i, "krylov_k": krylov_k_i,
                "solver_name": solver_name}
    except Exception as exc:
        return {"error": f"Unexpected error: {exc}", "m": m_i,
                "order_A": order_A_i, "krylov_k": krylov_k_i,
                "solver_name": solver_name}


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build series specifications
# ──────────────────────────────────────────────────────────────────────────────

def _build_series_specs(p: dict, s: dict, cmp: dict) -> list[dict]:
    base_mt  = p["matrix_type"]
    base_st  = p["structure"]
    base_sp  = p["struct_param"]
    base_sol = s["solver_name"]
    axis     = cmp["axis"]

    if axis == "matrix_type":
        specs = []
        for mt in cmp["matrix_type_values"]:
            st_name = base_st if base_st in compatible_structures(mt) else "Dense"
            sp      = base_sp if st_name == base_st else 1
            specs.append({
                "label":        mt,
                "matrix_type":  mt,
                "structure":    st_name,
                "struct_param": sp,
                "solver_name":  base_sol,
            })
        return specs

    if axis == "structure":
        return [
            {
                "label":        structure_label(sv["name"], sv["param"]),
                "matrix_type":  base_mt,
                "structure":    sv["name"],
                "struct_param": sv["param"],
                "solver_name":  base_sol,
            }
            for sv in cmp["structure_values"]
        ]

    if axis == "solver":
        return [
            {
                "label":        sol,
                "matrix_type":  base_mt,
                "structure":    base_st,
                "struct_param": base_sp,
                "solver_name":  sol,
            }
            for sol in cmp["solver_values"]
        ]

    return [{
        "label":        _single_label(p, s),
        "matrix_type":  base_mt,
        "structure":    base_st,
        "struct_param": base_sp,
        "solver_name":  base_sol,
    }]


def _single_label(p: dict, s: dict) -> str:
    device = "GPU" if p.get("use_gpu") else "CPU"
    source = "imported" if p.get("import_A") else p["matrix_type"]
    return " | ".join([
        source,
        structure_label(p["structure"], p["struct_param"]),
        s["solver_name"],
        device,
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Experiment execution
# ──────────────────────────────────────────────────────────────────────────────

if run:
    p   = prob_params
    s   = solver_params
    sw  = p["sweep"]
    cmp = compare

    series_specs = _build_series_specs(p, s, cmp)
    _eff_k_vals  = k_vals

    combos = list(itertools.product(
        sw["m_values"],
        sw["order_A_values"],
        _eff_k_vals,
    ))

    total_runs  = len(series_specs) * len(combos)
    bar         = st.progress(0, text="Running…")
    run_idx     = 0
    series_list = []

    for spec in series_specs:
        instances = []
        for (m_i, oA_i, k_i) in combos:
            run_idx += 1
            bar.progress(
                run_idx / total_runs,
                text=(f"[{run_idx}/{total_runs}]  "
                      f"Series: {spec['label']}  |  "
                      f"m={m_i}, ord_A={oA_i}, k={k_i}  "
                      f"{'[GPU]' if p.get('use_gpu') else '[CPU]'}"),
            )
            inst = _run_instance(
                p, s,
                solver_name    = spec["solver_name"],
                matrix_type_i  = spec["matrix_type"],
                structure_i    = spec["structure"],
                struct_param_i = spec["struct_param"],
                m_i            = m_i,
                order_A_i      = oA_i,
                krylov_k_i     = k_i,
            )
            instances.append(inst)

        series_list.append({
            "label":     spec["label"],
            "instances": instances,
            "spec":      spec,
        })

    bar.empty()
    st.session_state.series_list = series_list
    st.session_state.compare     = cmp


# ──────────────────────────────────────────────────────────────────────────────
# Sweep / instance label helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_sweep_param(series_list: list) -> str | None:
    try:
        return series_list[0]["instances"][0]["prob_params"]["sweep"]["param"]
    except (IndexError, KeyError):
        return None


def _instance_label(inst: dict, n_total: int, idx: int,
                    sweep_param: str | None) -> str:
    if n_total == 1:
        return f"m={inst['m']}"
    if sweep_param == "m":
        return f"m = {inst['m']}"
    if sweep_param == "perturb_A_order":
        return f"ord_A = 10^{inst['order_A']}"
    if sweep_param == "krylov_k":
        return f"k = {inst.get('krylov_k', '?')}"
    return f"#{idx + 1}"


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 & 2 — single instance renderer
# ──────────────────────────────────────────────────────────────────────────────

def _render_instance(inst: dict) -> None:
    if inst.get("error"):
        st.error(inst["error"])
        return

    A      = inst["A"]       # always CPU numpy
    result = inst["result"]
    info   = matrix_info(A)
    m, n   = info["shape"]

    device_badge = "🟢 GPU" if inst.get("use_gpu") else "🔵 CPU"
    A_badge      = "📂 imported" if inst.get("imported_A") else "🔧 generated"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Problem creation (A)")
        st.caption(f"Device: {device_badge}  ·  A: {A_badge}")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Shape",     f"{m} × {n}")
        c2.metric("dtype",     info["dtype"])
        c3.metric("Non-zeros", f"{info['nnz']:,}")
        c4.metric("Density",   f"{info['density']:.2%}")
        c5.metric("Memory",    _bytes_to_human(info["memory_bytes"]))

        tab_tbl, tab_heat = st.tabs(["Entries", "Heatmap"])
        with tab_tbl:
            render_array("A", A)
        with tab_heat:
            render_heatmap("A", A)

        with st.expander("Zoom: crop submatrix  A[n_low : n_high, n_low : n_high]", expanded=False):
            crop_bound = min(A.shape[0], A.shape[1])
            crop_c1, crop_c2 = st.columns(2)
            with crop_c1:
                crop_low = int(st.number_input(
                    "n_low  (inclusive)", min_value=0,
                    max_value=crop_bound - 1, value=0, step=1, key="crop_low"))
            with crop_c2:
                crop_high = int(st.number_input(
                    "n_high  (exclusive)", min_value=1,
                    max_value=crop_bound, value=min(crop_bound, 10), step=1,
                    key="crop_high"))
            if crop_low >= crop_high:
                st.warning("n_low must be strictly less than n_high.")
            else:
                crop_sub = A[crop_low:crop_high, crop_low:crop_high]
                crop_sz  = crop_high - crop_low
                st.caption(
                    f"A[{crop_low}:{crop_high}, {crop_low}:{crop_high}]  —  "
                    f"{crop_sz} × {crop_sz}"
                )
                crop_tab_e, crop_tab_h = st.tabs(["Entries", "Heatmap"])
                with crop_tab_e:
                    render_array(f"A[{crop_low}:{crop_high}]", crop_sub)
                with crop_tab_h:
                    render_heatmap(f"A[{crop_low}:{crop_high}]", crop_sub)

        if inst["delta_A"] is not None:
            st.markdown("**Perturbation ΔA**")
            tab_tbl_d, tab_heat_d = st.tabs(["ΔA entries", "ΔA heatmap"])
            with tab_tbl_d:
                render_array("ΔA", inst["delta_A"])
            with tab_heat_d:
                render_heatmap("ΔA", inst["delta_A"])

    with col2:
        st.subheader(f"2. Eigenpairs via {result['method']}")

        if result["success"]:
            st.success(result["message"])
        else:
            st.warning(result["message"])

        vals    = result["eigenvalues"]
        vecs    = result["eigenvectors"]
        n_pairs = len(vals)

        sens = inst["sensitivity"]
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Eigenvalues computed", str(n_pairs))
        mc2.metric("Spectral radius ρ(A)", f"{sens['spectral_radius']:.4e}")
        mc3.metric("Relative spectral gap",
                   f"{sens['spectral_gap_rel']:.4f}" if n_pairs >= 2 else "N/A")

        render_spectrum_plot(vals, title=result["method"])

        with st.expander("Filter eigenvalues by range", expanded=False):
            ef_n_total = len(vals)
            ef_re_all  = vals.real
            ef_im_all  = vals.imag
            st.caption(
                f"Select a rectangle in the complex plane.  "
                f"Total eigenvalues: {ef_n_total}."
            )
            ef_c1, ef_c2 = st.columns(2)
            with ef_c1:
                st.markdown("**Real part**")
                ef_re_lo = st.number_input("Re  min", value=float(ef_re_all.min()),
                                           format="%.4e", key="ef_re_lo")
                ef_re_hi = st.number_input("Re  max", value=float(ef_re_all.max()),
                                           format="%.4e", key="ef_re_hi")
            with ef_c2:
                st.markdown("**Imaginary part**")
                ef_im_lo = st.number_input("Im  min", value=float(ef_im_all.min()),
                                           format="%.4e", key="ef_im_lo")
                ef_im_hi = st.number_input("Im  max", value=float(ef_im_all.max()),
                                           format="%.4e", key="ef_im_hi")

            ef_mask = (
                (ef_re_all >= ef_re_lo) & (ef_re_all <= ef_re_hi) &
                (ef_im_all >= ef_im_lo) & (ef_im_all <= ef_im_hi)
            )
            ef_n_in = int(ef_mask.sum())
            ef_pct  = 100.0 * ef_n_in / ef_n_total if ef_n_total > 0 else 0.0

            st.metric("In range", f"{ef_n_in} / {ef_n_total}  ({ef_pct:.1f}%)")

            if ef_n_in == 0:
                st.warning("No eigenvalues in this range — adjust the bounds.")
            else:
                ef_in  = vals[ef_mask]
                ef_out = vals[~ef_mask]

                ef_fig, ef_ax = plt.subplots(figsize=(5, 4))
                if len(ef_out) > 0:
                    ef_ax.scatter(ef_out.real, ef_out.imag, color="#cccccc",
                                  s=25, linewidths=0, zorder=2,
                                  label=f"Outside  ({len(ef_out)})")
                ef_ax.scatter(ef_in.real, ef_in.imag, color="#e74c3c",
                              s=55, edgecolors="#922b21", linewidths=0.6,
                              zorder=4, label=f"In range  ({ef_n_in},  {ef_pct:.1f}%)")
                ef_ax.axhline(0, color="#bbbbbb", lw=0.7, ls="--")
                ef_ax.axvline(0, color="#bbbbbb", lw=0.7, ls="--")
                ef_ax.plot(
                    [ef_re_lo, ef_re_hi, ef_re_hi, ef_re_lo, ef_re_lo],
                    [ef_im_lo, ef_im_lo, ef_im_hi, ef_im_hi, ef_im_lo],
                    color="#e74c3c", lw=1.2, ls="--", zorder=5,
                )
                ef_ax.set_xlabel("Re(λ)", fontsize=9)
                ef_ax.set_ylabel("Im(λ)", fontsize=9)
                ef_ax.set_title(
                    f"Eigenvalue filter — {ef_n_in}/{ef_n_total} selected ({ef_pct:.1f}%)",
                    fontsize=9, fontweight="bold",
                )
                ef_ax.legend(fontsize=8)
                ef_ax.tick_params(labelsize=8)
                ef_fig.tight_layout()
                st.pyplot(ef_fig, use_container_width=True)
                plt.close(ef_fig)

                with st.expander(f"Table of {ef_n_in} filtered eigenvalues", expanded=False):
                    render_eigenvalue_table(ef_in)

        with st.expander("Eigenvalue table", expanded=False):
            render_eigenvalue_table(vals)

        if vecs is not None and vecs.shape[1] > 0:
            MAX_SHOW_ALL = 20
            if n_pairs <= MAX_SHOW_ALL:
                eig_idx = st.selectbox(
                    f"Select eigenvector  (1 – {n_pairs})",
                    options     = range(n_pairs),
                    format_func = lambda i: (
                        f"λ_{i+1} = {_fmt(vals[i])}  (|λ|={abs(vals[i]):.4e})"
                    ),
                )
            else:
                eig_idx = st.number_input(
                    f"Eigenvector index (1 – {n_pairs})",
                    min_value=1, max_value=n_pairs, value=1, step=1,
                ) - 1

            v = vecs[:, eig_idx]
            col_v, col_empty = st.columns([1, 5])
            with col_v:
                st.metric("‖v‖₂", f"{float(np.linalg.norm(v)):.4f}")
                tab_tv, tab_hv = st.tabs(["Entries", "Heatmap"])
                with tab_tv:
                    render_array(f"v_{eig_idx+1}", v)
                with tab_hv:
                    render_vector_heatmap(f"v_{eig_idx+1}", v)

        quality = inst["quality"]
        with st.expander("Per-eigenpair quality metrics", expanded=False):
            res_norms = quality.get("residual_norms", [])
            rq_accs   = quality.get("rayleigh_accuracies", [])
            orth_err  = quality.get("orthogonality_error")

            eps = np.finfo(float).eps
            if res_norms:
                df_rows = []
                for i, (rn, ra) in enumerate(zip(res_norms, rq_accs)):
                    df_rows.append({
                        "i":        i + 1,
                        "λᵢ":       _fmt(vals[i]),
                        "|λᵢ|":     f"{abs(vals[i]):.4e}",
                        "‖Avᵢ−λvᵢ‖/(‖A‖‖vᵢ‖)": f"{rn:.2e}",
                        "|λ̃−ρ(v)|/max(|λ|,1)":  f"{ra:.2e}",
                    })
                import pandas as pd
                st.dataframe(pd.DataFrame(df_rows), use_container_width=True,
                             hide_index=True)

            if orth_err is not None:
                orth_ratio = orth_err / eps
                st.metric("‖VᴴV − I‖_F", f"{orth_err:.2e}",
                          help="Orthogonality of eigenvector matrix.")
                if orth_ratio < 100:
                    st.success(f"Near-perfect orthogonality (ratio = {orth_ratio:.1e}).")
                elif orth_ratio < 1e6:
                    st.info(f"Moderate orthogonality loss (ratio = {orth_ratio:.1e}).")
                else:
                    st.warning(f"Severe orthogonality loss (ratio = {orth_ratio:.1e}).")

            st.caption(f"Residuals computed in: **{quality.get('residual_prec', 'N/A')}**")

    st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# Sections 3–8
# ──────────────────────────────────────────────────────────────────────────────

def _render_section3(series_list: list) -> None:
    st.subheader("3. Problem-specific sensitivity metrics")
    sweep_param = _get_sweep_param(series_list)
    st.markdown("**Spectral radius  ρ(A)**")
    render_spectral_radius_plot(series_list, sweep_param)
    st.markdown("**Spectral gap  (|λ₁|−|λ₂|)/|λ₁|**")
    render_spectral_gap_plot(series_list, sweep_param)
    st.markdown("**Eigenvalue condition numbers  κ(λᵢ)**")
    render_eig_cond_plot(series_list, sweep_param)


def _render_section4(series_list: list) -> None:
    st.subheader("4. Solution quality metrics")
    sweep_param = _get_sweep_param(series_list)
    st.markdown("**Eigenvector residual**")
    _make_quality_plot(
        series_list, sweep_param,
        metric_fn    = lambda inst: (
            float(np.mean(inst["quality"]["residual_norms"]))
            if inst["quality"]["residual_norms"] else None
        ),
        title        = "Mean eigenvector residual",
        ylabel       = "log10(mean ||Av_i - l_i v_i|| / (||A|| ||v_i||))",
        color_single = "#2980b9",
        single_label = "Residual",
    )
    st.markdown("**Rayleigh quotient accuracy**")
    _make_quality_plot(
        series_list, sweep_param,
        metric_fn    = lambda inst: (
            float(np.mean(inst["quality"]["rayleigh_accuracies"]))
            if inst["quality"]["rayleigh_accuracies"] else None
        ),
        title        = "Mean Rayleigh quotient deviation",
        ylabel       = "log10(mean |RQ - rho(v)| / max(|lambda|, 1))",
        color_single = "#e67e22",
        single_label = "RQ deviation",
    )
    st.markdown("**Eigenvector orthogonality  ‖VᴴV−I‖_F**")
    _make_quality_plot(
        series_list, sweep_param,
        metric_fn    = lambda inst: inst["quality"].get("orthogonality_error"),
        title        = "Eigenvector orthogonality error",
        ylabel       = "log10(||V^H V - I||_F)",
        color_single = "#8e44ad",
        single_label = "Orth. error",
    )


def _render_section5(active_inst: dict) -> None:
    st.subheader("5. Solver behaviour metrics")
    if active_inst.get("error"):
        st.error(active_inst["error"])
        return
    render_convergence_plot(active_inst)


def _render_section6(active_inst: dict) -> None:
    st.subheader("6. Structural metrics")
    if active_inst.get("error"):
        st.error(active_inst["error"])
        return

    A    = active_inst["A"]   # CPU numpy
    vals = active_inst["result"]["eigenvalues"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Eigenvalue distribution by magnitude**")
        mags = np.abs(vals)
        sns.set_theme(style="whitegrid", font_scale=0.9)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.hist(np.log10(mags + 1e-300), bins=min(len(mags), 30),
                color="#2980b9", alpha=0.8, edgecolor="#1a5276", linewidth=0.4)
        ax.set_xlabel("log10(|lambda|)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title("Distribution of log10(|lambda|)", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.markdown("**Eigenvalue gaps (consecutive |λ| differences)**")
        mags_sorted = np.sort(mags)[::-1]
        if len(mags_sorted) >= 2:
            gaps     = np.diff(mags_sorted)
            rel_gaps = np.abs(gaps) / (mags_sorted[:-1] + 1e-300)
            fig2, ax2 = plt.subplots(figsize=(5, 3.2))
            ax2.semilogy(np.arange(1, len(rel_gaps) + 1), rel_gaps + 1e-300,
                         color="#e74c3c", lw=1.5, marker="o",
                         markersize=3, markerfacecolor="white")
            ax2.set_xlabel("Gap index (magnitude-sorted)", fontsize=9)
            ax2.set_ylabel("Relative gap |li - li+1| / |li|", fontsize=9)
            ax2.set_title("Consecutive relative eigenvalue gaps", fontsize=10,
                          fontweight="bold")
            ax2.tick_params(labelsize=8)
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
        else:
            st.caption("Need at least 2 eigenvalues for gap plot.")

    st.markdown("**Matrix normality check**")
    A_c      = A.astype(complex)
    comm     = float(np.linalg.norm(A_c.conj().T @ A_c - A_c @ A_c.conj().T, ord="fro"))
    norm_A   = float(np.linalg.norm(A_c, ord="fro"))
    rel_comm = comm / (norm_A ** 2 + 1e-300)

    col_n1, col_n2 = st.columns(2)
    col_n1.metric("‖AᴴA − AAᴴ‖_F", f"{comm:.4e}",
                  help="Zero for normal matrices.")
    col_n2.metric("Relative commutator", f"{rel_comm:.4e}",
                  help="‖AᴴA−AAᴴ‖_F / ‖A‖_F².")

    eps = np.finfo(float).eps
    if rel_comm < 100 * eps:
        st.success("A is effectively normal — all eigenvalue condition numbers κ(λᵢ) ≈ 1.")
    elif rel_comm < 1e-4:
        st.info("A is approximately normal.")
    else:
        st.warning(
            "A is non-normal — eigenvalues may be sensitive to perturbations.  "
            "Eigenvalue condition numbers κ(λᵢ) > 1."
        )


def _render_section7(active_inst: dict, series_list: list) -> None:
    st.header("7. Summary")
    if active_inst.get("error"):
        st.error(active_inst["error"])
        return

    p    = active_inst.get("prob_params", {})
    s    = active_inst.get("solver_params", {})
    res  = active_inst["result"]
    sens = active_inst["sensitivity"]
    qual = active_inst["quality"]
    vals = res["eigenvalues"]
    n    = len(vals)

    device_str = "GPU" if active_inst.get("use_gpu") else "CPU"
    source_str = "imported (.npy)" if active_inst.get("imported_A") else active_inst.get("matrix_type", "N/A")

    lines = [
        "=== Numerical Ax = λx Lab — Experiment Summary ===",
        "",
        "-- MATRIX --",
        f"  Source        : {source_str}",
        f"  Structure     : {active_inst.get('structure', 'N/A')} "
        f"(param={active_inst.get('struct_param', 'N/A')})",
        f"  Size          : {active_inst['m']} × {active_inst['m']}",
        f"  dtype         : {p.get('dtype_A', 'N/A')}",
        f"  Hermitian     : {p.get('make_hermitian', False)}",
        f"  Positive def. : {p.get('make_pd', False)}",
        f"  Perturbed     : {p.get('perturb_A', False)}"
        + (f" (order 10^{active_inst.get('order_A', 'N/A')})" if p.get("perturb_A") else ""),
        f"  Device        : {device_str}",
        "",
        "-- SOLVER --",
        f"  Method        : {res.get('method', 'N/A')}",
        f"  Sort by       : {s.get('sort_by', 'magnitude')}",
        f"  Converged     : {res.get('success', 'N/A')}",
        f"  Converged at  : {res.get('converged_at', 'N/A')} iterations",
        f"  Message       : {res.get('message', '')}",
        "",
        "-- SENSITIVITY METRICS (Section 3) --",
        f"  Spectral radius ρ(A)   : {sens['spectral_radius']:.6e}",
        f"  Spectral gap (abs)     : {sens['spectral_gap_abs']:.6e}",
        f"  Spectral gap (rel)     : {sens['spectral_gap_rel']:.6e}",
        f"  ‖A‖₂                   : {sens['norm_A']:.6e}",
        "",
        "-- SOLUTION QUALITY METRICS (Section 4) --",
    ]

    res_norms = qual.get("residual_norms", [])
    rq_accs   = qual.get("rayleigh_accuracies", [])
    orth_err  = qual.get("orthogonality_error")
    prec      = qual.get("residual_prec", "N/A")

    if res_norms:
        lines.append(f"  Mean eigenvector residual  : {float(np.mean(res_norms)):.4e}")
        lines.append(f"  Max  eigenvector residual  : {float(np.max(res_norms)):.4e}")
    if rq_accs:
        lines.append(f"  Mean RQ deviation          : {float(np.mean(rq_accs)):.4e}")
        lines.append(f"  Max  RQ deviation          : {float(np.max(rq_accs)):.4e}")
    if orth_err is not None:
        lines.append(f"  Eigenvector orth. error    : {orth_err:.4e}")
    lines.append(f"  Residual precision         : {prec}")

    lines += ["", "-- EIGENVALUES (top 10 by magnitude) --"]
    for i in range(min(10, n)):
        lam = vals[i]
        lines.append(
            f"  λ_{i+1:2d}: Re={lam.real:+.6e}  Im={lam.imag:+.6e}  |λ|={abs(lam):.6e}"
        )
    if n > 10:
        lines.append(f"  ... ({n - 10} more eigenvalues not shown)")

    lines += [
        "",
        "-- EXPERIMENT SCALE --",
        f"  Series  : {len(series_list)}",
        f"  Instances per series: {len(series_list[0]['instances']) if series_list else 0}",
        "",
        "=== END SUMMARY ===",
    ]

    st.text_area(
        "Copy and paste this into an LLM to verify results:",
        value="\n".join(lines),
        height=420,
    )


def _render_section8(active_inst: dict, series_list: list) -> None:
    st.header("8. Save results")
    st.caption("Coming soon — PDF export of all plots and the summary text.")


# ──────────────────────────────────────────────────────────────────────────────
# Main display
# ──────────────────────────────────────────────────────────────────────────────

series_list = st.session_state.series_list

if series_list is None:
    st.info(
        "Configure the problem and solver in the sidebar, "
        "then click **Run Experiment**."
    )
else:
    sweep_param = _get_sweep_param(series_list)

    n_series = len(series_list)
    if n_series == 1:
        active_series = series_list[0]
    else:
        series_labels = [s["label"] for s in series_list]
        si = st.selectbox(
            f"Series  ({n_series} total — compare axis)",
            options     = range(n_series),
            format_func = lambda i: series_labels[i],
            index       = 0,
        )
        active_series = series_list[si]

    instances = active_series["instances"]
    n_inst    = len(instances)

    if n_inst == 1:
        active_inst = instances[0]
    else:
        inst_labels = [
            _instance_label(inst, n_inst, i, sweep_param)
            for i, inst in enumerate(instances)
        ]
        ii = st.selectbox(
            f"Instance  ({n_inst} per series — sweep axis)",
            options     = range(n_inst),
            format_func = lambda i: inst_labels[i],
            index       = 0,
        )
        active_inst = instances[ii]

    _render_instance(active_inst)

    col3, col4 = st.columns(2)
    with col3:
        _render_section3(series_list)
    with col4:
        _render_section4(series_list)
    st.divider()

    col5, col6 = st.columns(2)
    with col5:
        _render_section5(active_inst)
    with col6:
        _render_section6(active_inst)
    st.divider()

    _render_section7(active_inst, series_list)
    st.divider()

    _render_section8(active_inst, series_list)
