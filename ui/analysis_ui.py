"""
ui/analysis_ui.py
=================
Streamlit display functions for eigenvalue experiment results.

Functions
---------
render_array(label, arr)            – scrollable dataframe
render_heatmap(label, arr)          – seaborn diverging heatmap
render_vector_heatmap(label, arr)   – single-column heatmap
render_spectrum_plot(eigenvalues)   – complex plane scatter + magnitude bar
render_eigenvector_display(v, idx)  – single eigenvector heatmap + stats
_fmt(val)                           – number formatter
_bytes_to_human(n)                  – byte → human string
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import seaborn as sns
import streamlit as st

from core.problem_creation import matrix_info

_ANNOT_MAX   = 20
_HEATMAP_MAX = 150
_DISPLAY_MAX = 200

_SERIES_COLORS = [
    "#2980b9", "#e74c3c", "#27ae60", "#8e44ad",
    "#e67e22", "#16a085", "#c0392b", "#2c3e50",
    "#f39c12", "#1abc9c",
]


# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(val, decimals: int = 6) -> str:
    if val is None:
        return "N/A"
    try:
        v = complex(val)
    except (TypeError, ValueError):
        return str(val)
    if np.isinf(v.real) or np.isinf(v.imag):
        return "∞"
    if np.isnan(v.real) or np.isnan(v.imag):
        return "NaN"
    if v.imag == 0:
        r = v.real
        if r == 0.0:
            return "0"
        return f"{r:.{decimals}e}"
    return f"{v.real:.4e} + {v.imag:.4e}i"


def _bytes_to_human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000:
            return f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


# ──────────────────────────────────────────────────────────────────────────────
# Seaborn helpers
# ──────────────────────────────────────────────────────────────────────────────

def _apply_style() -> None:
    sns.set_theme(style="white", font_scale=0.9)


def _diverging_cmap():
    return sns.diverging_palette(220, 20, as_cmap=True)


def _annotate_fmt(data: np.ndarray) -> str:
    absmax = np.nanmax(np.abs(data))
    if absmax == 0:
        return ".2f"
    if absmax < 0.01 or absmax >= 1e4:
        return ".2e"
    return ".3f"


def _symlog_norm(data: np.ndarray):
    """
    SymLogNorm centred at zero.
    Linear in [-linthresh, linthresh] so exact zeros read as white;
    log-scaled outside so order-of-magnitude differences get strong colour contrast.
    Returns (norm, absmax).  norm is None when the data is all-zero.
    """
    absmax = float(np.nanmax(np.abs(data)))
    if absmax == 0:
        return None, 1.0
    linthresh = max(absmax * 1e-3, np.finfo(float).tiny * 10)
    norm = mcolors.SymLogNorm(
        linthresh=linthresh, linscale=0.5,
        vmin=-absmax, vmax=absmax, base=10,
    )
    return norm, absmax


# ──────────────────────────────────────────────────────────────────────────────
# 2-D matrix heatmap
# ──────────────────────────────────────────────────────────────────────────────

def render_heatmap(label: str, arr: np.ndarray) -> None:
    _apply_style()
    is_cx  = np.iscomplexobj(arr)
    parts  = [("Re", arr.real), ("Im", arr.imag)] if is_cx else [("", arr)]
    m, n   = arr.shape

    if m > _HEATMAP_MAX or n > _HEATMAP_MAX:
        ncols = 2 if is_cx else 1
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
        if ncols == 1:
            axes = [axes]
        for ax, (part_label, data) in zip(axes, parts):
            ax.spy(data, markersize=max(1, 180 // max(m, n)), color="#2166ac")
            ax.set_title(f"{label}  [{part_label}]" if is_cx else label,
                         fontsize=10, fontweight="bold", pad=8)
            ax.set_xlabel("column", fontsize=8)
            ax.set_ylabel("row",    fontsize=8)
            ax.xaxis.set_label_position("bottom")
            ax.xaxis.tick_bottom()
        st.caption(
            f"Matrix is {m}×{n} — showing non-zero pattern"
            + ("  (Re | Im)." if is_cx else ".")
        )
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    annot     = (m <= _ANNOT_MAX and n <= _ANNOT_MAX)
    cell_size = max(0.4, min(0.9, 8.0 / max(m, n)))
    ncols = 2 if is_cx else 1
    fig_w = min(n * cell_size * ncols + 1.5 * ncols, 14.0)
    fig_h = min(m * cell_size + 1.2, 8.0)

    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, fig_h))
    if ncols == 1:
        axes = [axes]

    for ax, (part_label, data) in zip(axes, parts):
        norm, absmax = _symlog_norm(data)
        fmt          = _annotate_fmt(data) if annot else ""
        norm_kw      = ({"norm": norm}
                        if norm is not None
                        else {"center": 0, "vmin": -absmax, "vmax": absmax})
        sns.heatmap(
            data, ax=ax, cmap=_diverging_cmap(),
            annot=annot, fmt=fmt,
            annot_kws={"size": max(6, min(10, int(80 / max(m, n))))},
            linewidths=0.4 if m <= 40 else 0.0,
            linecolor="#cccccc", square=True,
            cbar_kws={"shrink": 0.75, "label": "value (log scale)"},
            xticklabels=n <= 30, yticklabels=m <= 30,
            **norm_kw,
        )
        ax.set_title(f"{label}  [{part_label}]" if is_cx else label,
                     fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("column", fontsize=8)
        ax.set_ylabel("row",    fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        if n > 10:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 1-D vector heatmap
# ──────────────────────────────────────────────────────────────────────────────

def render_vector_heatmap(label: str, arr: np.ndarray) -> None:
    _apply_style()
    data = arr.real if np.iscomplexobj(arr) else arr
    m    = data.shape[0]

    if m > _HEATMAP_MAX:
        fig, ax  = plt.subplots(figsize=(1.2, 5))
        d2d      = data[:_HEATMAP_MAX].reshape(-1, 1)
        norm, _  = _symlog_norm(d2d)
        norm_kw  = ({"norm": norm} if norm is not None
                    else {"center": 0, "vmin": -float(np.nanmax(np.abs(d2d))) or 1.0,
                          "vmax":  float(np.nanmax(np.abs(d2d))) or 1.0})
        sns.heatmap(
            d2d, ax=ax, cmap=_diverging_cmap(),
            annot=False, linewidths=0.0, square=False,
            cbar_kws={"shrink": 0.6, "label": "value (log scale)"},
            xticklabels=False, yticklabels=False,
            **norm_kw,
        )
        ax.set_title(label, fontsize=9, fontweight="bold", pad=6)
        st.caption(f"Showing first {_HEATMAP_MAX} of {m} entries (real part).")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    annot        = m <= _ANNOT_MAX
    fmt          = _annotate_fmt(data) if annot else ""
    norm, absmax = _symlog_norm(data.reshape(-1, 1))
    norm_kw      = ({"norm": norm} if norm is not None
                    else {"center": 0, "vmin": -absmax, "vmax": absmax})

    cell_h = max(0.35, min(0.7, 6.0 / m))
    fig_h  = min(m * cell_h + 1.0, 9.0)

    fig, ax = plt.subplots(figsize=(1.6, fig_h))
    sns.heatmap(
        data.reshape(-1, 1), ax=ax, cmap=_diverging_cmap(),
        annot=annot, fmt=fmt,
        annot_kws={"size": max(6, min(9, int(60 / m)))},
        linewidths=0.4 if m <= 40 else 0.0,
        linecolor="#cccccc", square=False,
        cbar_kws={"shrink": 0.6, "label": "value (log scale)"},
        xticklabels=False, yticklabels=m <= 30,
        **norm_kw,
    )
    ax.set_title(label, fontsize=9, fontweight="bold", pad=6)
    ax.set_ylabel("index", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Raw array display
# ──────────────────────────────────────────────────────────────────────────────

def render_array(label: str, arr: np.ndarray) -> None:
    st.markdown(f"**{label}**")
    if arr.ndim == 1:
        truncated = arr.shape[0] > _DISPLAY_MAX
        data = arr[:_DISPLAY_MAX] if truncated else arr
        if np.iscomplexobj(data):
            df = pd.DataFrame({"Re": data.real, "Im": data.imag})
        else:
            df = pd.DataFrame(data, columns=["value"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        if truncated:
            st.caption(f"Showing first {_DISPLAY_MAX} of {arr.shape[0]} entries.")
    elif arr.ndim == 2:
        m, n = arr.shape
        data = arr[:_DISPLAY_MAX, :_DISPLAY_MAX]
        if np.iscomplexobj(data):
            st.caption("Re(A)  —  real part:")
            st.dataframe(pd.DataFrame(data.real), use_container_width=True, hide_index=True)
            st.caption("Im(A)  —  imaginary part:")
            st.dataframe(pd.DataFrame(data.imag), use_container_width=True, hide_index=True)
        else:
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        if m > _DISPLAY_MAX or n > _DISPLAY_MAX:
            st.caption(
                f"Showing [{min(m, _DISPLAY_MAX)}×{min(n, _DISPLAY_MAX)}] "
                f"of [{m}×{n}]."
            )
    else:
        st.warning(f"Cannot display array with ndim={arr.ndim}.")


# ──────────────────────────────────────────────────────────────────────────────
# Spectrum plot — complex plane + magnitude bar
# ──────────────────────────────────────────────────────────────────────────────

def render_spectrum_plot(eigenvalues: np.ndarray, title: str = "Spectrum") -> None:
    """
    Two-panel spectrum visualisation:
      Left  — complex plane scatter (Re on x-axis, Im on y-axis)
      Right — eigenvalue magnitudes |λᵢ| as a horizontal bar chart
    """
    _apply_style()
    vals = np.asarray(eigenvalues, dtype=complex)
    n    = len(vals)
    mags = np.abs(vals)

    fig, (ax_c, ax_m) = plt.subplots(1, 2, figsize=(9, 4))

    # ── Complex plane ─────────────────────────────────────────────────────────
    colors = plt.cm.viridis(mags / (mags.max() + 1e-14))
    sc = ax_c.scatter(vals.real, vals.imag, c=mags, cmap="viridis",
                      s=60, edgecolors="#333333", linewidths=0.5, zorder=4)
    ax_c.axhline(0, color="#aaaaaa", lw=0.8, ls="--", zorder=2)
    ax_c.axvline(0, color="#aaaaaa", lw=0.8, ls="--", zorder=2)

    # Unit circle reference
    theta = np.linspace(0, 2 * np.pi, 300)
    ax_c.plot(np.cos(theta), np.sin(theta), color="#cccccc",
              lw=1.0, ls=":", zorder=1, label="unit circle")

    ax_c.set_xlabel("Re(lambda)", fontsize=9)
    ax_c.set_ylabel("Im(lambda)", fontsize=9)
    ax_c.set_title(f"{title} — complex plane", fontsize=9, fontweight="bold")
    ax_c.set_aspect("equal", adjustable="datalim")
    ax_c.tick_params(labelsize=8)
    fig.colorbar(sc, ax=ax_c, shrink=0.7, label="|lambda|")

    # ── Magnitude bar chart ───────────────────────────────────────────────────
    idx_sorted = np.argsort(mags)[::-1]
    x_pos      = np.arange(min(n, 50))   # show at most 50 bars
    mags_show  = mags[idx_sorted[:50]]

    bars = ax_m.bar(x_pos, mags_show, color="#2980b9", alpha=0.8,
                    edgecolor="#1a5276", linewidth=0.4)
    ax_m.set_xlabel("Eigenvalue index (magnitude-sorted)", fontsize=9)
    ax_m.set_ylabel("|lambda|", fontsize=9)
    ax_m.set_title(f"{title} — magnitudes", fontsize=9, fontweight="bold")
    ax_m.tick_params(labelsize=8)
    if n > 50:
        ax_m.set_title(f"{title} — magnitudes (top 50 of {n})", fontsize=9,
                       fontweight="bold")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Eigenvalue table
# ──────────────────────────────────────────────────────────────────────────────

def render_eigenvalue_table(eigenvalues: np.ndarray, max_rows: int = 50) -> None:
    """Display eigenvalues as a scrollable table: index, Re, Im, |λ|."""
    vals = np.asarray(eigenvalues, dtype=complex)
    n    = min(len(vals), max_rows)
    df   = pd.DataFrame({
        "index":  np.arange(n),
        "Re(λ)":  vals[:n].real,
        "Im(λ)":  vals[:n].imag,
        "|λ|":    np.abs(vals[:n]),
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    if len(vals) > max_rows:
        st.caption(f"Showing first {max_rows} of {len(vals)} eigenvalues.")


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 plots — sensitivity metrics
# ──────────────────────────────────────────────────────────────────────────────

def render_spectral_radius_plot(series_list: list, sweep_param: str | None) -> None:
    """Gauge (single) or line plot (multi) of spectral radius ρ(A) = max|λ|."""
    _apply_style()
    all_good = [(s, [i for i in s["instances"] if not i.get("error")])
                for s in series_list]
    all_good = [(s, insts) for s, insts in all_good if insts]
    if not all_good:
        st.warning("No valid instances.")
        return

    single = sum(len(insts) for _, insts in all_good) == 1

    if single and len(all_good) == 1:
        rho     = all_good[0][1][0]["sensitivity"]["spectral_radius"]
        norm_A  = all_good[0][1][0]["sensitivity"]["norm_A"]
        ratio   = rho / norm_A if norm_A > 0 else 0.0

        fig, ax = plt.subplots(figsize=(5, 2.2))
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(grad, aspect="auto", extent=[0, 1, -0.4, 0.4],
                  cmap="RdYlGn_r", alpha=0.22, zorder=0)
        ax.axvline(min(ratio, 0.999), color="#2980b9", lw=2.5, zorder=3)
        ax.text(min(ratio, 0.95) + 0.01, 0.27,
                f"ρ(A) = {rho:.4e}", fontsize=8, color="#2980b9")
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("rho(A) / ||A||_2", fontsize=8)
        ax.set_yticks([])
        ax.set_title(f"Spectral radius  rho(A) = {_fmt(rho)}", fontsize=9,
                     fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    all_x = []
    for si, (series, insts) in enumerate(all_good):
        color  = _SERIES_COLORS[si % len(_SERIES_COLORS)]
        rhos   = [i["sensitivity"]["spectral_radius"] for i in insts]
        x_vals = _sweep_x_values(insts, sweep_param)
        all_x.extend(x_vals)
        ax.plot(x_vals, rhos, color=color, lw=2.0, marker="o",
                markersize=5, markerfacecolor="white", markeredgewidth=1.8,
                zorder=4, label=series["label"])
        if len(insts) <= 10:
            for xv, rv in zip(x_vals, rhos):
                ax.annotate(f"{rv:.2e}", xy=(xv, rv), xytext=(0, 6),
                            textcoords="offset points",
                            fontsize=7, ha="center", color=color)
    ax.set_xlabel(_sweep_axis_label(sweep_param), fontsize=9)
    ax.set_ylabel("rho(A) = max|lambda|", fontsize=9)
    ax.set_title("Spectral radius", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)
    _format_x_ticks(ax, sorted(set(all_x)), sweep_param)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_spectral_gap_plot(series_list: list, sweep_param: str | None) -> None:
    """Plot relative spectral gap (|λ₁|−|λ₂|) / |λ₁|."""
    _apply_style()
    all_good = [(s, [i for i in s["instances"] if not i.get("error")])
                for s in series_list]
    all_good = [(s, insts) for s, insts in all_good if insts]
    if not all_good:
        st.warning("No valid instances.")
        return

    single = sum(len(insts) for _, insts in all_good) == 1

    if single and len(all_good) == 1:
        gap = all_good[0][1][0]["sensitivity"]["spectral_gap_rel"]
        fig, ax = plt.subplots(figsize=(5, 2.2))
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(grad, aspect="auto", extent=[0, 1, -0.4, 0.4],
                  cmap="RdYlGn", alpha=0.22, zorder=0)
        ax.axvline(min(gap, 0.999), color="#27ae60", lw=2.5, zorder=3)
        ax.text(min(gap, 0.95) + 0.01, 0.27,
                f"gap = {gap:.4f}", fontsize=8, color="#27ae60")
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("Relative gap  (|l1| - |l2|) / |l1|", fontsize=8)
        ax.set_yticks([])
        ax.set_title(f"Spectral gap = {gap:.4f}", fontsize=9, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        if gap < 0.01:
            st.warning("Very small spectral gap — eigenvalues nearly equal.  "
                       "Power-type methods will converge slowly.")
        elif gap < 0.1:
            st.info("Moderate spectral gap — convergence may be slow.")
        else:
            st.success("Healthy spectral gap — power-type methods converge well.")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    all_x = []
    for si, (series, insts) in enumerate(all_good):
        color  = _SERIES_COLORS[si % len(_SERIES_COLORS)]
        gaps   = [i["sensitivity"]["spectral_gap_rel"] for i in insts]
        x_vals = _sweep_x_values(insts, sweep_param)
        all_x.extend(x_vals)
        ax.plot(x_vals, gaps, color=color, lw=2.0, marker="o",
                markersize=5, markerfacecolor="white", markeredgewidth=1.8,
                zorder=4, label=series["label"])
    ax.set_xlabel(_sweep_axis_label(sweep_param), fontsize=9)
    ax.set_ylabel("Relative spectral gap", fontsize=9)
    ax.set_title("Spectral gap  (|l1| - |l2|) / |l1|", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)
    _format_x_ticks(ax, sorted(set(all_x)), sweep_param)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_eig_cond_plot(series_list: list, sweep_param: str | None) -> None:
    """
    Plot eigenvalue condition numbers κ(λᵢ) for the selected instance.
    Only available for NumPy eig (requires left eigenvectors).
    """
    _apply_style()
    all_good = [(s, [i for i in s["instances"] if not i.get("error")])
                for s in series_list]
    all_good = [(s, insts) for s, insts in all_good if insts]
    if not all_good:
        st.warning("No valid instances.")
        return

    # Check if any instance has condition number data
    first_inst = all_good[0][1][0]
    conds = first_inst["sensitivity"].get("eigenvalue_conds")
    if conds is None:
        st.caption(
            "Eigenvalue condition numbers require left eigenvectors.  "
            "Use **NumPy eig** as the solver to see this plot."
        )
        return

    single = sum(len(insts) for _, insts in all_good) == 1

    if single and len(all_good) == 1:
        conds_v = np.array(conds, dtype=float)
        conds_v = np.clip(conds_v, 1.0, 1e16)
        log_c   = np.log10(conds_v)
        n       = len(log_c)

        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.bar(np.arange(n), log_c, color="#8e44ad", alpha=0.8,
               edgecolor="#6c3483", linewidth=0.4)
        ax.axhline(0, color="#27ae60", lw=1.0, ls="--", label="kappa=1 (normal matrix)")
        ax.set_xlabel("Eigenvalue index", fontsize=9)
        ax.set_ylabel("log10(kappa(lambda_i))", fontsize=9)
        ax.set_title("Eigenvalue condition numbers kappa(lambda_i) = 1/|y_i^H x_i|",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        max_c = float(np.max(conds_v))
        if max_c > 1e8:
            st.warning(f"max κ(λ) ≈ {max_c:.2e} — some eigenvalues are "
                       "very sensitive to perturbations in A.")
        elif max_c > 1e4:
            st.info(f"max κ(λ) ≈ {max_c:.2e} — moderate eigenvalue sensitivity.")
        else:
            st.success("All κ(λᵢ) near 1 — eigenvalues are well-conditioned.")
        return

    # Multi-series: plot max κ per instance
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    all_x = []
    for si, (series, insts) in enumerate(all_good):
        color  = _SERIES_COLORS[si % len(_SERIES_COLORS)]
        max_cs = []
        for inst in insts:
            c = inst["sensitivity"].get("eigenvalue_conds")
            if c is not None:
                max_cs.append(float(np.log10(max(np.clip(c, 1.0, 1e16)))))
            else:
                max_cs.append(0.0)
        x_vals = _sweep_x_values(insts, sweep_param)
        all_x.extend(x_vals)
        ax.plot(x_vals, max_cs, color=color, lw=2.0, marker="o",
                markersize=5, markerfacecolor="white", markeredgewidth=1.8,
                zorder=4, label=series["label"])
    ax.set_xlabel(_sweep_axis_label(sweep_param), fontsize=9)
    ax.set_ylabel("log10(max kappa(lambda))", fontsize=9)
    ax.set_title("Max eigenvalue condition number", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)
    _format_x_ticks(ax, sorted(set(all_x)), sweep_param)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 plots — solution quality metrics
# ──────────────────────────────────────────────────────────────────────────────

def _make_quality_plot(
    series_list:  list,
    sweep_param:  str | None,
    metric_fn,           # inst -> float
    title:        str,
    ylabel:       str,
    color_single: str,
    single_label: str,
    log_scale:    bool = True,
) -> None:
    _apply_style()
    eps     = np.finfo(float).eps
    log_eps = np.log10(eps)

    all_good = [(s, [i for i in s["instances"] if not i.get("error")])
                for s in series_list]
    all_good = [(s, insts) for s, insts in all_good if insts]
    if not all_good:
        st.warning("No valid instances.")
        return

    def _safe_log(v: float) -> float:
        if v is None or np.isnan(v): return log_eps
        if np.isinf(v) or v <= 0:   return 0.0
        return np.log10(v)

    single = sum(len(insts) for _, insts in all_good) == 1

    if single and len(all_good) == 1:
        val     = metric_fn(all_good[0][1][0])
        log_val = _safe_log(val) if log_scale else val
        lo, hi  = log_eps - 1, 2.0

        fig, ax = plt.subplots(figsize=(5, 2.0))
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(grad, aspect="auto", extent=[lo, hi, -0.4, 0.4],
                  cmap="RdYlGn", alpha=0.22, zorder=0)
        ax.axvline(log_val, color=color_single, lw=2.5, zorder=3)
        ax.axvline(log_eps, color="#7f8c8d",    lw=1.2, ls="--", zorder=2)
        ax.text(log_val + 0.1, 0.28, f"log10 = {log_val:.1f}",
                fontsize=8, color=color_single)
        ax.text(log_eps + 0.1, -0.30, "eps_mach", fontsize=7, color="#7f8c8d")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel(f"log10({single_label})", fontsize=8)
        ax.set_yticks([])
        ax.set_title(f"{title}  =  {_fmt(val)}", fontsize=9, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if log_val <= log_eps + 1:
            st.success(f"{single_label} ≈ ε_mach — excellent.")
        elif log_val <= log_eps + 4:
            st.info(f"{single_label} ≈ {_fmt(val)} — good.")
        else:
            st.warning(f"{single_label} ≈ {_fmt(val)} — large; check solver / conditioning.")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.axhline(log_eps, color="#7f8c8d", lw=1.0, ls="--", alpha=0.7, label="eps_mach")
    ax.axhspan(log_eps - 2, log_eps + 1, color="#27ae60", alpha=0.07)

    all_x = []
    for si, (series, insts) in enumerate(all_good):
        color    = _SERIES_COLORS[si % len(_SERIES_COLORS)]
        vals     = [metric_fn(i) for i in insts]
        log_vals = [_safe_log(v) for v in vals] if log_scale else vals
        x_vals   = _sweep_x_values(insts, sweep_param)
        all_x.extend(x_vals)
        ax.plot(x_vals, log_vals, color=color, lw=2.0, marker="o",
                markersize=5, markerfacecolor="white", markeredgewidth=1.8,
                zorder=4, label=series["label"])
        if len(insts) <= 10:
            for xv, lv in zip(x_vals, log_vals):
                ax.annotate(f"{lv:.1f}", xy=(xv, lv), xytext=(0, 6),
                            textcoords="offset points",
                            fontsize=7, ha="center", color=color)

    ax.set_xlabel(_sweep_axis_label(sweep_param), fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.tick_params(labelsize=8)
    _format_x_ticks(ax, sorted(set(all_x)), sweep_param)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(f"ε_mach = {eps:.2e}.")


# ──────────────────────────────────────────────────────────────────────────────
# Section 5 — Convergence history
# ──────────────────────────────────────────────────────────────────────────────

def render_convergence_plot(instance: dict) -> None:
    """
    Plot residual norm vs iteration for iterative solvers.
    Shows 'not applicable' for direct solvers.
    """
    _apply_style()
    result = instance.get("result", {})
    history = result.get("history")

    if history is None:
        st.caption(f"**{result.get('method', 'solver')}** — direct solver, no iteration history.")
        return

    if len(history) == 0:
        st.caption("No convergence history recorded.")
        return

    eps = np.finfo(float).eps
    fig, ax = plt.subplots(figsize=(6, 3.5))

    iters = np.arange(1, len(history) + 1)
    safe_history = [max(v, 1e-300) for v in history]

    ax.semilogy(iters, safe_history, color="#2980b9", lw=2.0,
                marker="o", markersize=4, markerfacecolor="white",
                markeredgewidth=1.5, zorder=4)
    ax.axhline(eps, color="#7f8c8d", lw=1.0, ls="--", alpha=0.7,
               label="eps_mach")
    ax.fill_between(iters, [eps] * len(iters), safe_history,
                    alpha=0.06, color="#2980b9")

    converged_at = result.get("converged_at")
    if converged_at is not None and converged_at <= len(history):
        ax.axvline(converged_at, color="#27ae60", lw=1.5, ls="--",
                   label=f"Converged at iter {converged_at}")

    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Residual / sub-diagonal norm", fontsize=9)
    ax.set_title(f"Convergence — {result.get('method', '')}", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Sweep axis helpers (shared across sections 3, 4, 5)
# ──────────────────────────────────────────────────────────────────────────────

def _sweep_axis_label(sweep_param: str | None) -> str:
    return {
        "m":               "Matrix size  m",
        "perturb_A_order": "Perturbation order  k  (||dA|| ~ 10^k * ||A||)",
        "krylov_k":        "Krylov dimension  k",
    }.get(sweep_param or "", "Instance")


def _sweep_x_values(instances: list, sweep_param: str | None) -> list:
    if sweep_param == "m":
        return [inst["m"] for inst in instances]
    if sweep_param == "perturb_A_order":
        return [inst["order_A"] for inst in instances]
    if sweep_param == "krylov_k":
        return [inst.get("krylov_k", 0) for inst in instances]
    return list(range(1, len(instances) + 1))


def _format_x_ticks(ax, x_vals: list, sweep_param: str | None) -> None:
    if sweep_param in ("perturb_A_order",):
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"10^{v}" for v in x_vals], fontsize=7)
