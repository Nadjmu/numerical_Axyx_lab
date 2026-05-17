# Numerical Ax = λx Lab

An interactive Streamlit application for studying eigenvalue algorithms, with emphasis on **spectral sensitivity**, **solver convergence**, and **numerical stability** for the eigenvalue problem **Ax = λx**.

Supports both **CPU** (NumPy / SciPy / LAPACK) and **GPU** (CuPy / cuSolver) execution, switchable via a toggle in the sidebar. Matrices can be generated internally or **imported from external `.npy` files**.

Part of the research project: *High Performance Data Reduction and Numerical Error Analysis for Memory Constrained Computational Physics Simulations.*

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Installation & Running](#2-installation--running)
3. [GPU Setup](#3-gpu-setup)
4. [Connecting via SSH](#4-connecting-via-ssh)
5. [Importing Custom Matrices](#5-importing-custom-matrices)
6. [Libraries Used](#6-libraries-used)
7. [Experiment Design](#7-experiment-design)
8. [Matrix Types](#8-matrix-types)
9. [Sparsity Structures](#9-sparsity-structures)
10. [Solvers](#10-solvers)
11. [Analysis Metrics](#11-analysis-metrics)
12. [High-Precision Eigenvector Residual](#12-high-precision-eigenvector-residual)
13. [UI Layout](#13-ui-layout)

---

## 1. Project Structure

```
eigen_lab/
├── app.py                    # Main Streamlit entry point, experiment orchestration
├── requirements.txt          # Python dependencies
├── GPU_SETUP.md              # Detailed GPU installation guide
├── core/
│   ├── __init__.py
│   ├── device.py             # CPU/GPU abstraction layer
│   ├── problem_creation.py   # Matrix generation, import, perturbation
│   ├── solvers.py            # All eigensolver implementations + SOLVERS registry
│   └── analysis.py           # Spectral sensitivity, quality metrics, high-precision residual
└── ui/
    ├── __init__.py
    ├── problem_ui.py         # Sidebar: device selector, import, matrix type, sweep/compare
    ├── solver_ui.py          # Sidebar: solver selection, parameters, solver compare
    └── analysis_ui.py        # Heatmaps, spectrum plots, metric plots
```

### Key design principles

**Matrix TYPE** defines the algebraic structure of the entries and the nature of the eigenvalues (Symmetric Random, Hilbert, Diagonal, etc.).  
**Matrix STRUCTURE** defines the sparsity pattern applied on top (tridiagonal, banded, etc.).  
These are orthogonal concepts enforced by the `COMPATIBILITY` dict in `problem_creation.py`.

**CPU / GPU** is an orthogonal axis controlled by `core/device.py`. The rest of the codebase uses `xp = get_array_module(use_gpu)` in place of `import numpy`, so the same algorithmic code runs on both devices.

**Import** is an orthogonal axis. Imported arrays bypass type/structure/size controls but still go through the perturbation pipeline.

There is no right-hand side vector **b** — the eigenvalue problem has none. The entire problem is defined by **A** alone.

---

## 2. Installation & Running

### CPU-only (no GPU required)

```bash
conda create -n eigen_lab python=3.11
conda activate eigen_lab
conda install -c conda-forge numpy scipy pandas matplotlib seaborn mpmath streamlit
streamlit run app.py
```

### CPU + GPU

```bash
conda create -n eigen_lab python=3.11
conda activate eigen_lab
conda install -c conda-forge numpy scipy pandas matplotlib seaborn mpmath streamlit cupy
streamlit run app.py
```

> **Why conda over pip?**  
> CuPy must be built against the exact CUDA version on the machine. Installing everything via `conda -c conda-forge` in a single command lets the solver pick mutually compatible versions automatically. Mixing `pip install` and `conda install` in the same environment can cause NumPy version conflicts.

---

## 3. GPU Setup

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA toolkit installed (`nvidia-smi` shows the version)

### Check your CUDA version

```bash
nvidia-smi          # top-right corner: "CUDA Version: XX.X"
```

### Install CuPy

```bash
# Via conda (recommended — auto-detects CUDA version)
conda install -c conda-forge cupy

# Or via pip if you know your exact CUDA version
pip install cupy-cuda11x   # CUDA 11.x
pip install cupy-cuda12x   # CUDA 12.x
```

### Verify

```bash
python -c "import cupy; print(cupy.__version__)"
python -c "import cupy; cupy.show_config()"
```

### Using the GPU toggle

Once CuPy is installed a **⚙️ Device** selector appears at the top of the sidebar with a **CPU / GPU** radio button. Switching to GPU:

- transfers matrices to the GPU after generation (or after import)
- runs all solver operations on the GPU via CuPy
- pulls eigenvalues, eigenvectors, and A back to CPU immediately after the solve, before any display or analysis

If CuPy is not installed the GPU option is disabled with a clear install message — the app always works in CPU-only mode.

### GPU solver mapping

| CPU                              | GPU                                       |
|----------------------------------|-------------------------------------------|
| `numpy.linalg.eig`               | `cupy.linalg.eig`                         |
| `numpy.linalg.eigh`              | `cupy.linalg.eigh`                        |
| `numpy.linalg.solve` (RQI)       | `cupy.linalg.solve`                       |
| `numpy.linalg.qr` (Pure/Prac QR) | `cupy.linalg.qr`                          |
| Arnoldi loop (A @ v)             | CuPy matmul — runs natively on GPU        |
| Lanczos loop (A @ v)             | CuPy matmul — runs natively on GPU        |

**Notes:**
- Hessenberg reduction in Practical QR uses `scipy.linalg.hessenberg` (no CuPy equivalent). This is a one-time O(m³) CPU cost; the iterative QR steps that dominate for large m run on the GPU.
- The small k×k Hessenberg/tridiagonal eigenvalue problem in Arnoldi/Lanczos is solved on CPU (`numpy.linalg.eig` / `numpy.linalg.eigh`). This is O(k³) and negligible.
- High-precision eigenvector residuals (mpmath / float128) always run on the CPU — no accuracy regression on the GPU path.

### Performance notes

- GPU speedup is most noticeable for **large dense matrices** (m ≥ 500) where direct solvers (eig, eigh) and the QR loop dominate.
- For Krylov solvers (Arnoldi, Lanczos), the GPU accelerates the matvec A @ v — beneficial for large sparse matrices.
- For **small matrices** (m < 100) the CPU is often faster due to GPU kernel launch overhead.

---

## 4. Connecting via SSH

When running on a remote GPU server, Streamlit's browser UI is accessed by forwarding a port over SSH.

### Start the app on the server

```bash
streamlit run app.py --server.port 8504
```

### Forward the port from your local machine

```bash
ssh -4 -L 8504:localhost:8504 username@server.address
```

The `-4` flag forces IPv4 and avoids the common `bind [::1]:XXXX: Cannot assign requested address` error.

### Open in your browser

```
http://localhost:8504
```

---

## 5. Importing Custom Matrices

A can be imported from an external `.npy` file. This lets you analyse any matrix produced outside the app — from a simulation, a finite element assembly, a PDE discretisation, or a hand-crafted example.

### How to create a .npy file

```python
import numpy as np

# Any square matrix you want to analyse
A = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]], dtype=np.float64)  # 1-D FDM Laplacian

np.save("A.npy", A)   # upload this in the sidebar
```

### Using import in the sidebar

A checkbox **Import A from .npy file** appears at the top of the **Matrix A** section. When checked, a file uploader replaces the type / size / structure / seed controls. Shape, dtype, and sparsity are all read from the file. The app shows a confirmation with shape, dtype, and non-zero count.

### What changes when importing

| Feature | Generated matrix | Imported matrix |
|---|---|---|
| Type / seed / size controls | Shown | Hidden |
| Structure selectbox | Shown | Hidden |
| Hermitian / PD checkboxes | Shown | Hidden |
| Perturbation | Uses selected structure mask | Inherits sparsity pattern of imported array |
| GPU transfer | After generation | After import, before solve |
| dtype cast | Via dtype selector | Via dtype selector (cast applied after load) |

### Perturbation on imported matrices

When an imported A is perturbed, the noise is masked to the **non-zero pattern of the original imported matrix**. This ensures the perturbation respects the structure of the matrix — a sparse FEM matrix stays sparse after perturbation — without requiring the user to manually specify a structure. Implemented via `sparsity_mask(A)` in `core/problem_creation.py`.

### Requirements for imported arrays

- File format: `.npy` (saved with `numpy.save`)
- Must be 2-D and square
- All values must be finite (no NaN or Inf)
- Any numeric dtype is accepted and cast to the selected dtype after loading

---

## 6. Libraries Used

| Library | Purpose | Docs |
|---|---|---|
| [NumPy](https://numpy.org/doc/stable/) | Matrix construction, dense linear algebra, float128 | [numpy.org](https://numpy.org/doc/stable/) |
| [SciPy](https://docs.scipy.org/doc/scipy/) | LAPACK wrappers (Hessenberg reduction) | [docs.scipy.org](https://docs.scipy.org/doc/scipy/) |
| [CuPy](https://cupy.dev/) | GPU array library — drop-in NumPy replacement | [cupy.dev](https://cupy.dev/) |
| [Streamlit](https://docs.streamlit.io/) | Web UI framework | [docs.streamlit.io](https://docs.streamlit.io/) |
| [mpmath](https://mpmath.org/doc/current/) | Arbitrary-precision arithmetic for eigenvector residuals | [mpmath.org](https://mpmath.org/doc/current/) |
| [Pandas](https://pandas.pydata.org/docs/) | DataFrames for eigenvalue tables and entry display | [pandas.pydata.org](https://pandas.pydata.org/docs/) |
| [Matplotlib](https://matplotlib.org/stable/) | Plot rendering backend | [matplotlib.org](https://matplotlib.org/stable/) |
| [Seaborn](https://seaborn.pydata.org/) | Heatmaps with diverging palettes | [seaborn.pydata.org](https://seaborn.pydata.org/) |

### Key functions worth knowing

- [`numpy.linalg.eig`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html) — all eigenvalues and right eigenvectors (LAPACK `dgeev`)
- [`numpy.linalg.eigh`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html) — all eigenvalues for symmetric matrices (LAPACK `dsyev`)
- [`numpy.linalg.qr`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html) — QR decomposition used in Pure/Practical QR algorithms
- [`scipy.linalg.hessenberg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.hessenberg.html) — upper Hessenberg reduction (LAPACK `dgehrd`)
- [`numpy.linalg.solve`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html) — shifted linear system for Rayleigh Quotient Iteration
- [`mpmath.matrix`](https://mpmath.org/doc/current/matrices.html) — arbitrary-precision matrix type for high-precision residuals
- [`numpy.float128`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.float128) — 80-bit extended precision on x86 Linux
- [`cupy.linalg.eig`](https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.eig.html) — GPU eigensolver
- [`cupy.linalg.eigh`](https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.eigh.html) — GPU symmetric eigensolver
- [`cupy.linalg.qr`](https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.qr.html) — GPU QR decomposition

---

## 7. Experiment Design

The app supports two axes of variation that produce multi-series, multi-instance experiments.

### Sweep axis (x-axis of plots)
Vary one parameter across instances within each series:
- **m** — matrix size (generated matrices only)
- **Perturbation order on A** — magnitude of noise added to A (10^k · ‖A‖)
- **Krylov dimension k** — number of Krylov vectors (Arnoldi / Lanczos only)

### Compare axis (legend of plots)
Vary one parameter across series:
- **Matrix type** — e.g. compare Symmetric Random vs Hilbert vs Diagonal
- **Structure** — e.g. compare Dense vs Tridiagonal vs Banded
- **Solver** — e.g. compare Pure QR vs Practical QR vs Arnoldi

### Data model

```
session_state.series_list = [
    {
        "label":     str,           # legend entry
        "instances": [              # one per sweep combo
            {
                "A":           ndarray,   # always CPU numpy
                "A_original":  ndarray,   # always CPU numpy
                "delta_A":     ndarray | None,
                "result":      dict,      # from solver (eigenvalues/vecs always CPU)
                "sensitivity": dict,      # from spectral_sensitivity
                "quality":     dict,      # from eigenpair_analysis
                "m":           int,
                "order_A":     int,
                "krylov_k":    int,
                "use_gpu":     bool,
                "imported_A":  bool,
                "error":       str | None,
            }
        ],
        "spec": dict,
    }
]
```

Sections 1–2 show one selected instance; Sections 3–6 plot all instances across all series.

### Solver parameters exposed in the sidebar

| Solver | Parameters |
|---|---|
| Rayleigh Quotient Iteration | Initial shift σ₀, convergence tolerance, max iterations |
| Pure QR | Convergence tolerance, max iterations |
| Practical QR | Convergence tolerance, max iterations |
| Arnoldi / Lanczos | Krylov dimension k (sweepable), convergence tolerance, max iterations |

---

## 8. Matrix Types

All types produce square (m × m) matrices. The **type** determines entry values and eigenvalue structure; the **structure** then applies a sparsity mask on top. Matrix generation always runs on the **CPU** regardless of device choice — the result is transferred to the GPU after construction. When a matrix is **imported**, type and structure controls are not shown.

### Random Gaussian
```
A[i,j] ~ N(0, 1)  i.i.d.
```
General non-symmetric matrix. Eigenvalues are complex in general, distributed approximately on a disk of radius √m (Girko's circular law).

**Reference:** [Girko's circular law — Wikipedia](https://en.wikipedia.org/wiki/Circular_law)

---

### Symmetric Random
```
A = (G + Gᵀ) / 2,   G[i,j] ~ N(0, 1)
```
Real eigenvalues guaranteed by symmetry. Eigenvalue distribution follows the Wigner semicircle law.

**Reference:** [Wigner semicircle law — Wikipedia](https://en.wikipedia.org/wiki/Wigner_semicircle_distribution)

---

### Random SPD
```
A = Q Λ Qᵀ,   Q = random orthogonal,   Λ = diag(logspace(0, k, m))   →   κ(A) = 10^k
```
Symmetric positive definite with prescribed condition number. Dense only.

---

### Diagonal (Prescribed)
```
D = diag(λ),   λ from logspace(−k/2, k/2, m) with random signs and permutation
```
Eigenvalues are exactly the diagonal entries — ideal for ground-truth benchmarking.

---

### Hilbert
```
H[i,j] = 1 / (i + j + 1)
```
Symmetric positive definite. Condition number κ(H) ~ (3.5)^m. The canonical severely ill-conditioned matrix.

**Reference:** [Hilbert matrix — Wikipedia](https://en.wikipedia.org/wiki/Hilbert_matrix)

---

### Toeplitz
```
T[i,j] = f(|i − j|),   f[k] = cos(k + φ) · exp(−k / 4)
```
Symmetric, constant along each diagonal. Appears in convolution operators and 1-D finite difference stencils.

---

### Circulant
```
A[i,j] = c[(j − i) mod m],   c[k] = N(0,1) · exp(−k / (m/4))
```
Eigenvalues = DFT of the first row (closed form). Appears in problems with periodic boundary conditions.

---

### Tridiagonal Symmetric
```
A = diag(d) + diag(e, 1) + diag(e, −1),   d[i] ~ N(0,1),   e[i] ~ N(0,1) × scale
```
Natural benchmark for the Lanczos algorithm. Models the 1-D FDM Laplacian. Dense only (already tridiagonal by construction).

---

### Compatibility table

| Matrix type | Dense | Sparse Tridiagonal | Sparse Block-Tridiagonal | Sparse Banded |
|---|:---:|:---:|:---:|:---:|
| Random Gaussian | ✅ | ✅ | ✅ | ✅ |
| Symmetric Random | ✅ | ✅ | ✅ | ✅ |
| Random SPD | ✅ | ❌ | ❌ | ❌ |
| Diagonal (Prescribed) | ✅ | ❌ | ❌ | ❌ |
| Hilbert | ✅ | ✅ | ✅ | ✅ |
| Toeplitz | ✅ | ✅ | ✅ | ✅ |
| Circulant | ✅ | ✅ | ✅ | ✅ |
| Tridiagonal Symmetric | ✅ | ❌ | ❌ | ❌ |
| **Custom (imported)** | — | — | — | — |

---

## 9. Sparsity Structures

Structures are **pure sparsity masks** — they zero out entries outside the pattern. Not applicable to imported matrices — their sparsity pattern is inherited directly from the array.

### Dense
No entries are zeroed. All m² entries retained.

### Sparse Tridiagonal
Only diagonals −1, 0, +1 are kept. 3m − 2 non-zeros.

### Sparse Block-Tridiagonal
Main block-diagonal and two neighbouring blocks retained. param = block size.

### Sparse Banded
`num_diags` diagonals centred on the main diagonal. Tridiagonal is a special case with `num_diags = 3`.

**Reference:** [Band matrix — Wikipedia](https://en.wikipedia.org/wiki/Band_matrix)

---

## 10. Solvers

All solvers share the same interface and work identically on generated and imported matrices:

```python
def solve_*(A, params: dict) -> dict:
    # A: numpy or cupy array
    # returns:
    # {
    #   "eigenvalues":  np.ndarray (complex, CPU)   — always CPU numpy
    #   "eigenvectors": np.ndarray | None (CPU)     — always CPU numpy
    #   "method":       str                         — includes [GPU] or [CPU]
    #   "success":      bool
    #   "message":      str
    #   "converged_at": int | None
    #   "history":      list[float] | None
    # }
```

The solver detects whether A is a CuPy array and dispatches to GPU routines automatically. All returned arrays are always CPU numpy.

### Solver comparison summary

| Solver | Requires | Returns | Cost | GPU support |
|---|---|---|---|:---:|
| NumPy eig | Square | All m pairs | O(m³) | ✅ |
| NumPy eigh | Symmetric | All m pairs | O(m³) | ✅ |
| Rayleigh QI | Square | 1 pair | O(m³)/iter | ✅ |
| Pure QR | Square | All m pairs | O(m³)/iter | ✅ |
| Practical QR | Square | All m pairs | O(m²)/iter | ✅ (QR steps on GPU, Hessenberg on CPU) |
| Arnoldi | Square | k pairs | O(m²k) | ✅ (matvec on GPU) |
| Lanczos | Symmetric | k pairs | O(m²k) | ✅ (matvec on GPU) |

---

## 11. Analysis Metrics

All metrics are computed in `core/analysis.py`. Works with CPU numpy arrays (GPU arrays are already pulled to CPU by the solvers before analysis runs). Works equally on generated and imported matrices.

### Section 3 — Problem-specific sensitivity metrics

**Spectral radius** ρ(A) = max|λᵢ|  
**Spectral gap** |λ₁| − |λ₂| (absolute) and (|λ₁|−|λ₂|)/|λ₁| (relative)  
**Eigenvalue condition numbers** κ(λᵢ) = 1/|yᵢᴴxᵢ| — requires left eigenvectors, available only from NumPy eig

### Section 4 — Solution quality metrics

**Eigenvector residual** ‖Avᵢ − λ̃ᵢvᵢ‖₂ / (‖A‖₂·‖vᵢ‖₂) — backward error per pair  
**Rayleigh quotient accuracy** |λ̃ᵢ − ρ(vᵢ)| / max(|λ̃ᵢ|, 1) — deviation from RQ  
**Eigenvector orthogonality** ‖VᴴV − I‖_F

### Section 6 — Structural metrics

**Eigenvalue distribution** — histogram of log₁₀|λᵢ|  
**Consecutive relative gaps** — |λᵢ−λᵢ₊₁|/|λᵢ| on log scale  
**Matrix normality** — ‖AᴴA − AAᴴ‖_F / ‖A‖_F²

---

## 12. High-Precision Eigenvector Residual

The eigenvector residual `r = Av − λv` suffers from catastrophic cancellation when (λ, v) is a good eigenpair. The precision strategy in `core/analysis.py` always runs on the **CPU**, even when the solve ran on the GPU:

```
if mpmath available and m ≤ 200:
    compute using mpmath at 50 decimal places  (~166 bits)
elif float128 is genuinely wider than float64:
    compute using numpy.float128  (80-bit extended, x86 Linux only)
else:
    compute using float64  (fallback)
```

There is **no accuracy regression** on the GPU path or the import path — the high-precision residual is identical regardless of how the matrix was obtained or which device was used for the solve.

**References:** [`mpmath.matrix`](https://mpmath.org/doc/current/matrices.html) · [`numpy.float128`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.float128) · Higham, *Accuracy and Stability of Numerical Algorithms*, Chapter 3

---

## 13. UI Layout

```
Sidebar                          Main area
───────────────────              ─────────────────────────────────────────────
⚙️ Device                        1. Problem creation (A)
  CPU / GPU toggle                  Device: 🔵 CPU / 🟢 GPU
                                    A: 🔧 generated / 📂 imported
Matrix A                            [shape  dtype  nnz  density  memory]
  [ ] Import A from .npy            A entries / heatmap
      └─ file uploader              ΔA entries / heatmap  (if perturbed)
  ── or, if not importing ──
  Type                           2. Eigenpairs via <solver>  [CPU] or [GPU]
  Seed                              [n_pairs  ρ(A)  spectral gap]
  Type param (if applicable)        Complex plane scatter + magnitude bar chart
  Size m  [Sweep checkbox]          Eigenvalue table (scrollable, top 50)
  Structure                         Eigenvector selector
  Hermitian / PD checkboxes         Selected eigenvector entries / heatmap
  dtype                             Per-pair quality metrics (expander)

Perturbation                     3. Problem-specific sensitivity metrics
  Perturb A  [order + Sweep]        ρ(A) gauge/line | Spectral gap | κ(λᵢ) bar

Compare                          4. Solution quality metrics
  None / Matrix type / Structure    Residual gauge/line | RQ accuracy | Orth. error

Solver                           5. Solver behaviour metrics
  Method                            Convergence curve (residual vs iteration)
  Eigenvalue ordering
    magnitude / algebraic        6. Structural metrics
                                    log|λ| histogram | consecutive gap plot
Solver parameters                   Matrix normality check
  RQI: shift, tol, max_iter
  QR:  tol, max_iter             7. Summary
  Krylov: k [Sweep], tol, iter      LLM-pasteable text block (includes device + source)

Compare solvers checkbox         8. Save results
                                    (coming soon — PDF export)
[Run Experiment]
```

### Gauge plots (single instance)
Horizontal gradient bar (green=good, red=bad) with a vertical marker at the current value. Used in Sections 3–4 when there is exactly one series and one sweep instance.

### Line plots (multiple instances or series)
One coloured line per series. Points annotated with log₁₀ values when ≤ 10 instances per series. Reference line at ε_mach.

### Spectrum plot (Section 2)
Two panels: complex plane scatter (Re/Im, coloured by |λ|, unit circle shown) and magnitude bar chart (|λᵢ| in magnitude-sorted order, top 50 for large problems).

### Convergence plot (Section 5)
Semi-log plot of the solver's internal residual or sub-diagonal norm per iteration. Direct solvers show "not applicable". Converged iteration marked with a dashed line.

### Heatmaps
Diverging colourmap (blue=negative, red=positive, white=zero). Cell annotations for matrices ≤ 20×20. Spy plot fallback for matrices > 150×150. Complex matrices show real part with a caption.
