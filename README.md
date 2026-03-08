# multi-step-mpc

Model Predictive Control (MPC) with a **smoothed block-input strategy** implemented in Julia. The controller groups micro-steps into blocks of size `nu`, computes a smoothed reference input `Uref` offline using null-space freedom, and solves a block QP at each MPC iteration using the Clarabel solver.

Two experiments are included, differing in the null-space dimension used for smoothing:

| Folder | Block size | Nullity | Strategy |
|--------|-----------|---------|----------|
| `nu=3` | nu_min + 1 | 1 | Single null vector, scalar alpha |
| `nu=4` | nu_min + 2 | 2 | Two null vectors, alpha solved via normal equations |

---

## Repository Structure

```
multi-step-mpc/
├── README.md
├── nu=3/
│   ├── run_mpc.jl          # Entry point
│   ├── src/
│   │   ├── Constants.jl
│   │   ├── SystemDynamics.jl
│   │   ├── BlockQP.jl
│   │   └── MPCPlotting.jl
│   └── output/             # Generated at runtime
│       ├── nu=3.html
│       └── nu=3_results.json
└── nu=4/
    ├── run_mpc.jl          # Entry point
    ├── src/
    │   ├── Constants.jl
    │   ├── SystemDynamics.jl
    │   ├── BlockQP.jl
    │   └── MPCPlotting.jl
    └── output/             # Generated at runtime
        ├── nu=4.html
        └── nu=4_results.json
```

---

## System

A 2D discrete-time linear system:

```
x_{k+1} = A x_k + B u_k
```

With parameters defined in `Constants.jl`:

```
A = [0.1  -0.5]      B = [1.0]
    [0.2   0.8]          [0.0]

X0 = [1.0, 0.5]      (initial state)
XT = [0.5, 1.0]      (target state)

Q = diag(5.0, 5.0)   (state cost)
R = 1.0               (input cost)
N = 20                (horizon)
max_micro_steps = 201
```

---

## Method

### 1. Reachability Index
Find the minimum `nu_min` such that the lifted input matrix `Bl` has full row rank and `(I - A^nu) * xT` is reachable.

### 2. Smoothed Uref (Offline)
Extend to `nu = nu_min + extra_steps` to create null-space freedom:

- **Particular solution:** `U0 = Bl^+ * d` (minimum norm via pseudoinverse)
- **Null basis:** `V = null(Bl)` — columns are null vectors satisfying `Bl * V = 0`
- **Optimal alpha:** minimise consecutive input gaps `J = sum(u_i - u_{i+1})^2`
  - `nu=3` (nullity 1): closed-form scalar solution
  - `nu=4` (nullity 2): normal equations `(DV)^T (DV) alpha = -(DV)^T (D U0)`
- **Final:** `Uref = U0 + V * alpha*` — still satisfies `Bl * Uref = d`

### 3. MPC Loop
At each block step, solve a QP (via Clarabel) with `Uref` as the input reference. Apply the first block of `nu` micro-steps, then re-solve with the updated state.

---

## Source Files

| File | Description |
|------|-------------|
| `Constants.jl` | System matrices, initial/target states, cost weights, output filenames |
| `SystemDynamics.jl` | `build_block_B`, `compute_reachability_index`, `compute_Uref_smoothed`, `apply_micro_step` |
| `BlockQP.jl` | `solve_block_QP` — builds and solves the block QP using JuMP + Clarabel |
| `MPCPlotting.jl` | `save_mpc_plot` — generates an interactive 4-panel HTML plot via PlotlyJS |
| `run_mpc.jl` | Top-level script — runs the full MPC pipeline and saves outputs |

---

## Output

Running the script produces two files in the `output/` subfolder:

- **HTML plot** — 4-panel interactive Plotly chart showing:
  - Position trajectory (micro-steps + block snapshots)
  - Velocity trajectory
  - Control input with Uref reference lines
  - log10 convergence error
- **JSON summary** — full run data including states, inputs, costs, and Uref

---

## Requirements

- Julia 1.9+
- Packages: `JuMP`, `Clarabel`, `PlotlyJS`, `LinearAlgebra`, `Printf`, `JSON`

Install dependencies:
```julia
using Pkg
Pkg.add(["JuMP", "Clarabel", "PlotlyJS", "JSON"])
```

---

## Running

```bash
# For nu=3
julia nu=3/run_mpc.jl

# For nu=4
julia nu=4/run_mpc.jl
```

Outputs are saved automatically to `nu=3/output/` and `nu=4/output/` respectively.
