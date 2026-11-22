# Code walkthrough — how the M3 code works, in plain English

This file explains every part of the code so that I can explain it in the M5 live-code check.
Read it next to the code. Order = the order in which data flows.

---

## The big picture (one sentence per step)

1. **Phases** `phi` (184 numbers) are the only thing we change.
2. `signals.multisine(phi)` turns them into one period of the force `u` (2048 samples).
3. `system.simulate(...)` pushes the mass with this force for 8 periods (Euler steps).
4. We keep the **last** period: the samples `z_k = [u_k, x1_k, x2_k]`, a 2048 × 3 array.
5. `cost.cost_and_grad_Z(Z)` measures how well these 2048 points cover the 20 × 20 × 20 grid.
6. The **gradient** tells us how the cost changes when each phase changes a tiny bit.
7. L-BFGS-B uses cost + gradient to pick better phases; repeat until it stops.

---

## `config.py`

Only numbers. Every setting of the paper and every choice I made (ε, P, seeds, optimizer
tolerances) lives here, so nothing is hidden in other files.

* `AMP = 160 * sqrt(2/184)`: a sine with amplitude A has power A²/2. 184 sines with equal A
  have total power 184·A²/2. Setting this equal to 160² gives A = 16.68 N. Because the sines
  have different frequencies, the power does **not** depend on the phases, so the std is
  exactly 160 N for every phase vector (check D1 confirms this).
* `P_PERIODS = 8`: I first used 5 (M2). The optimized signal needed 6 to pass my steady-state
  test, so I raised it to 8 as my M2 protocol says.
* `FMINCON_GRAD_BUDGET = 16`: 3000 evaluations / (184 + 1) evaluations per finite-difference
  gradient. Used to test why the paper's final cost is higher than ours.

## `system.py`

* `f(x1, x2, u)`: Eq. (9). The spring term `s*l*x1/sqrt(x1² + a²)` is the nonlinear part.
  **1/m multiplies only the velocity equation** (instructor's correction of Eq. 11).
* `simulate(u, x0)`: Euler, Eq. (10): `x_next = x + Ts * f(x, u)`. Both new states are computed
  from the *old* state (the tuple assignment `x1, x2 = ..., ...` does that). I use Python floats,
  not NumPy, inside the loop because for 2 numbers NumPy is slower.
* `simulate_backward(X, gX)`: the **hand-made gradient** (see the section below).
* `steady_state_response(u, P)`: repeats the period P times (`np.tile`), simulates, returns the last period.
* `steady_state_check(u)`: my M2 rule. Compare period p with period p−1; the largest change divided by
  the largest state value must be ≤ 1e-6.

## `signals.py`

* `_ANGLE[l, k] = 2π·l·k/N`: computed once (184 × 2048 table).
* `multisine(phi) = sum over l of A·sin(angle + phi_l)` → Eq. (12).
* `multisine_grad(phi, g_u)`: since du_k/dφ_l = A·cos(angle + φ_l), the chain rule gives
  dC/dφ_l = A·Σ_k (dC/du_k)·cos(angle + φ_l). That is one matrix–vector product.
* `schroeder_phases`: Eq. (13), two index conventions (the paper does not say which one).
* `crest_factor(u) = max|u| / rms(u)`.

## `cost.py`

* `make_grid()`: `linspace(lo, hi, 20)` on each axis, all 8000 combinations.
* `densities(Z)`: Eq. (7). Trick: divide every coordinate by its σ (40, 0.2, 2). Then the
  weighted distance becomes a normal squared distance, and
  |a − b|² = |a|² + |b|² − 2a·b lets NumPy compute all 8000 × 2048 distances with one matrix product.
  `E[i, k] = exp(−½·distance²)`; `d_i` = row sum of E.
* `cost`: Eq. (8), `mean(1/(eps + d))`. If a grid point has no data near it, d_i ≈ 0 and it adds
  ≈ 1/ε = 1000 to the sum → empty grid points are punished strongly.
* `cost_and_grad_Z`: derivative of the cost with respect to every sample (rules below).
* `objective(phi)`: the full chain phases → u → states → cost and gradient. This is what the optimizer calls.
* `coverage`: κ = share of grid points with d_i ≥ 1 (at least "one sample worth" of data nearby).

## `optimize.py`

* `optimize_phases`: calls `scipy.optimize.minimize(..., method="L-BFGS-B", jac=True)`.
  `jac=True` means "my function returns the cost AND the gradient". The callback stores the cost
  after each iteration; `eval_costs` stores every single evaluation.
* `best_within_budget`: lowest cost in the first 16 evaluations (fmincon budget test).
* `adam`: a simple first-order optimizer for the cross-check (different method, same start).

## Scripts

`run_sec44.py` (T1, T2) → `run_adam.py` → `run_sec45.py` (T6) → `validation.py` →
`run_sec46.py` (T7) → `make_figures.py` (figures, table, `numbers.tex`). `main.py` runs them all.
`diagnostics.py` holds the checks D1–D8.

---

## The hand-made gradient (the part most likely to be asked about)

**Why:** L-BFGS-B needs the gradient of the cost with respect to 184 phases. Finite differences
would need 185 cost evaluations per gradient (≈ 50 s); the exact gradient costs about one evaluation.
PyTorch (automatic differentiation) was planned in M2 but could not be installed, so I wrote the
chain rule myself. This is the same thing autograd does internally ("reverse mode").

**Step A — cost to samples.** With w = z/σ and a_i = dC/dd_i = −1/(n(ε + d_i)²):
dd_i/dw_k = E_ik·(w̃_i − w_k), so dC/dw_k = Σ_i a_i E_ik (w̃_i − w_k), and dC/dz_k = dC/dw_k / σ.

**Step B — samples to inputs, backwards through time.** x_{k+1} depends on x_k and u_k:

    x1[k+1] = x1[k] + Ts·x2[k]
    x2[k+1] = x2[k] + Ts·f2(x1[k], x2[k], u[k])

Let λ_k = total derivative of the cost with respect to x_k. Going from the end back to the start:

    λ1[k] = (direct part from the cost) + λ1[k+1] + Ts·(∂f2/∂x1)·λ2[k+1]
    λ2[k] = (direct part from the cost) + Ts·λ1[k+1] + (1 − Ts·c/m)·λ2[k+1]
    dC/du[k] = (Ts/m)·λ2[k+1]

with ∂f2/∂x1 = (−s + s·l·a²/(x1² + a²)^{3/2})/m. This is the transpose of the Euler Jacobian
applied backwards — the "adjoint" equation.

**Step C — periodic input.** u at step p·N + k is the same number as u_k, so the gradients of all
8 copies are added: `gU_full.reshape(P, N).sum(axis=0)`, plus the direct part (u is the first
column of z).

**Step D — inputs to phases:** `signals.multisine_grad`.

**How I know it is right:** check D6 compares it with central finite differences
(C(φ + h) − C(φ − h))/(2h) on 10 random phases: error ≈ 1e-10 of the gradient size.

---

## Likely questions and short answers

* **Why is your optimized cost (≈1.2) lower than the paper's (3.88)?** Our initial cost matches the
  paper (252.6 vs 287.15, and our 20 seeds span the paper's value), so the cost function is the same.
  Our optimizer passes 3.88 after ~17 iterations and continues. fmincon's defaults allow 3000
  evaluations; with finite-difference gradients that is only 16 gradients. Within a 16-gradient
  budget our costs fall in the range of the paper's Fig. 8 → most likely the paper stopped early.
  It is a hypothesis: the paper does not report its options.
* **Why 69× vs 74×?** 287.15/3.88 = 74.0; the text says 69. It is an inconsistency in the paper.
* **Why P = 8?** The optimized data visit |x1| ≈ 2 m where the spring is stiff and the Euler
  transient decays slowly; P = 5 failed my criterion (2.5e-5 > 1e-6), so I raised P (protocol rule).
* **Why L-BFGS-B and not fmincon?** No MATLAB; L-BFGS-B is also a gradient-based quasi-Newton
  method. The Adam cross-check reaches a similarly low cost, so the result is not optimizer-specific.
* **What is ε for?** It stops 1/d_i from becoming infinite at grid points with no data. It was not
  given in the paper; since C ≤ 1/ε and the paper reports 287.15, ε ≤ 3.5e-3. I use 1e-3; V1 shows its effect.
* **Is there an estimation/validation split?** No model is identified in this part of the paper.
  Instead: 20 random seeds, an independent finer grid (30³), and a different initial state.
* **What does the Schroeder result show?** Same spectrum, crest factor 1.88 (paper 1.87); it spreads
  x1 and x2 but not u. The optimized signal spreads u more (Fig. 4 bottom vs middle rows).
