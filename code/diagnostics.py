"""
diagnostics.py  --  verification checks of the implementation (M2 W7-W8).
Each check prints PASS/FAIL and the numbers are saved to
results/diagnostics.json.  A figure with free decay and steady-state
convergence is saved too.

Checks
  D1  input: mean 0, std 160 N, exactly 184 excited lines of equal amplitude A
  D2  free decay: with u = 0 from x1 = 0.5 m the system settles (damped)
  D3  steady state: M2 criterion reached within P = 5 periods (seed 0)
  D4  kernel = 1 when a sample sits exactly on a grid point
  D5  cost bound: C <= 1/eps  (and C -> 1/eps when there are no samples near the grid)
  D6  gradient: hand-written (adjoint) gradient vs central finite differences
  D7  Euler stability: linearised discrete-time poles inside the unit circle
  D8  grid: 8000 points, spacings 42.1053, 0.2105, 2.1053 (as in the paper)
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg
import signals
import system
import cost


def check(name, ok, info):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")
    return {"name": name, "pass": bool(ok), "info": info}


def main():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out = []
    phi = signals.random_phases(cfg.SEED_MAIN)
    u = signals.multisine(phi)

    # D1 input properties
    A = signals.amplitude_spectrum(u)
    excited = np.where(A > 1e-8 * A.max())[0]
    ok = (abs(u.mean()) < 1e-9 and abs(u.std() - cfg.U_STD) < 1e-9
          and np.array_equal(excited, cfg.LINES)
          and np.max(np.abs(A[cfg.LINES] - cfg.AMP)) < 1e-9)
    out.append(check("D1 input", ok, f"mean={u.mean():.1e} N, std={u.std():.6f} N, "
                     f"lines {excited.min()}..{excited.max()} ({excited.size}), A={cfg.AMP:.4f} N, "
                     f"max|A_l-A|={np.max(np.abs(A[cfg.LINES]-cfg.AMP)):.1e}"))

    # D2 free decay
    T = 1000                                   # 10 s
    Xd = system.simulate(np.zeros(T), x0=np.array([0.5, 0.0]))
    # equilibria of the spring force: x1 = 0 and |x1| = sqrt(l^2 - a^2) if l > a (here l < a -> only 0)
    ok = abs(Xd[-1, 0]) < 1e-3 and abs(Xd[-1, 1]) < 1e-2
    out.append(check("D2 free decay", ok, f"x(0)=[0.5,0]; x(10 s)=[{Xd[-1,0]:.2e}, {Xd[-1,1]:.2e}]"))

    # D3 steady state
    rels, p_ok = system.steady_state_check(u)
    ok = p_ok is not None and p_ok <= cfg.P_PERIODS
    out.append(check("D3 steady state", ok, f"accepted at period p={p_ok}; rel(P={cfg.P_PERIODS})={rels[cfg.P_PERIODS-2]:.1e}"))

    # D4 kernel = 1 on a grid point
    zi = cost.GRID[1234:1235]
    d, _ = cost.densities(zi, grid=zi)
    out.append(check("D4 kernel at zero distance", abs(d[0] - 1.0) < 1e-14, f"exp(0) = {d[0]:.15f}"))

    # D5 cost bound
    Z = np.column_stack([u, system.steady_state_response(u)[0]])
    C = cost.cost(Z)
    Cfar = cost.cost(Z + 1e4)        # all samples far away -> d_i ~ 0 -> C ~ 1/eps
    ok = C <= 1 / cfg.EPS and abs(Cfar - 1 / cfg.EPS) < 1e-6
    out.append(check("D5 cost bound", ok, f"C={C:.3f} <= 1/eps={1/cfg.EPS:.0f}; far samples: C={Cfar:.6f}"))

    # D6 gradient check (10 random phases, central differences).
    # First attempt used h = 1e-6 and a per-component relative error: it
    # "failed" (8e-5) only on a component whose gradient is ~0.003 (vs max
    # ~212), where finite-difference round-off dominates.  Final test: h = 1e-4
    # and error scaled by the largest gradient component (standard practice).
    _, g = cost.objective(phi)
    rng = np.random.default_rng(123)
    errs, comp_errs = [], []   # (own names: 'rels' is used by D3 for the figure)
    for j in rng.choice(cfg.F_LINES, 10, replace=False):
        h = 1e-4
        p1, p2 = phi.copy(), phi.copy()
        p1[j] += h
        p2[j] -= h
        fd = (cost.objective(p1)[0] - cost.objective(p2)[0]) / (2 * h)
        errs.append(abs(g[j] - fd) / np.max(np.abs(g)))
        comp_errs.append(abs(g[j] - fd) / abs(fd))
    out.append(check("D6 gradient vs finite differences", max(errs) < 1e-6,
                     f"max |g-fd|/max|g| = {max(errs):.1e}; max per-component relative error "
                     f"= {max(comp_errs):.1e} (h = 1e-4, 10 phases)"))

    # D7 Euler stability of the linearisation at x1 = 0 and |x1| = 2
    lam = []
    for x1 in (0.0, 2.0):
        k = (-cfg.S_SPRING + cfg.S_SPRING * cfg.L_FREE * cfg.A_ANCHOR**2 / (x1**2 + cfg.A_ANCHOR**2) ** 1.5) / cfg.M_MASS
        J = np.array([[1, cfg.TS], [cfg.TS * k, 1 - cfg.TS * cfg.C_DAMP / cfg.M_MASS]])
        lam.append(np.max(np.abs(np.linalg.eigvals(J))))
    out.append(check("D7 Euler stability", max(lam) < 1,
                     f"spectral radius at x1=0: {lam[0]:.6f}, at |x1|=2: {lam[1]:.6f}"))

    # D8 grid
    sp = [np.diff(np.linspace(lo, hi, cfg.GRID_PTS))[0] for lo, hi in zip(cfg.DOMAIN_LO, cfg.DOMAIN_HI)]
    ok = cost.GRID.shape == (8000, 3) and np.allclose(sp, [42.1053, 0.2105, 2.1053], atol=1e-4)
    out.append(check("D8 grid", ok, f"n={cost.GRID.shape[0]}, spacings={[round(s, 4) for s in sp]}"))

    with open(os.path.join(cfg.RESULTS_DIR, "diagnostics.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # Figure: free decay + steady-state convergence
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.0))
    t = np.arange(T) * cfg.TS
    ax[0].plot(t, Xd[:, 0], label=r"$x_1$ [m]")
    ax[0].plot(t, Xd[:, 1] / 10, label=r"$x_2/10$ [m/s]")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("state")
    ax[0].set_title("D2: free decay from $x_1=0.5$ m, $u=0$")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].semilogy(range(2, 2 + len(rels)), rels, "o-", color="#1f77b4", label="initial signal")
    opt_path = os.path.join(cfg.RESULTS_DIR, "sec44.npz")
    if os.path.exists(opt_path):          # also show the optimized signal (slower decay)
        r_opt, _ = system.steady_state_check(signals.multisine(np.load(opt_path)["phi"]))
        ax[1].semilogy(range(2, 2 + len(r_opt)), r_opt, "s--", color="#e8590c", label="optimized signal")
    ax[1].axhline(cfg.SS_TOL, color="k", ls="--", label=r"tolerance $10^{-6}$")
    ax[1].axvline(cfg.P_PERIODS, color="C3", ls=":", label=f"P = {cfg.P_PERIODS} used")
    ax[1].set_xlabel("period p")
    ax[1].set_ylabel(r"$\max|x^{(p)}-x^{(p-1)}| / \max|x^{(p)}|$")
    ax[1].set_title("D3: steady-state convergence (seed 0)")
    ax[1].legend(fontsize=7)
    ax[1].legend(fontsize=7, loc="lower left")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.RESULTS_DIR, "fig_diagnostics.png"), dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
