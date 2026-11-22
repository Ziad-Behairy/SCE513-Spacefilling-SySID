"""
validation.py  --  cheap checks on the FINAL seed-0 signals (no re-optimization).

V1  epsilon sensitivity: cost of the initial and optimized signal for
    eps in {1e-4, 1e-3, 3e-3, 1e-2}  (eps is not given in the paper)
V2  finer independent grid (30^3 = 27000 points): does the optimized signal
    still beat the initial one, i.e. the optimizer did not "exploit" the 20^3 grid?
V3  different initial state x0 = [1.5, -10]: same steady-state samples?
V4  more periods (P = 12 instead of the P used, 8): same cost?
Results -> results/validation.json
"""
import json
import os
import numpy as np
import config as cfg
import cost
import signals
import system


def cost_chunked(Z, grid, eps=cfg.EPS, chunk=4000):
    """Same as cost.cost but over grid chunks (the 30^3 grid is large)."""
    d = np.concatenate([cost.densities(Z, grid[i:i + chunk])[0] for i in range(0, len(grid), chunk)])
    return np.mean(1.0 / (eps + d))


def main():
    d44 = np.load(os.path.join(cfg.RESULTS_DIR, "sec44.npz"))
    Z0, Z1 = d44["Z0"], d44["Z1"]
    out = {}

    out["V1_eps"] = []
    for eps in (1e-4, 1e-3, 3e-3, 1e-2):
        c0, c1 = cost.cost(Z0, eps=eps), cost.cost(Z1, eps=eps)
        out["V1_eps"].append({"eps": eps, "C_init": c0, "C_opt": c1, "ratio": c0 / c1})
        print(f"V1 eps={eps:7.0e}: C_init={c0:9.3f}  C_opt={c1:7.4f}  ratio={c0 / c1:7.1f}")

    g30 = cost.make_grid(30)
    c0, c1 = cost_chunked(Z0, g30), cost_chunked(Z1, g30)
    out["V2_grid30"] = {"C_init": c0, "C_opt": c1, "ratio": c0 / c1}
    print(f"V2 30^3 grid: C_init={c0:.3f}  C_opt={c1:.4f}  ratio={c0 / c1:.1f}")

    out["V3_x0"] = {}
    out["V4_P12"] = {}
    for name, phi, Z in (("init", d44["phi0"], Z0), ("opt", d44["phi"], Z1)):
        u = signals.multisine(phi)
        X3 = system.steady_state_response(u, cfg.P_PERIODS, x0=np.array([1.5, -10.0]))[0]
        out["V3_x0"][name] = float(np.max(np.abs(X3 - Z[:, 1:])))
        X4 = system.steady_state_response(u, 12)[0]
        out["V4_P12"][name] = {"max_state_diff": float(np.max(np.abs(X4 - Z[:, 1:]))),
                               "C_P": cost.cost(Z), "C_P12": cost.cost(np.column_stack([u, X4]))}
    print("V3 max |x(x0=[1.5,-10]) - x(x0=0)|:", out["V3_x0"])
    print("V4 P=12 vs P=8:", out["V4_P12"])
    with open(os.path.join(cfg.RESULTS_DIR, "validation.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    return out


if __name__ == "__main__":
    main()
