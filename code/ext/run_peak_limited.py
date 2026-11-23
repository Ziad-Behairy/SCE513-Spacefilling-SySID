"""
run_peak_limited.py  --  extension E4b: space-filling design with an actuator peak limit.

The optimised signals have a higher crest factor (about 3) than Schroeder (1.88),
i.e. higher force peaks for the same power.  A real actuator has a peak limit, so
we add a penalty to the cost of Eq. (8):

    J(phi) = C(phi) + rho * sum_k max(0, |u_k| - U_max)^2,     U_max = CF_max * 160 N

(the amplitudes, hence the power, stay fixed; only the phases change).  For every
seed 1-20 and CF_max in {inf, 2.5, 1.9}, L-BFGS-B runs 100 iterations from the same
random phases phi0(s); the designed input is then applied to the true system and the
model is identified and tested exactly as in run_extension.py (40 dB).
Output: results/ext_peak_limited/seed_XX.json
Usage: python -m ext.run_peak_limited first_seed last_seed
"""
import json, os, sys
import numpy as np
from scipy.optimize import minimize
import config as cfg
import cost, signals
from ext.run_extension import MCP, evaluate, experiment

CF_LIMITS = [np.inf, 2.5, 1.9]
RHO = 1e-2
ITERS = 100
SNR = 40.0
OUT = os.path.join(cfg.RESULTS_DIR, "ext_peak_limited")


def objective(phi, umax):
    C, g = cost.objective(phi)
    if not np.isfinite(umax):
        return C, g
    u = signals.multisine(phi)
    v = np.maximum(np.abs(u) - umax, 0.0)
    pen = RHO * np.sum(v ** 2)
    gU = 2 * RHO * v * np.sign(u)                  # d penalty / d u_k
    return C + pen, g + signals.multisine_grad(phi, gU)


def run_seed(s):
    mc = json.load(open(os.path.join(MCP, f"seed_{s:02d}.json")))
    phi0 = np.array(mc["phi0"])
    res = {"seed": s}
    for cfmax in CF_LIMITS:
        umax = cfmax * cfg.U_STD
        r = minimize(lambda p: objective(p, umax), phi0, jac=True, method="L-BFGS-B",
                     options={"maxiter": ITERS, "ftol": cfg.OPT_FTOL, "gtol": cfg.OPT_GTOL, "maxcor": 10})
        u = signals.multisine(r.x)
        theta, lam, Ztrue = experiment(u, s + 5000 + 100, SNR)   # same noise seed as Table 6 at 40 dB
        ev = evaluate(theta, Ztrue)
        key = "inf" if not np.isfinite(cfmax) else f"{cfmax:g}"
        res[key] = {"CF": float(signals.crest_factor(u)), "C_true": float(cost.cost(Ztrue)),
                    "kappa": float(cost.coverage(Ztrue)[0]), "e_uni": ev["e_uni"],
                    "e_sim160": ev["e_sim160"], "e_sim240": ev["e_sim240"],
                    "div": ev["div160"] + ev["div240"], "nit": int(r.nit)}
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    a, b = int(sys.argv[1]), int(sys.argv[2])
    for s in range(a, b + 1):
        p = os.path.join(OUT, f"seed_{s:02d}.json")
        if os.path.exists(p):
            continue
        r = run_seed(s)
        json.dump(r, open(p, "w"))
        print(s, {k: (round(v["CF"], 2), round(v["e_uni"], 4)) for k, v in r.items() if k != "seed"}, flush=True)
