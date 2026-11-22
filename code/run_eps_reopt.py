"""
run_eps_reopt.py  --  M4: epsilon sensitivity WITH re-optimisation (seed 0).
epsilon is not reported in the paper.  For each value the phases are
optimised from the same seed-0 start with the same L-BFGS-B settings
(capped at 300 iterations to bound run time; the 1e-3 baseline at 300
iterations is also recorded from the same script for a like-for-like table).
"""
import json, os
import numpy as np
import config as cfg
import cost, signals
from optimize import optimize_phases

EPS_LIST = [1e-4, 1e-3, 3e-3, 1e-2]
ITERS = 300

if __name__ == "__main__":
    out = []
    phi0 = signals.random_phases(cfg.SEED_MAIN)
    for e in EPS_LIST:
        r = optimize_phases(phi0, maxiter=ITERS, eps=e, verbose=False)
        Z0, u0, _ = cost.samples_from_phases(phi0)
        Z1, u1, _ = cost.samples_from_phases(r["phi"])
        row = {"eps": e, "C_init": r["C0"], "C_opt": r["C"], "ratio": r["C0"] / r["C"],
               "CF_opt": float(signals.crest_factor(u1)), "kappa_opt": float(cost.coverage(Z1)[0]),
               "C_opt_at_eps1e-3": float(cost.cost(Z1, eps=1e-3)), "nit": r["nit"],
               "it_to_ratio74": int(np.argmax(r["history"] <= r["C0"] / 74.0)) if np.any(r["history"] <= r["C0"] / 74.0) else None}
        out.append(row)
        print(row, flush=True)
    json.dump(out, open(os.path.join(cfg.RESULTS_DIR, "eps_reopt.json"), "w"), indent=1)
