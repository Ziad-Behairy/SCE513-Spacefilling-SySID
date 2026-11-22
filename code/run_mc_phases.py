"""
run_mc_phases.py  --  M4: Monte Carlo rerun that also SAVES the phases.

Same optimizer and settings as run_sec46.py (T7), but every run stores
  phi0   : initial random phases (seed)
  phi_16 : phases after 16 iterations  (fmincon-default-like budget, H1)
  phi_20 : phases after 20 iterations  (early stop, paper-like cost)
  phi    : final phases (<= 1000 iterations)
so the designed signals can be reused by the extension (identification).
Usage: python run_mc_phases.py first_seed last_seed
"""
import json, os, sys, time
import numpy as np
from scipy.optimize import minimize
import config as cfg
import cost, signals


def run(seed, maxiter=cfg.OPT_MAXITER):
    phi0 = signals.random_phases(seed)
    snaps, hist = {}, []

    def cb(intermediate_result):
        r = intermediate_result
        hist.append(float(r.fun))
        if len(hist) in (16, 20):
            snaps[len(hist)] = np.array(r.x, copy=True)

    t0 = time.time()
    C0 = cost.objective(phi0)[0]
    res = minimize(lambda p: cost.objective(p), phi0, jac=True, method="L-BFGS-B", callback=cb,
                   options={"maxiter": maxiter, "ftol": cfg.OPT_FTOL, "gtol": cfg.OPT_GTOL, "maxcor": 10})
    return dict(seed=seed, C_init=float(C0), C_opt=float(res.fun), nit=int(res.nit),
                time_s=time.time() - t0, history=[C0] + hist,
                phi0=phi0.tolist(), phi_16=snaps[16].tolist(), phi_20=snaps[20].tolist(), phi=res.x.tolist())


if __name__ == "__main__":
    folder = os.path.join(cfg.RESULTS_DIR, "mc_phases")
    os.makedirs(folder, exist_ok=True)
    a, b = int(sys.argv[1]), int(sys.argv[2])
    for s in range(a, b + 1):
        path = os.path.join(folder, f"seed_{s:02d}.json")
        if os.path.exists(path):
            continue
        out = run(s)
        json.dump(out, open(path, "w"))
        print(f"seed {s}: {out['C_init']:.2f} -> {out['C_opt']:.3f} ({out['time_s']:.0f}s)", flush=True)
