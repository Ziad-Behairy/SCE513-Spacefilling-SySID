"""
run_h1_fmincon.py  --  M4: closer test of hypothesis H1 (early stopping).

H1: the paper's optimum (3.88) is higher than ours (1.217) because MATLAB
fmincon was run with default options: no user gradient -> forward finite
differences (184+1 cost evaluations per gradient) and
MaxFunctionEvaluations = 3000, i.e. only ~16 gradients.

Test: a quasi-Newton method (SciPy BFGS, dense BFGS Hessian as in fmincon's
default Hessian approximation) with FORWARD finite-difference gradients and a
hard cap of 3000 cost evaluations.  No exact gradient is used here.
Usage: python run_h1_fmincon.py seed [seed ...]
"""
import json, os, sys, time
import numpy as np
from scipy.optimize import minimize
import config as cfg
import cost, signals

MAX_FEV = 3000


class Budget(Exception):
    pass


def run(seed):
    phi0 = signals.random_phases(seed)
    state = {"n": 0, "best": np.inf, "trace": []}

    def f(phi):
        if state["n"] >= MAX_FEV:
            raise Budget
        Z, _, _ = cost.samples_from_phases(phi)
        C = float(cost.cost(Z))
        state["n"] += 1
        state["best"] = min(state["best"], C)
        state["trace"].append(state["best"])
        return C

    t0 = time.time()
    try:
        minimize(f, phi0, method="BFGS", jac="2-point", options={"maxiter": 10000, "gtol": 1e-6})
    except Budget:
        pass
    return {"seed": seed, "C_init": state["trace"][0], "C_final": state["best"], "nfev": state["n"],
            "n_grad_equiv": state["n"] / (cfg.F_LINES + 1), "time_s": time.time() - t0,
            "trace_every_185": state["trace"][::cfg.F_LINES + 1]}


if __name__ == "__main__":
    folder = os.path.join(cfg.RESULTS_DIR, "h1_fmincon")
    os.makedirs(folder, exist_ok=True)
    for s in map(int, sys.argv[1:]):
        p = os.path.join(folder, f"seed_{s:02d}.json")
        if os.path.exists(p):
            continue
        r = run(s)
        json.dump(r, open(p, "w"))
        print(f"H1 seed {s}: {r['C_init']:.2f} -> {r['C_final']:.3f} ({r['nfev']} evals, {r['time_s']:.0f}s)", flush=True)
