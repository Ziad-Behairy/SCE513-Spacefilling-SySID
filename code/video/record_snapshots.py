"""
record_snapshots.py -- re-run the seed-0 L-BFGS-B optimization (same settings as
run_sec44.py) for the first 300 iterations and store the phases after every
iteration, so the video can show how the data spread during optimization.
L-BFGS-B is deterministic, so these are exactly the first 300 iterations of the
reported run.  Output: results/video_snapshots.npz
"""
import os, sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
import signals
import cost

phi0 = signals.random_phases(cfg.SEED_MAIN)
snaps, costs = [phi0.copy()], [cost.objective(phi0)[0]]

def cb(intermediate_result):
    res = intermediate_result
    snaps.append(res.x.copy())
    costs.append(res.fun)

minimize(lambda p: cost.objective(p), phi0, jac=True, method="L-BFGS-B", callback=cb,
         options={"maxiter": 300, "ftol": cfg.OPT_FTOL, "gtol": cfg.OPT_GTOL, "maxcor": 10})
np.savez(os.path.join(cfg.RESULTS_DIR, "video_snapshots.npz"), phis=np.array(snaps), costs=np.array(costs))
print("saved", len(snaps), "snapshots; cost", costs[0], "->", costs[-1])
