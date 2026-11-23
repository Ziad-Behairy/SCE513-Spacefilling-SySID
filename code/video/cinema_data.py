"""
cinema_data.py  --  everything the cinematic video draws, computed once with the project code.
No new results: seed-1 signals, models and trajectories, recomputed exactly as in the experiments.
Output: results/video/cinema_data.npz
"""
import json, os
import numpy as np
from scipy.optimize import minimize
import config as cfg
import cost, signals, system
from ext import ident, ext_config as ec
from ext import run_extension as RE
from ext import run_peak_limited as PK
from gimbal import gimbal as G
from gimbal import run_gimbal as RG
from video.make_story import track_log

OUT = os.path.join(cfg.RESULTS_DIR, "video", "cinema_data.npz")


def main():
    mc = json.load(open(os.path.join(cfg.RESULTS_DIR, "mc_phases", "seed_01.json")))
    D = {}
    phis = {"R": np.array(mc["phi0"]), "O": np.array(mc["phi"])}
    # peak-limited design, CF <= 1.9 (same settings as run_peak_limited.py)
    umax = 1.9 * cfg.U_STD
    r = minimize(lambda p: PK.objective(p, umax), phis["R"], jac=True, method="L-BFGS-B",
                 options={"maxiter": PK.ITERS, "ftol": cfg.OPT_FTOL, "gtol": cfg.OPT_GTOL, "maxcor": 10})
    phis["P"] = r.x
    for k, ph in phis.items():
        u = signals.multisine(ph)
        X = system.steady_state_response(u)[0]
        Z = np.column_stack([u, X])
        d, _ = cost.densities(Z)
        D[f"u_{k}"], D[f"Z_{k}"], D[f"empty_{k}"] = u, Z, cost.GRID[d < 1.0]
        D[f"phi_{k}"] = ph
        D[f"kappa_{k}"] = cost.coverage(Z)[0]
        D[f"C_{k}"] = cost.cost(Z)
        D[f"CF_{k}"] = signals.crest_factor(u)
    # models at 40 dB, seed 1 (same noise seed as Table 6)
    nseed = 1 + ec.NOISE_SEED_OFFSET + 100 * list(ec.SNR_DB).index(40.0)
    for k in ["R", "O"]:
        th, lam, _ = RE.experiment(D[f"u_{k}"], nseed, 40.0)
        D[f"theta_{k}"] = th
    # model surface slice at u = 0
    x1 = np.linspace(-2, 2, 61)
    x2 = np.linspace(-20, 20, 61)
    X1, X2 = np.meshgrid(x1, x2)
    Zq = np.column_stack([np.zeros(X1.size), X1.ravel(), X2.ravel()])
    D["surf_x1"], D["surf_x2"] = X1, X2
    D["surf_true"] = ident.g_true(Zq).reshape(X1.shape)
    for k in ["R", "O"]:
        D[f"surf_{k}"] = ident.predict(D[f"theta_{k}"], Zq).reshape(X1.shape)
    # noise illustration: 4 measured periods of x1 at 30 dB (the E4 noise level)
    X = system.steady_state_response(D["u_O"])[0]
    sd = RE.X_REF_STD * 10 ** (-30 / 20)
    rng = np.random.default_rng(7)
    D["noise_clean"] = X[:, 0]
    D["noise_periods"] = np.stack([X[:, 0] + rng.standard_normal(cfg.N) * sd[0] for _ in range(4)])
    # gimbal: seed 1, test motion 2001 of task B, models R and Otask
    ref = G.reference(2001, "B")
    phi0 = G.random_phases(1)
    thR, _ = RG.experiment(phi0, 1)
    grid = G.task_points(RG.DREFS, thR, True)
    thT, _ = RG.experiment(G.design(phi0, thR, True, 20, grid), 1)
    D["gim_ref"] = ref
    D["gim_R"] = track_log(ref, thR, True)
    D["gim_T"] = track_log(ref, thT, True)
    np.savez_compressed(OUT, **D)
    print({k: float(v) for k, v in D.items() if np.ndim(v) == 0})


if __name__ == "__main__":
    main()
