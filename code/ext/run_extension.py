"""
run_extension.py  --  EXTENSION experiments (bonus).

For every Monte-Carlo seed s (1..20) and every noise level:
  1. build the input designs (all: same 184 lines, same amplitudes, std 160 N)
       R      random phases phi0(s)                       (paper's initial signal)
       S      Schroeder phases                            (classical low-crest-factor design)
       O16    optimised on the TRUE model, 16 iterations  (fmincon-default-like budget)
       O20    optimised on the TRUE model, 20 iterations  (paper-like cost ~3.9)
       Ofull  optimised on the TRUE model, <=1000 iterations
       Osur   optimised on a model IDENTIFIED from the R experiment (Q2),
              20 iterations from phi0(s)   [+ Osur100 at 40 dB: 100 iterations]
       Oiter  second round: model re-identified from the Osur experiment,
              20 more iterations on it (iterative identification + design)
  2. apply each input to the TRUE system, keep one steady-state period,
     add the SAME absolute sensor noise (fixed std, independent of the design)
  3. identify g(u,x1,x2) with the RBF ridge-LS model (ident.py)
  4. evaluate on data never used for fitting:
       e_uni  one-step NRMSE of g over 20000 uniform points in D_z (region of interest)
       e_sim  free-run simulation NRMSE (x1,x2) on 5 new random-phase multisines,
              std 160 N (design level) and 240 N (stress test); divergences counted
       error-vs-density statistics (why the errors differ)
Results: results/ext/seed_XX.json  (one file per seed, restart-safe)
Usage: python -m ext.run_extension first_seed last_seed
"""
import json, os, sys, time
import numpy as np
from scipy.optimize import minimize
import config as cfg
import cost, signals, system
from ext import ext_config as ec, ident, surrogate

OUT = os.path.join(cfg.RESULTS_DIR, "ext")
MCP = os.path.join(cfg.RESULTS_DIR, "mc_phases")

# ---- fixed evaluation data (identical for every design and seed) ----------
_rng = np.random.default_rng(ec.UNIFORM_SEED)
Z_UNI = cfg.DOMAIN_LO + (cfg.DOMAIN_HI - cfg.DOMAIN_LO) * _rng.random((ec.N_UNIFORM, 3))
G_UNI = ident.g_true(Z_UNI)
G_SD = float(G_UNI.std())
F_UNI = ident.features(Z_UNI)

# absolute sensor noise: std = (state std of the seed-0 random-phase reference) * 10^(-SNR/20)
_u_ref = signals.multisine(signals.random_phases(cfg.SEED_MAIN))
X_REF_STD = system.steady_state_response(_u_ref)[0].std(axis=0)

TESTS = []
for sd in ec.TEST_STD:
    for ts in ec.TEST_SEEDS:
        u = signals.multisine(signals.random_phases(ts)) * (sd / cfg.U_STD)
        TESTS.append((sd, u, system.steady_state_response(u)[0]))

LOGD_EDGES = np.array([-np.inf, -3, -2, -1, 0, 1, 2, np.inf])   # bins of log10 density


def density_at(Zq, Ztr, chunk=4000):
    """kernel density d(z) of the training samples at query points (paper Eq. 7 kernel)."""
    out = np.empty(len(Zq))
    Wt = Ztr / cfg.SIGMA
    for i in range(0, len(Zq), chunk):
        Wq = Zq[i:i + chunk] / cfg.SIGMA
        D2 = (Wq ** 2).sum(1)[:, None] + (Wt ** 2).sum(1)[None, :] - 2 * Wq @ Wt.T
        out[i:i + chunk] = np.exp(-0.5 * np.maximum(D2, 0)).sum(1)
    return out


def sim_model(theta, u):
    X = surrogate.sim(theta, np.tile(u, cfg.P_PERIODS))
    return X[(cfg.P_PERIODS - 1) * cfg.N:]


def evaluate(theta, Ztr):
    e = F_UNI @ theta - G_UNI
    out = {"e_uni": float(np.sqrt(np.mean(e ** 2)) / G_SD)}
    # error vs training-data density
    logd = np.log10(density_at(Z_UNI, Ztr) + 1e-300)
    b = np.digitize(logd, LOGD_EDGES[1:-1])
    out["dens_sse"] = [float(np.sum(e[b == i] ** 2)) for i in range(len(LOGD_EDGES) - 1)]
    out["dens_n"] = [int(np.sum(b == i)) for i in range(len(LOGD_EDGES) - 1)]
    for sd in ec.TEST_STD:
        errs, div = [], 0
        for s2, u, Xt in TESTS:
            if s2 != sd:
                continue
            Xh = sim_model(theta, u)
            if (not np.all(np.isfinite(Xh))) or np.max(np.abs(Xh)) > ec.DIVERGE:
                div += 1
                errs.append(np.inf)
                continue
            errs.append(float(np.mean(np.sqrt(np.mean((Xh - Xt) ** 2, 0)) / Xt.std(0))))
        out[f"e_sim{int(sd)}"] = float(np.median(errs))
        out[f"e_sim{int(sd)}_all"] = errs
        out[f"div{int(sd)}"] = div
    return out


def experiment(u, noise_seed, snr):
    """apply u to the TRUE system, measure with sensor noise, identify."""
    X = system.steady_state_response(u)[0]
    Ztrue = np.column_stack([u, X])
    if np.isfinite(snr):
        X = X + np.random.default_rng(noise_seed).standard_normal(X.shape) * X_REF_STD * 10 ** (-snr / 20)
    Z, y = ident.regression_data(u, X)
    theta, lam, _ = ident.fit(Z, y)
    return theta, lam, Ztrue


def design_on_surrogate(phi0, theta, iters):
    hist = []
    res = minimize(lambda p: surrogate.objective_hat(p, theta), phi0, jac=True, method="L-BFGS-B",
                   callback=lambda intermediate_result: hist.append(float(intermediate_result.fun)),
                   options={"maxiter": iters, "ftol": cfg.OPT_FTOL, "gtol": cfg.OPT_GTOL, "maxcor": 10})
    return res.x, float(res.fun)


def run_seed(seed):
    mc = json.load(open(os.path.join(MCP, f"seed_{seed:02d}.json")))
    designs = {"R": np.array(mc["phi0"]), "S": signals.schroeder_phases(),
               "O16": np.array(mc["phi_16"]), "O20": np.array(mc["phi_20"]), "Ofull": np.array(mc["phi"])}
    res = {"seed": seed, "levels": {}}
    for li, snr in enumerate(ec.SNR_DB):
        key = "inf" if not np.isfinite(snr) else str(int(snr))
        nseed = seed + ec.NOISE_SEED_OFFSET + 100 * li
        L = {}
        thetaR = None
        todo = (list(designs.items()) + [("Osur", None), ("Oiter", None)]
                + ([("Osur100", None)] if snr == 40.0 else []))
        for name, phi in todo:
            t0 = time.time()
            extra = {}
            if name.startswith("Osur"):
                iters = ec.SUR_ITERS if name == "Osur" else 100
                phi, C_hat = design_on_surrogate(designs["R"], thetaR, iters)
                extra["C_hat"] = C_hat
            if name == "Oiter":
                # round 2: re-identify from the Osur experiment, re-design from phi_sur
                phi, C_hat = design_on_surrogate(phi_sur, theta_sur, ec.SUR_ITERS)
                extra["C_hat"] = C_hat
            u = signals.multisine(phi)
            theta, lam, Ztrue = experiment(u, nseed, snr)      # same noise sequence for every design
            if name == "R":
                thetaR = theta
            if name == "Osur":
                phi_sur, theta_sur = phi, theta
            r = evaluate(theta, Ztrue)
            r.update(extra)
            r["C_true"] = float(cost.cost(Ztrue))
            r["kappa"] = float(cost.coverage(Ztrue)[0])
            r["CF"] = float(signals.crest_factor(u))
            r["lam"] = lam
            r["time_s"] = time.time() - t0
            L[name] = r
        res["levels"][key] = L
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    a, b = int(sys.argv[1]), int(sys.argv[2])
    for s in range(a, b + 1):
        path = os.path.join(OUT, f"seed_{s:02d}.json")
        if os.path.exists(path):
            continue
        while not os.path.exists(os.path.join(MCP, f"seed_{s:02d}.json")):
            time.sleep(20)
        t = time.time()
        out = run_seed(s)
        json.dump(out, open(path, "w"))
        lv = out["levels"]["40"]
        print(f"seed {s} ({time.time()-t:.0f}s) e_uni@40dB: " +
              " ".join(f"{k}={v['e_uni']:.3f}" for k, v in lv.items()), flush=True)
