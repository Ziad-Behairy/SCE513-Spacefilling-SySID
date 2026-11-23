"""
run_noise_remedies.py  --  extension E4a: where does the noise error come from, and what fixes it?

At 30 dB the model error grows (Table 6).  Two candidate causes:
  * BIAS (errors-in-variables): noisy states sit inside phi(z), so LS is biased.
    Course remedy: instrumental variables (IV).
  * VARIANCE: the target (x2[k+1]-x2[k])/Ts amplifies the sensor noise by sqrt(2)/Ts.
    Remedy that uses the periodic input: measure M periods and average them
    (noise std drops by sqrt(M); the signal is the same in every period).

For seeds 1-20, designs R, S, O20, Ofull at 30 dB we compare
  LS-1   : ridge LS on one measured period (the setting of Table 6)
  IV-1   : ridge IV on one period, instruments phi([u_k, x_{k-1}]) (lagged noisy states,
           independent of the noise at k and k+1 because the noise is white)
  LS-avg : ridge LS on the average of M = 4 measured periods
and a bias/variance split for seed 1 (model from the mean theta over 15 noise draws).
Output: results/ext_noise_remedies.json
"""
import json, os
import numpy as np
import config as cfg
import signals, system
from ext import ext_config as ec, ident
from ext.run_extension import MCP, F_UNI, G_UNI, G_SD, X_REF_STD

SNR = 30.0
M_AVG = 4
IV_LAMBDAS = ec.LAMBDAS


def e_uni(theta):
    return float(np.sqrt(np.mean((F_UNI @ theta - G_UNI) ** 2)) / G_SD)


def measure(u, seed, n_periods=1):
    """steady-state period + n_periods independent noise realisations, averaged"""
    X = system.steady_state_response(u)[0]
    rng = np.random.default_rng(seed)
    sd = X_REF_STD * 10 ** (-SNR / 20)
    noise = np.mean([rng.standard_normal(X.shape) * sd for _ in range(n_periods)], axis=0)
    return X + noise


def fit_ls(u, X, return_lambda=False):
    Z, y = ident.regression_data(u, X)
    theta, lam, _ = ident.fit(Z, y)
    return (theta, lam) if return_lambda else theta


def fit_iv(u, X):
    """ridge IV; lambda by blocked CV of the IV fit on the training data only"""
    Z, y = ident.regression_data(u, X)
    Zl = np.roll(Z, 1, axis=0)
    Zl[:, 0] = Z[:, 0]                       # the input is known exactly; only the states are lagged
    F, P = ident.features(Z), ident.features(Zl)
    n = len(y)
    idx = np.array_split(np.arange(n), ec.CV_FOLDS)

    def solve(rows, lam):
        reg = lam * np.eye(F.shape[1]); reg[0, 0] = 0.0
        return np.linalg.solve(P[rows].T @ F[rows] + reg * len(rows), P[rows].T @ y[rows])

    cv = []
    for lam in IV_LAMBDAS:
        err = 0.0
        for te in idx:
            tr = np.setdiff1d(np.arange(n), te)
            err += np.sum((F[te] @ solve(tr, lam) - y[te]) ** 2)
        cv.append(err)
    return solve(np.arange(n), IV_LAMBDAS[int(np.argmin(cv))])


def main():
    out = {"snr": SNR, "M": M_AVG, "per_seed": {}}
    for s in range(1, 21):
        mc = json.load(open(os.path.join(MCP, f"seed_{s:02d}.json")))
        des = {"R": np.array(mc["phi0"]), "S": signals.schroeder_phases(),
               "O20": np.array(mc["phi_20"]), "Ofull": np.array(mc["phi"])}
        row = {}
        for d, phi in des.items():
            u = signals.multisine(phi)
            nseed = s + ec.NOISE_SEED_OFFSET + 200          # same noise seed as Table 6 at 30 dB
            X1 = measure(u, nseed, 1)
            Xa = measure(u, nseed, M_AVG)
            t1, l1 = fit_ls(u, X1, True)
            ta, la = fit_ls(u, Xa, True)
            row[d] = {"LS1": e_uni(t1), "IV1": e_uni(fit_iv(u, X1)), "LSavg": e_uni(ta),
                      "lam_LS1": l1, "lam_LSavg": la}
        out["per_seed"][s] = row
        print(s, {d: {k: round(v, 4) for k, v in r.items()} for d, r in row.items()}, flush=True)
    # bias / variance split, seed 1
    mc = json.load(open(os.path.join(MCP, "seed_01.json")))
    split = {}
    for d, phi in {"R": np.array(mc["phi0"]), "Ofull": np.array(mc["phi"])}.items():
        u = signals.multisine(phi)
        th = np.array([fit_ls(u, measure(u, 9000 + k, 1)) for k in range(15)])
        clean = e_uni(fit_ls(u, system.steady_state_response(u)[0]))
        split[d] = {"clean": clean, "single_mean": float(np.mean([e_uni(t) for t in th])),
                    "mean_theta": e_uni(th.mean(0))}
        print("split", d, split[d], flush=True)
    out["split"] = split
    json.dump(out, open(os.path.join(cfg.RESULTS_DIR, "ext_noise_remedies.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
