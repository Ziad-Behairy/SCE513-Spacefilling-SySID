"""
run_fixed_lambda.py  --  robustness check for the extension (reviewer item):
is the accuracy gain of the space-filling design an artefact of the
cross-validated ridge parameter?  Here every design is identified with the
SAME fixed lambda (two values), for the designs whose phases are stored
(R, S, O20, Ofull), all seeds and noise levels.  Output: results/ext_fixed_lambda.json
"""
import json, os
import numpy as np
import config as cfg
import signals
from ext import ext_config as ec, ident
import system
from ext.run_extension import MCP, F_UNI, G_UNI, G_SD, X_REF_STD


def fit_fixed(u, noise_seed, snr, lam):
    """same experiment as run_extension.experiment, but with a fixed ridge lambda"""
    X = system.steady_state_response(u)[0]
    if np.isfinite(snr):
        X = X + np.random.default_rng(noise_seed).standard_normal(X.shape) * X_REF_STD * 10 ** (-snr / 20)
    Z, y = ident.regression_data(u, X)
    return ident._ridge(ident.features(Z), y, lam)

LAMS = [1e-6, 1e-4]


def main():
    out = {str(l): {} for l in LAMS}
    for lam in LAMS:
        for li, snr in enumerate(ec.SNR_DB):
            key = "inf" if not np.isfinite(snr) else str(int(snr))
            E = {d: [] for d in ("R", "S", "O20", "Ofull")}
            for s in range(1, 21):
                mc = json.load(open(os.path.join(MCP, f"seed_{s:02d}.json")))
                des = {"R": np.array(mc["phi0"]), "S": signals.schroeder_phases(),
                       "O20": np.array(mc["phi_20"]), "Ofull": np.array(mc["phi"])}
                for d, phi in des.items():
                    theta = fit_fixed(signals.multisine(phi), s + ec.NOISE_SEED_OFFSET + 100 * li, snr, lam)
                    E[d].append(float(np.sqrt(np.mean((F_UNI @ theta - G_UNI) ** 2)) / G_SD))
            out[str(lam)][key] = E
            r = np.array(E["R"]); o = np.array(E["Ofull"])
            print(f"lambda {lam:g} {key}: median e_uni R {np.median(r):.3f} S {np.median(E['S']):.3f} "
                  f"O20 {np.median(E['O20']):.3f} Ofull {np.median(o):.3f} paired factor {np.median(r / o):.1f}", flush=True)
    json.dump(out, open(os.path.join(cfg.RESULTS_DIR, "ext_fixed_lambda.json"), "w"))


if __name__ == "__main__":
    main()
