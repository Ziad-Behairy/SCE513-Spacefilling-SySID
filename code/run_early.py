"""
run_early.py  --  test of hypothesis H1 (early stopping), seed 0.

The paper's optimized cost (3.88) is reached by our L-BFGS-B after ~18
iterations, and fmincon's default budget allows ~16 finite-difference
gradients.  Here the SAME optimization (same start, same settings) is simply
stopped after a few iterations, and cost, ratio and crest factor are compared
with the paper's values.  L-BFGS-B is deterministic, so these are exactly the
intermediate points of the full run in run_sec44.py.
Result -> results/early.json
"""
import json
import os
import config as cfg
import signals
import cost
from optimize import optimize_phases


def main(stops=(16, 20, 25)):
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    phi0 = signals.random_phases(cfg.SEED_MAIN)
    out = []
    for it in stops:
        r = optimize_phases(phi0, maxiter=it, verbose=False)
        Z, u, _ = cost.samples_from_phases(r["phi"])
        kap, inside = cost.coverage(Z)
        out.append({"iterations": it, "C": r["C"], "ratio": r["C0"] / r["C"],
                    "CF": signals.crest_factor(u), "kappa": kap, "inside": inside})
        print(f"[H1] stop after {it:2d} iterations: C = {r['C']:.3f}, ratio = {r['C0'] / r['C']:.1f}, "
              f"CF = {out[-1]['CF']:.2f}")
    with open(os.path.join(cfg.RESULTS_DIR, "early.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    return out


if __name__ == "__main__":
    main()
