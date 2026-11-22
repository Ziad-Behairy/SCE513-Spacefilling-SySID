"""
run_sec45.py  --  target T6 (paper Sec. 4.5): Schroeder multisine.

Same amplitudes as Eq. (12), phases from Eq. (13).  Two index conventions
(see signals.schroeder_phases).  We report crest factor and cost of both,
next to the random-phase and the optimized signal of run_sec44.py.
"""
import json
import os
import numpy as np
import config as cfg
import signals
import system
import cost


def main():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out = {}
    for idx in ("standard", "line"):
        phi = signals.schroeder_phases(idx)
        u = signals.multisine(phi)
        Z, _, _ = cost.samples_from_phases(phi)
        kap, inside = cost.coverage(Z)
        _, p_ok = system.steady_state_check(u)
        out[idx] = {"CF": signals.crest_factor(u), "C": cost.cost(Z), "std": float(u.std()),
                    "kappa": kap, "inside": inside, "ss_period": p_ok}
        np.save(os.path.join(cfg.RESULTS_DIR, f"Z_schroeder_{idx}.npy"), Z)
        print(f"[Sec 4.5] Schroeder ({idx:8s}): CF = {out[idx]['CF']:.3f}, C = {out[idx]['C']:.3f}, "
              f"kappa = {kap:.3f}")
    with open(os.path.join(cfg.RESULTS_DIR, "sec45.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    return out


if __name__ == "__main__":
    main()
