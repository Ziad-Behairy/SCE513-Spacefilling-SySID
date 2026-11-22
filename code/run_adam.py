"""
run_adam.py  --  optimizer cross-check (M2 plan): Adam (lr 0.05) on seed 0.

If a completely different optimizer, started from the same phases, also
goes far below the paper's 3.88, the low cost is a property of the problem,
not a trick of L-BFGS-B.  Result -> results/adam.json, results/adam_history.npy
"""
import json
import os
import time
import numpy as np
import config as cfg
import signals
import system
from optimize import adam


def main():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    phi, hist = adam(signals.random_phases(cfg.SEED_MAIN))
    _, p_ok = system.steady_state_check(signals.multisine(phi))
    out = {"lr": cfg.ADAM_LR, "steps": cfg.ADAM_STEPS, "C_init": hist[0], "C_final": hist[-1],
           "C_at_16": hist[cfg.FMINCON_GRAD_BUDGET], "time_s": time.time() - t0,
           "CF": signals.crest_factor(signals.multisine(phi)), "ss_period": p_ok}
    np.save(os.path.join(cfg.RESULTS_DIR, "adam_history.npy"), hist)
    with open(os.path.join(cfg.RESULTS_DIR, "adam.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"[Adam] {hist[0]:.2f} -> {hist[-1]:.3f} in {cfg.ADAM_STEPS} steps ({out['time_s']:.0f} s)")
    return out


if __name__ == "__main__":
    main()
