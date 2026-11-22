"""
run_sec46.py  --  target T7 (paper Sec. 4.6): Monte Carlo over random
initial phases (seeds 1..20, same optimizer and settings as Sec. 4.4).

Usage:  python run_sec46.py [first_seed last_seed]
Each finished run is saved to results/mc/seed_XX.json (cost history included),
so runs can be split over several processes and restarted safely: seeds
that already have a file are skipped.
"""
import json
import os
import sys
import config as cfg
import signals
from optimize import optimize_phases, best_within_budget


def main(seeds=cfg.SEEDS_MC, maxiter=cfg.OPT_MAXITER):
    folder = os.path.join(cfg.RESULTS_DIR, "mc")
    os.makedirs(folder, exist_ok=True)
    for seed in seeds:
        path = os.path.join(folder, f"seed_{seed:02d}.json")
        if os.path.exists(path):
            continue
        r = optimize_phases(signals.random_phases(seed), maxiter=maxiter, verbose=False)
        out = {"seed": seed, "C_init": r["C0"], "C_opt": r["C"], "nit": r["nit"], "nfev": r["nfev"],
               "time_s": r["time_s"], "message": r["message"],
               "C_budget16": best_within_budget(r["eval_costs"]),
               "history": r["history"].tolist()}
        with open(path, "w") as fh:
            json.dump(out, fh)
        print(f"[MC] seed {seed:2d}: {r['C0']:.2f} -> {r['C']:.3f}  (16-grad budget: {out['C_budget16']:.3f}; "
              f"{r['nit']} it, {r['time_s']:.0f} s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(list(range(int(sys.argv[1]), int(sys.argv[2]) + 1)))
    else:
        main()
