"""
run_sec44.py  --  targets T1 and T2 (paper Sec. 4.4), seed 0.

1. draw random initial phases (seed 0)
2. optimize them with L-BFGS-B
3. verify steady state of the final signal (M2 criterion)
4. save everything to results/sec44.npz and results/sec44.json
"""
import json
import os
import numpy as np
import config as cfg
import signals
import system
import cost
from optimize import optimize_phases, best_within_budget


def main():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    phi0 = signals.random_phases(cfg.SEED_MAIN)
    print(f"[Sec 4.4] seed {cfg.SEED_MAIN}: initial cost = {cost.objective(phi0)[0]:.4f}")
    r = optimize_phases(phi0)
    print(f"[Sec 4.4] final cost = {r['C']:.4f} after {r['nit']} iterations "
          f"({r['nfev']} evaluations, {r['time_s']:.0f} s) -- {r['message']}")

    u0, u1 = signals.multisine(phi0), signals.multisine(r["phi"])
    rels0, p0 = system.steady_state_check(u0)
    rels1, p1 = system.steady_state_check(u1)
    Z0, _, _ = cost.samples_from_phases(phi0)
    Z1, _, _ = cost.samples_from_phases(r["phi"])
    kap0, in0 = cost.coverage(Z0)
    kap1, in1 = cost.coverage(Z1)
    dA = np.max(np.abs(signals.amplitude_spectrum(u1) - signals.amplitude_spectrum(u0)))

    summary = {
        "seed": cfg.SEED_MAIN, "eps": cfg.EPS, "P": cfg.P_PERIODS,
        "C_init": r["C0"], "C_opt": r["C"], "ratio": r["C0"] / r["C"],
        "nit": r["nit"], "nfev": r["nfev"], "time_s": r["time_s"], "message": r["message"],
        "CF_init": signals.crest_factor(u0), "CF_opt": signals.crest_factor(u1),
        "std_init": float(u0.std()), "std_opt": float(u1.std()),
        "max_abs_dA": float(dA),
        "kappa_init": kap0, "kappa_opt": kap1, "inside_init": in0, "inside_opt": in1,
        "ss_period_init": p0, "ss_period_opt": p1,
        "ss_rel_at_P_init": rels0[cfg.P_PERIODS - 2], "ss_rel_at_P_opt": rels1[cfg.P_PERIODS - 2],
        "C_budget16": best_within_budget(r["eval_costs"]),
        "first_iter_below_paper": int(np.argmax(r["history"] <= cfg.PAPER["C_opt"])),
    }
    np.savez(os.path.join(cfg.RESULTS_DIR, "sec44.npz"), phi0=phi0, phi=r["phi"],
             history=r["history"], eval_costs=r["eval_costs"], Z0=Z0, Z1=Z1, rels0=rels0, rels1=rels1)
    with open(os.path.join(cfg.RESULTS_DIR, "sec44.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    for k, v in summary.items():
        print(f"  {k:18s} {v}")
    return summary


if __name__ == "__main__":
    main()
