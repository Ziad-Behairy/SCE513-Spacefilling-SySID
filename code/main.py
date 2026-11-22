"""
main.py  --  ONE script that regenerates every M3 result, in order.

    python main.py            # full M3 run (about 3.5 h on a 2-core laptop CPU;
                              #  most of it is the 20 Monte Carlo runs)
    python main.py --quick    # smoke test: 20 iterations, 2 Monte Carlo runs (~5 min)
                              #  -> results_quick/  (numbers are NOT the report's)

Order:
  1. diagnostics.py   verification checks D1-D8 (+ figure)
  2. run_sec44.py     T1, T2: seed-0 optimization (L-BFGS-B, max 1000 iterations)
  3. run_adam.py      optimizer cross-check on seed 0 (Adam, 400 steps)
     run_early.py     H1 test: the same optimization stopped after 16/20/25 iterations
  4. run_sec45.py     T6: Schroeder multisine (crest factor, cost)
  5. validation.py    V1-V4: epsilon sensitivity, 30^3 grid, initial state, P = 12
  6. run_sec46.py     T7: Monte Carlo, seeds 1-20 (max 1000 iterations each)
  7. make_figures.py  all figures + comparison table + numbers.tex for the report
Tip: run_sec46.py can be split over two terminals:
     python run_sec46.py 1 10   and   python run_sec46.py 11 20
"""
import sys
import time
import config as cfg

QUICK = "--quick" in sys.argv
if QUICK:
    cfg.OPT_MAXITER = 20
    cfg.ADAM_STEPS = 20
    cfg.RESULTS_DIR = cfg.RESULTS_DIR.replace("results", "results_quick")

import diagnostics
import run_sec44
import run_adam
import run_early
import run_sec45
import validation
import run_sec46
import make_figures

MC_SEEDS = cfg.SEEDS_MC[:2] if QUICK else cfg.SEEDS_MC

t0 = time.time()
for name, step in [("diagnostics", diagnostics.main),
                   ("Sec. 4.4 (T1, T2)", run_sec44.main),
                   ("Adam cross-check", run_adam.main),
                   ("H1 early-stopping test", run_early.main),
                   ("Sec. 4.5 (T6)", run_sec45.main),
                   ("validation", validation.main),
                   ("Sec. 4.6 (T7)", lambda: run_sec46.main(MC_SEEDS, cfg.OPT_MAXITER)),
                   ("figures", make_figures.main)]:
    print(f"\n===== {name} =====", flush=True)
    step()
print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. Results in '{cfg.RESULTS_DIR}/'.")
