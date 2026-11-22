"""
run_all.py  --  ONE main script that regenerates every principal result of the report.

Numbered workflow (each step is restart-safe: finished seeds are skipped):
  1. main.py               M3 core: diagnostics D1-D8, targets T1-T7, validation V1-V6, M3 figures
  2. run_parallel.py       M4: Monte Carlo with stored phases (seeds 1-20), H1 fmincon-like test
                           (seeds 0-9), epsilon re-optimisation, extension (seeds 1-20)
  3. ext/run_fixed_lambda  extension robustness check (fixed ridge lambda)
  4. m4_analysis.py        M4 figures, tables, statistics, results/numbers_m4.tex
  5. make_illustrations.py 3-D drawings (needs PyVista; skipped with a warning if missing)
  6. pdflatex/bibtex       report/report.pdf (skipped if LaTeX is not installed)

Usage (from the package root):
  python code/run_all.py              # everything (~6 h on 2 cores, ~1.5-2 h on 8 cores)
  python code/run_all.py --workers 8  # number of parallel processes for step 2
  python code/run_all.py --analysis   # steps 3-6 only, from the stored results (minutes)
  python code/run_all.py --quick      # 5-minute smoke test of step 1 (writes results_quick/)
"""
import argparse, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def step(title, cmd, cwd=HERE, optional=False):
    print(f"\n=== {title} ===", flush=True)
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        msg = f"step failed: {' '.join(cmd)} (exit {r.returncode})"
        if optional:
            print("WARNING:", msg, flush=True)
        else:
            sys.exit(msg)


def build_report():
    if shutil.which("pdflatex") is None:
        print("\nWARNING: pdflatex not found - report not rebuilt (report/report.pdf is included).")
        return
    rep = os.path.join(ROOT, "report")
    for cmd in (["pdflatex", "-interaction=nonstopmode", "report.tex"], ["bibtex", "report"],
                ["pdflatex", "-interaction=nonstopmode", "report.tex"], ["pdflatex", "-interaction=nonstopmode", "report.tex"]):
        subprocess.run(cmd, cwd=rep, stdout=subprocess.DEVNULL)
    print("\nreport/report.pdf rebuilt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--analysis", action="store_true", help="only steps 3-6, from stored results")
    ap.add_argument("--quick", action="store_true", help="smoke test of step 1 only")
    a = ap.parse_args()
    if a.quick:
        step("1. M3 core (quick smoke test)", [PY, "main.py", "--quick"])
        sys.exit(0)
    if not a.analysis:
        step("1. M3 core reproduction", [PY, "main.py"])
        step("2. M4 runs and extension (parallel)", [PY, "run_parallel.py", str(a.workers)])
    step("3. extension robustness (fixed lambda)", [PY, "-m", "ext.run_fixed_lambda"])
    step("4. M4 analysis", [PY, "m4_analysis.py"])
    step("5. illustrations", [PY, "make_illustrations.py"], optional=True)
    build_report()
    print("\nAll principal results regenerated in results/.")
