"""
run_parallel.py  --  run the M4 jobs on many CPU cores at once.

Each job is one seed of one script, started as its own process, so N cores
run N seeds at the same time.  Finished seeds (existing result files) are
skipped by the scripts themselves, so this is restart-safe and the results
are identical to running the scripts one after another (same seeds).

Usage:  python run_parallel.py [n_workers]      (default: number of cores - 1)
"""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

JOBS = ([[PY, "run_mc_phases.py", str(s), str(s)] for s in range(1, 21)]          # Monte Carlo with phases
        + [[PY, "run_h1_fmincon.py", str(s)] for s in range(0, 10)]               # H1 fmincon-like test
        + [[PY, "run_eps_reopt.py"]]                                              # epsilon re-optimisation
        + [[PY, "-m", "ext.run_extension", str(s), str(s)] for s in range(1, 21)])  # extension (waits for its MC seed)


def done(job):
    R = os.path.join(HERE, "..", "results")
    name = job[1] if job[1] != "-m" else job[2]
    s = job[-1]
    f = {"run_mc_phases.py": f"mc_phases/seed_{int(s):02d}.json", "run_h1_fmincon.py": f"h1_fmincon/seed_{int(s):02d}.json",
         "run_eps_reopt.py": "eps_reopt.json", "ext.run_extension": f"ext/seed_{int(s):02d}.json"}[name]
    return os.path.exists(os.path.join(R, f))


def run(job):
    if done(job):
        return
    t = time.time()
    r = subprocess.run(job, cwd=HERE, env=ENV, capture_output=True, text=True)
    print(f"[{time.strftime('%H:%M:%S')}] {' '.join(job[1:])}: exit {r.returncode}, {time.time()-t:.0f}s "
          f"{(r.stdout.strip().splitlines() or [''])[-1]} {r.stderr.strip()[-300:] if r.returncode else ''}", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, (os.cpu_count() or 2) - 1)
    print(f"{len(JOBS)} jobs on {n} workers", flush=True)
    with ThreadPoolExecutor(n) as ex:
        list(ex.map(run, JOBS))
    print("all jobs finished", flush=True)
