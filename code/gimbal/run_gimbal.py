"""
run_gimbal.py  --  validation of the extension on the gimbal axis (gimbal.py).

For every seed s:
  R      random phases                          (what most people would do)
  S      Schroeder phases                       (classical design, same spectrum)
  Osur   designed on a model identified from the R experiment, 20 iterations
         (the TRUE plant is never used for design: the realistic case)
  Otask  TASK-AWARE design: same as Osur, but the points to cover are the
         operating points of the control tasks (5 + 5 design target motions,
         seeds 3000-3004, mapped to [u, theta, w] with the R model), not the box
Each input is applied to the TRUE plant, the states are measured with 40 dB
sensor noise, and g(u, theta, w) is identified with the RBF ridge-LS model.

Tests (never used for fitting or for choosing lambda):
  e_uni   one-step error of g on 20000 uniform points of the domain
  e_lim   the same, only for points beyond the joint limit (|theta| > TL)
  e_taskA/B  one-step error of g at the true operating points of the TEST motions
  Task A = normal aiming (to the joint limit, 0.2-1.2 Hz),
  Task B = aggressive slews (into the bumpers, 0.5-1.5 Hz); both fixed before any run.
  e_trk   RMS tracking error of a model-based (computed-torque) controller that
          uses the identified model, on the TRUE plant, 5 target motions
Usage: python -m gimbal.run_gimbal first_seed last_seed
"""
import json, os, sys, time
import numpy as np
from gimbal import gimbal as G

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "results", "gimbal")
SNR = 40.0
KP, KD = 25.0, 10.0                         # weak feedback: the model must do most of the work

_r = np.random.default_rng(777)
ZU = G.LO + (G.HI - G.LO) * _r.random((20000, 3))
GU = np.array([G.g_true(*z)[0] for z in ZU])
GSD = GU.std()
FU = G.features(ZU)
LIM = np.abs(ZU[:, 1]) > G.TL
REFS = {k: [G.reference(2000 + i, k) for i in range(5)] for k in "AB"}   # test motions
DREFS = [G.reference(3000 + i, k) for k in "AB" for i in range(5)]       # design motions
ZT = {k: G.task_points(REFS[k], G._NO, False, n=4000, seed=1) for k in "AB"}
GT = {k: np.array([G.g_true(*z)[0] for z in ZT[k]]) for k in "AB"}
FT = {k: G.features(ZT[k]) for k in "AB"}
TASK_TRUE = G.task_points(DREFS, G._NO, False)
XREF = G.steady(G.multisine(G.random_phases(0))).std(0)


def evaluate(theta):
    e = FU @ theta - GU
    return {"e_uni": float(np.sqrt(np.mean(e ** 2)) / GSD),
            "e_lim": float(np.sqrt(np.mean(e[LIM] ** 2)) / GSD),
            **{f"e_task{k}": float(np.sqrt(np.mean((FT[k] @ theta - GT[k]) ** 2)) / GT[k].std())
               for k in "AB"},
            **{f"e_trk{k}": [float(G.track(r, theta, True, KP, KD)) for r in REFS[k]] for k in "AB"}}


def experiment(phi, seed):
    u = G.multisine(phi)
    X = G.steady(u)
    Zt = np.column_stack([u, X])
    Xm = X + np.random.default_rng(seed + 5000).standard_normal(X.shape) * XREF * 10 ** (-SNR / 20)
    return G.identify(u, Xm), Zt


def run_seed(s):
    phi0 = G.random_phases(s)
    out = {"seed": s, "designs": {}}
    todo = ["R", "S", "Osur", "Otask"]
    th = {}
    for name in todo:
        t0 = time.time()
        if name == "R":
            phi = phi0
        elif name == "S":
            phi = G.schroeder()
        elif name == "Osur":
            phi = G.design(phi0, th["R"], True, 20)
        elif name == "Otask":
            grid = G.task_points(DREFS, th["R"], True)
            phi = G.design(phi0, th["R"], True, 20, grid)

        theta, Zt = experiment(phi, s)
        th[name] = theta
        r = evaluate(theta)
        r.update(C=G.cost(Zt), C_task=G.cost(Zt, TASK_TRUE), kappa=G.kappa(Zt), lim_share=float(np.mean(np.abs(Zt[:, 1]) > G.TL)),
                 time_s=time.time() - t0)
        if s == 1:
            r["phi"] = [float(v) for v in phi]
        out["designs"][name] = r
    # model-free floors for the control task
    out["trk_true_model"] = {k: [float(G.track(r, G._NO, False, KP, KD)) for r in REFS[k]] for k in "AB"}
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    a, b = int(sys.argv[1]), int(sys.argv[2])
    for s in range(a, b + 1):
        p = os.path.join(OUT, f"seed_{s:02d}.json")
        if os.path.exists(p):
            continue
        t = time.time()
        o = run_seed(s)
        json.dump(o, open(p, "w"))
        print(f"seed {s} {time.time()-t:.0f}s " + " ".join(
            f"{k}:uni={v['e_uni']:.2f},lim={v['e_lim']:.2f},A={v['e_taskA']:.2f},B={v['e_taskB']:.2f},"
            f"trkA={np.median(v['e_trkA'])*57.3:.1f},trkB={np.median(v['e_trkB'])*57.3:.1f}"
            for k, v in o["designs"].items()), flush=True)
