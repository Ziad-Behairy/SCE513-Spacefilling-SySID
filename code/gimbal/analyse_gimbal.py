"""
analyse_gimbal.py  --  figure + summary numbers of the gimbal validation.
Reads results/gimbal/seed_*.json (run_gimbal.py) and writes
  results/gimbal/fig_gimbal.png      (model error and tracking error per design)
  results/gimbal/summary.json        (medians, wins over R, one-sided Wilcoxon p)
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

D_ = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "gimbal")
RUNS = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(D_, "seed_*.json")))]
KS = ["R", "S", "Osur", "Otask"]
LAB = {"R": "random", "S": "Schroeder", "Osur": "SF box", "Otask": "SF task"}
COL = {"R": "#7a8793", "S": "#4b6a88", "Osur": "#e8590c", "Otask": "#0f766e"}
DEG = 180 / np.pi


def vals(key):
    if key.startswith("e_trk"):
        return {k: np.array([np.median(r["designs"][k][key]) for r in RUNS]) * DEG for k in KS}
    return {k: np.array([r["designs"][k][key] for r in RUNS]) * 100 for k in KS}


def main():
    panels = [("e_uni", "model error, whole box [%]"), ("e_lim", "model error, beyond joint limit [%]"),
              ("e_trkA", "tracking error, task A [deg]"), ("e_trkB", "tracking error, task B [deg]")]
    summ = {"n_seeds": len(RUNS)}
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.7))
    for ax, (key, title) in zip(axs, panels):
        v = vals(key)
        summ[key] = {}
        for i, k in enumerate(KS):
            ax.boxplot(v[k], positions=[i], widths=0.55, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=COL[k], alpha=0.35, edgecolor=COL[k]),
                       medianprops=dict(color=COL[k], lw=2), whiskerprops=dict(color=COL[k]),
                       capprops=dict(color=COL[k]))
            ax.scatter(np.full(len(v[k]), i) + np.linspace(-0.12, 0.12, len(v[k])), v[k], s=9, color=COL[k], zorder=3)
            s = {"median": float(np.median(v[k]))}
            if k != "R":
                s["wins_vs_R"] = int((v[k] < v["R"]).sum())
                s["p_less_than_R"] = float(wilcoxon(v[k], v["R"], alternative="less").pvalue)
            summ[key][k] = s
        if key.startswith("e_trk"):
            fl = np.median([np.median(r["trk_true_model"][key[-1]]) for r in RUNS]) * DEG
            ax.axhline(fl, color="k", ls=":", lw=1)
            ax.text(3.45, fl, "true model", fontsize=7, va="bottom", ha="right")
            summ[key]["true_model"] = float(fl)
        ax.set_xticks(range(4), [LAB[k] for k in KS], fontsize=7)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(D_, "fig_gimbal.png"), dpi=200)
    for k in ["kappa", "C", "lim_share"]:
        summ[k] = {d: float(np.median([r["designs"][d][k] for r in RUNS])) for d in KS}
    json.dump(summ, open(os.path.join(D_, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
