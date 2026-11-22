"""
make_figures.py  --  all figures and the comparison table from saved results.

Colours follow the paper: blue = initial random phase, orange = optimized,
purple = Schroeder.  Figures go to results/*.png, the table to
results/comparison_table.csv.
"""
import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import config as cfg
import signals

C_INIT, C_OPT, C_SCH = "#1f77b4", "#e8590c", "#7b3fa0"
R = cfg.RESULTS_DIR
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load():
    s44 = json.load(open(os.path.join(R, "sec44.json")))
    d44 = np.load(os.path.join(R, "sec44.npz"))
    s45 = json.load(open(os.path.join(R, "sec45.json")))
    return s44, d44, s45


def fig3_signal(d44):
    """T1: initial vs optimized input, time and frequency domain (paper Fig. 3)."""
    u0, u1 = signals.multisine(d44["phi0"]), signals.multisine(d44["phi"])
    t = np.arange(cfg.N) * cfg.TS
    f = np.arange(cfg.N) * cfg.FS / cfg.N
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.6))
    ax[0].plot(t, u0, color=C_INIT, lw=0.9, label="initial (random phase, seed 0)")
    ax[0].plot(t, u1, color=C_OPT, lw=0.9, ls="--", label="optimized")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Force u (N)")
    ax[0].set_xlim(0, t[-1])
    ax[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8, frameon=False)
    # same convention as the paper: 20*log10|FFT| over 0..fs (two-sided)
    m0 = 20 * np.log10(np.abs(np.fft.fft(u0)) + 1e-300)
    m1 = 20 * np.log10(np.abs(np.fft.fft(u1)) + 1e-300)
    ax[1].plot(f, m0, color=C_INIT, lw=1.2, label="initial")
    ax[1].plot(f, m1, color=C_OPT, lw=1.2, ls="--", label="optimized")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Magnitude (dB)")
    ax[1].set_xlim(0, cfg.FS)
    ax[1].legend(loc="center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig3_signal.png"), dpi=200)
    plt.close(fig)


def fig_history(d44):
    """Seed 0: best cost so far vs. number of cost+gradient evaluations, for
    L-BFGS-B and the Adam cross-check, with the fmincon-default budget (H1)."""
    ev = np.minimum.accumulate(d44["eval_costs"])
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    ax.loglog(np.arange(1, len(ev) + 1), ev, color=C_OPT, lw=1.6, label="L-BFGS-B (main)")
    ap = os.path.join(R, "adam_history.npy")
    if os.path.exists(ap):
        ha = np.minimum.accumulate(np.load(ap))
        ax.loglog(np.arange(1, len(ha) + 1), ha, color="#2a9d8f", lw=1.4, ls="-.", label="Adam, lr 0.05 (cross-check)")
    ax.axhline(cfg.PAPER["C_init"], color=C_INIT, ls=":", lw=1, label="paper: initial 287.15")
    ax.axhline(cfg.PAPER["C_opt"], color="k", ls="--", lw=1, label="paper: optimized 3.88")
    ax.axvline(cfg.FMINCON_GRAD_BUDGET, color="0.45", ls=(0, (1, 1.5)), lw=1.2)
    ax.text(cfg.FMINCON_GRAD_BUDGET * 1.08, 60, "fmincon default\nbudget: 16 gradients", fontsize=7, color="0.3")
    ax.set_xlabel("cost + gradient evaluations (log scale)")
    ax.set_ylabel(r"best cost $C_\theta$ (log scale)")
    ax.legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.62))
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig_cost_history.png"), dpi=200)
    plt.close(fig)


def _box(ax, i, j):
    lo, hi = cfg.DOMAIN_LO, cfg.DOMAIN_HI
    ax.add_patch(Rectangle((lo[i], lo[j]), hi[i] - lo[i], hi[j] - lo[j], fill=False, color="k", lw=1))


def fig4_side_views(d44):
    """T3: side views of the input-state samples (paper Fig. 4)."""
    sets = [(d44["Z0"], C_INIT, "initial"), (d44["Z1"], C_OPT, "optimized"),
            (np.load(os.path.join(R, "Z_schroeder_standard.npy")), C_SCH, "Schroeder")]
    planes = [(1, 2, r"$x_1$ (m)", r"$x_2$ (m/s)"), (0, 1, "u (N)", r"$x_1$ (m)"), (0, 2, "u (N)", r"$x_2$ (m/s)")]
    fig, ax = plt.subplots(3, 3, figsize=(6.8, 6.0))
    for r, (Z, col, name) in enumerate(sets):
        for c, (i, j, xl, yl) in enumerate(planes):
            a = ax[r, c]
            a.plot(Z[:, i], Z[:, j], "+", color=col, ms=2.5, mew=0.6)
            _box(a, i, j)
            a.set_xlabel(xl, labelpad=1)
            a.set_ylabel(yl, labelpad=1)
            a.set_xlim(cfg.DOMAIN_LO[i] * 1.25, cfg.DOMAIN_HI[i] * 1.25)
            a.set_ylim(cfg.DOMAIN_LO[j] * 1.3, cfg.DOMAIN_HI[j] * 1.3)
        ax[r, 0].set_title(name, loc="left", fontsize=9, color=col, fontweight="bold")
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    fig.savefig(os.path.join(R, "fig4_side_views.png"), dpi=200)
    plt.close(fig)


def fig5_3d(d44):
    """T4: optimized samples in 3-D with the 20^3 grid (paper Fig. 5)."""
    import cost
    Z = d44["Z1"]
    G = cost.GRID
    fig = plt.figure(figsize=(5.2, 4.2))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(G[:, 0], G[:, 1], G[:, 2], s=1.2, color="#d4a017", alpha=0.6, label="grid points (20$^3$)")
    ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=4, marker="x", color="k", lw=0.5, label="optimized samples")
    ax.set_xlabel("u (N)")
    ax.set_ylabel(r"$x_1$ (m)")
    ax.set_zlabel(r"$x_2$ (m/s)")
    ax.view_init(elev=18, azim=-60)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig5_3d.png"), dpi=200)
    plt.close(fig)


def fig6_slices(d44):
    """T5: x1-x2 cross-sections in 8 bins along u (paper Fig. 6). Returns counts per bin."""
    edges = np.linspace(-457.1429, 457.1429, 9)       # bin width 114.29 N
    Z0, Z1 = d44["Z0"], d44["Z1"]
    fig, ax = plt.subplots(4, 2, figsize=(6.2, 8.0), sharex=True, sharey=True)
    counts = []
    for b in range(8):
        a = ax.flat[b]
        m0 = (Z0[:, 0] >= edges[b]) & (Z0[:, 0] < edges[b + 1])
        m1 = (Z1[:, 0] >= edges[b]) & (Z1[:, 0] < edges[b + 1])
        counts.append((edges[b], edges[b + 1], int(m0.sum()), int(m1.sum())))
        a.plot(Z0[m0, 1], Z0[m0, 2], "*", color=C_INIT, ms=3, mew=0.4, label="initial")
        a.plot(Z1[m1, 1], Z1[m1, 2], "s", mfc="none", color=C_OPT, ms=3, mew=0.6, label="optimized")
        _box(a, 1, 2)
        a.set_title(f"{edges[b]:.1f} ≤ u < {edges[b+1]:.1f} N  (n={m0.sum()}/{m1.sum()})", fontsize=8)
        a.set_xlim(-2.6, 2.6)
        a.set_ylim(-30, 30)
    for a in ax[-1]:
        a.set_xlabel(r"$x_1$ (m)")
    for a in ax[:, 0]:
        a.set_ylabel(r"$x_2$ (m/s)")
    ax.flat[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig6_slices.png"), dpi=200)
    plt.close(fig)
    return counts


def fig7_schroeder():
    """T6: Schroeder multisine in time and frequency (paper Fig. 7)."""
    u = signals.multisine(signals.schroeder_phases("standard"))
    t = np.arange(cfg.N) * cfg.TS
    f = np.arange(cfg.N) * cfg.FS / cfg.N
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 3.8))
    ax[0].plot(t, u, color=C_SCH, lw=0.8)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Force u (N)")
    ax[0].set_xlim(0, t[-1])
    ax[1].plot(f, 20 * np.log10(np.abs(np.fft.fft(u)) + 1e-300), color=C_SCH, lw=1.2)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Magnitude (dB)")
    ax[1].set_xlim(0, cfg.FS)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig7_schroeder.png"), dpi=200)
    plt.close(fig)


def fig8_boxplot():
    """T7: Monte Carlo box plots of initial and final costs (paper Fig. 8)."""
    folder = os.path.join(R, "mc")
    if not os.path.isdir(folder) or not os.listdir(folder):
        return None
    runs = [json.load(open(os.path.join(folder, f))) for f in sorted(os.listdir(folder))]
    ci = np.array([r["C_init"] for r in runs])
    co = np.array([r["C_opt"] for r in runs])
    cb = np.array([r["C_budget16"] for r in runs])
    # cost of the same runs after 20 iterations (stop chosen from seed 0 in run_early.py, before the MC runs)
    c20 = np.array([r["history"][min(20, len(r["history"]) - 1)] for r in runs])
    # paper Fig. 8 values read from the axes (approximate): whiskers, quartiles, median, mean
    paper_final = {"min": 2.95, "q1": 3.28, "med": 3.65, "q3": 4.2, "max": 5.5, "mean": 3.8}

    def box(a, pos, data, col):
        a.boxplot(data, positions=[pos], widths=0.55, patch_artist=True,
                  boxprops=dict(facecolor=col + "33", edgecolor=col), medianprops=dict(color=col, lw=1.5),
                  whiskerprops=dict(color=col), capprops=dict(color=col), flierprops=dict(markeredgecolor=col))
        a.plot(pos, data.mean(), "o", mfc="#f2c200", mec="k", ms=5.5, zorder=5)

    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.0), gridspec_kw={"width_ratios": [1, 2.6]})
    box(ax[0], 1, ci, C_INIT)
    ax[0].axhline(cfg.PAPER["C_init"], color="k", ls="--", lw=0.9)
    ax[0].set_xticks([1], ["initial"])
    ax[0].set_ylabel("Cost")
    ax[0].set_title("initial (random phases)", fontsize=8.5)
    pf = paper_final
    ax[1].add_patch(Rectangle((0.45, pf["q1"]), 0.6, pf["q3"] - pf["q1"], color="0.5", alpha=0.25))
    ax[1].plot([0.75, 0.75], [pf["min"], pf["max"]], color="0.4", lw=1)
    ax[1].plot([0.45, 1.05], [pf["med"]] * 2, color="0.3", lw=1.5)
    ax[1].plot(0.75, pf["mean"], "o", mfc="#f2c200", mec="k", ms=5.5)
    box(ax[1], 2, cb, "#2a9d8f")
    box(ax[1], 3, c20, "#6a4c93")
    box(ax[1], 4, co, C_OPT)
    ax[1].set_xticks([0.75, 2, 3, 4], ["paper Fig. 8\n(read, approx.)", "ours: 16\ngradients", "ours: 20\niterations",
                                       f"ours: full\n(≤{cfg.OPT_MAXITER} it.)"], fontsize=7)
    ax[1].set_ylabel("Cost")
    ax[1].set_title("after optimization", fontsize=8.5)
    for a in ax:
        a.grid(alpha=0.3, axis="y")
    ax[0].plot([], [], "o", mfc="#f2c200", mec="k", label="mean")
    ax[0].plot([], [], "k--", lw=0.9, label="paper seed")
    ax[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig8_boxplot.png"), dpi=200)
    plt.close(fig)

    def st(x):
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        return {"mean": x.mean(), "median": med, "q1": q1, "q3": q3, "min": x.min(), "max": x.max()}
    stats = {"n": len(runs), "C_init": st(ci), "C_opt": st(co), "C_budget16": st(cb), "C_it20": st(c20),
             "ratio_mean": float(np.mean(ci / co)), "ratio_budget_mean": float(np.mean(ci / cb)),
             "nit_mean": float(np.mean([r["nit"] for r in runs])),
             "time_mean_s": float(np.mean([r["time_s"] for r in runs])),
             "n_budget_in_paper_range": int(np.sum((cb >= 2.9) & (cb <= 5.5)))}
    # iterations each run needs to reach the paper's final cost 3.88
    k_reach = np.array([int(np.argmax(np.array(r["history"]) <= cfg.PAPER["C_opt"])) for r in runs])
    stats["iters_to_paper"] = {"median": float(np.median(k_reach)), "min": int(k_reach.min()),
                               "max": int(k_reach.max()), "all": k_reach.tolist()}
    json.dump(stats, open(os.path.join(R, "sec46_stats.json"), "w"), indent=2, default=float)
    return stats


def comparison_table(s44, s45):
    """Published vs reproduced, absolute and relative difference -> CSV."""
    P = cfg.PAPER
    rows = [
        ("C_init (seed 0)", P["C_init"], s44["C_init"]),
        ("C_opt", P["C_opt"], s44["C_opt"]),
        ("C_opt within 16-gradient budget (H1)", P["C_opt"], s44["C_budget16"]),
        ("Ratio C_init/C_opt (vs implied 74.0)", P["ratio_implied"], s44["ratio"]),
        ("Ratio C_init/C_opt (vs text 69)", P["ratio_text"], s44["ratio"]),
        ("Crest factor Schroeder (l=1..184)", P["CF_schroeder"], s45["standard"]["CF"]),
        ("Crest factor Schroeder (l=21..204)", P["CF_schroeder"], s45["line"]["CF"]),
        ("Crest factor optimized", P["CF_opt"], s44["CF_opt"]),
    ]
    with open(os.path.join(R, "comparison_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "published", "reproduced", "abs_diff", "rel_diff_percent"])
        for name, p, r in rows:
            w.writerow([name, f"{p:.4g}", f"{r:.4g}", f"{r - p:.4g}", f"{100 * (r - p) / p:.1f}"])
    return rows


def export_numbers(s44, s45, mc):
    """Write results/numbers.tex: every number used in the report as a LaTeX
    macro, so the report is filled directly from the results (no copy errors)."""
    val = json.load(open(os.path.join(R, "validation.json")))
    diag = json.load(open(os.path.join(R, "diagnostics.json")))
    m = {
        "Cinit": f"{s44['C_init']:.2f}", "Copt": f"{s44['C_opt']:.3f}", "Ratio": f"{s44['ratio']:.1f}",
        "Nit": f"{s44['nit']}", "Nfev": f"{s44['nfev']}", "Tmin": f"{s44['time_s'] / 60:.1f}",
        "CFinit": f"{s44['CF_init']:.2f}", "CFopt": f"{s44['CF_opt']:.2f}",
        "CFschA": f"{s45['standard']['CF']:.3f}", "CFschB": f"{s45['line']['CF']:.3f}",
        "Csch": f"{s45['standard']['C']:.1f}", "CschB": f"{s45['line']['C']:.1f}",
        "KapInit": f"{100 * s44['kappa_init']:.1f}", "KapOpt": f"{100 * s44['kappa_opt']:.1f}",
        "KapSch": f"{100 * s45['standard']['kappa']:.1f}",
        "InInit": f"{100 * s44['inside_init']:.1f}", "InOpt": f"{100 * s44['inside_opt']:.1f}",
        "InSch": f"{100 * s45['standard']['inside']:.1f}",
        "dA": f"{s44['max_abs_dA']:.1e}", "SSrelOpt": f"{s44['ss_rel_at_P_opt']:.1e}",
        "SSpOpt": f"{s44['ss_period_opt']}", "SSpInit": f"{s44['ss_period_init']}",
        "RelCinit": f"{100 * (s44['C_init'] / cfg.PAPER['C_init'] - 1):.1f}",
        "RelCopt": f"{100 * (s44['C_opt'] / cfg.PAPER['C_opt'] - 1):.1f}",
        "RelCFsch": f"{100 * (s45['standard']['CF'] / cfg.PAPER['CF_schroeder'] - 1):.1f}",
        "RelCFopt": f"{100 * (s44['CF_opt'] / cfg.PAPER['CF_opt'] - 1):.1f}",
        "GridInit": f"{val['V2_grid30']['C_init']:.1f}", "GridOpt": f"{val['V2_grid30']['C_opt']:.3f}",
        "GridRatio": f"{val['V2_grid30']['ratio']:.1f}",
        "XzeroInit": f"{val['V3_x0']['init']:.1e}", "XzeroOpt": f"{val['V3_x0']['opt']:.1e}",
        "PtenOpt": f"{val['V4_P12']['opt']['max_state_diff']:.1e}",
        "DiagPass": f"{sum(d['pass'] for d in diag)}", "DiagN": f"{len(diag)}",
    }
    for e in val["V1_eps"]:
        tag = {1e-4: "A", 1e-3: "B", 3e-3: "C", 1e-2: "D"}[e["eps"]]
        m[f"EpsInit{tag}"] = f"{e['C_init']:.1f}"
        m[f"EpsOpt{tag}"] = f"{e['C_opt']:.3f}"
        m[f"EpsRatio{tag}"] = f"{e['ratio']:.1f}"
    h = np.load(os.path.join(R, "sec44.npz"))["history"]
    m["CoptAtCap"] = f"{h[min(150, len(h) - 1)]:.3f}"
    with open(os.path.join(R, "fig6_bin_counts.csv")) as fh:
        rows = list(csv.reader(fh))[1:]
    m["BinLoInit"], m["BinLoOpt"] = rows[0][2], rows[0][3]
    m["BinHiInit"], m["BinHiOpt"] = rows[-1][2], rows[-1][3]
    m["RelCbudget"] = f"{100 * (s44['C_budget16'] / cfg.PAPER['C_opt'] - 1):.1f}"
    m["Cbudget"] = f"{s44['C_budget16']:.2f}"
    m["FirstBelow"] = f"{s44['first_iter_below_paper']}"
    m["P"] = f"{cfg.P_PERIODS}"
    m["SSrelInit"] = f"{s44['ss_rel_at_P_init']:.1e}"
    old = os.path.join(R, "old_P5", "sec44.json")
    if os.path.exists(old):
        o = json.load(open(old))
        m["OldCopt"] = f"{o['C_opt']:.3f}"
        m["OldSSrel"] = f"{o['ss_rel_at_P_opt']:.1e}"
        m["OldSSp"] = f"{o['ss_period_opt']}"
    ea = os.path.join(R, "early.json")
    if os.path.exists(ea):
        for tag, e in zip(("Ea", "Eb", "Ec"), json.load(open(ea))):
            m[tag + "C"] = f"{e['C']:.2f}"
            m[tag + "R"] = f"{e['ratio']:.1f}"
            m[tag + "CF"] = f"{e['CF']:.2f}"
            m[tag + "It"] = str(e["iterations"])
    ad = os.path.join(R, "adam.json")
    if os.path.exists(ad):
        a = json.load(open(ad))
        m["AdamFinal"] = f"{a['C_final']:.3f}"
        m["AdamSixteen"] = f"{a['C_at_16']:.1f}"
        m["AdamSteps"] = f"{a['steps']}"
        m["AdamCF"] = f"{a['CF']:.2f}"
    if mc:
        for k, p, fmt in (("C_init", "MCi", ".1f"), ("C_opt", "MCo", ".2f"), ("C_budget16", "MCb", ".2f"), ("C_it20", "MCt", ".2f")):
            for s, name in (("mean", "Mean"), ("median", "Median"), ("q1", "Qone"), ("q3", "Qthree"),
                            ("min", "Min"), ("max", "Max")):
                m[p + name] = format(mc[k][s], fmt)
        m["MCn"] = str(mc["n"])
        m["MCratio"] = f"{mc['ratio_mean']:.1f}"
        m["MCratioB"] = f"{mc['ratio_budget_mean']:.1f}"
        m["MCnit"] = f"{mc['nit_mean']:.0f}"
        m["MCtime"] = f"{mc['time_mean_s'] / 60:.1f}"
        m["MCinRange"] = str(mc["n_budget_in_paper_range"])
        m["MCkMed"] = f"{mc['iters_to_paper']['median']:g}"
        m["MCkMin"] = str(mc["iters_to_paper"]["min"])
        m["MCkMax"] = str(mc["iters_to_paper"]["max"])
    with open(os.path.join(R, "numbers.tex"), "w") as fh:
        fh.write("% generated by make_figures.py -- do not edit by hand\n")
        for k, v in m.items():
            fh.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")


def main():
    s44, d44, s45 = load()
    fig3_signal(d44)
    fig_history(d44)
    fig4_side_views(d44)
    fig5_3d(d44)
    counts = fig6_slices(d44)
    with open(os.path.join(R, "fig6_bin_counts.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["u_lo", "u_hi", "n_initial", "n_optimized"])
        w.writerows(counts)
    fig7_schroeder()
    mc = fig8_boxplot()
    export_numbers(s44, s45, mc)
    for row in comparison_table(s44, s45):
        print(f"  {row[0]:42s} paper {row[1]:8.3f}   ours {row[2]:8.3f}")


if __name__ == "__main__":
    main()
