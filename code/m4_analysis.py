"""
m4_analysis.py  --  every M4 figure, table and number (after run_mc_phases.py,
run_h1_fmincon.py, run_eps_reopt.py and ext/run_extension.py).

Writes results/numbers_m4.tex (LaTeX macros used by the report),
results/m4_summary.json, results/ext_table.csv and the figures
fig_h1.png, fig_ext_accuracy.png, fig_ext_mechanism.png.
"""
import csv, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import config as cfg
import cost, signals, system

R = cfg.RESULTS_DIR
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
MAC = {}
DIG = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))


WORD = {"16": "sixteen", "20": "twenty", "30": "thirty", "40": "forty", "160": "onesixty", "240": "twoforty"}


def mac(name, value, fmt="{:.2f}"):
    name = re.sub(r"\d+", lambda mm: WORD.get(mm.group(), "".join(DIG[c] for c in mm.group())), name)
    name = re.sub(r"[^A-Za-z]", "", name)
    MAC["\\m" + name] = value if isinstance(value, str) else fmt.format(value)


def q(x):
    x = np.asarray(x, float)
    q1, m, q3 = np.percentile(x, [25, 50, 75])
    return {"median": m, "q1": q1, "q3": q3, "mean": float(np.mean(x)), "min": float(x.min()), "max": float(x.max())}


def loadfolder(name):
    f = os.path.join(R, name)
    if not os.path.isdir(f):
        return []
    return [json.load(open(os.path.join(f, x))) for x in sorted(os.listdir(f)) if x.endswith(".json")]


# --------------------------------------------------------------------------- reproduction (M4 closure)
def repro():
    runs = loadfolder("mc_phases")
    out = {"n_mc": len(runs)}
    mac("NMC", str(len(runs)))
    # determinism: the rerun must give the same optimum as the M3 run (same seed, same settings)
    old = {r["seed"]: r for r in loadfolder("mc")}
    # Seeds 1-4 and 11-15 were rerun on the same Linux machine as M3; seeds 5-10 and 16-20 on a
    # Windows laptop (different BLAS/compiler).  Compare final costs and the cost after 20 iterations.
    pairs = [(np.array(r["history"]), np.array(old[r["seed"]]["history"])) for r in runs if r["seed"] in old]
    fin = np.array([abs(a[-1] - b[-1]) / b[-1] for a, b in pairs])
    it20 = np.array([abs(a[20] - b[20]) / b[20] for a, b in pairs])
    out["rerun"] = {"n_identical": int(np.sum(fin == 0)), "final_max": float(fin.max()),
                    "final_median_nonzero": float(np.median(fin[fin > 0])) if np.any(fin > 0) else 0.0,
                    "it20_max": float(it20.max())}
    mac("RerunIdentical", str(out["rerun"]["n_identical"]))
    mac("RerunFinalMax", 100 * out["rerun"]["final_max"], "{:.1f}")
    mac("RerunFinalMed", 100 * out["rerun"]["final_median_nonzero"], "{:.1f}")
    mac("RerunItTwentyMax", out["rerun"]["it20_max"], "{:.0e}")
    rows = {"16": [], "20": [], "full": [], "init": []}
    for r in runs:
        for key, ph in [("init", "phi0"), ("16", "phi_16"), ("20", "phi_20"), ("full", "phi")]:
            phi = np.array(r[ph])
            Z, u, _ = cost.samples_from_phases(phi)
            kap, ins = cost.coverage(Z)
            rows[key].append({"C": float(cost.cost(Z)), "CF": float(signals.crest_factor(u)), "kappa": kap, "inside": ins,
                              "x1max": float(np.max(np.abs(Z[:, 1])))})
    for key, L in rows.items():
        for m in ("C", "CF", "kappa"):
            s = q([x[m] for x in L])
            out[f"{m}_{key}"] = s
            f = "{:.1f}" if m == "kappa" else ("{:.3g}" if m == "C" else "{:.2f}")
            scale = 100 if m == "kappa" else 1
            mac(f"{m}{key}Med", s["median"] * scale, f)
            mac(f"{m}{key}Min", s["min"] * scale, f)
            mac(f"{m}{key}Max", s["max"] * scale, f)
    # numbers that the report used to type by hand, now read from the result files
    D = {x["name"].split()[0]: x["info"] for x in json.load(open(os.path.join(R, "diagnostics.json")))}
    mac("GradErr", float(re.search(r"= ([0-9.e+-]+);", D["D6"]).group(1)), "{:.1e}")
    rad = re.findall(r"([0-9]\.[0-9]+)", D["D7"])
    mac("RadZero", float(rad[0]), "{:.4f}"); mac("RadTwo", float(rad[-1]), "{:.4f}")
    s44 = json.load(open(os.path.join(R, "sec44.json")))
    mac("EvalTime", s44["time_s"] / s44["nfev"], "{:.2f}")
    mac("LineDB", 20 * np.log10(cfg.AMP * cfg.N / 2), "{:.1f}")
    # Schroeder through the same diagnostics
    uS = signals.multisine(signals.schroeder_phases())
    XS = system.steady_state_response(uS)[0]
    ZS = np.column_stack([uS, XS])
    rels, p_ok = system.steady_state_check(uS)
    out["schroeder"] = {"C": float(cost.cost(ZS)), "CF": float(signals.crest_factor(uS)),
                        "kappa": float(cost.coverage(ZS)[0]), "ss_period": p_ok}
    mac("SchSS", str(p_ok))
    ci = np.array([x["C"] for x in rows["init"]])
    cf = np.array([x["C"] for x in rows["full"]])
    mac("NSchBelowRand", str(int(np.sum(out["schroeder"]["C"] < ci))))
    mac("NOptBelowSch", str(int(np.sum(cf < out["schroeder"]["C"]))))
    # iterations to reach the paper's 3.88
    k = [int(np.argmax(np.array(r["history"]) <= cfg.PAPER["C_opt"])) for r in runs]
    out["iters_to_paper"] = q(k)
    mac("ItPaperMed", out["iters_to_paper"]["median"], "{:.1f}")

    # H1: fmincon-like optimiser (BFGS, forward differences, 3000 evaluations)
    h1 = loadfolder("h1_fmincon")
    out["n_h1"] = len(h1)
    mac("NHone", str(len(h1)))
    if h1:
        cf = np.array([r["C_final"] for r in h1])
        out["h1"] = q(cf)
        mac("HoneMed", out["h1"]["median"]); mac("HoneMin", out["h1"]["min"]); mac("HoneMax", out["h1"]["max"])
        mac("HoneQOne", out["h1"]["q1"]); mac("HoneQThree", out["h1"]["q3"])
        mac("HoneGrad", np.mean([r["n_grad_equiv"] for r in h1]), "{:.1f}")
        mac("HoneTime", np.mean([r["time_s"] for r in h1]) / 60, "{:.1f}")
        seeds = ", ".join(str(r["seed"]) for r in h1)
        mac("HoneSeeds", seeds)
        mac("HoneAbove", str(int(np.sum(cf > 5.5))))
        paper = {"min": 2.95, "q1": 3.28, "med": 3.65, "q3": 4.2, "max": 5.5, "mean": 3.8}
        c16 = [r["history"][16] for r in runs]
        c20 = [r["history"][20] for r in runs]
        cfull = [r["C_opt"] for r in runs]
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        ax.add_patch(plt.Rectangle((0.7, paper["q1"]), 0.6, paper["q3"] - paper["q1"], color="0.5", alpha=0.3))
        ax.plot([1, 1], [paper["min"], paper["max"]], color="0.4", lw=1)
        ax.plot([0.7, 1.3], [paper["med"]] * 2, color="0.2", lw=1.5)
        data = [cf, c16, c20, cfull]
        cols = ["#2a9d8f", "#577590", "#6a4c93", "#e8590c"]
        for i, (d, c) in enumerate(zip(data, cols)):
            ax.boxplot(d, positions=[i + 2], widths=0.55, patch_artist=True,
                       boxprops=dict(facecolor=c + "33", edgecolor=c), medianprops=dict(color=c, lw=1.5),
                       whiskerprops=dict(color=c), capprops=dict(color=c), flierprops=dict(markeredgecolor=c, ms=3))
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 3, 4, 5], ["paper\nFig. 8", "BFGS-FD\n3000 ev.", "L-BFGS\n16 it.", "L-BFGS\n20 it.",
                                         "L-BFGS\nfull"], fontsize=6.5)
        ax.axhline(cfg.PAPER["C_opt"], color="k", ls=":", lw=0.8)
        ax.set_ylabel(r"final cost $C_\theta$")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(R, "fig_h1.png"), dpi=250)
        plt.close(fig)
    # epsilon with re-optimisation
    p = os.path.join(R, "eps_reopt.json")
    if os.path.exists(p):
        E = json.load(open(p))
        out["eps_reopt"] = E
        mac("EpsKapMin", 100 * min(r["kappa_opt"] for r in E), "{:.1f}")
        mac("EpsKapMax", 100 * max(r["kappa_opt"] for r in E), "{:.1f}")
        for row, nm in zip(E, ["A", "B", "C", "D"]):
            mac(f"Eps{nm}Init", row["C_init"], "{:.1f}"); mac(f"Eps{nm}Opt", row["C_opt"], "{:.3g}")
            mac(f"Eps{nm}Ratio", row["ratio"], "{:.0f}"); mac(f"Eps{nm}CF", row["CF_opt"])
            mac(f"Eps{nm}Kap", row["kappa_opt"] * 100, "{:.1f}"); mac(f"Eps{nm}Cref", row["C_opt_at_eps1e-3"], "{:.3g}")
    return out


# --------------------------------------------------------------------------- extension
DES = ["R", "S", "O16", "O20", "Ofull", "Osur", "Oiter", "Osur100"]
LAB = {"R": "random", "S": "Schroeder", "O16": "opt. 16 it.", "O20": "opt. 20 it.", "Ofull": "opt. full",
       "Osur": "surrogate", "Oiter": "iterative", "Osur100": "surr. 100 it."}
COL = {"R": "#1f77b4", "S": "#7b3fa0", "O16": "#f4a261", "O20": "#e76f51", "Ofull": "#e8590c",
       "Osur": "#2a9d8f", "Oiter": "#264653", "Osur100": "#8ab17d"}
LEV = [("inf", "noise-free"), ("40", "40 dB"), ("30", "30 dB")]


PVALS = []


def extension():
    runs = loadfolder("ext")
    out = {"n_ext": len(runs)}
    mac("NExt", str(len(runs)))
    if not runs:
        return out
    table = []
    for lk, ll in LEV:
        for d in DES:
            vals = {m: [r["levels"][lk][d][m] for r in runs if d in r["levels"][lk]]
                    for m in ("e_uni", "e_sim160", "e_sim240", "C_true", "kappa", "CF", "div160", "div240")}
            if not vals["e_uni"]:
                continue
            row = {"level": ll, "design": LAB[d]}
            for m in ("e_uni", "e_sim160", "e_sim240", "C_true", "kappa", "CF"):
                s = q(np.array(vals[m], float))
                row[m] = s["median"]; row[m + "_q1"] = s["q1"]; row[m + "_q3"] = s["q3"]
                mac(f"X{d}_{lk}_{m}", s["median"] * (100 if m.startswith("e_") or m == "kappa" else 1),
                    "{:.1f}" if (m.startswith("e_") or m == "kappa") else "{:.3g}")
                if m.startswith("e_"):
                    mac(f"X{d}_{lk}_{m}QOne", s["q1"] * 100, "{:.1f}"); mac(f"X{d}_{lk}_{m}QThree", s["q3"] * 100, "{:.1f}")
            row["div160"] = int(np.sum(vals["div160"])); row["div240"] = int(np.sum(vals["div240"]))
            mac(f"X{d}_{lk}_divTot", str(row["div160"] + row["div240"]))
            # paired one-sided Wilcoxon test vs random phase (H0: no improvement)
            if d != "R":
                for m in ("e_uni", "e_sim160", "e_sim240"):
                    a = np.array([r["levels"][lk][d][m] for r in runs if d in r["levels"][lk]], float)
                    b = np.array([r["levels"][lk]["R"][m] for r in runs if d in r["levels"][lk]], float)
                    # a diverged simulation (inf) is ranked as the worst possible result, not dropped
                    a = np.where(np.isfinite(a), a, 1e6); b = np.where(np.isfinite(b), b, 1e6)
                    ok = np.ones(len(a), bool)
                    pv = stats.wilcoxon(a, b, alternative="less").pvalue
                    pv2 = stats.wilcoxon(a, b, alternative="greater").pvalue
                    PVALS.append((d, lk, m, pv))
                    row[f"p_{m}"] = pv; row[f"pworse_{m}"] = pv2
                    mac(f"P{d}_{lk}_{m}", "$<0.001$" if pv < 1e-3 else f"$={pv:.3f}$")
                    mac(f"Pw{d}_{lk}_{m}", "$<0.001$" if pv2 < 1e-3 else f"$={pv2:.3f}$")
                    mac(f"Win{d}_{lk}_{m}", str(int(np.sum(a[ok] < b[ok]))))
            table.append(row)
    # improvement factors (median of paired ratios R / design)
    for lk, _ in LEV:
        for d in DES[1:]:
            rr = [r["levels"][lk]["R"]["e_uni"] / r["levels"][lk][d]["e_uni"] for r in runs if d in r["levels"][lk]]
            if rr:
                mac(f"F{d}_{lk}", float(np.median(rr)), "{:.1f}")
    # rank correlation: space-filling cost on the true system vs one-step error (all designs, all seeds)
    for lk, _ in LEV:
        c = [r["levels"][lk][d]["C_true"] for r in runs for d in DES if d in r["levels"][lk]]
        e = [r["levels"][lk][d]["e_uni"] for r in runs for d in DES if d in r["levels"][lk]]
        rho, pv = stats.spearmanr(c, e)
        out[f"spearman_{lk}"] = (float(rho), float(pv))
        mac(f"Rho_{lk}", rho, "{:.2f}")
    for d in ("R", "O20", "Ofull", "Osur"):
        c = [r["levels"]["40"][d]["C_true"] for r in runs]; e = [r["levels"]["40"][d]["e_uni"] for r in runs]
        rho, pv = stats.spearmanr(c, e)
        mac(f"RhoIn_{d}", rho, "{:.2f}"); mac(f"RhoInP_{d}", "$<0.001$" if pv < 1e-3 else f"$={pv:.2f}$")
    opt = [(r["levels"]["40"][d]["C_true"], r["levels"]["40"][d]["e_uni"]) for r in runs for d in ("O16", "O20", "Ofull")]
    rho, pv = stats.spearmanr(*zip(*opt))
    mac("RhoOpt", rho, "{:.2f}"); mac("RhoOptP", "$<0.001$" if pv < 1e-3 else f"$={pv:.2f}$")
    a = [r["levels"]["40"]["R"]["e_sim160"] < r["levels"]["40"]["Ofull"]["e_sim160"] for r in runs]
    mac("WinRvsOfullsim", str(int(np.sum(a))))
    mac("ChatOsur", float(np.median([r["levels"]["40"]["Osur"]["C_hat"] for r in runs])), "{:.3g}")
    pe = [p for (d, lk, m, p) in PVALS if m == "e_uni"]
    mac("NTests", str(len(PVALS))); mac("MaxPeuni", "10^{%d}" % int(np.ceil(np.log10(max(pe)))))
    mac("BonfOK", "yes" if max(pe) <= 0.01 / len(PVALS) else "no")
    fl = os.path.join(R, "ext_fixed_lambda.json")
    if os.path.exists(fl):
        FL = json.load(open(fl))
        for lam, nm in (("1e-06", "A"), ("0.0001", "B")):
            for lk in ("inf", "40", "30"):
                E = FL[lam][lk]
                mac(f"FixF{nm}_{lk}", float(np.median(np.array(E["R"]) / np.array(E["Ofull"]))), "{:.1f}")
                mac(f"FixS{nm}_{lk}", 100 * float(np.median(E["S"])), "{:.1f}")
    with open(os.path.join(R, "ext_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for t in table for k in t}, key=lambda k: (k not in ("level", "design"), k)))
        w.writeheader(); w.writerows(table)
    out["table"] = table
    # LaTeX table for the report (generated, never typed by hand)
    def cell(lk, d, m, scale=100, f="{:.1f}"):
        v = np.array([r["levels"][lk][d][m] for r in runs if d in r["levels"][lk]], float) * scale
        fin = v[np.isfinite(v)]
        return f.format(np.median(v)) if len(fin) == len(v) else f.format(np.median(v)) + "$^\\dagger$"
    def sig(lk, d, m):
        if d == "R":
            return ""
        a = np.array([r["levels"][lk][d][m] for r in runs], float); b = np.array([r["levels"][lk]["R"][m] for r in runs], float)
        a = np.where(np.isfinite(a), a, 1e6); b = np.where(np.isfinite(b), b, 1e6)
        p_better = stats.wilcoxon(a, b, alternative="less").pvalue
        p_worse = stats.wilcoxon(a, b, alternative="greater").pvalue
        return "$^{*}$" if p_better < 0.01 else ("$^{\\circ}$" if p_worse < 0.01 else "")
    lines = ["\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule",
             " & \\multicolumn{3}{c}{$e_{\\mathrm{uni}}$ [\\%]} & \\multicolumn{2}{c}{$e_{\\mathrm{sim}}$ [\\%], 40 dB} & & \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
             "Design & clean & 40\\,dB & 30\\,dB & 160\\,N & 240\\,N & $C_\\theta$ & $\\kappa$[\\%]\\\\\\midrule"]
    for d in DES:
        row = [LAB[d]]
        for lk in ("inf", "40", "30"):
            row.append(cell(lk, d, "e_uni") + sig(lk, d, "e_uni") if d in runs[0]["levels"][lk] else "--")
        row.append(cell("40", d, "e_sim160") + sig("40", d, "e_sim160"))
        row.append(cell("40", d, "e_sim240") + sig("40", d, "e_sim240"))
        row.append(cell("40", d, "C_true", 1, "{:.3g}"))
        row.append(cell("40", d, "kappa", 100, "{:.0f}"))
        lines.append(" & ".join(row) + "\\\\")
    lines.append("\\bottomrule\\end{tabular}")
    open(os.path.join(R, "ext_table.tex"), "w").write("% generated by m4_analysis.py\n" + "\n".join(lines) + "\n")

    # ---- Fig. A: accuracy over D_z for every design and noise level + cost vs error
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [2.2, 1]})
    show = ["R", "S", "O20", "Ofull", "Osur", "Oiter"]
    w = 0.8 / len(show)
    for j, (lk, ll) in enumerate(LEV):
        for i, d in enumerate(show):
            v = [r["levels"][lk][d]["e_uni"] * 100 for r in runs]
            ax[0].boxplot(v, positions=[j + (i - (len(show) - 1) / 2) * w], widths=w * 0.85, patch_artist=True,
                          boxprops=dict(facecolor=COL[d] + "55", edgecolor=COL[d]), medianprops=dict(color=COL[d], lw=1.4),
                          whiskerprops=dict(color=COL[d]), capprops=dict(color=COL[d]), flierprops=dict(markeredgecolor=COL[d], ms=2))
    for d in show:
        ax[0].plot([], [], "s", color=COL[d], label=LAB[d])
    ax[0].set_xticks(range(len(LEV)), [l for _, l in LEV])
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"one-step NRMSE of $\hat g$ on $\mathcal{D}_z$ [%]")
    ax[0].legend(ncol=6, fontsize=6.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0),
                 handletextpad=0.3, columnspacing=0.9)
    ax[0].grid(alpha=0.3, axis="y")
    lk = "40"
    for d in DES:
        c = [r["levels"][lk][d]["C_true"] for r in runs if d in r["levels"][lk]]
        e = [r["levels"][lk][d]["e_uni"] * 100 for r in runs if d in r["levels"][lk]]
        ax[1].loglog(c, e, "o", ms=2.5, color=COL[d], alpha=0.8)
    ax[1].set_xlabel(r"cost $C_\theta$ on the true system")
    for d in DES:
        ax[1].plot([], [], "o", ms=3, color=COL[d], label=LAB[d])
    ax[1].legend(fontsize=5.5, frameon=False, loc="upper left", handletextpad=0.1, labelspacing=0.2)
    ax[1].set_ylabel("NRMSE [%] (40 dB)")
    ax[1].set_title(fr"Spearman $\rho$ = {out['spearman_40'][0]:.2f}", fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig_ext_accuracy.png"), dpi=250)
    plt.close(fig)

    # ---- Fig. B: mechanism (error vs local data density) + free-run simulation error
    from ext.run_extension import LOGD_EDGES
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    centers = ["<-3", "[-3,-2)", "[-2,-1)", "[-1,0)", "[0,1)", "[1,2)", "≥2"]
    for d in ["R", "S", "Ofull", "Osur", "Oiter"]:
        sse = np.sum([r["levels"][lk][d]["dens_sse"] for r in runs], axis=0)
        n = np.sum([r["levels"][lk][d]["dens_n"] for r in runs], axis=0)
        G_SD = 1.0
        rm = np.where(n > 50, np.sqrt(sse / np.maximum(n, 1)), np.nan)
        out[f"dens_{d}"] = {"rmse": rm.tolist(), "n": n.tolist()}
        ax[0].semilogy(range(len(rm)), rm, "o-", ms=3, color=COL[d], label=LAB[d])
    ax[0].set_xticks(range(len(centers)), centers, fontsize=6.5)
    ax[0].set_xlabel(r"$\log_{10}$ of local training-data density $d(z)$")
    ax[0].set_ylabel(r"RMSE of $\hat g$ [m/s$^2$]")
    ax[0].legend(fontsize=6.5, frameon=False)
    ax[0].grid(alpha=0.3, which="both")
    show2 = ["R", "S", "O20", "Ofull", "Osur", "Oiter"]
    for j, m in enumerate(["e_sim160", "e_sim240"]):
        for i, d in enumerate(show2):
            v = np.array([r["levels"][lk][d][m] for r in runs], float) * 100
            v = np.where(np.isfinite(v), v, 1e3)
            ax[1].boxplot(v, positions=[j + (i - 2.5) * 0.13], widths=0.11, patch_artist=True,
                          boxprops=dict(facecolor=COL[d] + "55", edgecolor=COL[d]), medianprops=dict(color=COL[d], lw=1.4),
                          whiskerprops=dict(color=COL[d]), capprops=dict(color=COL[d]), flierprops=dict(markeredgecolor=COL[d], ms=2))
    for d in show2:
        ax[1].plot([], [], "s", color=COL[d], label=LAB[d])
    ax[1].legend(ncol=3, fontsize=6, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0),
                 handletextpad=0.3, columnspacing=0.8)
    ax[1].set_xticks([0, 1], ["test std 160 N\n(design level)", "test std 240 N\n(stress test)"])
    ax[1].set_yscale("log")
    ax[1].set_ylabel("free-run NRMSE [%] (40 dB)")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig_ext_mechanism.png"), dpi=250)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- extension E4: practical limits
def practical():
    out = {}
    nr = os.path.join(R, "ext_noise_remedies.json")
    if os.path.exists(nr):
        N = json.load(open(nr))
        seeds = sorted(N["per_seed"], key=int)
        for d in ("R", "S", "O20", "Ofull"):
            for m in ("LS1", "IV1", "LSavg"):
                v = [N["per_seed"][k][d][m] for k in seeds]
                mac(f"NR_{d}_{m}", 100 * float(np.median(v)), "{:.1f}")
            a = np.array([N["per_seed"][k][d]["LSavg"] for k in seeds]); b = np.array([N["per_seed"][k][d]["LS1"] for k in seeds])
            mac(f"NRwin_{d}", str(int(np.sum(a < b))))
            mac(f"NRp_{d}", "$<0.001$" if stats.wilcoxon(a, b, alternative="less").pvalue < 1e-3
                else "$=%.3f$" % stats.wilcoxon(a, b, alternative="less").pvalue)
            ai = np.array([N["per_seed"][k][d]["IV1"] for k in seeds])
            mac(f"NRivwin_{d}", str(int(np.sum(ai < b))))
        for d, v in N["split"].items():
            for m in ("clean", "single_mean", "mean_theta"):
                mac(f"Split_{d}_{m}", 100 * v[m], "{:.1f}")
        mac("NRM", str(N["M"]))
        out["noise"] = N
    pk = loadfolder("ext_peak_limited")
    if pk:
        ext = {r["seed"]: r for r in loadfolder("ext")}
        mac("NPeak", str(len(pk)))
        for key, nm in (("inf", "Inf"), ("2.5", "TwoFive"), ("1.9", "OneNine")):
            for m in ("CF", "e_uni", "e_sim160", "e_sim240", "C_true"):
                v = np.array([r[key][m] for r in pk], float)
                sc, f = (100, "{:.1f}") if m.startswith("e_") else (1, "{:.2f}" if m == "CF" else "{:.3g}")
                mac(f"Pk_{nm}_{m}", sc * float(np.median(v)), f)
            mac(f"Pk_{nm}_CFmax", float(max(r[key]["CF"] for r in pk)), "{:.2f}")
            mac(f"Pk_{nm}_div", str(int(sum(r[key]["div"] for r in pk))))
            rr = [ext[r["seed"]]["levels"]["40"]["R"]["e_uni"] / r[key]["e_uni"] for r in pk if r["seed"] in ext]
            mac(f"Pk_{nm}_F", float(np.median(rr)), "{:.1f}")
        # figure: practical limits
        fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
        keys = [("inf", "no limit"), ("2.5", "CF $\\leq$ 2.5"), ("1.9", "CF $\\leq$ 1.9")]
        rv = [ext[r["seed"]]["levels"]["40"]["R"]["e_uni"] * 100 for r in pk]
        sv = [ext[r["seed"]]["levels"]["40"]["S"]["e_uni"] * 100 for r in pk]
        data = [rv, sv] + [[r[k]["e_uni"] * 100 for r in pk] for k, _ in keys]
        cols = [COL["R"], COL["S"], "#e8590c", "#f08c4a", "#f4b183"]
        labs = ["random", "Schroeder\nCF 1.88"] + [f"opt. 100 it.\n{l}" for _, l in keys]
        for i, (d, c) in enumerate(zip(data, cols)):
            ax[0].boxplot(d, positions=[i], widths=0.55, patch_artist=True, boxprops=dict(facecolor=c + "55", edgecolor=c),
                          medianprops=dict(color=c, lw=1.4), whiskerprops=dict(color=c), capprops=dict(color=c),
                          flierprops=dict(markeredgecolor=c, ms=2))
        ax[0].set_xticks(range(5), labs, fontsize=6)
        ax[0].set_yscale("log"); ax[0].set_ylabel("$e_{\\mathrm{uni}}$ [%] (40 dB)"); ax[0].grid(alpha=0.3, axis="y")
        ax[0].set_title("(a) actuator peak limit", fontsize=8)
        from matplotlib.ticker import FuncFormatter, LogLocator
        for a_ in ax:
            a_.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            a_.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            a_.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        if "noise" in out:
            N = out["noise"]; seeds = sorted(N["per_seed"], key=int)
            ds = ["R", "S", "O20", "Ofull"]; ms = [("LS1", "LS, 1 period"), ("IV1", "IV, 1 period"), ("LSavg", f"LS, mean of {N['M']} periods")]
            w = 0.26
            for j, (m, lab) in enumerate(ms):
                med = [np.median([N["per_seed"][k][d][m] for k in seeds]) * 100 for d in ds]
                ax[1].bar(np.arange(4) + (j - 1) * w, med, w, label=lab, color=["#8d99a6", "#2a9d8f", "#264653"][j])
            ax[1].set_xticks(range(4), [LAB[d] for d in ds], fontsize=7)
            ax[1].set_yscale("log"); ax[1].set_ylabel("median $e_{\\mathrm{uni}}$ [%] (30 dB)")
            ax[1].legend(fontsize=6, frameon=False); ax[1].grid(alpha=0.3, axis="y")
            ax[1].set_title("(b) noise: IV vs. periodic averaging", fontsize=8)
        for a_ in ax:
            a_.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            a_.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            a_.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        fig.tight_layout()
        fig.savefig(os.path.join(R, "fig_ext_practical.png"), dpi=250)
        plt.close(fig)
    return out


if __name__ == "__main__":
    S = {"repro": repro(), "ext": extension(), "practical": practical()}
    with open(os.path.join(R, "numbers_m4.tex"), "w") as fh:
        fh.write("% generated by m4_analysis.py -- do not edit by hand\n")
        for k, v in MAC.items():
            fh.write(f"\\newcommand{{{k}}}{{{v}}}\n")
    json.dump(S, open(os.path.join(R, "m4_summary.json"), "w"), indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))
    print(f"{len(MAC)} macros written; MC runs {S['repro']['n_mc']}, H1 runs {S['repro'].get('n_h1')}, ext seeds {S['ext']['n_ext']}")
