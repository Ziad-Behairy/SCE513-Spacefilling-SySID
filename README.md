# SCE 513 Course Project: Reproducibility Package (M4)

**Student:** Ziad Aboelkhir (g202622300) · **Course:** SCE 513 System Identification, Term 261, KFUPM ·
**Instructor:** Prof. Mujahed M. Al-Dhaifallah

**Paper reproduced:** M. Kiss, R. Tóth, M. Schoukens, "Space-filling input design for nonlinear state-space
identification", *IFAC-PapersOnLine* 58(15), 562–567, 2024 (SYSID 2024).
[doi:10.1016/j.ifacol.2024.08.589](https://doi.org/10.1016/j.ifacol.2024.08.589) · [arXiv:2405.18207](https://arxiv.org/abs/2405.18207)

**Approved scope:** Secs. 4.4–4.6, targets T1–T7 (M2 protocol). **Optional extension (bonus):** does better
coverage give better identified models, also when the design model is itself identified?

| Deliverable | File |
|---|---|
| Final report (IFAC-style, two columns) | `report/report.pdf` (source `report/report.tex`, `report/refs.bib`) |
| Every figure, table and number of the report | `results/` (generated; `numbers.tex`, `numbers_m4.tex`) |
| Source code | `code/` (walkthrough: `code/CODE_WALKTHROUGH.md`) |
| Data | none external: see `data_instructions.md` |
| Unit tests | `tests/` (`make test`) |
| Presentation (M5) | `presentation/` |

## 1. Environment

| Item | Version used |
|---|---|
| Python | 3.13 (3.10+ works) |
| NumPy / SciPy / Matplotlib | 2.5.3 / 1.18.1 / 3.11.2 |
| Numba | 0.68.1 (extension only) |
| LaTeX | pdflatex + bibtex (TeX Live), packages: mathptmx, natbib, booktabs, titlesec |
| Hardware | CPU only, float64. Timings below: 2 cores |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test          # 10 unit tests, < 1 min
```
No MATLAB, toolbox, GPU or PyTorch. All gradients are derived by hand (adjoint method) and checked
against finite differences.

## 2. Run order (regenerates everything)

**One main script:** `python code/run_all.py` runs the numbered workflow below in order
(`--workers 8` for more cores, `--analysis` to redo only steps 3–6 from the stored results in ~1 min,
`--quick` for a 5-minute smoke test; `--analysis` reuses the stored peak-limited runs instead of rerunning them). The table lists the same steps for running them one by one.

| Step | Command | Produces | Time (2 cores) |
|---|---|---|---|
| 1 | `make repro` (= `cd code && python main.py`) | M3 core: diagnostics D1–D8, T1–T7, validation V1–V6, figures, `numbers.tex` | ~3.5 h |
| 2 | `make m4` | MC rerun saving phases (`results/mc_phases/`), H1 fmincon-like test (`results/h1_fmincon/`), ε re-optimisation (`eps_reopt.json`) | ~3 h |
| 3 | `make extension` | extension experiments, `results/ext/seed_XX.json` | ~1 h |
| 4 | `make report` | M4 figures, `numbers_m4.tex`, `ext_table.csv`, 3-D drawings (`make_illustrations.py`), `report/report.pdf` | 1–2 min |

`make all` runs 1–4. `make quick` is a 5-minute smoke test of step 1 (writes `results_quick/`, not used
by the report). Long steps are restart-safe (finished seeds are skipped) and can be split over cores, e.g.
`cd code && python run_mc_phases.py 1 10` and `python run_mc_phases.py 11 20` in two terminals.

## 3. Seeds and determinism
All randomness uses `numpy.random.default_rng` with fixed seeds (table in `data_instructions.md`).
On the same machine every run is bit-for-bit identical (the M4 rerun of seeds 1-4 and 11-15 reproduced
the M3 optima exactly). Seeds 5-10 and 16-20 were rerun on a Windows laptop: the costs agree to about 1e-6
after 20 iterations, but round-off differences grow along the non-convex optimisation path, so the
converged costs differ by a few percent (reported in the report, Sec. 3). Use the stored
`results/mc_phases/*.json` to reproduce the exact figures.

## 4. Code map

| File | Content | Paper |
|---|---|---|
| `code/config.py` | every setting, seed and published value | Secs. 4.1–4.3 |
| `code/system.py` | Euler simulation, steady state, adjoint gradient | Eqs. 9–10 |
| `code/signals.py` | multisine, Schroeder phases, crest factor, du/dφ | Eqs. 12–13 |
| `code/cost.py` | grid, kernel densities, cost (5)–(8) and its gradient | Eqs. 5–8 |
| `code/optimize.py` | L-BFGS-B wrapper, Adam cross-check | Sec. 4.4 |
| `code/run_sec44.py … run_sec46.py` | T1–T7 | Secs. 4.4–4.6 |
| `code/diagnostics.py`, `validation.py` | checks D1–D8, V1–V6 | n/a |
| `code/run_mc_phases.py` | M4: MC rerun that stores phases after 16, 20 and all iterations | Sec. 4.6 |
| `code/run_h1_fmincon.py` | M4: fmincon-like BFGS with forward differences, 3000 evaluations (H1) | n/a |
| `code/run_eps_reopt.py` | M4: ε sensitivity with re-optimisation | n/a |
| `code/ext/ident.py` | extension: RBF ridge least squares with cross-validated λ | n/a |
| `code/ext/surrogate.py` | extension: simulation + adjoint gradient of the identified model | n/a |
| `code/ext/run_extension.py` | extension experiments (8 designs × 3 noise levels × 20 seeds) | n/a |
| `code/ext/run_fixed_lambda.py` | extension robustness: same fixed ridge λ for every design | n/a |
| `code/ext/run_noise_remedies.py` | E4b: bias vs variance of the noise error; IV vs averaging of measured periods | n/a |
| `code/ext/run_peak_limited.py` | E4a: space-filling design with an actuator peak limit (crest factor ≤ 2.5, ≤ 1.9) | n/a |
| `code/run_parallel.py` | runs all M4/extension seeds on many cores (`make parallel N=8`) | n/a |
| `code/m4_analysis.py` | M4 figures, tables, statistics, `numbers_m4.tex` | n/a |
| `code/make_illustrations.py` | 3-D drawing of the test system and 3-D view of the samples (T4) | Sec. 4.1, Fig. 5 |
| `code/run_all.py` | **one main script**: runs the whole numbered workflow | n/a |

## 5. Results → report map

| Report item | File(s) | Script |
|---|---|---|
| Tables 1–3 (result map, settings, checks) | `config.py`, `diagnostics.json`, `validation.json` | `diagnostics.py`, `validation.py` |
| Table 4 (published vs. reproduced) | `numbers.tex`, `numbers_m4.tex`, `comparison_table.csv` | `make_figures.py`, `m4_analysis.py` |
| Table 5 (ε re-optimisation) | `eps_reopt.json` | `run_eps_reopt.py` |
| Table 6 (extension) | `ext_table.tex`, `ext_table.csv`, `ext/` | `m4_analysis.py` |
| Fig. 1 (3-D system drawing, optimisation loop) | `fig_system_3d.png` (loop drawn in LaTeX) | `make_illustrations.py` |
| Fig. 2 (side views, T3) | `fig4_side_views.png` | `make_figures.py` |
| Fig. 3 (signal, T1) | `fig3_signal.png` | `make_figures.py` |
| Fig. 4 (slices T5, Schroeder T6, cost history) | `fig6_slices.png`, `fig7_schroeder.png`, `fig_cost_history.png` | `make_figures.py` |
| Fig. 5 (3-D samples, T4) | `fig_states_3d.png` | `make_illustrations.py` |
| Fig. 6 (T7 + H1) | `fig_h1.png`, `h1_fmincon/`, `mc_phases/` | `m4_analysis.py` |
| Fig. 7 (surrogate loop) | drawn in LaTeX | n/a |
| Fig. 8 (extension) | `fig_ext_accuracy.png`, `fig_ext_mechanism.png` | `m4_analysis.py` |
| Extension robustness (fixed λ) | `ext_fixed_lambda.json` | `ext/run_fixed_lambda.py` |
| E4 (peak limit, noise remedies) and its figure | `ext_peak_limited/`, `ext_noise_remedies.json`, `fig_ext_practical.png` | `ext/run_peak_limited.py`, `ext/run_noise_remedies.py`, `m4_analysis.py` |
| In the package only | `fig5_3d.png` (M3 version of T4), `fig8_boxplot.png` (M3 version of T7), `fig_diagnostics.png` | `make_figures.py` |

## 6. Attribution and AI use
No code of the authors exists or is used; the method and parameters are from the paper (cited). Paper
figures are not redistributed. Libraries are installed from PyPI under their own licences (see `LICENSE`).
Claude (Anthropic) assisted with code drafting, debugging and report drafting; see the AI-use statement in
the report. All results were run and checked by the student.
