# SCE 513 Course Project — Reproducibility Package (M4)

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
| Data | none external — see `data_instructions.md` |
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

| Step | Command | Produces | Time (2 cores) |
|---|---|---|---|
| 1 | `make repro` (= `cd code && python main.py`) | M3 core: diagnostics D1–D8, T1–T7, validation V1–V6, figures, `numbers.tex` | ~3.5 h |
| 2 | `make m4` | MC rerun saving phases (`results/mc_phases/`), H1 fmincon-like test (`results/h1_fmincon/`), ε re-optimisation (`eps_reopt.json`) | ~3 h |
| 3 | `make extension` | extension experiments, `results/ext/seed_XX.json` | ~1 h |
| 4 | `make report` | M4 figures, `numbers_m4.tex`, `ext_table.csv`, `report/report.pdf` | 1 min |

`make all` runs 1–4. `make quick` is a 5-minute smoke test of step 1 (writes `results_quick/`, not used
by the report). Long steps are restart-safe (finished seeds are skipped) and can be split over cores, e.g.
`cd code && python run_mc_phases.py 1 10` and `python run_mc_phases.py 11 20` in two terminals.

## 3. Seeds and determinism
All randomness uses `numpy.random.default_rng` with fixed seeds (table in `data_instructions.md`).
The M4 rerun of the 20 Monte-Carlo optimisations reproduces the M3 optima exactly (relative difference
reported as `\mRerunDiff` in the report).

## 4. Code map

| File | Content | Paper |
|---|---|---|
| `code/config.py` | every setting, seed and published value | Secs. 4.1–4.3 |
| `code/system.py` | Euler simulation, steady state, adjoint gradient | Eqs. 9–10 |
| `code/signals.py` | multisine, Schroeder phases, crest factor, du/dφ | Eqs. 12–13 |
| `code/cost.py` | grid, kernel densities, cost (5)–(8) and its gradient | Eqs. 5–8 |
| `code/optimize.py` | L-BFGS-B wrapper, Adam cross-check | Sec. 4.4 |
| `code/run_sec44.py … run_sec46.py` | T1–T7 | Secs. 4.4–4.6 |
| `code/diagnostics.py`, `validation.py` | checks D1–D8, V1–V6 | — |
| `code/run_mc_phases.py` | M4: MC rerun that stores phases after 16, 20 and all iterations | Sec. 4.6 |
| `code/run_h1_fmincon.py` | M4: fmincon-like BFGS with forward differences, 3000 evaluations (H1) | — |
| `code/run_eps_reopt.py` | M4: ε sensitivity with re-optimisation | — |
| `code/ext/ident.py` | extension: RBF ridge least squares with cross-validated λ | — |
| `code/ext/surrogate.py` | extension: simulation + adjoint gradient of the identified model | — |
| `code/ext/run_extension.py` | extension experiments (8 designs × 3 noise levels × 20 seeds) | — |
| `code/m4_analysis.py` | M4 figures, tables, statistics, `numbers_m4.tex` | — |

## 5. Results → report map

| Report item | File(s) | Script |
|---|---|---|
| Table 1 (comparison) | `numbers.tex`, `numbers_m4.tex`, `comparison_table.csv` | `make_figures.py`, `m4_analysis.py` |
| Fig. 1 (side views, T3) | `fig4_side_views.png` | `make_figures.py` |
| Fig. 2 (T7 + H1) | `fig_h1.png`, `h1_fmincon/`, `mc_phases/` | `m4_analysis.py` |
| Table 2, Figs. 3–4 (extension) | `ext_table.csv`, `fig_ext_accuracy.png`, `fig_ext_mechanism.png`, `ext/` | `m4_analysis.py` |
| Other M3 figures (T1, T4–T6, diagnostics) | `fig3_signal.png`, `fig5_3d.png`, `fig6_slices.png`, `fig7_schroeder.png`, `fig_diagnostics.png` | `make_figures.py` |

## 6. Attribution and AI use
No code of the authors exists or is used; the method and parameters are from the paper (cited). Paper
figures are not redistributed. Libraries are installed from PyPI under their own licences (see `LICENSE`).
Claude (Anthropic) assisted with code drafting, debugging and report drafting; see the AI-use statement in
the report. All results were run and checked by the student.
