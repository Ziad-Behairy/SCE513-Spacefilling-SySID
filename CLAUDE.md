# SCE 513 Course Project — Claude Code guide

Individual course project of **Ziad Aboelkhir (g202622300)**, SCE 513 System Identification,
KFUPM Term 261, instructor Prof. Mujahed M. Al-Dhaifallah.

Reproduction of: M. Kiss, R. Tóth, M. Schoukens, "Space-filling input design for nonlinear
state-space identification", IFAC-PapersOnLine 58(15):562–567, 2024 (SYSID 2024),
doi:10.1016/j.ifacol.2024.08.589, arXiv:2405.18207. Approved scope: Secs. 4.4–4.6 (targets T1–T7).
Extension (bonus): "From coverage to accuracy" — see `report/report.tex` §5 and `code/ext/`.

## Milestones
- M1 selection, M2 protocol, M3 checkpoint: done.
- **M4** final report + package: `report/report.pdf` + this repository (tag `m4`).
- **M5** seminar (10 min + 5 min questions) + live-code demo: `presentation/` (next).

## First thing in a new Claude Code session
If this folder is not yet on GitHub, run:
```bash
bash scripts/setup_github.sh            # creates PRIVATE repo sce513-spacefilling-sysid and pushes, with tags
```
(needs `gh auth login` once). Keep the repo private: the project is individual graded work.

## Layout
```
code/            all source (M3 core + M4 additions); run from the repo root via `make`
  config.py      every setting, seed and published value of the reproduction
  system.py signals.py cost.py optimize.py   model, input, cost, optimiser (paper Eqs. 5-13)
  run_*.py       one script per experiment; main.py runs everything in order
  ext/           extension: ident.py (RBF ridge LS), surrogate.py (identified-model adjoint),
                 run_extension.py (experiments), ext_config.py (all extension settings)
  m4_analysis.py all M4 figures/tables + results/numbers_m4.tex
results/         every number and figure in the report (generated, never edited by hand)
report/          report.tex (IFAC-style two-column), refs.bib, report.pdf
tests/           pytest unit tests (fast)   ->  make test
presentation/    M5 slides (to do)
```

## Commands
```bash
make test        # unit tests, < 1 min
make quick       # smoke test of the whole pipeline (~5 min) -> results_quick/
make all         # full regeneration (~6 h on 2 cores, ~1.5 h on 8 cores; see README)
make report      # analysis figures + numbers_m4.tex + LaTeX -> report/report.pdf
```

## Rules for any agent working here
1. **Never type a result number by hand** into the report. Numbers come from `results/numbers.tex`
   and `results/numbers_m4.tex`, written by `make_figures.py` and `m4_analysis.py`.
2. **Never edit files in `results/` by hand.** Re-run the script that produces them.
3. Keep negative/discrepant results. Never tune settings after seeing test results; the
   extension's test data (uniform D_z points, test seeds 1001–1005) are never used for fitting
   or for choosing λ (λ is chosen by CV on training data only).
4. Every code change keeps `make test` green; add a test for new math (gradients: finite differences).
5. Ziad must be able to explain every line in the oral defense: when you add code, add a
   plain-language docstring and update `code/CODE_WALKTHROUGH.md`.
6. AI use must stay disclosed in the report (Section "Authorship and AI-use statement").
7. Style for written text: short, clear, student voice; no marketing language.

## Subagents (`.claude/agents/`)
- `verifier` — independent check: clean clone, `make test`, every number in report vs. result files,
  requirement checklist (Sec. 15 of the course requirements). Use before any submission.
- `report-editor` — edits `report/report.tex` within the 6–8 page IFAC limit.
- `slides-builder` — builds the M5 10-minute deck from the report.
- `defense-coach` — drills Ziad with likely oral-defense questions and checks his answers.
