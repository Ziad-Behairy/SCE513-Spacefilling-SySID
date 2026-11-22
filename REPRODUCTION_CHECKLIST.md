# Reproduction checklist (course requirements, Sec. 15)

| ✓ | Requirement | Evidence |
|---|---|---|
| ☑ | Paper satisfies Route A (SYSID 2024, IFAC-PapersOnLine 58(15)) and was approved | Approved 22 Sep 2026 (M1/M2); report §1 |
| ☑ | Exact reproduced targets identified by figure / table / metric | T1 Fig. 3, T2 Sec. 4.4 costs, T3 Fig. 4, T4 Fig. 5, T5 Fig. 6, T6 Sec. 4.5 / Fig. 7, T7 Sec. 4.6 / Fig. 8; report Table 1 |
| ☑ | Reported and reproduced results compared quantitatively | Report Table 1, Fig. 2; `results/comparison_table.csv`, `results/numbers*.tex` |
| ☑ | Estimation and validation data handled correctly | Reproduction: no model fitted → 20 seeds, independent 30³ grid, other x0, P = 12 (V1–V6). Extension: λ by CV on training data only; tests on 20 000 uniform D_z points and 5 independent random-phase realisations at two amplitudes (160 N, 240 N), never used for fitting |
| ☑ | Discrepancies and deviations reported, not hidden | Report §3 (deviations), §4.1 (H1), §6; unsuccessful attempts (P = 5, first gradient check) kept in `results/old_P5/` and M3 report |
| ☑ | Optional extension asks a new technical question and uses the reproduction as baseline | Report §5: designs come from the reproduced optimiser (`results/mc_phases/`); random-phase and Schroeder baselines of identical spectrum |
| ☑ | Report, code, data instructions, results, README, presentation included | `report/report.pdf`, `code/`, `data_instructions.md`, `results/`, `README.md`, `presentation/` |
| ☑ | Main results can be regenerated from the package | `make all` (or steps in README §2); `make test`, `make quick` for fast checks; CI workflow `.github/workflows/ci.yml` |
| ☑ | External code, data, figures and ideas cited and licensed | No author code exists; paper cited; libraries cited in report; `LICENSE` attribution; paper figures not redistributed |
| ☑ | Individual authorship and AI use disclosed | Report, "Authorship and AI-use statement" |

## Diagnostic checks (all pass)
D1 input dimensions/units · D2 free decay · D3 steady state · D4 kernel unit test · D5 cost bound ·
D6 adjoint vs finite differences · D7 Euler stability · D8 grid: `code/diagnostics.py`, `results/diagnostics.json`,
and the unit tests in `tests/` (also surrogate-model gradient check).
