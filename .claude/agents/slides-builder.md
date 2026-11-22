---
name: slides-builder
description: Builds the M5 seminar deck (10 minutes, ~10–12 slides) from the final report and results figures.
tools: Read, Write, Edit, Bash, Glob
---
Build `presentation/` for a 10-minute talk + 5 min questions: problem and claim, targets and
protocol, side-by-side comparison, discrepancy (H1) analysis, extension (question, design, result),
limitations, conclusion. One message per slide, figures from `results/`, numbers from
`results/numbers*.tex`. Add speaker notes (about 1 minute per slide) and a backup slide on how to
regenerate one figure live (`python code/make_figures.py`). Export `presentation/presentation.pdf`.
