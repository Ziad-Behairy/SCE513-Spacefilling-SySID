PY ?= python3
.PHONY: test quick all repro m4 extension analysis report clean parallel

test:            ## unit tests (< 1 min)
	cd tests && $(PY) -m pytest -q

quick:           ## smoke test of the M3 pipeline (~5 min) -> results_quick/
	cd code && $(PY) main.py --quick

repro:           ## M3 core: T1-T7, diagnostics, validation, figures (~3.5 h on 2 cores)
	cd code && $(PY) main.py

m4:              ## M4 additions to the reproduction (MC rerun with phases, H1 fmincon-like, epsilon re-opt)
	cd code && $(PY) run_mc_phases.py 1 20
	cd code && $(PY) run_h1_fmincon.py 0 1 2 3 4 5 6 7 8 9
	cd code && $(PY) run_eps_reopt.py

extension:       ## bonus extension (needs results/mc_phases), ~1 h on 2 cores
	cd code && $(PY) -m ext.run_extension 1 20
	cd code && $(PY) -m ext.run_fixed_lambda

analysis:        ## M4 figures, tables, results/numbers_m4.tex and 3-D drawings
	cd code && $(PY) m4_analysis.py
	-cd code && $(PY) make_illustrations.py

report: analysis ## LaTeX -> report/report.pdf
	cd report && pdflatex -interaction=nonstopmode report.tex >/dev/null; bibtex report >/dev/null; \
	pdflatex -interaction=nonstopmode report.tex >/dev/null; pdflatex -interaction=nonstopmode report.tex | grep -E "Output written|Error" || true

all: repro m4 extension report

clean:
	rm -rf results_quick report/*.aux report/*.log report/*.bbl report/*.blg report/*.out

parallel:        ## all M4 + extension jobs on many cores (restart-safe), e.g. make parallel N=8
	cd code && $(PY) run_parallel.py $(N)
