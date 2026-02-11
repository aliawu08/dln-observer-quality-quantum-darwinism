# Convenience targets for building the papers, reproducing figures, and running checks.

.PHONY: preprint pra figs test clean install check

install:
	python -m pip install --upgrade pip
	pip install -r requirements.lock.txt

check: test figs

preprint:
	cd paper/preprint && pdflatex -interaction=nonstopmode paper4a_observer_quality_major_revision_preprint.tex
	cd paper/preprint && /usr/bin/bibtex.original paper4a_observer_quality_major_revision_preprint
	cd paper/preprint && pdflatex -interaction=nonstopmode paper4a_observer_quality_major_revision_preprint.tex
	cd paper/preprint && pdflatex -interaction=nonstopmode paper4a_observer_quality_major_revision_preprint.tex

pra:
	cd paper/pra && pdflatex -interaction=nonstopmode paper4a_observer_quality_major_revision_PRA_revtex.tex
	cd paper/pra && /usr/bin/bibtex.original paper4a_observer_quality_major_revision_PRA_revtex
	cd paper/pra && pdflatex -interaction=nonstopmode paper4a_observer_quality_major_revision_PRA_revtex.tex
	cd paper/pra && pdflatex -interaction=nonstopmode paper4a_observer_quality_major_revision_PRA_revtex.tex

figs:
	python scripts/central_spin_example.py
	# copy figures into paper folders for LaTeX build
	cp figures/*.png paper/preprint/ || true
	cp figures/*.png paper/pra/ || true

test:
	pytest -q

clean:
	find paper -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" -o -name "*.bbl" -o -name "*.blg" -o -name "*Notes.bib" | xargs -r rm -f
