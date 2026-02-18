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

PRA_FIGS := central_spin_redundancy_vs_time.pdf \
            central_spin_m_required_vs_time.pdf \
            central_spin_robustness_vs_p.pdf \
            inverted_sophistication_crossover.pdf

PREPRINT_FIGS := central_spin_redundancy_vs_time.pdf \
                 inverted_sophistication_crossover.pdf

figs:
	SOURCE_DATE_EPOCH=0 python scripts/central_spin_example.py
	SOURCE_DATE_EPOCH=0 python -m scripts.dynamical_redundancy
	# copy only the figures each paper variant actually references
	$(foreach f,$(PREPRINT_FIGS),cp figures/$(f) paper/preprint/;)
	$(foreach f,$(PRA_FIGS),cp figures/$(f) paper/pra/;)

test:
	pytest -q

clean:
	find paper -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" -o -name "*.bbl" -o -name "*.blg" -o -name "*Notes.bib" | xargs -r rm -f
