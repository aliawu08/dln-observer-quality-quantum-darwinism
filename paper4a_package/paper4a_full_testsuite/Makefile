.PHONY: test reproduce clean

test:
	pytest -q

reproduce:
	python scripts/reproduce_results_ABC.py
	python scripts/reproduce_central_spin_figure.py --outdir figures

clean:
	rm -f figures/*.png figures/*.csv
