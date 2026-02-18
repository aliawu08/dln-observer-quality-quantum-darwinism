# Observer Quality as a Resource Variable in Quantum Darwinism

[![CI](https://github.com/aliawu08/observer-quality-quantum-darwinism/actions/workflows/ci.yml/badge.svg)](https://github.com/aliawu08/observer-quality-quantum-darwinism/actions/workflows/ci.yml)
[![LaTeX Build](https://github.com/aliawu08/observer-quality-quantum-darwinism/actions/workflows/latex.yml/badge.svg)](https://github.com/aliawu08/observer-quality-quantum-darwinism/actions/workflows/latex.yml)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Paper-CC_BY_4.0-lightgrey.svg)](LICENSE-CC-BY-4.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18610548.svg)](https://doi.org/10.5281/zenodo.18610548)

This repository accompanies the paper *"Observer Quality as a Resource Variable in Quantum Darwinism: &epsilon;-SBS Robustness, Chernoff-Optimal Decoding, and a Central-Spin Worked Example"* (Wu, 2026).

**Contact:** Alia Wu &mdash; wut08@nyu.edu
**ORCID:** [0009-0005-4424-102X](https://orcid.org/0009-0005-4424-102X)

---

## Key Results

| Result | Description |
|--------|-------------|
| **Observer-quality triple** | Q = (R, &Lambda;, &tau;) &mdash; record-access fraction, calibration channel, temporal horizon &mdash; as a resource variable for classical objectivity |
| **SBS as factor DAG** | The conditional-independence structure of spectrum broadcast structure states is a bipartite factor DAG whose latent node is the pointer variable, instantiating the DLN belief-dependency graph without modification |
| **Strict decoder hierarchy** | Dec_D &subne; Dec_L &subne; Dec_N &mdash; memoryless, product, and collective decoding classes with strict nesting |
| **Tight exponent gap** | Factor-of-two Chernoff exponent separation (&xi;_N / &xi;_L = 2) between repeated single-copy Helstrom readout and collective decoding for pure-state records (optimal adaptive individual strategies can match collective exponents; the gap is specific to this fixed readout) |
| **Bounded recovery** | Revision-graph formalism governing adaptive transitions between measurement strategies under changing coherence conditions |

---

## Repository Structure

```
.
├── paper/
│   ├── preprint/                        # Single-column preprint (PhilSci-Archive)
│   └── pra/                             # Physical Review A (RevTeX 4.2) submission
├── scripts/
│   └── central_spin_example.py          # Central-spin worked example
├── tests/                               # Unit tests (bounds, limiting cases)
├── notebooks/
│   └── central_spin_derivations.ipynb   # Numerical cross-checks
├── figures/                             # Generated figures (via `make figs`)
└── .github/workflows/                   # CI: tests, figures, LaTeX builds
```

---

## Quickstart

**Requirements:** Python &ge; 3.11

```bash
# Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt

# Run tests
python -m pytest -q

# Reproduce central-spin figures
make figs

# Build papers
make preprint   # single-column preprint PDF
make pra        # RevTeX 4.2 PRA submission PDF
```

`requirements.txt` is minimal and unpinned for convenience; `requirements.lock.txt` is the canonical frozen set for bit-for-bit reproducibility.

---

## Related Work

This paper builds on the DLN framework developed across:

> Wu, A. (2026). *Compression Efficiency and Structural Learning as a Computational Model of DLN Cognitive Stages*.
> bioRxiv: [10.64898/2026.02.01.703168](https://www.biorxiv.org/content/10.64898/2026.02.01.703168v1)
> Repository: [github.com/aliawu08/dln-compression-model](https://github.com/aliawu08/dln-compression-model)

> Wu, A. (2026). *Cognitive Architecture as Hidden Moderator: Reconciling Contradictory Emotion&ndash;Cognition Findings with the DLN Framework*.
> PsychArchives: [10.23668/psycharchives.21641](https://doi.org/10.23668/psycharchives.21641)
> Repository: [github.com/aliawu08/dln-emotion-cognition](https://github.com/aliawu08/dln-emotion-cognition)

---

## Docker (optional)

```bash
docker build -t observer-quality .
docker run --rm -it observer-quality bash -lc "make figs && make preprint && make pra"
```

---

## Citation

```bibtex
@misc{wu_observer_quality_2026,
  title  = {Observer Quality as a Resource Variable in Quantum Darwinism:
            $\varepsilon$-SBS Robustness, Chernoff-Optimal Decoding,
            and a Central-Spin Worked Example},
  author = {Wu, Alia},
  year   = {2026},
  doi    = {10.5281/zenodo.18610548},
  url    = {https://doi.org/10.5281/zenodo.18610548}
}
```

---

## License

This project is dual-licensed:

- **Code** (source code, scripts, tests): [MIT License](LICENSE)
- **Documentation, figures, and paper content**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-CC-BY-4.0)
