# Reproducibility

This repository is designed to make Paper 4A fully checkable: figures, bounds, and PDFs can be rebuilt from source.

## Quick start (Python)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
python -m pytest -q
make figs
```

## Repository layout

| Path | Contents |
|------|----------|
| `paper/preprint/` | Single-column preprint (PhilSci-Archive) |
| `paper/pra/` | Physical Review A (RevTeX 4.2) submission |
| `scripts/` | Central-spin worked example and dynamical redundancy scripts |
| `tests/` | Unit tests for bounds and limiting cases |
| `notebooks/` | Numerical cross-checks of analytic expressions |
| `figures/` | Generated figures (via `make figs`) |

## Figure provenance

### Central-spin worked example

- Script: `scripts/central_spin_example.py`
- Outputs (generated):
  - `figures/central_spin_redundancy_vs_time.png`
  - `figures/central_spin_m_required_vs_time.png`
  - `figures/central_spin_robustness_vs_p.png`

### Dynamical redundancy and inverted sophistication

- Script: `scripts/dynamical_redundancy.py`
- Outputs (generated):
  - `figures/dynamical_redundancy_xi_eff.png`
  - `figures/inverted_sophistication_crossover.png`
  - `figures/pointer_resolution_gap.png`
  - `figures/robustness_decoherence_models.png`

### Paper figure copies

The `make figs` target copies all generated figures into both paper folders for LaTeX compilation.
The CI workflow rebuilds figures and fails if the committed paper figures do not match regenerated outputs.

## Build the PDFs

### Preprint
```bash
make preprint
```

### PRA (RevTeX)
```bash
make pra
```

## Docker (optional)

```bash
docker build -t paper4a .
docker run --rm paper4a
```

To build PDFs inside the container:

```bash
docker run --rm -it paper4a bash -lc "make figs && make preprint && make pra"
```

## Determinism

Both scripts use fixed RNG seeds for any randomized parameters so that regenerated figures are deterministic.
