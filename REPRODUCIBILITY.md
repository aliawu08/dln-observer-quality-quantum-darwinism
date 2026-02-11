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
| `dln-compression-model/` | Paper 1: DLN compression model (code + paper) |
| `scripts/` | Central-spin worked example script |
| `tests/` | Unit tests for bounds and limiting cases |
| `notebooks/` | Numerical cross-checks of analytic expressions |
| `notes/` | Internal drafting notes (retained for provenance) |

## Figure provenance

The worked example is a central-spin pure-dephasing model.

- Script: `scripts/central_spin_example.py`
- Outputs (generated):
  - `figures/central_spin_redundancy_vs_time.png`
  - `figures/central_spin_m_required_vs_time.png`

The `make figs` target also copies these into the paper folders for LaTeX compilation:

- `paper/preprint/central_spin_redundancy_vs_time.png`
- `paper/preprint/central_spin_m_required_vs_time.png`
- `paper/pra/central_spin_redundancy_vs_time.png`
- `paper/pra/central_spin_m_required_vs_time.png`

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

## DLN compression model (Paper 1)

The `dln-compression-model/` subdirectory contains the code and paper source for Paper 1. See its own [`README.md`](dln-compression-model/README.md) for build and run instructions.

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

The central-spin script uses a fixed RNG seed for any randomized parameters so that regenerated figures are deterministic.
