# Paper 4A – Full Local Test Suite & Reproducibility Package

This package is a **local, runnable** test and reproducibility suite for the numerical and
operational claims used in the Paper 4A “observer quality / DLN-stage decoding” framework.

It is designed to be:

- **Deterministic** (fixed seeds for baseline checks).
- **Self-contained** (no internet; minimal dependencies).
- **Fast** (tests finish in seconds on a typical laptop).
- **Transparent** (tests validate explicit formulas and code paths used in figures / quoted numbers).

## What this test suite checks

### Mathematical / operational invariants
1. **Trace-distance contraction** under a depolarizing CPTP map.
2. **Chernoff data-processing inequality (DPI)** in a numerically checkable form:
   for random qubit states ρ, σ and depolarizing channel Λₚ,
   `QCBcoef(Λₚ(ρ), Λₚ(σ)) ≥ QCBcoef(ρ, σ)` (within numerical tolerance).
3. **Decision-theoretic continuity**: for a fixed measurement event,
   `|Prρ(E) − Prσ(E)| ≤ Dtr(ρ, σ)` (numerically verified).
4. **Pure-record “factor-of-two” exponent gap** *for the specific q_L class used here*:
   collective exponent `−log(c²)` versus **local-Helstrom-per-fragment** product exponent `−log(c)`.
   (This is the exact gap that your manuscript’s DLN mapping intends to capture.)

### Model-specific, reproducible checks (central-spin pure dephasing)
Using a fixed RNG seed and parameters consistent with the manuscript’s “Numerical verification” section:
- `N=200`, `g_k ~ Uniform(0.8,1.2)`, `t=0.5`, and “best-m fragments” selection.
The suite checks:
1. The per-fragment exponents `xi_N/m` and `xi_L/m` match the manuscript’s quoted values
   (to tight tolerance).
2. The **pointer-accessibility gap fraction** over a time grid on `[0,2]` reproduces the
   reported ~19.5% (m=8), ~17.8% (m=12), ~10.0% (m=25), ~3.0% (m=50) results.

### Result-A / Result-B ensemble-vs-typical exponents
The tests compute both:
- `xi_eff` (typical / quenched: average of `−log(2 P_e,t)`), and
- `xi_ens` (ensemble / annealed: `−(1/m) log(2 E[P_e,t])`),
and verify the Jensen relation `xi_ens ≤ xi_eff`, plus reproduce the numerical values
used for the unmonitored-collapse example.

## Installation

### Option A (recommended): create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate      # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: use your existing Python environment
Just run:
```bash
pip install -r requirements.txt
```

## Running tests

### Run all tests (pytest)
```bash
pytest -q
```

### Run a subset (examples)
```bash
pytest -q tests/test_qcb_properties.py
pytest -q tests/test_central_spin_model.py
```

### Run via the convenience wrapper (works even if you prefer not to type pytest)
```bash
python tests/run_all.py
```

## Reproducing figures / numbers

### Central-spin redundancy vs time (writes a PNG + CSV)
```bash
python scripts/reproduce_central_spin_figure.py --outdir figures
```

### Reproduce Results A–C numbers printed to console
```bash
python scripts/reproduce_results_ABC.py
```

## If tests fail: diagnostic directions

Most failures fall into one of these categories:

### 1) Floating-point / BLAS differences (rare)
Symptoms:
- Small violations of inequalities at the ~1e−8 to ~1e−6 scale.

Actions:
- Re-run once to confirm determinism:
  ```bash
  pytest -q -k qcb --maxfail=1
  ```
- If it is a strict inequality check, increase tolerance slightly:
  in `tests/conftest.py`, the default tolerance is centralized as `TOL`.

### 2) Numpy version differences
This suite assumes modern NumPy (>=1.23).
If you are on a very old version, upgrade:
```bash
pip install --upgrade numpy
```

### 3) Wrong Python is running
Confirm the interpreter:
```bash
python -c "import sys; print(sys.executable)"
python -c "import numpy as np; print(np.__version__)"
```

### 4) You changed model semantics (expected)
If you modify:
- the definition of q_L (product class),
- the time grid `[0,2]`,
- the “best-m fragments” selection rule,
- or the seed/parameter choices,

then the central-spin baseline checks will fail **by design**.

Actions:
- Update the baseline values in `tests/test_central_spin_model.py` (they are explicit and documented).
- Or switch those checks to “property tests” (monotonicity/ratio only) instead of exact numbers.

## File layout

```
paper4a_full_testsuite/
  paper/      LaTeX source (for reference)
  src/        Reusable code (QCB, distances, central-spin model)
  scripts/    Reproducibility scripts that generate figures / numbers
  tests/      pytest suite + a wrapper runner
  figures/    Output directory for generated plots/CSV
  baselines/  Baseline metadata (seed/params) used by tests
```

## Notes on scientific scope

These tests validate **internal mathematical and numerical consistency** of the
manuscript’s operational objects (exponents, bounds, accessibility, monitoring models).
They do not “prove the theorems” (the proof burden is in the manuscript), but they
ensure the *numerical instantiations* and *inequality claims* used as evidence are correct,
reproducible, and stable under small implementation perturbations.
