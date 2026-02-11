import json
import math
import os

import numpy as np

from paper4a_qd.central_spin import (
    couplings_uniform,
    xi_collective_per_fragment,
    xi_local_helstrom_per_fragment,
    best_m_sum,
    pointer_gap_fraction,
)

def load_meta():
    meta_path = os.path.join(os.path.dirname(__file__), "..", "baselines", "metadata.json")
    meta_path = os.path.abspath(meta_path)
    return json.load(open(meta_path, "r"))["central_spin"]


def test_central_spin_exponents_match_baseline(tol):
    meta = load_meta()
    g = couplings_uniform(meta["N"], meta["g_low"], meta["g_high"], seed=meta["seed"])
    t = float(meta["t_for_exponent"])
    m = int(meta["m_for_exponent"])

    xiN = best_m_sum(xi_collective_per_fragment(g, t), m) / m
    xiL = best_m_sum(xi_local_helstrom_per_fragment(g, t), m) / m

    # Baseline values for seed=2, t=0.5, best-25 selection.
    # These are deterministic given the coupling seed and selection rule.
    assert abs(xiN - 1.9150369414824993) <= 50 * tol
    assert abs(xiL - 0.9575184707412496) <= 50 * tol
    assert abs((xiN / xiL) - 2.0) <= 50 * tol


def test_pointer_accessibility_gap_fractions(tol):
    meta = load_meta()
    g = couplings_uniform(meta["N"], meta["g_low"], meta["g_high"], seed=meta["seed"])
    tmin, tmax, num = meta["time_grid"]
    delta = float(meta["delta"])

    # Expected fractions for seed=2, N=200, grid=2001 points on [0,2], δ=1e-3.
    expected = {
        8: 0.19490254872563717,
        12: 0.17791104447776113,
        25: 0.10044977511244378,
        50: 0.029985007496251874,
    }
    for m, exp_frac in expected.items():
        frac = pointer_gap_fraction(g=g, tmin=tmin, tmax=tmax, num=num, m=m, delta=delta)
        assert abs(frac - exp_frac) <= 2000 * tol
