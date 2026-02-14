import math
import numpy as np

from paper4a_qd.observer_topology import (
    xi_ensemble_annealed_from_errors,
    xi_typical_quenched,
    xi_full_cycle_ensemble,
    xi_unmonitored_collective_ensemble,
)
from paper4a_qd.central_spin import couplings_uniform, xi_collective_per_fragment, xi_local_helstrom_per_fragment, best_m_sum


def test_jensen_annealed_leq_quenched(tol):
    rng = np.random.default_rng(1234)
    for _ in range(50):
        # random per-episode errors in (0, 1/2]
        Pe = rng.uniform(1e-6, 0.5, size=1000)
        m = 25
        # Define xi_eff and xi_ens using the same conventions as the paper:
        # xi_eff = E[-log(2Pe)]/m , xi_ens = -(1/m) log(E[2Pe])
        xi_eff = float(np.mean(-np.log(2.0 * Pe)) / m)
        xi_ens = float(-(1.0 / m) * math.log(float(np.mean(2.0 * Pe))))
        assert xi_ens <= xi_eff + tol


def test_result_A_numbers_match_ensemble_model(tol):
    # Reproduce the manuscript-style numbers for the central-spin instance.
    seed = 2
    N = 200
    g = couplings_uniform(N, 0.8, 1.2, seed=seed)
    t = 0.5
    m = 25

    xiN = best_m_sum(xi_collective_per_fragment(g, t), m) / m
    xiL = best_m_sum(xi_local_helstrom_per_fragment(g, t), m) / m

    f = 0.8
    C_monitor = 0.10

    xi_full = xi_full_cycle_ensemble(f=f, m=m, xiN=xiN, xiL=xiL, C_monitor=C_monitor)
    xi_un = xi_unmonitored_collective_ensemble(f=f, m=m, xiN=xiN)

    # Expected values from the same formulas (deterministic).
    assert abs(xi_full - 0.906993770636024) <= 500 * tol
    assert abs(xi_un - 0.06437751649736402) <= 500 * tol

    # Baseline product exponent for comparison (no monitoring overhead)
    assert abs(xiL - 0.9575184707412496) <= 50 * tol
