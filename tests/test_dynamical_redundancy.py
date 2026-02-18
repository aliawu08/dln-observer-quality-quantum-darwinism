"""Tests for dynamical redundancy bounds and inverted sophistication.

Verifies the core physics claims:
    1. xi_N / xi_L = 2 for pure states (existing result, new context)
    2. Unmonitored Dec_N error saturates at (1-f_coh)/2 (garbage floor)
    3. For f_coh < 1, Dec_L beats unmonitored Dec_N at sufficiently large m
    4. Full-cycle observer always has positive xi_eff
    5. Monitoring cost is strictly positive (Principle 1: learning is never free)
    6. Dec_N resolves pointer distinctions that Dec_L cannot (Result C)
"""
import math
import numpy as np
import pytest

from scripts.central_spin_example import (
    CentralSpinParams,
    sample_couplings,
    per_qubit_overlap,
    per_qubit_xi,
)
from scripts.dynamical_redundancy import (
    error_prob_collective,
    error_prob_product,
    error_prob_unmonitored_collective,
    error_prob_unmonitored_continuous,
    error_prob_depolarized_collective,
    error_prob_depolarized_product,
    error_prob_full_cycle,
    effective_exponent,
    inverted_sophistication_crossover,
    pointer_resolution_gap,
    robustness_f_star,
)


@pytest.fixture
def central_spin_setup():
    """Standard central-spin configuration for tests."""
    p = CentralSpinParams()
    g = sample_couplings(p)
    t = 0.5
    c = per_qubit_overlap(g, t)
    xi = per_qubit_xi(c)
    return g, t, c, xi


# ------------------------------------------------------------------
# Result A tests: Dynamical redundancy
# ------------------------------------------------------------------

class TestExponentRatio:
    """The factor-of-2 exponent ratio for pure states."""

    def test_xi_N_equals_2_xi_L(self, central_spin_setup):
        """xi_N / xi_L = 2 exactly for pure-state records."""
        _, _, _, xi = central_spin_setup
        m = 20
        xi_sorted = np.sort(xi)[::-1]
        xi_N = float(np.sum(xi_sorted[:m]))
        xi_L = xi_N / 2.0
        pe_coll = error_prob_collective(xi, m)
        pe_prod = error_prob_product(xi, m)
        # Check exponent ratio
        eff_N = effective_exponent(pe_coll, m)
        eff_L = effective_exponent(pe_prod, m)
        assert abs(eff_N / eff_L - 2.0) < 0.01

    def test_collective_always_better_than_product(self, central_spin_setup):
        """P_e^coll <= P_e^prod for all m."""
        _, _, _, xi = central_spin_setup
        for m in [1, 5, 10, 20, 50]:
            pe_c = error_prob_collective(xi, m)
            pe_p = error_prob_product(xi, m)
            assert pe_c <= pe_p + 1e-15


class TestGarbageExponent:
    """Unmonitored Dec_N has effective exponent -> 0."""

    def test_unmonitored_error_floor(self, central_spin_setup):
        """Unmonitored Dec_N error -> (1-f_coh)/2 for large m."""
        _, _, _, xi = central_spin_setup
        m = 50  # large enough for collective error to be negligible
        for f_coh in [0.5, 0.8, 0.9, 0.95]:
            pe = error_prob_unmonitored_collective(xi, m, f_coh)
            expected_floor = (1.0 - f_coh) / 2.0
            # Error should be close to the floor (collective part is negligible)
            assert abs(pe - expected_floor) < 0.01 * expected_floor + 1e-10

    def test_unmonitored_exponent_vanishes(self, central_spin_setup):
        """Effective exponent of unmonitored Dec_N -> 0 for large m."""
        _, _, _, xi = central_spin_setup
        f_coh = 0.9
        xi_effs = []
        for m in [10, 20, 50, 100]:
            pe = error_prob_unmonitored_collective(xi, min(m, 200), f_coh)
            xi_effs.append(effective_exponent(pe, min(m, 200)))
        # Exponent should decrease toward 0 as m grows
        assert xi_effs[-1] < xi_effs[0]
        assert xi_effs[-1] < 0.1  # effectively zero

    def test_perfect_coherence_recovers_collective(self, central_spin_setup):
        """At f_coh = 1.0, unmonitored Dec_N = perfect Dec_N."""
        _, _, _, xi = central_spin_setup
        m = 20
        pe_unmon = error_prob_unmonitored_collective(xi, m, f_coh=1.0)
        pe_coll = error_prob_collective(xi, m)
        assert abs(pe_unmon - pe_coll) < 1e-15


class TestMonitoringCost:
    """Monitoring always has a cost (Principle 1)."""

    def test_monitoring_reduces_exponent(self, central_spin_setup):
        """Full-cycle with monitoring < Dec_N without monitoring cost.

        At f_coh=1.0, full-cycle uses Dec_N but on fewer fragments.
        """
        _, _, _, xi = central_spin_setup
        m = 25
        f_monitor = 0.1
        pe_fc = error_prob_full_cycle(xi, m, f_coh=1.0, f_monitor=f_monitor)
        pe_coll = error_prob_collective(xi, m)
        # Full-cycle on (1-f)*m fragments > collective on m fragments
        assert pe_fc > pe_coll

    def test_monitoring_cost_is_positive(self, central_spin_setup):
        """xi_eff(full-cycle) < xi_N for all f_monitor > 0."""
        _, _, _, xi = central_spin_setup
        m = 25
        for f_mon in [0.05, 0.1, 0.2]:
            pe_fc = error_prob_full_cycle(xi, m, f_coh=1.0, f_monitor=f_mon)
            pe_coll = error_prob_collective(xi, m)
            xi_fc = effective_exponent(pe_fc, m)
            xi_coll = effective_exponent(pe_coll, m)
            assert xi_fc < xi_coll  # monitoring always costs

    def test_full_cycle_always_positive_exponent(self, central_spin_setup):
        """Full-cycle xi_eff > 0 for all f_coh > 0."""
        _, _, _, xi = central_spin_setup
        m = 25
        f_monitor = 0.1
        for f_coh in [0.01, 0.1, 0.5, 0.8, 0.95, 1.0]:
            pe = error_prob_full_cycle(xi, m, f_coh, f_monitor)
            xi_eff = effective_exponent(pe, m)
            assert xi_eff > 0


# ------------------------------------------------------------------
# Result B tests: Inverted sophistication
# ------------------------------------------------------------------

class TestInvertedSophistication:
    """Dec_N without monitoring can be strictly worse than Dec_L."""

    def test_inversion_exists(self, central_spin_setup):
        """For f_coh < 1, there exists m where Dec_L < Dec_N-no-monitor."""
        _, _, _, xi = central_spin_setup
        f_coh = 0.9
        # Find m where inversion occurs
        found_inversion = False
        for m in range(1, 51):
            pe_prod = error_prob_product(xi, m)
            pe_unmon = error_prob_unmonitored_collective(xi, m, f_coh)
            if pe_prod < pe_unmon:
                found_inversion = True
                break
        assert found_inversion, "Inversion should occur for f_coh=0.9"

    def test_f_star_increases_with_m(self, central_spin_setup):
        """f* (critical coherence) increases toward 1 as m grows."""
        g, t, _, _ = central_spin_setup
        cross = inverted_sophistication_crossover(g, t=t,
                                                   m_grid=np.arange(1, 31))
        f_star = cross["f_star"]
        valid = ~np.isnan(f_star)
        f_valid = f_star[valid]
        # f* should be non-decreasing (modulo numerical noise)
        for i in range(1, len(f_valid)):
            assert f_valid[i] >= f_valid[i - 1] - 1e-10

    def test_f_star_approaches_one(self, central_spin_setup):
        """For large m, f* -> 1 (need nearly perfect coherence)."""
        g, t, _, _ = central_spin_setup
        cross = inverted_sophistication_crossover(g, t=t,
                                                   m_grid=np.arange(1, 51))
        # At m=30+, f* should be very close to 1
        f_star_late = cross["f_star"][29:]  # m >= 30
        valid = ~np.isnan(f_star_late)
        if np.any(valid):
            assert np.all(f_star_late[valid] > 0.999)


# ------------------------------------------------------------------
# Result C tests: Pointer resolution gap
# ------------------------------------------------------------------

class TestPointerResolution:
    """Dec_N resolves pointer distinctions that Dec_L cannot."""

    def test_gap_exists(self, central_spin_setup):
        """There exist times where Dec_N resolves but Dec_L does not."""
        g, _, _, _ = central_spin_setup
        t_grid = np.linspace(0.01, 2.0, 200)
        res = pointer_resolution_gap(g, t_grid, m=8, delta=1e-3)
        assert res["gap_fraction"] > 0, "Resolution gap should exist"

    def test_dec_N_resolves_superset_of_dec_L(self, central_spin_setup):
        """Whenever Dec_L resolves, Dec_N also resolves (xi_N = 2*xi_L)."""
        g, _, _, _ = central_spin_setup
        t_grid = np.linspace(0.01, 2.0, 200)
        res = pointer_resolution_gap(g, t_grid, m=10, delta=1e-3)
        # Dec_L resolves => Dec_N resolves (but not vice versa)
        assert np.all(res["resolvable_N"] | ~res["resolvable_L"])

    def test_gap_narrows_with_m(self, central_spin_setup):
        """Resolution gap decreases as fragment budget increases."""
        g, _, _, _ = central_spin_setup
        t_grid = np.linspace(0.01, 2.0, 200)
        gaps = []
        for m in [5, 10, 20, 50]:
            res = pointer_resolution_gap(g, t_grid, m=m, delta=1e-3)
            gaps.append(res["gap_fraction"])
        # Gap should generally decrease (or stay same) with more fragments
        assert gaps[-1] <= gaps[0] + 0.01

    def test_xi_N_equals_2_xi_L_in_resolution(self, central_spin_setup):
        """The resolution computation uses xi_N = 2*xi_L correctly."""
        g, _, _, _ = central_spin_setup
        t_grid = np.array([0.5])
        res = pointer_resolution_gap(g, t_grid, m=10, delta=1e-3)
        ratio = res["xi_total_N"][0] / res["xi_total_L"][0]
        assert abs(ratio - 2.0) < 1e-10


# ------------------------------------------------------------------
# Robustness tests: alternative decoherence models
# ------------------------------------------------------------------

class TestRobustness:
    """Inverted sophistication survives under relaxed decoherence models."""

    def test_continuous_model_inversion_at_half(self, central_spin_setup):
        """Model B: inversion threshold is f* = 1/2 exactly."""
        _, _, _, xi = central_spin_setup
        m = 15
        # At f_coh = 0.49, continuous model should give worse P_e than product
        pe_prod = error_prob_product(xi, m)
        pe_cont_low = error_prob_unmonitored_continuous(xi, m, f_coh=0.49)
        assert pe_prod < pe_cont_low, "Dec_L should beat continuous-unmon at f=0.49"
        # At f_coh = 0.51, continuous model should beat product
        pe_cont_high = error_prob_unmonitored_continuous(xi, m, f_coh=0.51)
        assert pe_cont_high < pe_prod, "Continuous-unmon should beat Dec_L at f=0.51"

    def test_continuous_recovers_collective_at_1(self, central_spin_setup):
        """Model B at f_coh=1.0 equals perfect collective."""
        _, _, _, xi = central_spin_setup
        m = 20
        pe_cont = error_prob_unmonitored_continuous(xi, m, f_coh=1.0)
        pe_coll = error_prob_collective(xi, m)
        assert abs(pe_cont - pe_coll) < 1e-15

    def test_depolarized_collective_between_models(self, central_spin_setup):
        """Model C f* lies between Model A (binary) and Model B (continuous)."""
        g, t, c, xi = central_spin_setup
        m = 10
        # Model A (binary) f*
        pe_coll = error_prob_collective(xi, m)
        pe_prod = error_prob_product(xi, m)
        f_star_binary = (pe_prod - 0.5) / (pe_coll - 0.5)
        # Model C: at f = f_star_binary, depolarized collective should be
        # better than depolarized product (since binary is the harshest model)
        pe_c_dep = error_prob_depolarized_collective(c, m, f_star_binary)
        pe_p_dep = error_prob_depolarized_product(c, m, f_star_binary)
        assert pe_c_dep <= pe_p_dep + 1e-10, \
            "At binary f*, depolarized collective should be at least as good as product"

    def test_depolarized_recovers_pure_at_fcoh_1(self, central_spin_setup):
        """Model C at f_coh=1.0 matches the pure-state results."""
        _, _, c, xi = central_spin_setup
        m = 10
        pe_dep_coll = error_prob_depolarized_collective(c, m, f_coh=1.0)
        pe_coll = error_prob_collective(xi, m)
        # Should be close (not exact due to numerical QCB optimization)
        assert abs(pe_dep_coll - pe_coll) / pe_coll < 0.05

    def test_observer_side_models_predict_inversion(self, central_spin_setup):
        """Models A and B (observer-side decoherence) predict inversion."""
        _, _, _, xi = central_spin_setup
        m = 15
        pe_prod = error_prob_product(xi, m)
        # Model A: inversion at f_coh=0.9
        pe_unmon_a = error_prob_unmonitored_collective(xi, m, f_coh=0.9)
        assert pe_prod < pe_unmon_a, "Model A: inversion at f=0.9"
        # Model B: inversion at f_coh=0.4
        pe_unmon_b = error_prob_unmonitored_continuous(xi, m, f_coh=0.4)
        assert pe_prod < pe_unmon_b, "Model B: inversion at f=0.4"

    def test_depolarized_no_inversion(self, central_spin_setup):
        """Model C (system-side decoherence): collective always beats product.

        When decoherence affects the fragment states themselves (depolarization),
        BOTH measurement types degrade symmetrically.  The collective advantage
        (factor ~2 in exponent) persists.  This is the key physical distinction:
        inversion requires OBSERVER-SIDE decoherence, not SYSTEM-SIDE.
        """
        _, _, c, _ = central_spin_setup
        for m in [5, 10, 20]:
            for f in [0.1, 0.3, 0.5, 0.8]:
                pe_c = error_prob_depolarized_collective(c, m, f)
                pe_p = error_prob_depolarized_product(c, m, f)
                # Collective should be better or equal (within noise)
                assert pe_c <= pe_p + 1e-5, \
                    f"Depolarized collective should beat product at m={m}, f={f}"

    def test_robustness_binary_strongest_inversion(self, central_spin_setup):
        """Binary model gives the strongest inversion (highest f*)."""
        g, t, _, _ = central_spin_setup
        rob = robustness_f_star(g, t=t, m_grid=np.arange(5, 16))
        for i in range(len(rob["m_grid"])):
            fb = rob["f_star_binary"][i]
            fc = rob["f_star_continuous"][i]
            fd = rob["f_star_depolarized"][i]
            if np.isnan(fb):
                continue
            # Binary f* >= continuous f* = 0.5
            assert fb >= fc - 0.01, \
                f"Binary f* should >= continuous at m={rob['m_grid'][i]}"
            # Depolarized f* should be 0 (no inversion)
            if not np.isnan(fd):
                assert fd < 0.1, \
                    f"Depolarized should show no genuine inversion at m={rob['m_grid'][i]}"
