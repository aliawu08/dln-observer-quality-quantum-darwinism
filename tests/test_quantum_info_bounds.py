"""Tests for fundamental quantum information inequalities used in the paper.

These tests verify mathematical claims that underlie the theoretical framework:
    1. Factor-of-two exponent ratio: -log(c^2) / (-log c) = 2 (pure arithmetic)
    2. Data processing inequality for the quantum Chernoff bound under depolarization
    3. Decision continuity: |Tr(M(rho - sigma))| <= D_tr(rho, sigma) for POVM effects
    4. Numerical regression against seed=2 central-spin values (cross-validation)

Adapted from paper4a_package/tests/run_tests.py; restructured for pytest.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# ------------------------------------------------------------------
# Linear algebra helpers (self-contained, no external dependencies)
# ------------------------------------------------------------------

def _is_hermitian(a: np.ndarray, tol: float = 1e-10) -> bool:
    return np.allclose(a, a.conj().T, atol=tol, rtol=0.0)


def _mat_pow_psd(a: np.ndarray, power: float) -> np.ndarray:
    """Fractional power for PSD Hermitian matrices via eigen-decomposition."""
    if not _is_hermitian(a):
        a = 0.5 * (a + a.conj().T)
    w, v = np.linalg.eigh(a)
    w = np.clip(w, 0.0, None)
    wp = np.zeros_like(w)
    mask = w > 0
    wp[mask] = np.power(w[mask], power)
    return (v * wp) @ v.conj().T


def _trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Trace distance D_tr = 1/2 ||rho - sigma||_1."""
    delta = rho - sigma
    if not _is_hermitian(delta):
        delta = 0.5 * (delta + delta.conj().T)
    w = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(w)))


def _random_density_matrix(rng: np.random.Generator, d: int = 2) -> np.ndarray:
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    x = a @ a.conj().T
    x = 0.5 * (x + x.conj().T)
    return x / np.trace(x)


def _depolarize(rho: np.ndarray, p: float) -> np.ndarray:
    d = rho.shape[0]
    return (1.0 - p) * rho + p * np.eye(d) / d


def _qcb_trace_term(rho: np.ndarray, sigma: np.ndarray, s: float) -> float:
    return float(np.real(np.trace(
        _mat_pow_psd(rho, s) @ _mat_pow_psd(sigma, 1.0 - s)
    )))


# ------------------------------------------------------------------
# Test 1: Factor-of-two gap (pure arithmetic identity)
# ------------------------------------------------------------------

def test_factor_two_gap_mathematical():
    """For overlap c in (0,1): -log(c^2) / (-log c) = 2 exactly.

    This is the pure-state identity that gives xi_N = 2 * xi_L.
    Tested on 100 random overlaps, independent of any code functions.
    """
    rng = np.random.default_rng(0)
    for _ in range(100):
        c = float(rng.uniform(1e-3, 0.999))
        xi_coll = -math.log(c ** 2)
        xi_local = -math.log(c)
        ratio = xi_coll / xi_local
        assert abs(ratio - 2.0) < 1e-12, f"ratio mismatch: {ratio}"


# ------------------------------------------------------------------
# Test 2: Data processing inequality for quantum Chernoff bound
# ------------------------------------------------------------------

def test_chernoff_dpi_depolarizing():
    """DPI: Tr[Lambda(rho)^s Lambda(sigma)^{1-s}] >= Tr[rho^s sigma^{1-s}].

    Under depolarization, the Chernoff overlap can only increase (i.e. the
    exponent can only decrease). This underlies our claim that system-side
    decoherence degrades BOTH measurement types symmetrically.
    """
    rng = np.random.default_rng(1)
    s_vals = [0.1, 0.25, 0.5, 0.75, 0.9]
    for _ in range(50):
        rho = _random_density_matrix(rng, 2)
        sigma = _random_density_matrix(rng, 2)
        p = float(rng.uniform(0.0, 0.4))
        rho_p = _depolarize(rho, p)
        sigma_p = _depolarize(sigma, p)
        for s in s_vals:
            lhs = _qcb_trace_term(rho_p, sigma_p, s)
            rhs = _qcb_trace_term(rho, sigma, s)
            assert lhs + 1e-10 >= rhs, (
                f"DPI violated: lhs={lhs}, rhs={rhs}, s={s}, p={p}"
            )


# ------------------------------------------------------------------
# Test 3: Decision continuity for POVM effects
# ------------------------------------------------------------------

def test_decision_continuity_single_effect():
    """For any POVM effect 0 <= M <= I: |Tr(M(rho - sigma))| <= D_tr(rho, sigma).

    This is the fundamental bound connecting measurement distinguishability
    to trace distance, used throughout the paper's error analysis.
    """
    rng = np.random.default_rng(2)
    for _ in range(50):
        rho = _random_density_matrix(rng, 2)
        sigma = _random_density_matrix(rng, 2)
        # Random POVM effect: sample PSD and scale to operator norm <= 1
        a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        m = a @ a.conj().T
        w = np.linalg.eigvalsh(m)
        m = m / float(np.max(w))
        m = 0.5 * (m + m.conj().T)
        diff = float(np.real(np.trace(m @ (rho - sigma))))
        dtr = _trace_distance(rho, sigma)
        assert abs(diff) <= dtr + 1e-10, (
            f"continuity violated: |diff|={abs(diff)}, Dtr={dtr}"
        )


# ------------------------------------------------------------------
# Test 4: Numerical regression (seed=2, cross-validation)
# ------------------------------------------------------------------

def _central_spin_best_exponents_seed2(m: int) -> tuple[float, float]:
    """Central-spin exponents using seed=2 (alternative package convention)."""
    rng = np.random.default_rng(2)
    g = rng.uniform(0.8, 1.2, size=200)
    c = np.abs(np.cos(2.0 * g * 0.5))
    c = np.clip(c, 1e-12, 1.0)
    xiN = -np.log(c ** 2)
    xiL = -np.log(c)
    idx = np.argsort(-xiN)[:m]
    return float(np.sum(xiN[idx])), float(np.sum(xiL[idx]))


def test_central_spin_regression_seed2():
    """Cross-validate against independently computed numerical outputs (seed=2).

    These values were independently computed and match the manuscript's
    'Numerical verification' section.
    """
    m = 25
    xiN_total, xiL_total = _central_spin_best_exponents_seed2(m)
    xiN_per = xiN_total / m
    xiL_per = xiL_total / m

    # Per-fragment exponents
    assert abs(xiN_per - 1.9150369414824993) < 1e-9
    assert abs(xiL_per - 0.9575184707412496) < 1e-9
    assert abs((xiN_per / xiL_per) - 2.0) < 1e-12

    # Unmonitored collective (binary model): P = f * 0.5 * e^{-xiN} + (1-f) * 0.5
    f = 0.8
    P_good = 0.5 * math.exp(-xiN_total)
    P_un = f * P_good + (1.0 - f) * 0.5
    xi_ens_un = -(1.0 / m) * math.log(2.0 * P_un)
    assert abs(xi_ens_un - 0.06437751649736402) < 1e-9

    # Full-cycle with monitoring: m_diag=3, m_dec=22
    m_dec = m - 3
    xiN_dec, xiL_dec = _central_spin_best_exponents_seed2(m_dec)
    P_good_dec = 0.5 * math.exp(-xiN_dec)
    P_bad_dec = 0.5 * math.exp(-xiL_dec)
    P_full = f * P_good_dec + (1.0 - f) * P_bad_dec
    xi_ens_full = -(1.0 / m) * math.log(2.0 * P_full)
    assert abs(xi_ens_full - 0.9124248656563532) < 1e-9
