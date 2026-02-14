#!/usr/bin/env python3
"""
Lightweight tests for the Paper 4A package.

These tests are designed to be:
  - deterministic (fixed seeds)
  - dependency-light (numpy only)
  - aligned with the mathematical statements used in the manuscript

Run:
  python -m tests.run_tests
or:
  python tests/run_tests.py
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ------------------------
# Linear algebra helpers
# ------------------------

def is_hermitian(a: np.ndarray, tol: float = 1e-10) -> bool:
    return np.allclose(a, a.conj().T, atol=tol, rtol=0.0)


def mat_pow_psd(a: np.ndarray, power: float) -> np.ndarray:
    """
    Fractional power for PSD Hermitian matrices via eigen-decomposition.
    """
    if not is_hermitian(a):
        a = 0.5 * (a + a.conj().T)
    w, v = np.linalg.eigh(a)
    w = np.clip(w, 0.0, None)
    wp = np.power(w, power, where=(w > 0))
    wp[w == 0] = 0.0
    return (v * wp) @ v.conj().T


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Trace distance D_tr = 1/2 ||rho - sigma||_1.
    """
    delta = rho - sigma
    if not is_hermitian(delta):
        delta = 0.5 * (delta + delta.conj().T)
    w = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(w)))


def random_density_matrix(rng: np.random.Generator, d: int = 2) -> np.ndarray:
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    x = a @ a.conj().T
    x = 0.5 * (x + x.conj().T)
    return x / np.trace(x)


def depolarize(rho: np.ndarray, p: float) -> np.ndarray:
    d = rho.shape[0]
    return (1.0 - p) * rho + p * np.eye(d) / d


def qcb_trace_term(rho: np.ndarray, sigma: np.ndarray, s: float) -> float:
    return float(np.real(np.trace(mat_pow_psd(rho, s) @ mat_pow_psd(sigma, 1.0 - s))))


# ------------------------
# Central-spin checks
# ------------------------

@dataclass(frozen=True)
class CentralSpinParams:
    seed: int = 2
    N: int = 200
    g_low: float = 0.8
    g_high: float = 1.2
    t: float = 0.5


def central_spin_best_exponents(m: int, p: CentralSpinParams = CentralSpinParams()) -> Tuple[float, float]:
    rng = np.random.default_rng(p.seed)
    g = rng.uniform(p.g_low, p.g_high, size=p.N)
    c = np.abs(np.cos(2.0 * g * p.t))
    c = np.clip(c, 1e-12, 1.0)
    xiN = -np.log(c**2)
    xiL = -np.log(c)
    idx = np.argsort(-xiN)[:m]
    return float(np.sum(xiN[idx])), float(np.sum(xiL[idx]))


def ensemble_exponent(P_e: float, m_total: int) -> float:
    return -(1.0 / m_total) * math.log(2.0 * P_e)


# ------------------------
# Tests
# ------------------------

def test_factor_two_gap() -> None:
    # For overlap c in (0,1), collective exponent per copy is -log(c^2)=2(-log c).
    rng = np.random.default_rng(0)
    for _ in range(100):
        c = float(rng.uniform(1e-3, 0.999))
        xi_coll = -math.log(c**2)
        xi_local = -math.log(c)
        ratio = xi_coll / xi_local
        assert abs(ratio - 2.0) < 1e-12, f"ratio mismatch: {ratio}"


def test_chernoff_dpi_depolarizing() -> None:
    # Spot-check the inequality Tr[Λ(ρ)^s Λ(σ)^{1-s}] >= Tr[ρ^s σ^{1-s}] for several s.
    rng = np.random.default_rng(1)
    s_vals = [0.1, 0.25, 0.5, 0.75, 0.9]
    for _ in range(50):
        rho = random_density_matrix(rng, 2)
        sigma = random_density_matrix(rng, 2)
        p = float(rng.uniform(0.0, 0.4))
        rho_p = depolarize(rho, p)
        sigma_p = depolarize(sigma, p)
        for s in s_vals:
            lhs = qcb_trace_term(rho_p, sigma_p, s)
            rhs = qcb_trace_term(rho, sigma, s)
            # Allow tiny numerical slack
            assert lhs + 1e-10 >= rhs, f"DPI violated: lhs={lhs}, rhs={rhs}, s={s}, p={p}"


def test_decision_continuity_single_effect() -> None:
    # For any POVM effect 0<=M<=I, |Tr(M(ρ-σ))| <= D_tr(ρ,σ).
    rng = np.random.default_rng(2)
    for _ in range(50):
        rho = random_density_matrix(rng, 2)
        sigma = random_density_matrix(rng, 2)
        # Random effect: sample PSD and scale to have operator norm <=1.
        a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        m = a @ a.conj().T
        w = np.linalg.eigvalsh(m)
        m = m / float(np.max(w))  # now max eigenvalue 1
        m = 0.5 * (m + m.conj().T)
        diff = float(np.real(np.trace(m @ (rho - sigma))))
        dtr = trace_distance(rho, sigma)
        assert abs(diff) <= dtr + 1e-10, f"continuity violated: |diff|={abs(diff)}, Dtr={dtr}"


def test_central_spin_reproduced_numbers() -> None:
    # Match manuscript-reported values (seed=2).
    m = 25
    xiN_total, xiL_total = central_spin_best_exponents(m)
    xiN_per = xiN_total / m
    xiL_per = xiL_total / m

    assert abs(xiN_per - 1.9150369414824993) < 1e-9
    assert abs(xiL_per - 0.9575184707412496) < 1e-9
    assert abs((xiN_per / xiL_per) - 2.0) < 1e-12

    # Unmonitored collective (observer-side binary model): P = f * 0.5 e^{-xiN} + (1-f)*0.5.
    f = 0.8
    P_good = 0.5 * math.exp(-xiN_total)
    P_un = f * P_good + (1.0 - f) * 0.5
    xi_ens_un = ensemble_exponent(P_un, m)
    assert abs(xi_ens_un - 0.06437751649736402) < 1e-9

    # Full-cycle with monitoring: m_diag=3, m_dec=22
    m_diag = 3
    m_dec = m - m_diag
    xiN_dec, xiL_dec = central_spin_best_exponents(m_dec)
    P_good_dec = 0.5 * math.exp(-xiN_dec)
    P_bad_dec = 0.5 * math.exp(-xiL_dec)
    P_full = f * P_good_dec + (1.0 - f) * P_bad_dec
    xi_ens_full = ensemble_exponent(P_full, m)
    assert abs(xi_ens_full - 0.9124248656563532) < 1e-9


def run_all() -> None:
    tests = [
        test_factor_two_gap,
        test_chernoff_dpi_depolarizing,
        test_decision_continuity_single_effect,
        test_central_spin_reproduced_numbers,
    ]
    for t in tests:
        t()
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    try:
        run_all()
    except AssertionError as e:
        print("TEST FAILED:", e, file=sys.stderr)
        sys.exit(1)
