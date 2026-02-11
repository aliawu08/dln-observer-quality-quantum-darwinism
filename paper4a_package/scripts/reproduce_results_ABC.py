#!/usr/bin/env python3
"""
Reproduce key numerical values quoted in the manuscript (Results A–C).

This script is intentionally lightweight and deterministic (seed=2 by default).
It matches the "Numerical verification" section in paper/paper4a_pra_reviewed.tex.

Dependencies: numpy
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class Params:
    seed: int = 2
    N: int = 200
    g_low: float = 0.8
    g_high: float = 1.2
    t: float = 0.5
    f: float = 0.8
    C_monitor: float = 0.10
    delta: float = 1e-3


def sample_couplings(p: Params) -> np.ndarray:
    rng = np.random.default_rng(p.seed)
    return rng.uniform(p.g_low, p.g_high, size=p.N)


def overlaps(g: np.ndarray, t: float) -> np.ndarray:
    c = np.abs(np.cos(2.0 * g * t))
    return np.clip(c, 1e-12, 1.0)


def best_subset_exponents(c: np.ndarray, m: int) -> Tuple[float, float]:
    """Return (xiN_total, xiL_total) on the best m fragments."""
    xiN = -np.log(c**2)
    xiL = -np.log(c)
    idx = np.argsort(-xiN)[:m]
    return float(xiN[idx].sum()), float(xiL[idx].sum())


def approx_bayes_error_from_exponent(xi_total: float) -> float:
    """Chernoff-style upper bound / exponent proxy: P_e ≈ 0.5 * exp(-xi_total)."""
    return 0.5 * math.exp(-xi_total)


def ensemble_exponent_from_error(P_e: float, m_total: int) -> float:
    return -(1.0 / m_total) * math.log(2.0 * P_e)


def result_A(p: Params, c: np.ndarray) -> Dict[str, float]:
    m = 25
    xiN_total, xiL_total = best_subset_exponents(c, m)
    xiN_per = xiN_total / m
    xiL_per = xiL_total / m

    # Unmonitored collective (binary episode-level observer-side decoherence):
    # good episodes: P_e ≈ 0.5 exp(-xiN_total); bad episodes: random, P_e=0.5.
    P_good = approx_bayes_error_from_exponent(xiN_total)
    P_un = p.f * P_good + (1.0 - p.f) * 0.5
    xi_ens_un = ensemble_exponent_from_error(P_un, m)

    # Fixed local decoding (robust across episodes):
    P_L = approx_bayes_error_from_exponent(xiL_total)
    xi_ens_L = ensemble_exponent_from_error(P_L, m)  # = xiL_total/m

    # Full-cycle with monitoring: m_diag=ceil(C_monitor*m), m_dec=m-m_diag
    m_diag = int(math.ceil(p.C_monitor * m))
    m_dec = m - m_diag
    xiN_dec, xiL_dec = best_subset_exponents(c, m_dec)
    P_good_dec = approx_bayes_error_from_exponent(xiN_dec)
    P_bad_dec = approx_bayes_error_from_exponent(xiL_dec)
    P_full = p.f * P_good_dec + (1.0 - p.f) * P_bad_dec
    xi_ens_full = ensemble_exponent_from_error(P_full, m)  # normalized by total m

    return dict(
        xiN_per=xiN_per,
        xiL_per=xiL_per,
        ratio=xiN_per / xiL_per,
        xi_ens_full=xi_ens_full,
        xi_ens_L=xi_ens_L,
        xi_ens_un=xi_ens_un,
        m_diag=float(m_diag),
        m_dec=float(m_dec),
    )


def result_B_thresholds(p: Params, c: np.ndarray) -> Dict[str, float]:
    # Binary episode-level threshold (per-episode) using exponent proxy errors.
    # f*_episode(m) = (P_L(m)-1/2)/(P_N(m)-1/2).
    out = {}
    for m in [5, 10]:
        xiN_total, xiL_total = best_subset_exponents(c, m)
        P_N = approx_bayes_error_from_exponent(xiN_total)
        P_L = approx_bayes_error_from_exponent(xiL_total)
        f_star = (P_L - 0.5) / (P_N - 0.5)
        out[f"f_star_episode_m{m}"] = float(f_star)

    # Continuous scaling model using exponent ratio; for pure-state Helstrom-local vs collective, ratio is ~1/2.
    out["f_star_continuous"] = 0.5
    return out


def pointer_access_gap_fraction(p: Params, m: int, delta: float, t_min: float = 0.0, t_max: float = 2.0, points: int = 1001) -> float:
    g = sample_couplings(p)
    ts = np.linspace(t_min, t_max, points)
    thr = math.log(1.0 / (2.0 * delta))
    gap = 0
    for t in ts:
        c_t = overlaps(g, float(t))
        xiN = -np.log(c_t**2)
        xiL = -np.log(c_t)
        idx = np.argsort(-xiN)[:m]
        okN = float(xiN[idx].sum()) >= thr
        okL = float(xiL[idx].sum()) >= thr
        if okN and (not okL):
            gap += 1
    return gap / len(ts)


def result_C(p: Params) -> Dict[str, float]:
    out = {}
    for m in [8, 12, 25, 50]:
        out[f"gap_fraction_m{m}"] = pointer_access_gap_fraction(p, m, p.delta)
    return out


def main() -> None:
    p = Params()
    g = sample_couplings(p)
    c = overlaps(g, p.t)

    A = result_A(p, c)
    B = result_B_thresholds(p, c)
    C = result_C(p)

    print("=== Result A (central-spin, seed=2) ===")
    print(f"xiN/m = {A['xiN_per']:.6f}")
    print(f"xiL/m = {A['xiL_per']:.6f}")
    print(f"ratio = {A['ratio']:.6f}")
    print(f"m_diag = {int(A['m_diag'])}, m_dec = {int(A['m_dec'])}")
    print(f"xi_ens_full = {A['xi_ens_full']:.6f}")
    print(f"xi_ens_L    = {A['xi_ens_L']:.6f}")
    print(f"xi_ens_un   = {A['xi_ens_un']:.6f}")

    print("\n=== Result B thresholds ===")
    print(f"f*_episode(m=5)  = {B['f_star_episode_m5']:.6f}")
    print(f"f*_episode(m=10) = {B['f_star_episode_m10']:.6f}")
    print(f"f*_continuous    = {B['f_star_continuous']:.3f}")

    print("\n=== Result C pointer-accessibility gap fractions (t in [0,2]) ===")
    for m in [8, 12, 25, 50]:
        print(f"gap_fraction(m={m}) = {C[f'gap_fraction_m{m}']*100:.3f}%")

if __name__ == "__main__":
    main()
