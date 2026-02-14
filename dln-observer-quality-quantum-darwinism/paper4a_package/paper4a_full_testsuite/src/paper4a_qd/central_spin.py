from __future__ import annotations

import math
import numpy as np

def couplings_uniform(N: int, low: float, high: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=N)

def overlap_c(g: np.ndarray, t: float) -> np.ndarray:
    """Single-fragment overlap c_k(t) = |cos(2 g_k t)| in the pure-dephasing central-spin model."""
    c = np.abs(np.cos(2.0 * g * t))
    return np.clip(c, 1e-12, 1.0)

def xi_collective_per_fragment(g: np.ndarray, t: float) -> np.ndarray:
    """Per-fragment Chernoff exponent for pure conditional records (collective optimal): ξ_k = -log(c_k^2)."""
    c = overlap_c(g, t)
    return -np.log(c ** 2)

def xi_local_helstrom_per_fragment(g: np.ndarray, t: float) -> np.ndarray:
    """Per-fragment exponent for local-Helstrom-per-copy decoding: ξ_k = -log(c_k)."""
    c = overlap_c(g, t)
    return -np.log(c)

def best_m_sum(xi_k: np.ndarray, m: int) -> float:
    """Sum of the top-m per-fragment exponents (best accessible subset of size m)."""
    if m <= 0:
        return 0.0
    idx = np.argsort(xi_k)[::-1][:m]
    return float(np.sum(xi_k[idx]))

def threshold(delta: float) -> float:
    """Chernoff-style sufficient condition threshold: require ξ >= log(1/(2δ))."""
    return float(math.log(1.0 / (2.0 * delta)))

def is_accessible_collective(g: np.ndarray, t: float, m: int, delta: float) -> bool:
    xi = xi_collective_per_fragment(g, t)
    return best_m_sum(xi, m) >= threshold(delta)

def is_accessible_local(g: np.ndarray, t: float, m: int, delta: float) -> bool:
    xi = xi_local_helstrom_per_fragment(g, t)
    return best_m_sum(xi, m) >= threshold(delta)

def pointer_gap_fraction(
    g: np.ndarray,
    tmin: float,
    tmax: float,
    num: int,
    m: int,
    delta: float,
) -> float:
    """Fraction of times where collective decoding is accessible but local-Helstrom product decoding is not."""
    ts = np.linspace(tmin, tmax, num)
    gap = 0
    for t in ts:
        accN = is_accessible_collective(g, float(t), m, delta)
        accL = is_accessible_local(g, float(t), m, delta)
        if accN and (not accL):
            gap += 1
    return float(gap / len(ts))
