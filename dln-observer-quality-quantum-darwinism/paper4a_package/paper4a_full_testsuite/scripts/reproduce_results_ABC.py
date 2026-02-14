#!/usr/bin/env python
from __future__ import annotations

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import json
import math
import os
from dataclasses import dataclass

import numpy as np

from paper4a_qd.central_spin import (
    couplings_uniform,
    xi_collective_per_fragment,
    xi_local_helstrom_per_fragment,
    best_m_sum,
    pointer_gap_fraction,
)
from paper4a_qd.observer_topology import (
    xi_full_cycle_ensemble,
    xi_unmonitored_collective_ensemble,
)

@dataclass(frozen=True)
class CentralSpinParams:
    seed: int
    N: int
    g_low: float
    g_high: float
    t_for_exponent: float
    m_for_exponent: int
    t_grid_min: float
    t_grid_max: float
    t_grid_num: int
    delta: float

def load_params() -> CentralSpinParams:
    meta_path = os.path.join(os.path.dirname(__file__), "..", "baselines", "metadata.json")
    meta_path = os.path.abspath(meta_path)
    meta = json.load(open(meta_path, "r"))
    cs = meta["central_spin"]
    tmin, tmax, num = cs["time_grid"]
    return CentralSpinParams(
        seed=cs["seed"],
        N=cs["N"],
        g_low=cs["g_low"],
        g_high=cs["g_high"],
        t_for_exponent=cs["t_for_exponent"],
        m_for_exponent=cs["m_for_exponent"],
        t_grid_min=tmin,
        t_grid_max=tmax,
        t_grid_num=num,
        delta=cs["delta"],
    )

def main() -> None:
    p = load_params()
    g = couplings_uniform(p.N, p.g_low, p.g_high, seed=p.seed)

    # Exponents at the manuscript point
    xiN_k = xi_collective_per_fragment(g, p.t_for_exponent)
    xiL_k = xi_local_helstrom_per_fragment(g, p.t_for_exponent)

    m = p.m_for_exponent
    xiN = best_m_sum(xiN_k, m) / m
    xiL = best_m_sum(xiL_k, m) / m

    print("=== Central-spin (pure dephasing) exponents at t=%.3f ===" % p.t_for_exponent)
    print(f"seed={p.seed}, N={p.N}, g~U({p.g_low},{p.g_high}), best-m selection, m={m}")
    print(f"xi_N/m = {xiN:.12f}")
    print(f"xi_L/m = {xiL:.12f}")
    print(f"xi_N/xi_L = {xiN/xiL:.12f}")
    print()

    # Result A ensemble exponents (matches the manuscript’s unmonitored-collapse numbers)
    f = 0.8
    C_monitor = 0.10
    xi_full = xi_full_cycle_ensemble(f=f, m=m, xiN=xiN, xiL=xiL, C_monitor=C_monitor)
    xi_un = xi_unmonitored_collective_ensemble(f=f, m=m, xiN=xiN)
    print("=== Result A (ensemble exponent model) ===")
    print(f"f={f}, m={m}, C_monitor={C_monitor}")
    print(f"xi_ens(full-cycle)      = {xi_full:.12f}")
    print(f"xi_ens(unmonitored coll)= {xi_un:.12f}")
    print(f"xi_L baseline           = {xiL:.12f}")
    print()

    # Pointer accessibility gap fractions
    print("=== Result C pointer-accessibility gap fractions ===")
    for m_gap in [8, 12, 25, 50]:
        frac = pointer_gap_fraction(
            g=g,
            tmin=p.t_grid_min,
            tmax=p.t_grid_max,
            num=p.t_grid_num,
            m=m_gap,
            delta=p.delta,
        )
        print(f"m={m_gap:>2d}: gap fraction over t∈[{p.t_grid_min},{p.t_grid_max}] = {100*frac:.3f}%")
    print()

    # Result B: illustrative critical fraction in the “continuous scaling” caricature
    # If xi_N = 2 xi_L (pure-state + local-Helstrom), then f* = xi_L / xi_N = 1/2.
    f_star = xiL / xiN
    print("=== Result B (continuous scaling caricature) ===")
    print(f"f* (approx) = xi_L/xi_N = {f_star:.12f}  (≈ 0.5 expected)")
    print()

if __name__ == "__main__":
    main()
