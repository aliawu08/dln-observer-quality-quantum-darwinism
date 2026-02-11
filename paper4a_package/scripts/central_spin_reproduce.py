#!/usr/bin/env python3
"""
Central-spin reproducibility script for Paper 4A.

Generates:
  - figures/central_spin_redundancy_vs_time.png
  - figures/central_spin_redundancy_vs_time.csv

Model:
  Central-spin pure dephasing with conditional fragment overlap c_k(t)=|cos(2 g_k t)|.
  Single-fragment Chernoff exponent: xi_k(t) = -log(c_k(t)^2).

We compute (accessible) redundancy as:
  redundancy(t) = m_max / m_req(t),
where m_req(t) is the minimum number of accessible fragments needed to satisfy:
  sum_{top m_req} xi_k(t) >= log(1/(2 delta)).

This is an operational sample-complexity redundancy proxy consistent with the manuscript.

Dependencies: numpy, matplotlib
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Params:
    seed: int = 2
    N: int = 200
    g_low: float = 0.8
    g_high: float = 1.2
    R_O: float = 0.25
    delta: float = 1e-3
    t_min: float = 0.0
    t_max: float = 2.0
    t_points: int = 1001


def sample_couplings(seed: int, N: int, g_low: float, g_high: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(g_low, g_high, size=N)


def choose_accessible_indices(seed: int, N: int, m_max: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 12345)
    return rng.choice(N, size=m_max, replace=False)


def exponents_for_time(g: np.ndarray, t: float) -> np.ndarray:
    c = np.abs(np.cos(2.0 * g * t))
    c = np.clip(c, 1e-12, 1.0)
    return -np.log(c**2)


def min_fragments_for_threshold(xi: np.ndarray, threshold: float) -> Tuple[int, float]:
    """
    Given per-fragment exponents xi (nonnegative), choose the best m fragments and
    return the smallest m such that cumulative sum >= threshold.
    Returns (m_req, achieved_sum). If threshold cannot be met, returns (0, max_sum).
    """
    xi_sorted = np.sort(xi)[::-1]  # descending
    cum = np.cumsum(xi_sorted)
    idx = np.searchsorted(cum, threshold, side="left")
    if idx >= len(xi_sorted):
        return 0, float(cum[-1]) if len(cum) else 0.0
    return int(idx + 1), float(cum[idx])


def main() -> None:
    p = Params()
    threshold = math.log(1.0 / (2.0 * p.delta))

    g = sample_couplings(p.seed, p.N, p.g_low, p.g_high)

    m_max = int(math.floor(p.R_O * p.N))
    if m_max <= 0:
        raise ValueError("R_O too small: m_max=0")

    accessible = choose_accessible_indices(p.seed, p.N, m_max)
    g_acc = g[accessible]

    ts = np.linspace(p.t_min, p.t_max, p.t_points)

    m_req_list = []
    red_list = []

    for t in ts:
        xi = exponents_for_time(g_acc, float(t))
        m_req, _ = min_fragments_for_threshold(xi, threshold)
        m_req_list.append(m_req)
        if m_req == 0:
            red_list.append(0.0)
        else:
            red_list.append(m_max / m_req)

    # Output paths
    repo_root = Path(__file__).resolve().parents[1]
    fig_dir = repo_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    png_path = fig_dir / "central_spin_redundancy_vs_time.png"
    csv_path = fig_dir / "central_spin_redundancy_vs_time.csv"

    # Save CSV
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "m_req", "redundancy"])
        for t, m_req, red in zip(ts, m_req_list, red_list):
            w.writerow([f"{t:.8f}", m_req, f"{red:.8f}"])

    # Plot
    plt.figure()
    plt.plot(ts, red_list)
    plt.xlabel("interaction time t")
    plt.ylabel("redundancy proxy (m_max / m_req)")
    plt.title("Central-spin redundancy proxy vs time")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
