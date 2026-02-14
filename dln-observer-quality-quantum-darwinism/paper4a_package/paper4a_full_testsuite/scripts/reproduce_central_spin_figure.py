#!/usr/bin/env python
from __future__ import annotations

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import argparse
import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

from paper4a_qd.central_spin import (
    couplings_uniform,
    xi_collective_per_fragment,
    best_m_sum,
    threshold,
)

def min_m_to_reach_threshold(xi_k: np.ndarray, thr: float, m_max: int) -> int | None:
    """Return minimal m <= m_max such that sum of top-m xi_k >= thr, else None."""
    if m_max <= 0:
        return None
    xi_sorted = np.sort(xi_k)[::-1]
    xi_sorted = xi_sorted[:m_max]
    cumsum = np.cumsum(xi_sorted)
    idx = np.searchsorted(cumsum, thr, side="left")
    if idx >= len(cumsum):
        return None
    return int(idx + 1)

def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce a central-spin redundancy-vs-time figure and CSV.")
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--g-low", type=float, default=0.8)
    ap.add_argument("--g-high", type=float, default=1.2)
    ap.add_argument("--R", type=float, default=0.25, help="Access fraction R_O.")
    ap.add_argument("--delta", type=float, default=1e-3, help="Target error δ.")
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=2.0)
    ap.add_argument("--num", type=int, default=801)
    ap.add_argument("--outdir", type=str, default="figures")
    ap.add_argument("--prefix", type=str, default="central_spin_redundancy_vs_time")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    g = couplings_uniform(args.N, args.g_low, args.g_high, seed=args.seed)
    m_max = int(math.floor(args.R * args.N))
    thr = threshold(args.delta)

    ts = np.linspace(args.tmin, args.tmax, args.num)
    rows = []
    for t in ts:
        xi_k = xi_collective_per_fragment(g, float(t))
        m_req = min_m_to_reach_threshold(xi_k, thr, m_max=m_max)
        if m_req is None:
            red = 0
            m_req_out = ""
        else:
            red = m_max // m_req
            m_req_out = m_req
        rows.append((float(t), m_req_out, int(red)))

    csv_path = os.path.join(args.outdir, f"{args.prefix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "m_required", "redundancy_accessible"])
        w.writerows(rows)

    # plot
    t_vals = np.array([r[0] for r in rows], dtype=float)
    red_vals = np.array([r[2] for r in rows], dtype=float)
    plt.figure()
    plt.plot(t_vals, red_vals)
    plt.xlabel("interaction time t")
    plt.ylabel("accessible redundancy (floor(m_max / m_required))")
    plt.title(f"Central-spin redundancy vs time (N={args.N}, R={args.R}, δ={args.delta}, seed={args.seed})")
    png_path = os.path.join(args.outdir, f"{args.prefix}.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Wrote: {png_path}")
    print(f"Wrote: {csv_path}")

if __name__ == "__main__":
    main()
