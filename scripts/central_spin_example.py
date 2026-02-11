#!/usr/bin/env python3
"""
Central-spin pure-dephasing toy model for the observer-quality formalism.

We consider a qubit system S interacting with N environment qubits via a controlled-phase
(pure dephasing) Hamiltonian. Environment qubit k has coupling g_k and starts in |+>.
Conditional on the system pointer value x in {0,1}, environment qubit k evolves to a pure
state |phi_k^{(x)}(t)>. The overlap between the two conditional states is
    c_k(t) = |<phi_k^{(0)}(t) | phi_k^{(1)}(t)>| = |cos(2 g_k t)|.

For a product of qubits, the quantum Chernoff coefficient equals the fidelity for pure states:
    Q(t; A) = prod_{k in A} c_k(t)^2,
so the (additive) Chernoff exponent is
    xi(t; A) = -log Q(t; A) = sum_{k in A} (-log c_k(t)^2).

Given an error target delta (equal priors), a conservative finite-m bound is
    P_e <= 1/2 exp(-xi(t; A)),
so it suffices that xi(t; A) >= log(1/(2 delta)).

This script:
- samples couplings g_k,
- computes the minimal number of accessible qubits needed to reach the target exponent,
  assuming the observer can choose the best (most informative) qubits,
- plots redundancy vs time for a fixed access fraction R.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CentralSpinParams:
    N: int = 200
    g0: float = 1.0
    rel_spread: float = 0.2  # +/- rel_spread around g0
    rng_seed: int = 7


def sample_couplings(p: CentralSpinParams) -> np.ndarray:
    rng = np.random.default_rng(p.rng_seed)
    low = (1.0 - p.rel_spread) * p.g0
    high = (1.0 + p.rel_spread) * p.g0
    return rng.uniform(low, high, size=p.N)


def per_qubit_overlap(g: np.ndarray, t: float) -> np.ndarray:
    # c_k(t) = |cos(2 g_k t)|
    return np.abs(np.cos(2.0 * g * t))


def per_qubit_xi(c: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # xi_k = -log c_k^2; clip to avoid log(0)
    c_clipped = np.clip(c, eps, 1.0)
    return -np.log(c_clipped ** 2)


def min_qubits_for_target(xi: np.ndarray, target: float, m_max: int) -> int:
    """
    Given per-qubit exponents xi_k, choose the best subset (largest xi_k)
    and return the minimal m such that sum_{k=1}^m xi_(k) >= target.
    """
    xi_sorted = np.sort(xi)[::-1]  # descending
    cumsum = np.cumsum(xi_sorted)
    m = int(np.searchsorted(cumsum, target, side="left") + 1)
    return min(m, m_max)


def redundancy_vs_time(
    g: np.ndarray,
    R: float,
    delta: float,
    t_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = g.size
    m_access = max(1, int(math.floor(R * N)))
    target = math.log(1.0 / (2.0 * delta))

    m_req = np.empty_like(t_grid, dtype=int)
    redundancy = np.empty_like(t_grid, dtype=float)
    xi_eff = np.empty_like(t_grid, dtype=float)

    for i, t in enumerate(t_grid):
        c = per_qubit_overlap(g, float(t))
        xi = per_qubit_xi(c)
        m = min_qubits_for_target(xi, target=target, m_max=m_access)
        m_req[i] = m
        # effective exponent per accessed qubit (best subset)
        xi_eff[i] = float(np.sort(xi)[::-1][:m].mean())
        redundancy[i] = math.floor(m_access / m)

    return m_req, xi_eff, redundancy


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    p = CentralSpinParams()
    g = sample_couplings(p)

    # Example observer and target error
    R = 0.25
    delta = 1e-3

    t_grid = np.linspace(0.0, 2.0, 500)
    m_req, xi_eff, redundancy = redundancy_vs_time(g, R=R, delta=delta, t_grid=t_grid)

    # Plot redundancy
    plt.figure()
    plt.plot(t_grid, redundancy)
    plt.xlabel("t (arb. units)")
    plt.ylabel("Redundancy  R_δ  (accessible copies)")
    plt.title(f"Central-spin toy model: redundancy vs time (N={p.N}, R={R}, δ={delta})")
    plt.tight_layout()
    fig_path = out_dir / "central_spin_redundancy_vs_time.png"
    plt.savefig(fig_path, dpi=200)

    # Plot required m
    plt.figure()
    plt.plot(t_grid, m_req)
    plt.xlabel("t (arb. units)")
    plt.ylabel("m required")
    plt.title(f"Required fragments vs time (N={p.N}, R={R}, δ={delta})")
    plt.tight_layout()
    fig2_path = out_dir / "central_spin_m_required_vs_time.png"
    plt.savefig(fig2_path, dpi=200)

    # Simple printouts
    print("Saved:", fig_path)
    print("Saved:", fig2_path)
    print("Example at t=1.0: m_req =", int(m_req[np.argmin(np.abs(t_grid-1.0))]))


if __name__ == "__main__":
    main()
