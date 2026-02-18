#!/usr/bin/env python3
"""
Central-spin pure-dephasing toy model used in Paper 4A (Observer Quality Formalism).

We consider a qubit system S interacting with N environment qubits via a controlled-phase
(pure dephasing) Hamiltonian. Environment qubit k has coupling g_k and starts in |+>.
Conditional on the system pointer value x in {0,1}, environment qubit k evolves to a pure
state |phi_k^{(x)}(t)>. The overlap between the two conditional states is

    c_k(t) = |<phi_k^{(0)}(t) | phi_k^{(1)}(t)>| = |cos(2 g_k t)|.

For pure-state records (no fragment noise), the quantum Chernoff coefficient equals the
fidelity:

    Q(t; A) = prod_{k in A} c_k(t)^2,

so the additive Chernoff exponent is

    xi(t; A) = -log Q(t; A) = sum_{k in A} (-log c_k(t)^2).

Given an error target delta (equal priors), a conservative finite-m bound is

    P_e <= 1/2 exp(-xi(t; A)),

so it suffices that xi(t; A) >= log(1/(2 delta)).

This script generates the figures used in the PRA paper:

- central_spin_redundancy_vs_time.png
- central_spin_m_required_vs_time.png
- central_spin_robustness_vs_p.png  (robustness sweep over noise p and coupling distributions)

The robustness sweep includes:
  - coupling distribution variation (uniform vs Gaussian), and
  - depolarizing readout noise p applied independently to each fragment.

For depolarizing noise, we compute a per-fragment Chernoff exponent using the fact that
(1) the depolarized conditional qubit states are isospectral and
(2) for such pairs the Chernoff minimizer is attained at s=1/2, so

    Q_k = Tr[ sqrt(rho_k^{(0)}) sqrt(rho_k^{(1)}) ]

and the multi-fragment exponent is additive.

(If you want a fully general QCB computation for arbitrary mixed states and heterogeneity,
see scripts/dynamical_redundancy.py in this repository.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.linalg as npla

# Ensure headless operation in CI / servers.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class CentralSpinParams:
    N: int = 200
    g0: float = 1.0
    rel_spread: float = 0.2  # +/- rel_spread around g0
    rng_seed: int = 7


def sample_couplings(p: CentralSpinParams) -> np.ndarray:
    """Backwards-compatible uniform coupling sampler."""
    rng = np.random.default_rng(p.rng_seed)
    low = (1.0 - p.rel_spread) * p.g0
    high = (1.0 + p.rel_spread) * p.g0
    return rng.uniform(low, high, size=p.N)


def sample_couplings_distribution(
    p: CentralSpinParams,
    distribution: str,
    rng_seed: int,
) -> np.ndarray:
    """Sample couplings using the requested distribution.

    Parameters
    ----------
    distribution:
        "uniform" or "gaussian".
    rng_seed:
        Seed used for deterministic sampling.

    Notes
    -----
    The Gaussian distribution is truncated to positive couplings.
    Its standard deviation is chosen to match the uniform distribution's
    variance for the same rel_spread.
    """
    rng = np.random.default_rng(rng_seed)

    if distribution.lower() == "uniform":
        low = (1.0 - p.rel_spread) * p.g0
        high = (1.0 + p.rel_spread) * p.g0
        return rng.uniform(low, high, size=p.N)

    if distribution.lower() == "gaussian":
        # Uniform on [g0(1-a), g0(1+a)] has std = a*g0/sqrt(3).
        std = p.rel_spread * p.g0 / math.sqrt(3.0)
        g = rng.normal(loc=p.g0, scale=std, size=p.N)
        # Truncate to positive couplings.
        g = np.clip(g, 1e-6, None)
        return g

    raise ValueError(f"Unknown distribution: {distribution!r}")


def per_qubit_overlap(g: np.ndarray, t: float) -> np.ndarray:
    """c_k(t) = |cos(2 g_k t)|."""
    return np.abs(np.cos(2.0 * g * t))


def per_qubit_xi_pure(c: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Pure-state per-qubit Chernoff exponents: xi_k = -log c_k^2."""
    c_clipped = np.clip(c, eps, 1.0)
    return -np.log(c_clipped**2)



# -----------------------------------------------------------------------------
# Backwards-compatible API (used by tests / earlier drafts)
# -----------------------------------------------------------------------------
def per_qubit_xi(c: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compatibility wrapper for the pure-record exponent xi_k = -log c_k^2."""
    return per_qubit_xi_pure(c, eps=eps)

def min_qubits_for_target(xi: np.ndarray, target: float, m_max: int) -> int:
    """Compatibility wrapper for best-subset selection."""
    return min_qubits_for_target_best_subset(xi, target=target, m_max=m_max)
def _qubit_density_from_overlap(c: float, p_dep: float) -> tuple[np.ndarray, np.ndarray]:
    """Canonical depolarized qubit pair with given overlap.

    We use |psi0> = |0>, |psi1> = c|0> + sqrt(1-c^2)|1>.
    Depolarization is unitary-covariant, so the Chernoff coefficient depends
    only on c and p_dep.
    """
    c = float(np.clip(c, 0.0, 1.0))
    s = math.sqrt(max(0.0, 1.0 - c * c))
    psi0 = np.array([[1.0], [0.0]], dtype=complex)
    psi1 = np.array([[c], [s]], dtype=complex)
    rho0 = psi0 @ psi0.conj().T
    rho1 = psi1 @ psi1.conj().T
    if p_dep <= 0.0:
        return rho0, rho1
    I = np.eye(2, dtype=complex)
    rho0 = (1.0 - p_dep) * rho0 + p_dep * I / 2.0
    rho1 = (1.0 - p_dep) * rho1 + p_dep * I / 2.0
    return rho0, rho1


def _sqrtm_2x2(rho: np.ndarray) -> np.ndarray:
    """Matrix square root for 2x2 PSD matrices via eigen-decomposition."""
    evals, evecs = npla.eigh(rho)
    evals = np.clip(evals, 0.0, None)
    return evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T


def qcb_coeff_depolarized_from_overlap(c: float, p_dep: float) -> float:
    """Chernoff coefficient for depolarized qubit pair, using s=1/2.

    For the depolarized pure-state pair used here, the two states are
    isospectral, and the Chernoff minimizer is attained at s=1/2.
    Thus the QCB coefficient equals Tr[sqrt(rho0) sqrt(rho1)].

    Returns a number in (0, 1].
    """
    rho0, rho1 = _qubit_density_from_overlap(c, p_dep)
    s0 = _sqrtm_2x2(rho0)
    s1 = _sqrtm_2x2(rho1)
    val = float(np.real(np.trace(s0 @ s1)))
    # Numerical safety.
    return float(np.clip(val, 1e-15, 1.0))


def per_qubit_xi_depolarized(c: np.ndarray, p_dep: float) -> np.ndarray:
    """Per-qubit Chernoff exponents under depolarizing noise."""
    if p_dep <= 0.0:
        return per_qubit_xi_pure(c)
    q = np.array([qcb_coeff_depolarized_from_overlap(float(ci), p_dep) for ci in c], dtype=float)
    return -np.log(q)


def min_qubits_for_target_best_subset(xi: np.ndarray, target: float, m_max: int) -> int:
    """Choose the best subset (largest xi_k) and return minimal m meeting target.

    Returns m_max if the target is not achievable within the access budget.
    """
    xi_sorted = np.sort(xi)[::-1]  # descending
    cumsum = np.cumsum(xi_sorted[:m_max])
    if cumsum.size == 0:
        return 0
    if cumsum[-1] < target:
        return m_max
    idx = int(np.searchsorted(cumsum, target, side="left"))
    return int(idx + 1)


def min_qubits_for_target_random_access(
    xi: np.ndarray,
    target: float,
    m_max: int,
    *,
    num_trials: int = 400,
    rng_seed: int = 0,
) -> int:
    """Typical required m under random-access sampling.

    Model:
      - The observer samples m fragments uniformly without replacement.
      - We estimate the median accumulated exponent over random samples.
      - Return the minimal m whose median (over Monte Carlo trials) meets target.

    Returns m_max if the target is not achievable within the access budget.

    Implementation trick:
      A random subset of size m is the first m elements of a random permutation.
      We therefore sample permutations and compute cumulative sums up to m_max.
    """
    if m_max <= 0:
        return 0

    rng = np.random.default_rng(rng_seed)

    # Collect cumsums for many trials.
    # Shape: (num_trials, m_max)
    cumsums = np.empty((num_trials, m_max), dtype=float)
    for i in range(num_trials):
        perm = rng.permutation(xi)
        cumsums[i, :] = np.cumsum(perm[:m_max])

    med = np.median(cumsums, axis=0)
    if med.size == 0 or med[-1] < target:
        return m_max
    idx = int(np.searchsorted(med, target, side="left"))
    return int(idx + 1)


def redundancy_vs_time(
    g: np.ndarray,
    R: float,
    delta: float,
    t_grid: np.ndarray,
    *,
    p_dep: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute required m(t) and redundancy under best-subset selection."""
    N = g.size
    m_access = max(1, int(math.floor(R * N)))
    target = math.log(1.0 / (2.0 * delta))

    m_req = np.empty_like(t_grid, dtype=int)
    redundancy = np.empty_like(t_grid, dtype=float)
    xi_eff = np.empty_like(t_grid, dtype=float)

    for i, t in enumerate(t_grid):
        c = per_qubit_overlap(g, float(t))
        xi = per_qubit_xi_depolarized(c, p_dep=p_dep)
        m = min_qubits_for_target_best_subset(xi, target=target, m_max=m_access)
        m_req[i] = m
        xi_eff[i] = float(np.sort(xi)[::-1][:m].mean()) if m > 0 else 0.0
        redundancy[i] = math.floor(m_access / max(1, m))

    return m_req, xi_eff, redundancy


def robustness_sweep_vs_p(
    *,
    p: CentralSpinParams,
    R: float,
    delta: float,
    t_fixed: float,
    p_grid: np.ndarray,
    n_coupling_draws: int = 25,
    num_trials_random_access: int = 400,
) -> dict[str, np.ndarray]:
    """Compute median required m vs depolarizing noise p for several variants."""
    N = p.N
    m_access = max(1, int(math.floor(R * N)))
    target = math.log(1.0 / (2.0 * delta))

    results: dict[str, list[float]] = {
        "uniform_best": [],
        "uniform_random": [],
        "gaussian_best": [],
        "gaussian_random": [],
    }

    # Deterministic seeds for coupling draws.
    base_seed = 10_000

    for p_dep in p_grid:
        m_uniform_best = []
        m_uniform_rand = []
        m_gauss_best = []
        m_gauss_rand = []

        for j in range(n_coupling_draws):
            seed_j = base_seed + j
            g_u = sample_couplings_distribution(p, "uniform", rng_seed=seed_j)
            g_g = sample_couplings_distribution(p, "gaussian", rng_seed=seed_j)

            # Overlaps at fixed t.
            c_u = per_qubit_overlap(g_u, t_fixed)
            c_g = per_qubit_overlap(g_g, t_fixed)

            xi_u = per_qubit_xi_depolarized(c_u, p_dep=float(p_dep))
            xi_g = per_qubit_xi_depolarized(c_g, p_dep=float(p_dep))

            m_uniform_best.append(min_qubits_for_target_best_subset(xi_u, target=target, m_max=m_access))
            m_gauss_best.append(min_qubits_for_target_best_subset(xi_g, target=target, m_max=m_access))

            # Random-access uses Monte Carlo; use deterministic per-(p_dep,j) seed.
            seed_ra = 1_000_000 + 1000 * int(round(1000 * float(p_dep))) + j
            m_uniform_rand.append(
                min_qubits_for_target_random_access(
                    xi_u,
                    target=target,
                    m_max=m_access,
                    num_trials=num_trials_random_access,
                    rng_seed=seed_ra,
                )
            )
            m_gauss_rand.append(
                min_qubits_for_target_random_access(
                    xi_g,
                    target=target,
                    m_max=m_access,
                    num_trials=num_trials_random_access,
                    rng_seed=seed_ra + 17,
                )
            )

        results["uniform_best"].append(float(np.median(m_uniform_best)))
        results["uniform_random"].append(float(np.median(m_uniform_rand)))
        results["gaussian_best"].append(float(np.median(m_gauss_best)))
        results["gaussian_random"].append(float(np.median(m_gauss_rand)))

    return {k: np.array(v, dtype=float) for k, v in results.items()}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    p = CentralSpinParams()
    g = sample_couplings(p)

    # Example observer and target error
    R = 0.25
    delta = 1e-3

    # --- Time-domain figures (pure records, best-subset selection) ---
    t_grid = np.linspace(0.0, 2.0, 500)
    m_req, _, redundancy = redundancy_vs_time(g, R=R, delta=delta, t_grid=t_grid, p_dep=0.0)

    plt.figure()
    plt.plot(t_grid, redundancy)
    plt.xlabel("t (arb. units)")
    plt.ylabel("Redundancy  R_δ  (accessible copies)")
    plt.title(f"Central-spin toy model: redundancy vs time (N={p.N}, R={R}, δ={delta})")
    plt.tight_layout()
    fig_path = out_dir / "central_spin_redundancy_vs_time.pdf"
    plt.savefig(fig_path)

    plt.figure()
    plt.plot(t_grid, m_req)
    plt.xlabel("t (arb. units)")
    plt.ylabel("m required")
    plt.title(f"Required fragments vs time (N={p.N}, R={R}, δ={delta})")
    plt.tight_layout()
    fig2_path = out_dir / "central_spin_m_required_vs_time.pdf"
    plt.savefig(fig2_path)

    # --- Robustness figure (noise + coupling distribution + access model) ---
    t_fixed = 0.5
    p_grid = np.linspace(0.0, 0.30, 7)
    sweep = robustness_sweep_vs_p(
        p=p,
        R=R,
        delta=delta,
        t_fixed=t_fixed,
        p_grid=p_grid,
        n_coupling_draws=25,
        num_trials_random_access=400,
    )

    plt.figure()
    plt.plot(p_grid, sweep["uniform_best"], label="uniform, best-subset")
    plt.plot(p_grid, sweep["uniform_random"], label="uniform, random-access")
    plt.plot(p_grid, sweep["gaussian_best"], label="gaussian, best-subset")
    plt.plot(p_grid, sweep["gaussian_random"], label="gaussian, random-access")
    plt.xlabel("depolarizing noise  p")
    plt.ylabel("m required (median over coupling draws)")
    plt.title(f"Robustness sweep at t={t_fixed} (N={p.N}, R={R}, δ={delta})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig3_path = out_dir / "central_spin_robustness_vs_p.pdf"
    plt.savefig(fig3_path)

    # Console output for quick checks
    print("Saved:", fig_path)
    print("Saved:", fig2_path)
    print("Saved:", fig3_path)
    i_t = int(np.argmin(np.abs(t_grid - 1.0)))
    print("Example at t=1.0: m_req =", int(m_req[i_t]))


if __name__ == "__main__":
    main()
