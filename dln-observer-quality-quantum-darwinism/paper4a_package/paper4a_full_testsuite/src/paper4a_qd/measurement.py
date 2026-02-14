from __future__ import annotations

import numpy as np

from .distances import trace_distance

def helstrom_projectors(rho0: np.ndarray, rho1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the Helstrom-optimal projective measurement {P0, P1} for equal priors.

    For equal priors, the Helstrom measurement is the projection onto the
    positive/negative eigenspaces of Δ = ρ0 - ρ1.
    """
    Delta = (rho0 - rho1)
    Delta = (Delta + Delta.conj().T) / 2
    w, V = np.linalg.eigh(Delta)
    # Positive eigenspace -> decide hypothesis 0
    pos = w > 0
    if np.any(pos):
        P0 = V[:, pos] @ V[:, pos].conj().T
    else:
        # If no positive eigenvalues, always decide 1
        P0 = np.zeros_like(rho0, dtype=complex)
    P1 = np.eye(rho0.shape[0], dtype=complex) - P0
    return P0, P1

def helstrom_error_equal_priors(rho0: np.ndarray, rho1: np.ndarray) -> float:
    """Helstrom optimal Bayes error for equal priors.

    P_e* = 1/2 (1 - 1/2 ||ρ0 - ρ1||_1) = 1/2 (1 - D_tr(ρ0,ρ1)).
    """
    return 0.5 * (1.0 - trace_distance(rho0, rho1))

def induced_classical_distributions(rho0: np.ndarray, rho1: np.ndarray, povm: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Classical distributions p(y|0), q(y|1) induced by POVM elements."""
    p = np.array([np.real_if_close(np.trace(M @ rho0)).real for M in povm], dtype=float)
    q = np.array([np.real_if_close(np.trace(M @ rho1)).real for M in povm], dtype=float)
    # numerical cleanup
    p = np.clip(p, 0.0, 1.0)
    q = np.clip(q, 0.0, 1.0)
    # normalize (guard against tiny drift)
    p = p / p.sum()
    q = q / q.sum()
    return p, q

def bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    """Bhattacharyya coefficient B(p,q) = sum_y sqrt(p_y q_y)."""
    return float(np.sum(np.sqrt(p * q)))

def classical_chernoff_coefficient(p: np.ndarray, q: np.ndarray, grid_points: int = 2001) -> tuple[float, float]:
    """Classical Chernoff coefficient min_s sum_y p^s q^{1-s}."""
    ss = np.linspace(0.0, 1.0, grid_points)
    vals = []
    for s in ss:
        vals.append(np.sum((p ** s) * (q ** (1.0 - s))))
    vals = np.array(vals, dtype=float)
    idx = int(np.argmin(vals))
    return float(vals[idx]), float(ss[idx])
