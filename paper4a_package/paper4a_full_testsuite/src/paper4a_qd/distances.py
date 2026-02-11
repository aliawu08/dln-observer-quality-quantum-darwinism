from __future__ import annotations

import numpy as np

def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Trace distance D_tr(ρ,σ) = 1/2 ||ρ-σ||_1.

    For Hermitian A = ρ-σ, ||A||_1 = sum_i |λ_i(A)|.
    """
    A = (rho - sigma)
    A = (A + A.conj().T) / 2  # enforce Hermiticity numerically
    eigvals = np.linalg.eigvalsh(A)
    return 0.5 * float(np.sum(np.abs(eigvals)))

def fidelity_pure(psi: np.ndarray, phi: np.ndarray) -> float:
    """Fidelity for pure states: |<psi|phi>|^2."""
    return float(np.abs(np.vdot(psi, phi)) ** 2)
