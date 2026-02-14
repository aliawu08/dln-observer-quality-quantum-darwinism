from __future__ import annotations

import numpy as np

def set_seed(seed: int) -> np.random.Generator:
    """Create a reproducible NumPy RNG."""
    return np.random.default_rng(seed)

def random_pure_state(d: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-ish random pure state vector in C^d via complex Gaussian."""
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    v = v / np.linalg.norm(v)
    return v

def pure_density(psi: np.ndarray) -> np.ndarray:
    """|psi><psi|"""
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T

def random_density(d: int, rng: np.random.Generator) -> np.ndarray:
    """Random full-rank density matrix using the Ginibre ensemble."""
    X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = X @ X.conj().T
    rho = rho / np.trace(rho)
    # Ensure exact Hermiticity numerically
    rho = (rho + rho.conj().T) / 2
    return rho

def depolarizing_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Depolarizing CPTP map: Λ_p(ρ) = (1-p)ρ + p I/d."""
    d = rho.shape[0]
    return (1.0 - p) * rho + p * np.eye(d, dtype=complex) / d

def kron_all(mats: list[np.ndarray]) -> np.ndarray:
    """Kronecker product of a list of matrices."""
    out = np.array([[1.0]], dtype=complex)
    for M in mats:
        out = np.kron(out, M)
    return out
