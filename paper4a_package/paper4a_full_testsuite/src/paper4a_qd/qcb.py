from __future__ import annotations

import math
import numpy as np

def _matrix_power_psd(rho: np.ndarray, power: float, eps: float = 1e-15) -> np.ndarray:
    """Fractional power for PSD matrices using eigen-decomposition.

    Parameters
    ----------
    rho:
        Hermitian PSD matrix.
    power:
        Exponent in [0,1] typically.
    eps:
        Eigenvalue floor to avoid NaNs for 0^power when power=0 etc.

    Notes
    -----
    - For eigenvalues extremely close to 0, we clamp to eps.
    - This is numerically stable for the low-dimensional (qubit / few-qubit)
      matrices used throughout this package.
    """
    rhoH = (rho + rho.conj().T) / 2
    w, V = np.linalg.eigh(rhoH)
    w = np.clip(w.real, 0.0, None)
    # For power == 0, interpret rho^0 on support as projector; eps clamp is fine here.
    w_safe = np.maximum(w, eps)
    w_p = w_safe ** power
    return (V @ np.diag(w_p) @ V.conj().T)

def qcb_trace_term(rho: np.ndarray, sigma: np.ndarray, s: float) -> float:
    """Compute Tr[rho^s sigma^(1-s)] for PSD rho,sigma."""
    A = _matrix_power_psd(rho, s)
    B = _matrix_power_psd(sigma, 1.0 - s)
    val = np.trace(A @ B)
    # Numerical noise may produce a tiny imaginary part
    return float(np.real_if_close(val, tol=1e6).real)

def qcb_coefficient(
    rho: np.ndarray,
    sigma: np.ndarray,
    grid_points: int = 2001,
    refine: bool = True,
) -> tuple[float, float]:
    """Quantum Chernoff coefficient Q(ρ,σ) = min_{s∈[0,1]} Tr[ρ^s σ^{1-s}].

    Returns
    -------
    (Q, s_opt)
    """
    # coarse grid
    ss = np.linspace(0.0, 1.0, grid_points)
    vals = np.array([qcb_trace_term(rho, sigma, float(s)) for s in ss], dtype=float)
    idx = int(np.argmin(vals))
    s_best = float(ss[idx])
    q_best = float(vals[idx])

    if not refine:
        return q_best, s_best

    # Local refinement around best grid point using golden-section search
    # This assumes local near-unimodality around the minimum, which is sufficient
    # for our low-dim test validation.
    left = float(ss[max(idx - 1, 0)])
    right = float(ss[min(idx + 1, grid_points - 1)])

    # Expand a bit to avoid edge lock
    left = max(0.0, left - (right - left))
    right = min(1.0, right + (right - left))

    phi = (1 + 5 ** 0.5) / 2
    invphi = 1 / phi

    a, b = left, right
    c = b - (b - a) * invphi
    d = a + (b - a) * invphi

    fc = qcb_trace_term(rho, sigma, c)
    fd = qcb_trace_term(rho, sigma, d)

    for _ in range(40):  # plenty for double precision
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * invphi
            fc = qcb_trace_term(rho, sigma, c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * invphi
            fd = qcb_trace_term(rho, sigma, d)

    s_opt = (a + b) / 2
    q_opt = qcb_trace_term(rho, sigma, s_opt)
    if q_opt < q_best:
        return float(q_opt), float(s_opt)
    return q_best, s_best

def qcb_exponent(rho: np.ndarray, sigma: np.ndarray, **kwargs) -> float:
    """Chernoff exponent ξ(ρ,σ) = -log Q(ρ,σ)."""
    Q, _ = qcb_coefficient(rho, sigma, **kwargs)
    # Clip to avoid log(0) in pathological numeric cases
    Q = max(Q, 1e-300)
    return float(-math.log(Q))
