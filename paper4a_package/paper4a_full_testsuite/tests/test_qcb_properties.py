import math
import numpy as np

from paper4a_qd.utils import set_seed, random_density, depolarizing_channel, kron_all
from paper4a_qd.distances import trace_distance
from paper4a_qd.qcb import qcb_coefficient


def test_trace_distance_contraction_under_depolarizing(tol):
    rng = set_seed(123)
    for _ in range(25):
        rho = random_density(2, rng)
        sigma = random_density(2, rng)
        p = float(rng.uniform(0.0, 0.8))
        D0 = trace_distance(rho, sigma)
        D1 = trace_distance(depolarizing_channel(rho, p), depolarizing_channel(sigma, p))
        assert D1 <= D0 + tol


def test_qcb_dpi_under_depolarizing(tol):
    rng = set_seed(321)
    for _ in range(15):
        rho = random_density(2, rng)
        sigma = random_density(2, rng)
        p = float(rng.uniform(0.0, 0.7))
        Q0, _ = qcb_coefficient(rho, sigma, grid_points=2001, refine=True)
        Q1, _ = qcb_coefficient(
            depolarizing_channel(rho, p),
            depolarizing_channel(sigma, p),
            grid_points=2001,
            refine=True,
        )
        # DPI says Q1 >= Q0 (coefficient increases / exponent decreases).
        assert Q1 + tol >= Q0


def test_qcb_iid_tensor_power_property(tol):
    """For i.i.d. tensor powers of the same hypothesis pair, QCB coefficient factorizes exactly:
      Q(ρ^{⊗2}, σ^{⊗2}) = Q(ρ,σ)^2
    because the minimizer s is shared.
    """
    rng = set_seed(999)
    for _ in range(10):
        rho = random_density(2, rng)
        sigma = random_density(2, rng)

        Q, _ = qcb_coefficient(rho, sigma, grid_points=2001, refine=True)

        rho2 = kron_all([rho, rho])
        sigma2 = kron_all([sigma, sigma])
        Q2, _ = qcb_coefficient(rho2, sigma2, grid_points=2001, refine=True)

        assert abs(Q2 - (Q * Q)) <= 200 * tol
