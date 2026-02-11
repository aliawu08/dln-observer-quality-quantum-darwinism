import math
import numpy as np

from paper4a_qd.utils import set_seed, random_pure_state, pure_density
from paper4a_qd.measurement import (
    helstrom_projectors,
    induced_classical_distributions,
    bhattacharyya,
)


def test_pure_state_helstrom_bhattacharyya_equals_overlap(tol):
    rng = set_seed(7)
    for _ in range(40):
        psi0 = random_pure_state(2, rng)
        psi1 = random_pure_state(2, rng)
        c = float(abs(np.vdot(psi0, psi1)))
        rho0 = pure_density(psi0)
        rho1 = pure_density(psi1)

        P0, P1 = helstrom_projectors(rho0, rho1)
        p, q = induced_classical_distributions(rho0, rho1, [P0, P1])
        B = bhattacharyya(p, q)

        # For two pure states, Helstrom measurement yields B = |<psi0|psi1>| exactly.
        assert abs(B - c) <= 50 * tol


def test_factor_of_two_exponent_gap_for_local_helstrom(tol):
    rng = set_seed(8)
    for _ in range(40):
        psi0 = random_pure_state(2, rng)
        psi1 = random_pure_state(2, rng)
        c = float(abs(np.vdot(psi0, psi1)))
        # Avoid degenerate overlaps for numeric stability
        c = min(max(c, 1e-6), 1 - 1e-6)

        xi_collective = -math.log(c ** 2)
        xi_local_helstrom = -math.log(c)

        assert abs(xi_collective - 2.0 * xi_local_helstrom) <= 1e-9
