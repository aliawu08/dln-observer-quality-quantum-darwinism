import numpy as np

from paper4a_qd.utils import set_seed, random_density
from paper4a_qd.distances import trace_distance


def random_projective_povm(d: int, rng: np.random.Generator) -> list[np.ndarray]:
    # Random unitary via QR of complex Gaussian
    X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(X)
    # Projectors onto columns of Q
    povm = []
    for k in range(d):
        v = Q[:, k].reshape(d, 1)
        P = v @ v.conj().T
        povm.append(P)
    return povm


def test_event_probability_continuity_under_trace_distance(tol):
    rng = set_seed(2026)
    for _ in range(30):
        rho = random_density(2, rng)
        # Make sigma close to rho by convex mixing with another state
        tau = random_density(2, rng)
        eps = float(rng.uniform(0.0, 0.2))
        sigma = (1.0 - eps) * rho + eps * tau
        sigma = (sigma + sigma.conj().T) / 2
        sigma = sigma / np.trace(sigma)

        D = trace_distance(rho, sigma)

        povm = random_projective_povm(2, rng)
        # Choose an event E = {outcome 0}
        M = povm[0]
        p_rho = float(np.real_if_close(np.trace(M @ rho)).real)
        p_sig = float(np.real_if_close(np.trace(M @ sigma)).real)
        assert abs(p_rho - p_sig) <= D + tol
