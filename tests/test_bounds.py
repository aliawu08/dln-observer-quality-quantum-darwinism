import math
import numpy as np

from scripts.central_spin_example import per_qubit_xi, per_qubit_overlap, min_qubits_for_target


def test_xi_monotone_in_overlap():
    c1 = np.array([0.9, 0.5, 0.1])
    c2 = np.array([0.95, 0.6, 0.2])
    xi1 = per_qubit_xi(c1)
    xi2 = per_qubit_xi(c2)
    # larger overlap -> smaller xi
    assert np.all(xi2 < xi1)


def test_min_qubits_for_target_basic():
    xi = np.array([1.0, 0.5, 0.25, 0.125])
    target = 1.2
    m = min_qubits_for_target(xi, target=target, m_max=4)
    # sorted xi: 1.0 + 0.5 >= 1.2 => m=2
    assert m == 2


def test_overlap_formula_range():
    g = np.array([0.1, 1.0, 2.0])
    t = 0.37
    c = per_qubit_overlap(g, t)
    assert np.all(c >= 0.0) and np.all(c <= 1.0)
