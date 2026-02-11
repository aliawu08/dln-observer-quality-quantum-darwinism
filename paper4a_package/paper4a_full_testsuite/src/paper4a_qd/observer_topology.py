from __future__ import annotations

import math

def xi_typical_quenched(f: float, xi_good: float, xi_bad: float) -> float:
    """Typical/quenched exponent: E[ -log(2 P_e,t) ] / m.

    If the per-episode exponents are (xi_good, xi_bad) and episodes are i.i.d.
    with good fraction f, then xi_eff = f*xi_good + (1-f)*xi_bad.
    """
    return f * xi_good + (1.0 - f) * xi_bad

def xi_ensemble_annealed_from_errors(
    f: float,
    m: int,
    twoPe_good: float,
    twoPe_bad: float,
) -> float:
    """Ensemble/annealed exponent: -(1/m) log( E[2 P_e,t] ).

    Here episodes are i.i.d. and in a good episode 2P_e = twoPe_good,
    in a bad episode 2P_e = twoPe_bad.
    """
    val = f * twoPe_good + (1.0 - f) * twoPe_bad
    val = max(val, 1e-300)
    return float(-(1.0 / m) * math.log(val))

def twoPe_from_exponent(xi: float, m: int) -> float:
    """Model the (bound) 2 P_e ≈ exp(-xi*m)."""
    return float(math.exp(-xi * m))

def xi_full_cycle_ensemble(
    f: float,
    m: int,
    xiN: float,
    xiL: float,
    C_monitor: float,
) -> float:
    """Ensemble exponent for a full-cycle policy with monitoring budget.

    We model monitoring as consuming m_diag = ceil(C_monitor*m) fragments,
    leaving m_eff = m - m_diag fragments for decoding. In good episodes we
    use collective exponent xiN, in bad episodes robust product exponent xiL.

    The ensemble exponent is computed from:
      2E[P_e] = f e^{-xiN*m_eff} + (1-f) e^{-xiL*m_eff}
      xi_ens = -(1/m) log(2E[P_e]).
    """
    m_diag = int(math.ceil(C_monitor * m))
    m_eff = max(m - m_diag, 1)
    twoPe_good = twoPe_from_exponent(xiN, m_eff)
    twoPe_bad = twoPe_from_exponent(xiL, m_eff)
    return xi_ensemble_annealed_from_errors(f=f, m=m, twoPe_good=twoPe_good, twoPe_bad=twoPe_bad)

def xi_unmonitored_collective_ensemble(
    f: float,
    m: int,
    xiN: float,
) -> float:
    """Ensemble exponent for an unmonitored collective decoder under observer-side decoherence.

    With probability f, the coherent collective measurement works and yields
    2P_e ≈ exp(-xiN*m). With probability (1-f), the apparatus decoheres and
    returns random (uninformative) outcomes, giving 2P_e = 1.

      2E[P_e] = f e^{-xiN*m} + (1-f)*1
      xi_ens = -(1/m) log(2E[P_e]).
    """
    twoPe_good = twoPe_from_exponent(xiN, m)
    twoPe_bad = 1.0
    return xi_ensemble_annealed_from_errors(f=f, m=m, twoPe_good=twoPe_good, twoPe_bad=twoPe_bad)
