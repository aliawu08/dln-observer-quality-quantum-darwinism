# Research Directions: Closing the DLN-QD Bridge
# DRAFT — for discussion before any paper changes

**Date:** 2026-02-10
**Status:** Proposal — not yet implemented

---

## Guiding Principles (Author's)

1. **Learning always has a cost.** Every transition in R_O, every monitoring
   episode, every coherence check dissipates resources. No free model revision.
   This must be a theorem, not an assumption.

2. **Network-Full fills the gap.** The complete revision cycle (factor
   compression + cumulative state tracking + structural revision) is the
   target. Partial structures (Linear-Plus, Network-standard,
   Network-NoContract) are comparison baselines that demonstrate why the full
   cycle is necessary.

3. **Topological invariance.** DLN stages are characterized by invariant
   signatures — the topology of G and R determines performance, independent
   of the specific algorithm. Quantifying the observer enables controlling
   quantum outcomes by designing the observer. Current experiments are
   "mostly linear at max."

---

## What We Have (Paper 4A, current preprint)

- Observer quality triple Q_O = (R_O, Lambda_O, tau_O)
- SBS-as-bipartite-factor-DAG (Proposition, Sec 3.3)
- Decoder hierarchy: Dec_D ⊊ Dec_L ⊊ Dec_N with strict nesting proof
- Tight exponent separation: xi_N / xi_L = 2 for pure states
- Stage-dependent redundancy: m_L = 2 m_N
- Calibration contraction (Thm 2): Lambda degrades Chernoff exponent
- epsilon-SBS robustness (Thm 3): additive error control
- Model space M_O and revision graph R_O definitions
- Bounded recovery proposition (qualitative)
- Monitoring protocol (L2.1-L2.3, descriptive)
- Central-spin worked example with explicit (R_O, C_O, tau_O)

## What We Need (the gap the editor identified)

Results where the DLN framework produces physics content that is not derivable
from quantum information theory alone. Specifically:

- A QUANTITATIVE result that depends on R_O topology (not just Q_O)
- A PREDICTION that is counterintuitive from the physics-only perspective
- COST terms that are bounded from below (Principle 1)

---

## Proposed New Results

### Result A: Dynamical Redundancy Bound (Direction 3)

**Theorem (Dynamical Redundancy under Non-Stationary Coherence).**

Let O be an observer with resource triple Q_O and revision graph R_O.
Suppose coherence conditions fluctuate: at each observation episode t,
the observer either has coherence (can implement Dec_N, probability
governed by a coherence process) or does not (restricted to Dec_L).

Let f_N denote the fraction of episodes in which the observer operates in
Dec_N. Define the time-averaged effective exponent:

    xi_eff(R_O) depends on the topology of R_O as follows:

(i) Full-cycle R_O (Dec_N <-> Dec_L, with monitoring):

    xi_eff = f_coh * xi_N + (1 - f_coh) * xi_L
             - C_monitor(f, W)

    where f_coh is the coherence fraction (determined by environment
    physics), and C_monitor > 0 is the monitoring overhead (Principle 1:
    never zero). The monitoring cost comes from reserving a fraction f of
    fragments for diagnostics, reducing the effective decoding set.

    Explicit form: C_monitor = f * [f_coh * xi_N + (1 - f_coh) * xi_L]
    (the exponent contribution lost by diverting fragments to monitoring).

    Net: xi_eff(full-cycle) = (1 - f) * [f_coh * xi_N + (1 - f_coh) * xi_L]

(ii) Expand-only R_O (Dec_N -> Dec_L, no return):

    After the first coherence loss event (expected time ~ 1/lambda_loss),
    the observer is permanently in Dec_L:

    xi_eff(expand-only) -> xi_L   as T -> infinity

    regardless of subsequent coherence recovery.

(iii) Fixed Dec_L (trivial R_O):

    xi_eff(fixed-L) = xi_L   always.

(iv) Fixed Dec_N without monitoring (trivial R_O, but using collective
     decoder):

    xi_eff(fixed-N, no monitor) = f_coh * xi_N + (1 - f_coh) * xi_garbage

    where xi_garbage <= 0 represents the exponent when a collective POVM
    designed for coherent states is applied to a decohered register. This
    can be NEGATIVE (error rate worse than random guessing) when the POVM
    is sufficiently mismatched.

**Key comparisons (the content that requires DLN):**

- Full-cycle > expand-only whenever coherence recovers at least once.
  The gap is determined by the R_O topology (cyclic vs acyclic).
  [This result requires the revision graph formalism.]

- Full-cycle > fixed Dec_N without monitoring whenever f_coh < 1.
  [This is the inverted sophistication prediction — Direction 5.]

- The monitoring cost C_monitor is the price of Principle 1.
  Full-cycle pays this cost and still dominates because the cost of
  NOT monitoring (xi_garbage) is worse.

**Dynamical redundancy requirement:**

    m_eff(delta, R_O) = log(1/(2*delta)) / xi_eff(R_O)

This interpolates between:
    m_N = log(1/(2*delta)) / xi_N   (always coherent, no monitoring needed)
    m_L = 2 * m_N                    (never coherent)

For the full-cycle observer:
    m_N < m_eff(full-cycle) < m_L

The exact position depends on f_coh and f (monitoring fraction).

**Why this is new:** Static QD gives m_N or m_L. The dynamical bound
gives m_eff as a function of the observer's revision architecture AND
the environment's coherence statistics. This does not exist in the
literature.


### Result B: Inverted Sophistication Theorem (Direction 5)

**Theorem (Collective Measurement Without Monitoring Can Be Worse Than
Product Measurement).**

Let the conditional fragment states be pure with overlap c in (0,1).
Suppose coherence is available with probability f_coh < 1, and when
coherence is unavailable, the collective POVM acts on a partially
decohered register.

There exists a critical coherence fraction f* in (0,1) such that:

    For f_coh < f*:
        P_e(Dec_N, no monitoring, m fragments)
        > P_e(Dec_L, m fragments)

That is, an observer committed to collective measurement WITHOUT
monitoring has strictly HIGHER error than an observer using product
measurement, whenever coherence is sufficiently unreliable.

The critical threshold:
    f* = xi_L / (xi_L - xi_garbage)

where xi_garbage is the (possibly negative) effective exponent of the
collective POVM applied to a decohered register.

**The DLN resolution:** A full-cycle Network observer (with monitoring)
avoids this trap by detecting coherence loss and switching to Dec_L.
Its error rate is always <= min(P_e(Dec_N when coherent), P_e(Dec_L)),
minus the monitoring overhead.

**Connection to Paper 1:** This is the exact QD analog of the
catastrophic failure in Paper 1 Sec 6.4-6.5:
- Linear-Plus at -49,000 = Dec_N without monitoring under regime change
- Linear at -9,400 = Dec_L under regime change
- Network-Full at +60 to +106 = Dec_N with full revision cycle

The mechanism is identical: exploiting structure (factor / coherence)
without verifying that the structure is still valid leads to worse
outcomes than not exploiting it at all.

**Why this is new and testable:** Standard QD says collective >= product,
always. This theorem says: ONLY IF the observer monitors coherence.
Without monitoring, the ordering can invert. This is a prediction that
(a) follows from DLN, (b) is not derivable from static QD analysis,
and (c) is testable in any QD experiment with fluctuating decoherence.


### Result C: Stage-Dependent Pointer Accessibility (Q6, elevated)

**Proposition (DLN Stage Constrains Accessible Pointer Observables).**

The pointer observable that an observer can resolve depends on their
DLN stage:

(i) A Dec_D observer (memoryless, single-fragment) can distinguish
    pointer values x, x' only if the single-fragment distinguishability
    is sufficient:
        ||sigma_k^(x) - sigma_k^(x')||_1 > 2(1 - 2*delta)
    for at least one accessible fragment k. This limits Dec_D to
    COARSE pointer observables — those that produce large per-fragment
    distinguishability.

(ii) A Dec_L observer (product measurement, m fragments) can resolve
     finer pointer distinctions by accumulating evidence across fragments.
     The resolution limit is set by the product-measurement Chernoff
     exponent: pointer values x, x' are resolvable if
         sum_k xi_k^(L)(x, x') >= log(1/(2*delta))

(iii) A Dec_N observer (collective measurement) can resolve the FINEST
      pointer distinctions, achieving the quantum Chernoff limit:
          sum_k xi_k^(N)(x, x') >= log(1/(2*delta))
      with xi_k^(N) = 2 * xi_k^(L) for pure states.

**Consequence:** For a fixed fragment budget m, a Dec_N observer can
resolve pointer distinctions that are invisible to a Dec_L observer,
and a Dec_L observer can resolve distinctions invisible to Dec_D.

**The design implication (Principle 3):** The effective pointer basis
is not a fixed property of the system-environment interaction alone —
it is jointly determined by the physics (decoherence/SBS formation) AND
the observer's DLN stage. By upgrading the observer's stage, we gain
access to finer pointer information from the same quantum environment.

This means: einselection (the environment selecting the pointer basis)
is necessary but not sufficient. The observer's topology determines
which of the environment-selected pointer values are actually accessible.

**Connection to "mostly linear at max":** Current experiments resolve
only the pointer observables accessible to Dec_L. A Dec_N observer
with the same fragment access would resolve a strictly larger set of
pointer values from the same environment. This is not a theoretical
curiosity — it's a concrete capability gap.


---

## Proposed Paper Structure (additions to current preprint)

### Option 1: Extend Paper 4A

Add to current preprint:
- New Sec 3.3.5: "Dynamical redundancy under non-stationary coherence"
  (Result A — theorem + proof)
- New Sec 3.3.6: "Inverted sophistication: when collective measurement
  fails" (Result B — theorem + proof)
- Extend Sec 3.3.1 or add Sec 3.3.7: "Stage-dependent pointer
  accessibility" (Result C — proposition)
- Extend Sec 7 (central-spin): compute xi_garbage, f*, C_monitor for
  the central-spin model as concrete numerical examples
- Update Discussion (Sec 8) to emphasize the three new results

Estimated addition: ~4-6 pages in preprint format.

### Option 2: Separate follow-up paper

Keep Paper 4A as the foundation (observer quality, static bounds,
DLN classification). Write Paper 4B focusing on:
- Dynamical redundancy (Result A)
- Inverted sophistication (Result B)
- Stage-dependent pointer accessibility (Result C)
- Extended central-spin numerics
- Experimental predictions

This separates the "framework" paper from the "predictions" paper.

### Recommendation

Option 1 is stronger for the PRA editor's concern: "what does DLN
tell you that physics doesn't?" — the answer (Results A, B, C) should
be in the same paper as the framework, not deferred.

Option 2 is safer if Paper 4A is already at a natural length limit
or if the author prefers to submit the current framework first and
the predictions second.

---

## Questions to Answer (prioritized)

### Must Answer (required for Results A and B)

Q1: What is xi_garbage explicitly?
    When a collective POVM designed for coherent pure states is applied
    to a partially decohered (mixed) register, what is the resulting
    error exponent? Compute for the central-spin model.

Q2: What is the critical coherence fraction f*?
    Derive the threshold below which Dec_N-without-monitoring is worse
    than Dec_L. Compute numerically for the central-spin model.

Q3: Is the ordering xi_eff(full-cycle) > xi_eff(expand-only) >
    xi_eff(fixed-L) > xi_eff(fixed-N-no-monitor, f_coh < f*)
    provable for general coherence processes, or model-specific?

### Should Answer (strengthens the bridge)

Q4: Can the mixed-state exponent ratio be expressed as a function of
    a single compressibility parameter?
    (Would formally close the bridge between Paper 1's K/F ratio and
    Paper 4A's exponent ratio.)

Q5: What is the thermodynamic lower bound on C_monitor via Landauer?
    (Grounds Principle 1 in fundamental physics.)

### Important (Principle 3 payoff)

Q6: For the central-spin model, what pointer distinctions can Dec_N
    resolve that Dec_L cannot?
    Compute the resolution gap explicitly. Show that for a given m and
    delta, Dec_N can distinguish pointer values x, x' that Dec_L
    cannot — and quantify the gap.

---

## Computational Work Needed

1. **Extend central_spin_example.py** to compute:
   - xi_garbage(t) for a decohered register under a collective POVM
   - f*(t, c) as a function of time and overlap
   - xi_eff(t, f_coh, f) for the dynamical bound
   - Resolution comparison between Dec_L and Dec_N

2. **New figures** for the paper:
   - xi_eff vs f_coh for different R_O topologies (full-cycle, expand-only,
     fixed-L, fixed-N-no-monitor) — this is the "money plot" for Direction 3
   - P_e comparison showing the inverted sophistication crossover — the
     "money plot" for Direction 5
   - Pointer resolution gap between Dec_L and Dec_N — the "money plot" for Q6

3. **New tests** to verify the bounds numerically.

---

## What This Achieves

If Results A, B, and C are proved and computed for the central-spin model:

1. **The PRA editor's question is answered.** DLN tells us:
   (A) how to compute redundancy under non-stationary conditions
       (requires R_O — not available from static QD),
   (B) that collective measurement without monitoring can be WORSE than
       product measurement (counterintuitive, testable, requires DLN),
   (C) that observer topology determines which pointer values are
       accessible (makes einselection observer-dependent in a
       quantifiable way).

2. **All three author principles are satisfied.**
   - Cost of learning enters explicitly as C_monitor > 0 (Principle 1)
   - Network-Full is the only observer achieving optimal xi_eff (Principle 2)
   - Results depend on R_O topology, not on specific algorithm (Principle 3)

3. **The bridge is load-bearing.** The results are derived FROM the DLN
   framework (revision graph formalism, monitoring protocol, stage
   classification). They are not re-derivable from quantum Chernoff
   theory alone because they concern DYNAMICS on measurement strategies,
   not static measurement performance.

---

*End of draft. For discussion before implementation.*
