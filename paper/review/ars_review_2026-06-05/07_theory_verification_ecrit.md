# Theory Verification — ε_crit Subcritical Certificate (thm:phase_transition, regime (a))

**Date:** 2026-06-05
**Scope:** Independent adversarial verification of one hypothesized soundness gap in the
subcritical contraction certificate. Verified line-by-line against the primary LaTeX
(`theory.tex`, `appendix/B_sensitivity.tex`, `appendix/A_preliminaries.tex`, `background.tex`)
plus rng-seeded numerics. This pass does **not** defer to the prior reviewer's framing; it
reaches its own verdict and refines the prior R4 finding.

---

## Claim under test

`thm:phase_transition`(a) asserts: under (A1)–(A3) **alone**, with κ := ‖J_z‖₂ and
ε_crit := (1−κ)/‖W‖₂, every structural perturbation with ‖δÂ‖_F = ε < ε_crit keeps the
perturbed operator a contraction with a unique fixed point. The proof's Step 2 (`eq:Jzp-bound`)
bounds ‖J_z'‖₂ ≤ ‖Â'‖₂‖W‖₂ ≤ (‖Â‖₂+ε)‖W‖₂ and asserts the RHS < 1 "precisely when
ε < 1/‖W‖₂ − ‖Â‖₂ = ε_crit."

Hypothesized gap: the equality `1/‖W‖₂ − ‖Â‖₂ = (1−κ)/‖W‖₂` requires κ = ‖Â‖₂‖W‖₂, which
holds only in the all-active case (A4); under partial ReLU κ can be strictly smaller, so the
stated ε_crit exceeds the radius `eq:Jzp-bound` actually proves.

---

## Independent analysis

### Point 1 — κ := ‖J_z‖₂ is unambiguous, and ‖J_z‖₂ < ‖Â‖₂‖W‖₂ is achievable (CONFIRMED)

κ is defined unambiguously as ‖J_z‖₂ in three places:
- `A_preliminaries.tex` (A3): "κ := ‖J_z‖₂ < 1, equivalently ‖Â‖₂‖W‖₂ < 1."
- `tab:notation`: "κ … contraction factor ‖J_z‖₂ < 1."
- `background.tex`: "κ = ‖J_z‖₂", and line 13 writes κ ≤ ‖Â‖₂‖W‖₂ < 1 with an **explicit ≤**.

The parenthetical "equivalently ‖Â‖₂‖W‖₂ < 1" is **not** a redefinition of κ; it is a second,
strictly stronger *sufficient* condition for contraction. The implication ‖Â‖₂‖W‖₂ < 1 ⇒ κ < 1
is one-directional; the two are equivalent only all-active. So the parenthetical does not rescue
the equality in `eq:Jzp-bound`.

Since J_z = diag(φ')(Â⊗W) with diag(φ') a 0/1 projection, ‖J_z‖₂ = ‖diag(φ')(Â⊗W)‖₂ ≤
‖Â⊗W‖₂ = ‖Â‖₂‖W‖₂, with equality iff the mask retains a maximizing singular direction of Â⊗W.
A 0/1 mask that zeros the output coordinate carrying the dominant left singular vector of Â⊗W
makes it strict.

**Numeric (seed 20260605):** ‖Â‖₂‖W‖₂ = 0.900; all-active κ = 0.900 (equality); with a single
unit on the dominant direction turned off, κ = 0.538 — strictly below the bound by 0.362.
**Confirmed:** κ can be strictly < ‖Â‖₂‖W‖₂ for a partially-active ReLU model.

### Point 2 — `eq:Jzp-bound` proves the smaller radius, not ε_crit (CONFIRMED)

The chain ‖J_z'‖₂ ≤ ‖Â'‖₂‖W‖₂ ≤ (‖Â‖₂+ε)‖W‖₂ forces ‖J_z'‖₂ < 1 only for
ε < **ε_suff := 1/‖W‖₂ − ‖Â‖₂**. The asserted identity ε_suff = (1−κ)/‖W‖₂ = ε_crit rearranges
to κ = ‖Â‖₂‖W‖₂ — exactly the all-active condition (A4). Under the *reported* operating regime
(partial ReLU, κ ∈ [0.14, 0.59] per `tab:cross_domain`, ‖Â‖₂ ≈ 1 for a symmetric-normalized
adjacency), the identity is false and ε_crit > ε_suff strictly.

**Numeric:** with ‖W‖₂ = 0.9, ‖Â‖₂ = 1: ε_suff = 0.111 (= all-active ε_crit, identity holds);
but with the reported partial κ = 0.538, the *stated* ε_crit = (1−κ)/‖W‖₂ = 0.513 — exceeding
the proven-safe radius by 0.402. On (ε_suff, ε_crit] the written proof establishes **no**
contraction bound, yet the certificate claims safety there. The "(A1)–(A3) alone" header is
contradicted: the equality silently consumes (A4).

This reproduces and confirms prior finding **R4 CRITICAL-1** (catalogued 2026-05-30) from the
primary source: line 56's step "RHS < 1 precisely when ε < 1/‖W‖₂ − ‖Â‖₂ = ε_crit" is the false
link, and line 70's downstream "1 − ‖J_z'‖₂ = ‖W‖₂(ε_crit − ε)" inherits the same all-active
dependence.

### Point 3 — the triangle-bound "fix" is ALSO unsound over a finite perturbation (REFINEMENT — new vs R4)

The natural repair ‖J_z'‖₂ ≤ ‖J_z‖₂ + ‖J_z'−J_z‖₂ ≤ κ + ε‖W‖₂ requires
J_z'−J_z = diag(φ')(δÂ⊗W) with a **single** activation mask shared by z* and z*'. Over a finite
ε up to ε_crit (which is O(1): the paper's per-dataset values reach ~0.66, 0.86) the equilibrium
crosses ReLU regions and the mask changes. Writing M for the mask at z* and M' for the mask at
the perturbed equilibrium z*', the correct decomposition is

  J_z' − J_z = M'(δÂ⊗W) + **(M'−M)(Â⊗W)**,

so the triangle route actually yields ‖J_z'‖₂ ≤ κ + ε‖W‖₂ + ‖(M'−M)(Â⊗W)‖₂. The mask-change
term is exactly what the single-mask fix drops, and it is **not** o(1) in ε: a newly activated
region can switch on units that were dormant at z*, injecting a contribution up to O(‖Â‖₂‖W‖₂).

**Numerics.**
- *Region crossing is real, not hypothetical (seed 7):* for a partially-active model with
  κ = 0.619, ε_crit = 0.544, the mask first flips at ε ≈ 0.22 — far inside the certified ball.
  Step 1's mask-stability argument ("stable for small ‖δÂ‖", measure-zero faces) is an
  infinitesimal/local statement and does **not** cover the finite range ε < ε_crit.
- *Triangle bound fails adversarially (seed 101, 4000 partial-ReLU models):* worst case found
  κ = 0.320, ε = 0.020 (well inside ε_crit = 0.883), true ‖J_z'‖₂ = **0.625** while the proposed
  bound κ + ε‖W‖₂ = **0.335**. The mask flipped and (M'−M)(Â⊗W) added ≈ 0.29. The triangle "fix"
  is violated by +0.29 — it is **not** a valid upper bound on ‖J_z'‖₂.

(In a single benign random direction the triangle bound happened to hold at every step; that is a
non-adversarial accident, not a proof. The adversarial search refutes it.)

**Consequence.** κ carries essentially no certified control over ‖J_z'‖₂ once the region changes,
because ‖J_z'‖₂ is governed by the mask M' **at the perturbed equilibrium**, not by κ (the mask at
the original equilibrium) plus a small increment. Of the two candidate bounds:
- mask-fixed triangle (κ + ε‖W‖₂): UNSOUND for finite partial-ReLU ε (mask-change term dropped);
- mask-agnostic (‖Â'‖₂‖W‖₂): SOUND and unconditional, because M' is a projection so
  ‖J_z'‖₂ = ‖M'(Â'⊗W)‖₂ ≤ ‖Â'‖₂‖W‖₂ for **every** mask — but it delivers only the smaller radius
  ε_suff = 1/‖W‖₂ − ‖Â‖₂, not ε_crit.

So **neither** bound cleanly delivers the stated ε_crit for a finite, partially-active
perturbation. The mask-agnostic bound is the only clean certificate and it equals ε_crit only
all-active.

### Point 4 — net verdict

This is a **real soundness gap as written**, not a benign typo: the certificate asserts safety on
a nonempty interval (ε_suff, ε_crit] on which the written proof establishes nothing, and the
defect is the same norm-vs-actual-contraction / all-active-vs-partial-ReLU conflation already
catalogued three times in this paper (R4 CRITICAL-1; the ‖z*‖ finiteness denominator; the
`thm:cf2s` upper side). It is *fixable* — the repair is one line and strictly shrinks the
certified radius without new assumptions — so it sits between (a) and (b): the math is recoverable,
but the current statement over-claims and the header's "(A1)–(A3) alone" is false for the written
ε_crit. Crucially, the obvious triangle-bound patch a reviewer would propose is **itself wrong**
over the finite ball, so a hostile reviewer who probes one step deeper than the surface gap will
find the paper has no correct proof of the stated radius on file.

---

## Verdict + severity

**Verdict: CONFIRM the gap, and REFINE it.** The prior finding (stated ε_crit exceeds the proven
radius; "(A1)–(A3) alone" silently uses (A4)) is correct and reproduced from source. The
refinement: the natural single-mask triangle repair is **also unsound** for finite partial-ReLU
perturbations because the equilibrium crosses ReLU regions inside the certified ball (mask flips at
ε ≈ 0.22 ≪ ε_crit = 0.54), injecting a dropped (M'−M)(Â⊗W) term (adversarial violation +0.29).
The only sound, assumption-clean certificate is the mask-agnostic radius ε_suff = max(0,
1/‖W‖₂ − ‖Â‖₂) ≤ ε_crit.

**Severity: MAJOR (from a hostile AAAI theory reviewer's seat).** The flagship "sufficient safety
boundary" is the headline of `thm:phase_transition`; as written it is unsound on a nonempty,
realistically-reachable interval, and the natural fix fails too. It is not CRITICAL-fatal because a
correct (smaller) certificate exists with a one-line change and the *qualitative* three-regime
picture survives. **It should block acceptance in the current wording** — a referee can exhibit a
partial-ReLU model where the certified-safe interval contains a budget at which ‖J_z'‖₂ > 1, and a
revised version with the corrected radius (or an explicit all-active scoping) is required before the
subcritical certificate can be trusted.

---

## Minimal correct fix

Replace the certified radius and Step-2 wording. Two clean options; (1) is preferred (keeps the
full nonlinear ReLU model, consumes only A1–A3):

**(1) Mask-agnostic radius (recommended).** Define
  ε_crit^suff := max(0, 1/‖W‖₂ − ‖Â‖₂),
and state Step 2 as: M' = diag(φ') at z*' is a 0/1 projection, hence unconditionally
‖J_z'‖₂ = ‖M'(Â'⊗W)‖₂ ≤ ‖Â'‖₂‖W‖₂ ≤ (‖Â‖₂ + ε)‖W‖₂ < 1 for ε < ε_crit^suff, for **every** φ'.
No single-mask assumption, no region-stability over the finite ball, no (A4). Demote
(1−κ)/‖W‖₂ to a (generally larger) "operating margin / spectral-radius-side heuristic," not a
certified contraction radius. Consumes: A1 (‖diag(φ')‖₂ ≤ 1), A2 (‖W‖₂ ≤ c), and the threat-model
bound ‖δÂ‖₂ ≤ ‖δÂ‖_F = ε. Does **not** consume A3 (κ is not needed for the radius) or A4.

**(2) Keep (1−κ)/‖W‖₂ but scope to all-active.** Add (A4) to `thm:phase_transition`(a) so
κ = ‖Â‖₂‖W‖₂ and the identity ε_suff = (1−κ)/‖W‖₂ becomes valid, and correct the header's
"every certificate uses (A1)–(A3) alone." Honest but weaker: it ties the headline safety boundary
to the all-active assumption the paper elsewhere tries to avoid for certificates.

**Do NOT** adopt the triangle bound ‖J_z'‖₂ ≤ κ + ε‖W‖₂: it is unsound over finite partial-ReLU
ε (shown above), because it silently assumes a single activation mask across a region boundary the
equilibrium provably crosses inside the certified ball.
