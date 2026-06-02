# Recommendations #4–#6: framing, theory, execution (for D-phase)

## #4 — Pre-empt "incremental over influence functions" (related-work paragraph)

**Relation to influence functions.** Influence functions (Koh & Liang 2017; Hampel
1974) quantify how *removing or up-weighting a training point* perturbs the trained
parameters or a test prediction, through the inverse *training* Hessian. AEGIS's
S_c differs on three counts. **(i)** It perturbs the *structure* (graph edges), not
the training set, and differentiates the *equilibrium* map z*(A) — an inference-time
fixed point — so the operative resolvent is (I−J_z)⁻¹ of the IGNN operator, not the
training Hessian. **(ii)** It is computed *matrix-free* at scale (Neumann + randomized
SVD, N=7,650), where influence functions need a parameter-Hessian inverse-solve.
**(iii)** It yields what an influence scalar cannot: a *sound certified radius*
(Thm Certify), the *SVD-optimal attack direction* (Prop attack), and a *closed-form
phase transition* (Thm 1). An influence score is one number per training point; S_c
is the full first-order *geometry* of the structural prediction map, from which
attack, defense, and certificate jointly follow. *(Composing the equilibrium
resolvent of S_c with a training-influence resolvent would give a structural
**poisoning** sensitivity — future work.)*

## #5 — The attack–defense duality as a proposition (turns the slogan into a theorem)

**Proposition (attack–defense coupling).** Let S_c=(I−J_z)⁻¹J_A P_c with leading
right singular pair (σ₁, v₁). For node v with margin m_v and head gap
Δw_v=W_{y_v}−W_c, the certified radius obeys ρ_v ≤ m_v/‖Δw_v S_{c,v}‖ (Prop Certify)
and the global worst-case first-order shift is ε·σ₁(S_c)/√2 (Prop attack). Both are
governed by the **same resolvent** (I−J_z)⁻¹:
- **(a) coupling.** The per-node sensitivity ‖S_{c,v}‖ that shrinks ρ_v is the v-block
  norm of the operator whose top singular value σ₁ is the attack gain; the
  least-certifiable nodes (smallest ρ_v) lie on the support of the optimal attack
  direction v₁.
- **(b) shared divergence.** As κ→1 (ε→ε_crit), ‖(I−J_z)⁻¹‖₂→∞, so σ₁→∞ (attack
  unbounded) *and* ρ_v→0 (no node certifiable) **simultaneously** — attacker's gain
  and defender's radius are reciprocal images of one resolvent, diverging at one
  phase transition.

So "where you can attack" and "what you can certify" are two readings of one object.
**Numerically verifiable** (cheap, no heavy GPU): (1) rank-correlate per-node ρ_v
against incidence on the top-|v₁| edges (expect strong negative correlation);
(2) along the κ-sweep, σ₁↑ and cert-fraction↓ track together toward ε_crit. I'll run
this as the validation for #5.

## #6 — 10-page execution plan (discipline, not content)

**Main (10pp) — the spine:**
- Intro + reframe (sensitivity geometry + certified attack–defense duality).
- Background + threat model.
- Theory: Thm 1 (phase transition) · Prop attack (SVD) · Prop transfer (the τ
  backbone) · **Prop Certify (NEW headline)** · duality prop (#5) · **Prop Universal
  (NEW, operator-agnostic)**.
- Experiments: tab:cross_domain (acc/κ/ε_crit) · four-quadrant (SVD≥PGD, 10/10) ·
  transfer τ 39-cell · **Certify non-vacuity + vs AGNNCert (#2)** · **regularized-
  defense frontier (#1, NEW — the working defense)** · **fraud case study (LEAD it,
  for ICDM-fit)**.
- Conclusion.

**Supplementary:** full proofs (Certify/transfer/duality/Universal) · **MonDEQ
breadth (#3)** · RL-value demo · AGNNCert full tables · σ₁-regularizer details ·
per-seed tables.

**Cut / shrink to reclaim space:** PF is gone (~1pp reclaimed → give it to Certify);
the defense section *shrinks* to the honest delocalization + the regularizer (the
defense that works), replacing the failed-portfolio discussion; Stackelberg-lite →
one paragraph (certified residual + beats centrality nulls) + supplementary.

**ICDM framing:** lead with *auditing a deployed fraud detector* (no labels, one
query) as the application; keep the geometry/certificate as the rigorous engine —
"applied security tooling with provable backing," not "a theory paper."
