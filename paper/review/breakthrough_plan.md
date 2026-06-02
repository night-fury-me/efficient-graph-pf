# AEGIS → Breakthrough: Research North Star

**Goal (non-negotiable, set 2026-06-02):** elevate AEGIS from a competent diagnostics
paper (borderline-accept) to a **breakthrough** result. **Deadlines and effort are
explicitly NOT constraints.** The only constraint is **truth**: a breakthrough claim that
collapses under scrutiny is worse than no claim. We pursue the deepest result the
mathematics actually supports, and we let the *proofs* — not our hopes — set the boldness
of the final claim. This raises the bar (more ruthless about rigor), it does not lower it.

## Why "breakthrough" needs new theory (not integration)
The *integrated* paper tops out at borderline-accept because: (1) the rigorous core is
**narrow** (contractive-IGNN-only; broad claims are empirical, not proven); (2) novelty is
**synthesis-driven** (IFT sensitivity + curvature certificate + Zargarbashi–Bojchevski CP
→ "competent combination" attribution risk); (3) effect sizes are **modest** (delocalized
vulnerability, 0–1.8% flips, small-ε certificate). Reframing fixes #2; only **new theory**
fixes #1 and #3. Breakthrough = converting *sufficient → sharp*, *sound → tight*,
*empirical → proven*, *IGNN → general*, and elevating "a tool" to "a law".

## The thesis (the law — working target)
The constrained resolvent sensitivity **S_c = (I−J_z)⁻¹ J_A P_c** is THE governing object
of adversarial robustness in implicit/equilibrium computation. A single quantity
**σ₁(S_c)** dictates:
1. **WHEN models break** — a *sharp* phase transition at **ε_crit = (1−κ)/‖W‖** (κ=‖J_z‖):
   provable smooth regime below, provable breaking attack above (two-sided), with an order
   parameter and a critical exponent.
2. **HOW FAR they're safe** — a **tight** per-node certificate (a matching attack achieves
   the radius ρ_v, not just a sound upper bound).
3. **HOW to protect them** — the **min-max optimal** defense: regularizing σ₁(S_c) is the
   optimal defender response to the σ₁-optimal attacker (Stackelberg saddle).
4. **For a whole class** — **universally**: the same law and critical scaling govern a
   general class of equilibrium operators (IGNN, monotone DEQ, contractive message
   passing, and beyond), architecture-independent — *universality* in the statistical-
   physics sense.

**One sentence:** *the spectrum of the constrained resolvent governs adversarial
robustness in implicit computation — criticality, tight certification, and optimal
defense are three faces of one operator, sharply and universally.*

## CRUX VERDICT (2026-06-02) — the naive law above is REFUTED; refined thesis below
Two independent theory agents (prove vs refute, `breakthrough_crux_C1C4.md` +
`breakthrough_crux_redteam.md`) **converged**. What was refuted, what survived:

**REFUTED (do NOT claim):**
- **Sharpness at ε_crit.** ε_crit=(1−κ)/‖W‖ is the *norm* contraction radius — a certified-
  safe LOWER bound on the true *spectral* breaking budget ε*=1/ρ(W)−ρ(Â). Gap = the
  non-normality index η∈[1.19,2.47]. Verified: the ReLU fixed point stays unique to
  1.6×ε_crit; trained models sit at 2–4× margin with amplification ≈1.0008 — the
  divergence regime is *not reached*. Loss-of-contraction ≠ loss-of-prediction (argmax
  survives via margins). So "transition at ε_crit" is norm-certificate loss, not a break.
- **Universal exponent.** γ is NOT universal: γ=1 (simple crossing, generic), γ=½
  (exceptional point), γ=m (order-m defective Jordan block). Non-normal J_z (pseudospectra/
  Kreiss) decouples resolvent onset from κ.
- **Universal operator class.** Collapses to {symmetric-Â, single-W, all-active IGNN}. The
  "Universal/non-GNN" probe was circular (S_lib = the AEGIS resolvent relabeled, ‖diff‖=0);
  monotone DEQ *violates* the law (ρ=2.949, resolvent doesn't exist). **P4 was near-
  tautological — corrected.**

**SURVIVED (proved, defensible):**
- **γ=1 in the generic (simple-crossing) case** — residue of a simple pole, σ₁(S_c)∼C/(ε*−ε).
- **Order parameter = the SPECTRAL margin g=1−ρ(J_z)** (not the norm margin 1−κ).
- **C4 unification (most robust):** one operator S_c drives attack (top singular mode),
  certificate (r_v=Θ(g)→0), and defense (∂ε_crit/∂‖W‖<0) — though partly definitional.
- The empirical continuous→discrete transfer (τ≈0.99) is untouched.

## REFINED THESIS (post-crux, honest + still novel)
Adversarial criticality in implicit models is a **resolvent / pseudospectral phenomenon,
not a norm one.** The breaking budget is the *spectral* ε* (ε_crit a certified lower bound,
slack = non-normality η); the order parameter is g=1−ρ(J_z); and the critical exponent is
a **spectral-geometric invariant classified by how the equilibrium destabilises**
(γ∈{½,1,m} — three "universality classes" of adversarial criticality). **Universal in the
MECHANISM (resolvent singularity), architecture-dependent in the CONSTANTS.** One operator
S_c unifies attack/certificate/defense across the classes.

## NEW load-bearing CRUX (gates the whole breakthrough) — REACHABILITY
Is adversarial criticality actually *reachable* in trained implicit models under a budget-
bounded **worst-case** attack, or do trained models always sit at the 2–4× margin
(amplification ≈1.0008 in current data) so the transition is never approached → practically
vacuous? If reachable → the classification result is a real breakthrough. If not → the
criticality framing is a mathematical curiosity and we pivot. **Debug-first:** verify the
existing phase experiment actually tried to reach ε* with the worst-case direction before
accepting "unreachable."

## Why this is a breakthrough (escapes all three caps)
- It is a **LAW about a class of systems**, not a tool. The IFT/CP machinery becomes
  *method*, not contribution → escapes the novelty-attribution trap (#2).
- It imports **critical phenomena / universality** into certified robustness of implicit
  models — a new, surprising, *predictive* lens (kills "modest significance", #3).
- **Universality** dissolves the niche/narrow-scope read (#1): the result is about
  implicit computation generally, not a small GNN subclass.
- **Sharp + tight + optimal + universal** is the rigor profile of a landmark paper.

## THE CRUX — settle this FIRST, honestly (everything is load-bearing on it)
Is the transition *genuinely sharp and universal*, or merely a sufficient bound that
stops applying at ε_crit? Four sub-questions:
- **C1 (sharpness / matching lower bound).** For ε>ε_crit, *construct* δA, ‖δA‖_F≤ε, that
  provably destroys the equilibrium (loss of existence/uniqueness, or ρ(J_z)≥1) or forces
  a discontinuous prediction change. Sharpness ⇔ ε_crit is both the last safe budget and
  (up to constants) the first breaking budget.
- **C2 (order parameter + critical exponent).** As ε→ε_crit⁻, characterize the blow-up of
  ‖(I−J_z(ε))⁻¹‖ and σ₁(S_c(ε)); identify the order parameter (cand.: σ_min(I−J_z) → 0)
  and the exponent γ in (ε_crit−ε)^{−γ} (heuristic γ=1, the resolvent/contraction-boundary
  rate — verify rigorously).
- **C3 (universality).** Do ε_crit and γ depend ONLY on coarse spectral data (κ,‖W‖),
  architecture-independent, across a formalized general equilibrium-operator class?
- **C4 (unification at criticality).** Is the divergence of σ₁(S_c) the COMMON cause of
  attack optimality (v₁), certificate collapse (ρ_v→0), and defense leverage
  (∂ε_crit/∂ regularization)?

**Decision rule:** if C1–C3 hold → the law is real → breakthrough. If they fail → we learn
exactly where, and the honest result is whatever survives (still likely strong: tight
certificate + optimal defense). **No effort is spent on the rest until the crux verdict is
in.**

## Pillars (strong forms to prove) — P1 sharp universal transition · P2 tight certificate · P3 min-max optimal defense · P4 rigorous universality
Each: prove the strong form, or state the strongest TRUE weaker form + measured gap.
Current evidence each is plausibly-true: transition appears empirically (phase exps);
certificate sound w/ 0 breaches at 10 seeds (gate hugs the bound → tightness plausible);
coupling measured (partial corr −0.646 → optimality plausible); Universal holds
empirically (τ=0.988, 10 seeds, incl. non-GNN operators → universality plausible).

## Method
- **Theory:** attempt C1–C4 (ml-theory-writer) AND adversarially try to refute them
  (ml-theory-reviewer) in parallel — prove-vs-refute on the same crux. Reconcile; let
  failures redirect the thesis.
- **Empirics:** once the theory predicts a specific exponent γ and order parameter, sweep
  ε *through* ε_crit on many trained models/architectures, measure the order parameter and
  γ, and test universality (same scaling across architectures). This SEES the transition —
  decisive either way. Follow the per-experiment protocol (impl→critique→verify→run→md).
- **Standard:** 10 preferred seeds; bulletproof; debug every surprise; the proof sets the
  claim. Hostile-reviewer test on every statement before it is trusted.

## Status / log
- **2026-06-02:** Goal set. Crux assault LAUNCHED — ml-theory-writer (prove C1–C4 →
  `breakthrough_crux_C1C4.md`) + ml-theory-reviewer (refute / counterexample →
  `breakthrough_crux_redteam.md`), in parallel.
- **2026-06-02 (verdict):** The two agents CONVERGED → naive sharp+universal law REFUTED
  (see CRUX VERDICT above). Refined thesis = pseudospectral criticality + γ-classification
  (universal mechanism, architecture-dependent constants). **New gate = REACHABILITY** —
  must settle whether criticality is reachable under a bounded worst-case attack before
  committing. P4 (Universal) corrected to near-tautological. NEXT: reachability go/no-go
  experiment (worst-case ε-sweep to ε* across architectures; debug the existing phase exp
  first).
- **2026-06-02 (reachability debug-first — KEY):** the "unreachable" verdict is an
  **ARTIFACT.** `exp_phase_transition.py` varies κ by **retraining** (KAPPA_VALUES
  0.30→0.99) and measures the **v₁ shift at fixed ε=0.01** — it never runs a critical-
  driving attack, so reachability was **never tested**. Threat model confirmed: the
  perturbation is added to the normalized **Â directly, no renormalization**
  (`ctx_pert={"A_hat": A_sub+dA}`) ⇒ **ρ(Â+δÂ) CAN exceed 1 → criticality is structurally
  reachable in principle.** GO/NO-GO experiment now = a **critical-driving PGD attack**
  (maximize ρ(J_z(Â+δÂ)) over the edge-support Frobenius ball), swept in ε, vs v₁/random
  baselines, measuring ε_reach (where ρ→1) against ε_crit and ε*. Status: IMPLEMENTING
  (`scripts/exp_reachability.py`) — impl→critique→verify→run.
- **2026-06-02 (reachability VERDICT, seed 42 — `reachability_findings.md`):** attack
  verified (toy hits analytic optimum; J_z bit-identical to compute_jacobian). Result is a
  **split verdict:**
  - **REAL phenomenon:** criticality IS reachable; resolvent diverges with **γ=1.02 on both
    κ₀∈{0.5,0.9}** (the predicted simple-pole exponent); nonlinear equilibrium genuinely
    breaks (50/50 flips). **ε\* governs** (ε_reach=1.5–1.7×ε*), **ε_crit is wrong/conservative**
    (2.3–9.5× under). Critical-driving attack necessary (v₁ caps at ρ=0.966 at κ₀=0.9).
  - **BUT criticality is a DISTANT limit:** ε_reach = **60–128% of ‖Â‖_F** = **5–11× the
    paper's realistic budget (ε≤0.2)**. Trained models sit at ρ₀=0.22–0.39 (large margin);
    realistic-budget robustness is governed by first-order σ₁(S_c), NOT criticality.
  - **⟹ "criticality governs robustness" is NOT supported — the grand criticality
    breakthrough is DEAD.** Honest, found before building on sand (crux-first discipline
    worked).
- **PIVOT (2026-06-02):** survivors = γ=1 exponent (proved+measured), spectral-margin ≫
  norm-certificate (2–10×), critical-driving attack (matching lower bound), C4 unification.
  New breakthrough-grade target: **the first TIGHT structural robustness certificate for
  implicit GNNs** — ε* + the 1.5–1.7× active-fraction correction predicts ε_reach within a
  small constant; critical-driving = the matching two-sided attack; spectral margin shown
  2–10× the conservative ε_crit. Tight + two-sided + γ=1 structure. AWAITING user steer on
  whether this (strong, certified-robustness contribution) clears their "breakthrough" bar.
- **2026-06-02 (user: "keep hunting a breakthrough — new framing, forget effort").** New
  reframe: the equilibrium resolvent (I−J_z)⁻¹ as the **master operator of implicit
  computation** (robustness was one face). Probed 2 new framings in parallel:
  - **H — resolvent unifies over-squashing + robustness** (`breakthrough_oversquash_{theory,
    novelty}.md`): decay law REAL, margin↔reach Pareto FUNDAMENTAL, but BOTH theory+novelty
    agents say **solid-not-breakthrough** (over-squashing leg = known finite-depth Âᵏ at the
    fixed point; novel core = the opposing-sign control law, adjacent to Arroyo'25 / EIGNN).
    ⟹ documented STRONG FALLBACK.
  - **A — mechanistic interpretability of equilibrium computation via the resolvent**
    (`breakthrough_interp_scout.md`): **BREAKTHROUGH-VIABLE, gap genuinely open** (fixed-pt-
    Jacobian interp exists for RNNs only; DEQ algorithmic-reasoning exists but never
    interpreted via resolvent). (I−J_z)⁻¹=ΣJ_zᵏ = literal causal trace. **Falsifiable by ONE
    smoke with existing bug-audited machinery.** Deepest risk (κ<1 fights algorithm
    expressivity) ties back to our criticality physics.
- **PIVOT → A (interpretability).** Gating smoke LAUNCHED (2026-06-02): IGNN on a known
  graph algorithm (connected-components / source-reachability), seed 42 — gates G1 (does
  κ<1 IGNN solve it?), C1 (resolvent gain recovers the causal dependency structure vs a
  baseline), C3 (predict-then-intervene faithfulness vs re-solve), C2 (do resolvent eigen-
  modes align with the algorithm's data structure?). `scripts/exp_interp_smoke.py`.
- **2026-06-02 (interp gate on connected-components: WEAK — `interp_smoke_findings.md`).**
  Honest negative, debugged (self-checks bit-exact: resolvent block vs autograd rel-err
  4.4e-16). G1 NO contraction wall (κ=0.836, acc 1.000); but C1 resolvent-gain AUC 1.000
  **ties** black-box gradient, C2 eigen-align 0.989 **ties** adjacency control (learned
  W/φ' adds nothing), C3 off ~50% on the discrete bridge deletion. CC's causal structure
  IS adjacency support ⟹ resolvent decoding near-definitional. **One discriminating test
  left:** SSSP (answer = shortest-path *tree*, NOT reducible to adjacency support/spectrum).
- **META-PATTERN (honest):** 4 deep ideas rigorously tested → universal law REFUTED, pseudo-
  spectral criticality real-but-irrelevant, over-squashing strong-not-breakthrough, interp-
  on-CC WEAK. Each reduces to known structure / a strong tool, not a paradigm shift. **SSSP
  is the LAST clean shot for the interp pivot;** if WEAK → STOP hunting, consolidate the
  genuinely-strong survivors (tight certificate + validated arsenal) into an excellent paper.
- **SSSP discriminating gate LAUNCHED** (`scripts/exp_interp_sssp.py`, seed 42).
- **2026-06-02 (SSSP gate: PIVOT DEAD — `interp_sssp_findings.md`).** Well-controlled
  honest negative (self-checks bit-exact). G1: κ<1 IGNN can't learn SSSP (corr 0.53) AND
  the unconstrained variant also fails (0.45) ⟹ architectural expressivity floor, not a
  clean κ<1 wall. C1' crux: resolvent SP-tree AUC 0.852 **ties** input-grad (0.848),
  **loses** to hop-distance (0.859). Both hostile algorithms (CC, SSSP) WEAK ⟹ mechanistic-
  decoder pivot CLOSED.
- **═══ BREAKTHROUGH HUNT EXHAUSTED (2026-06-02) ═══** 5 deep ideas rigorously tested to
  destruction: (1) universal law REFUTED, (2) pseudospectral criticality real-but-
  practically-vacuous, (3) over-squashing unification STRONG-not-breakthrough, (4) interp/CC
  WEAK, (5) interp/SSSP WEAK. Every deep claim reduces to known structure (IFT sensitivity,
  adjacency support/spectrum) or is practically irrelevant. **S_c is a strong TOOL, not a
  paradigm-breakthrough seed — now known rigorously.** RECOMMENDATION: **STOP hunting,
  CONSOLIDATE** the genuinely-strong survivors into an excellent top-tier paper:
  - **Flagship:** first TIGHT two-sided structural robustness certificate for implicit GNNs
    (ε* + active-fraction correction; critical-driving = matching lower bound; spectral
    margin 2–10× the norm cert).
  - AEGIS-Conformal (10-seed runner-up); the γ=1 spectral-margin characterization; the
    over-squashing duality (novel bonus); the original one-query diagnostics + 39/39 transfer.
  AWAITING user call: consolidate (recommended) vs keep hunting (would require a genuinely
  NEW problem/idea outside this machinery — a different project, not an AEGIS sharpening).
