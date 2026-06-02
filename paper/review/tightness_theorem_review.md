# Adversarial Review — Flagship "Tight Two-Sided Bracket" Theorem (AEGIS, ICLR/COLT)

**Reviewer stance.** Hostile, top-tier theory referee. The draft is already adversarially
self-reviewed (it pre-empts U1–U3), so this review targets what the self-review still gets
wrong, still overclaims, or papers over. Numerical checks below were run on random
certified-regime instances (sandbox, not committed).

**Bottom-line verdict: NEEDS-REFRAME (not flagship-ready as written; reframe is cheap and the
core is sound).** The mathematics of the *bracket* (`\ecrit \le \ebreak \le (C/\beta)\ecrit`)
is correct and the lower side is airtight. But three things will get it shredded in review if
submitted as-is: (1) the word **"tight"** with a constant that empirically reaches ~10–18×;
(2) the upper bound holds only for the **all-active linear surrogate**, so the deployed ReLU
model's `\ebreak` is *not* what the theorem brackets; (3) a **wrong justification word**
("concavity") in the one genuinely novel step. None of these is fatal to the result; all are
fatal to the *framing* if left. Demote the nonlinear budget to empirical, rename "tight," fix
the convexity slip, and the flagship survives as a strong two-sided certificate with an
explicit computable constant.

---

## Per-attack verdicts

### Attack 1 — Is "tight" honest? **SERIOUS (framing-lethal if unaddressed).**

No. "Tight" is an overclaim and the single most attackable word in the draft.

- The bracket is `\ecrit \le \ebreak \le (C/\beta)\,\ecrit`. On the paper's own suite
  `C \in [1.3, \approx 10]`; on random certified instances with `\kappa` up to 0.9 I measured
  `\espec/\ecrit` ranging **[1.02, 18.2]** (median 2.25), and the *certified* constant `C/\beta`
  is strictly larger. A factor-up-to-10-to-18 two-sided enclosure is a legitimate and publishable
  result, but in the COLT/ICLR theory vocabulary **"tight"** means "matching up to constants that
  are 1, or up to `1+o(1)`, or with a matching-construction lower bound that *attains* the upper
  constant." This bracket does not: the lower side is `\ecrit`, the realized break is `\espec` (or
  `\espec/\beta`), and the ratio is exactly the `O(1)` slack `C` — which is *not* shown to be
  attained in both directions simultaneously except in the degenerate normal/dilute/aligned corner.
- A reviewer's one-line kill: *"You call a 10× enclosure 'tight'; tight has a technical meaning and
  this isn't it. Either prove the upper constant is attained (matching lower bound on `\ebreak` from
  above) or drop the word."*
- **Honest descriptor to use instead:** "**two-sided with an explicit, computable, `O(1)`
  (dimension-free) constant**," or "**constant-factor two-sided characterization**," or "**rate-tight
  in the normal/all-active regime, and constant-factor two-sided in general.**" The phrase the
  abstract already uses elsewhere — *"rate sharp in the all-active [normal] case"* — is the only
  place "sharp/tight" is defensible; confine it there.

### Attack 2 — A4 (all-active): fatal to the matching upper bound? **SERIOUS, borderline LETHAL.**

This is the most dangerous objection in the whole package, jointly with Attack 3.

- The upper side (ii) proves a break **only for `J_z = \Ahat\otimes W` with `\phi'\equiv1`**. The
  identity `\rho(J_z')=\rho(\Ahat')\rho(W)` is *false* off the all-active set: with a diagonal mask
  `D_a=\diag(\phi')` the operator is `D_a(\Ahat'\otimes W)`, whose spectral radius is **not**
  `\rho(\Ahat')\rho(W)` and does **not** factor through the Kronecker spectrum. So the constructed
  `\delta\Ahat^\star=\espec\,u_1u_1^\top` is proven to push the *surrogate's* radius to 1, not the
  deployed model's.
- Consequence: the theorem as boxed brackets `\ebreak` of the **all-active linearization**, an
  object that is not the deployed ReLU IGNN's stability boundary. The empirics confirm the gap is
  real and large: `\ereach/\espec \in [1.5,1.7]`. A hostile reviewer rejects the "matching lower
  bound" claim on exactly this: *"Your two-sided result is for a linear surrogate; for the model you
  ship, only the lower (safe) side is proved. The upper side is a different operator."*
- **Why it is "borderline" not outright lethal:** the *lower* side (i) is fully nonlinear-valid
  (only needs `\|\diag(\phi')\|\le1`), so the *certificate* the paper actually sells (safety up to
  `\ecrit`) is untouched. The damage is confined to the word "matching/two-sided" being claimed for
  the *deployed* model. The fix (Attack 3) is to scope the upper side explicitly to A4 and demote
  the nonlinear extension to empirical — then nothing false is claimed.
- **Can a reviewer reject the whole matching lower bound on this?** They can reject it *for the
  deployed model*. They cannot reject it *for the all-active operator*, where (ii) is a correct,
  exact construction. So: not a refutation of the math, but a refutation of the scope if the scope
  is left implicit. **Make the scope explicit in the theorem statement, not just §4.**

### Attack 3 — U1, the nonlinear hole: is `\ereach` part of a "theorem"? **LETHAL if presented as theorem; SURVIVABLE if demoted.**

The draft already concedes §3 is "modelled, not proved." Good — but it is not demoted *far
enough*, and the proof gap must be named precisely.

- **Exactly where the proof is missing.** Two independent gaps, not one:
  1. The `1/a` mean-field correction `\ereach\approx \espec/a` is asserted from "the active
     sub-block scales the per-unit spectral push by `\approx a`." There is **no lemma** that the
     dominant eigenvalue of `D_a(\Ahat'\otimes W)` scales as `a\cdot\rho(\Ahat')\rho(W)`. This is
     false in general (the active set is *correlated* with the equilibrium and with `u_1`; the
     active sub-block's Perron value is not `a` times the full one). It is a heuristic calibrated to
     two data points (`\kappa=0.5,0.9`). Presenting `\ereach\approx\espec/a` inside a section titled
     "reconciliation" adjacent to a boxed Theorem invites the reading that it is part of the theorem.
  2. Even granting the spectral identity, the jump from "an eigenvalue of the *linearized* Jacobian
     reaches 1" to "the *true nonlinear* equilibrium destabilizes / prediction flips 50/50" is a
     bifurcation claim about the nonlinear fixed-point map, supported only by the seed-42 reachability
     run. That is empirics, full stop.
- **Verdict on presentation:** the `1/a` budget and `\ereach` **must be demoted to an explicitly
  empirical claim** ("Observation," not "Definition+Theorem-adjacent derivation"), physically
  separated from the boxed theorem, with the heuristic flagged as a *conjecture* with a calibrated
  constant. As written, a referee will say *"§3 dresses a two-point curve fit as theory and lets it
  borrow the theorem's credibility."* That single sentence can sink the flagship.
- **Defensible version:** keep §2 (all-active, proved) as the theorem; move §3 to an "Empirical
  reconciliation / Conjecture" box: *"Empirically `\ereach\approx \espec/a` with `a\approx0.6`; we
  conjecture the active-fraction scaling and verify it on N instances; a proof for the masked
  operator is open."* Do **not** let `\ereach` appear in any sentence containing the word "theorem."

### Attack 4 — `\beta` vacuity: is the upper bound vacuous when `\beta\to0`? **SERIOUS.**

Yes, this is a real, under-defended hole, and worse than the draft admits.

- The upper bound is `\ebreak \le \espec/\beta`. If `\beta\to0` (the leading eigenvector `u_1` of
  `\Ahat` has vanishing mass on the edge-supported subspace), the bound `\to\infty` and is
  **vacuous**. The draft gives `\beta\approx0.62` on *one* 40-node test and asserts `\beta\to1` in
  the "dense-graph / edge-aligned limit." There is **no lower bound `\beta\ge\beta_0>0`** proved for
  any realistic graph family.
- This is structurally serious because `\beta` small is *not* pathological: `u_1` of the normalized
  adjacency is the Perron vector, which is dense and positive; `P_E` projects onto the **edge set
  pattern** (off-diagonal support on existing edges only). For sparse graphs the edge set is `O(N)`
  of the `O(N^2)` symmetric entries, so the projected mass `\beta` can be small and can *shrink with
  N*. The draft's `O(1)` non-vacuity claim (§4) silently assumes `\beta` is `N`-independent — that is
  an **unstated assumption** and a reviewer attack: *"`C` is dimension-free but `C/\beta` is your
  actual constant and you never bound `\beta` away from 0 as the graph grows; the certified upper
  bound may be vacuous on exactly the sparse graphs you test."*
- **Required fix:** either (a) prove a graph-dependent lower bound `\beta\ge \beta_0(\text{graph})`
  (e.g., via Perron-vector edge-mass / spectral-gap of `\Ahat`), or (b) define the threat model so
  the extremizer is the **leading singular mode of the constrained perturbation map `P_E`-restricted**
  (as Risk #4 in the crux already flags), in which case the relevant quantity is `\sigma_1` of a
  feasible operator and `\beta` disappears — replaced by a quantity that is `>0` by construction.
  Option (b) is cleaner and is what the algorithm actually computes; use it.

### Attack 5 — `C` blow-up as `\kappa\to1`; is `g_W` controlled? **SURVIVABLE (defended), one residual.**

The draft's defense here is largely correct, but it conflates two distinct points and leaves one
exposed.

- `C=\gW\frac{1+\kappa}{1-\kappa}\to\infty` as `\kappa\to1`. The draft excludes this via (A3)
  (`\kappa\le0.59` audited). That is legitimate **but note it cuts against "tight":** the constant
  is large *precisely in the regime that matters for robustness* (a near-critical, highly contractive
  model is the interesting one). At `\kappa=0.59`, `\frac{1+\kappa}{1-\kappa}=3.88`; times
  `\gW\le2.47` gives `C\le9.6`. So the "up to ~10" is driven by `\kappa` being allowed up to ~0.6,
  not a pathology — defensible, but it confirms Attack 1 (the constant is genuinely ~10 in-suite).
- **Is `\gW` (nonnormality of `W`) controlled in trained models?** This is the residual hole. The
  draft asserts `\gW=\norm{W}_2/\rho(W)\in[1.19,2.47]` *empirically* and `\gW\le\kappa_2(V_W)`. But
  spectral normalization controls `\norm{W}_2`, **not** `\rho(W)` from below, so nothing in training
  prevents `\rho(W)\to0` with `\norm{W}_2` fixed, i.e. `\gW\to\infty`. A reviewer: *"You normalize
  the numerator but never bound the denominator `\rho(W)` away from 0; `\gW` is an audited
  observation, not a controlled quantity, so `C` is not a priori bounded."* The honest move is to
  **report `\gW` as an audited per-model constant** (which the draft does) and explicitly state `C`
  is *a posteriori* computable, not *a priori* bounded by the training procedure. Do not imply
  spectral normalization bounds `C`.

### Attack 6 — Scope / overclaim (stability vs classification). **SURVIVABLE (well-handled).**

The draft handles this honestly and this is its strongest section. The "Scope honesty banner"
and §4 explicitly state `\ebreak` is the **well-posedness/contraction** boundary, sits at
`\norm{\delta\Ahat}_F\approx\norm{\Ahat}_2` (order one, must rewrite a constant fraction of the
graph), and is **not** a per-node classification-flip certificate (that is `\rad`, Prop. radius).
This pre-empts the "norm certs understate robustness by `C\times`" objection by scoping the claim
to stability.

- Residual risk (**moderate**): the headline reading *"the norm certificate `\ecrit` understates
  the true safety margin by `C`"* is still rhetorically a robustness claim. A skeptic notes the
  boundary lives at `\|\delta\Ahat\|\approx\|\Ahat\|` — *far* from any realistic attack budget — so
  the `C\times` understatement is operationally irrelevant to deployed robustness (nobody perturbs a
  constant fraction of the graph). **Recommendation:** state plainly that the contribution is a
  *sharp characterization of the stability boundary's location*, of theoretical/structural interest,
  and explicitly **not** a claim that practical robustness is `C\times` larger than previously
  certified. The draft is 90% there; tighten the headline sentence so "understates" cannot be
  misread as a practical-robustness gain.

### Attack 7 — Novelty / triviality. **SURVIVABLE, but real exposure on the upper bound.**

A reviewer *can* and likely *will* call the upper-bound construction routine, and they would be
partly right.

- The rank-one top-eigenvector bump `\delta\Ahat=\espec\,u_1u_1^\top` driving `\rho(\Ahat+t u_1u_1^\top)
  =\rho(\Ahat)+t` is **elementary** symmetric perturbation theory (exact because `u_1u_1^\top` shares
  the eigenbasis). The Kronecker spectral multiplicativity `\rho(A\otimes B)=\rho(A)\rho(B)` is
  textbook. The lower side is the existing Thm 1(a) restated. The additive gap identity §2.2 is one
  line. So *each ingredient is standard* — a referee can write *"this is a routine perturbation-theory
  exercise dressed as a flagship."*
- **Where genuine novelty survives, and how to foreground it:** (a) the *identification* that the
  norm-vs-radius gap for the IGNN equals exactly `W`'s nonnormality `\gW` and nothing from the graph
  (because `\Ahat` symmetric) — clean and non-obvious; (b) the unification C4 (one resolvent governs
  attack direction, radius, and defense) — this is the actual contribution; (c) the matching
  construction being *edge-feasible* (the `\beta`/`P_E` machinery) which is *not* trivial. **The
  novelty is the unification and the exact attribution of the gap, not the bracket per se.** Re-aim
  the framing around C4; sell the bracket as the corollary that makes C4 quantitative. If the bracket
  is the headline, novelty is weak; if C4 is the headline, novelty is solid.

---

## Two NEW defects the self-review missed

These were not in U1–U3 and are independently checkable:

**N1 — §2.3 "concavity" is the wrong word (convexity).** The draft writes: *"by concavity of the
top eigenvalue along a fixed symmetric direction the increment is at least linear."* The largest
eigenvalue `\lambda_{\max}(\Ahat+tB)` is a **convex** function of `t` (a pointwise max of linear
functionals `v^\top(\Ahat+tB)v`), **not concave**. Numerically: the "concave upper bound"
`\lambda_{\max}(\Ahat+tB)\le \lambda_{\max}(\Ahat)+t\,u_1^\top B u_1` is **violated in 42000/42000**
test points, while the **convex tangent lower bound** `\lambda_{\max}(\Ahat+tB)\ge
\lambda_{\max}(\Ahat)+t\,u_1^\top Bu_1` holds with **0 violations**. The *inequality the proof needs*
(`\rho(\Ahat')\ge\rho(\Ahat)+\beta t`, a lower bound) is therefore **correct** — but it follows from
**convexity** (tangent-line-below-the-curve), not concavity. As written, the justification is
mathematically false and a referee who checks will flag it as a sign error that "happens to land
right." **Fix:** replace "by concavity ... the increment is at least linear" with "by **convexity**
of `\lambda_{\max}` along a fixed direction, the curve lies **above** its tangent at `t=0`, so
`\rho(\Ahat+tB)\ge\rho(\Ahat)+t\,u_1^\top Bu_1=\rho(\Ahat)+\beta t`." One-word fix, but leaving it is
a credibility hit on the only novel step.

**N2 — (iii)'s equality conditions conflate "endpoints coincide" with "constant `C/\beta=1`."**
The draft says the bracket "collapses to equality (`\ecrit=\ebreak=\espec`) iff every gap factor is
1: `\gW=1`, `\kappa\to0^+`, and `\beta=1`." This is **over-stated**. The endpoint equality
`\ecrit=\espec` holds, by the additive identity §2.2, **iff `\gW=1` (W normal) alone** — `\kappa` and
`\beta` are irrelevant to whether the two budgets coincide. I verified: symmetric `W` (`\gW=1`),
`\kappa=0.5`, `\beta=1` gives `\ecrit=\espec=1.9706` **exactly**, yet the boxed constant
`C=\frac{1+0.5}{1-0.5}=3\ne1`. So `\kappa\to0` and `\beta=1` are the conditions for the *bound's
multiplicative constant `C/\beta` to numerically equal 1*, **not** conditions for the *true endpoints
to coincide*. The draft conflates "the phase boundary is exactly two-sided" (a statement about
`\ecrit,\espec,\ebreak`) with "my upper-bound constant is 1" (a statement about the slack in my
inequality). **Fix:** split (iii) into (iii-a) **exact two-sided boundary**: `\ecrit=\ebreak=\espec`
iff `\gW=1` and `\beta=1` (need `\beta=1` so the *feasible* break equals `\espec`, not for endpoint
coincidence per se; `\kappa` plays no role); and (iii-b) **constant tightness**: `C/\beta=1`
additionally iff `\kappa\to0`. Conflating these lets a referee say *"your stated equality conditions
are wrong — `\ecrit=\espec` at `\kappa=0.5` too."*

---

## The single most dangerous objection

**"Your 'tight two-sided matching' theorem brackets the all-active *linear surrogate*, not the
deployed ReLU IGNN; the only fully-nonlinear half is the lower (safe) side you already had; the
upper/matching half is a linearization plus a two-point `1/a` curve fit; and the slack you call
'tight' is empirically ~10–18×. Strip the surrogate and the curve fit, and the new content over the
existing Thm 1 is a one-line symmetric rank-one perturbation bounding `\espec\le C\,\ecrit`."**

This is the objection that fuses Attacks 1+2+3+7. If a referee writes this, the rebuttal must be:
(1) the theorem is *explicitly* scoped to the all-active operator **in its statement** (not buried in
§4); (2) `\ereach`/`1/a` is *explicitly* an empirical conjecture in a separate box, never called a
theorem; (3) the contribution is re-aimed at **C4 unification + exact gap attribution to `\gW`**,
with the bracket as the quantitative corollary; (4) "tight" is replaced by "constant-factor
two-sided, rate-sharp in the normal/all-active regime." With those four moves the objection is
defused. Without them, it is a reject.

---

## What MUST be fixed or reframed before submission (priority order)

1. **(LETHAL-if-left) Rename "tight."** Replace every "tight" with "constant-factor two-sided" /
   "two-sided with an explicit computable `O(1)` constant"; reserve "sharp" for the
   normal/all-active equality case only. (Attack 1.)
2. **(LETHAL-if-left) Scope the upper side to A4 in the theorem statement itself.** State (ii) as:
   *"For the all-active operator `J_z=\Ahat\otimes W`, there exists ..."* so no reader infers a
   matching bound for the deployed ReLU model. (Attack 2.)
3. **(LETHAL-if-left) Demote `\ereach`/`1/a` (§3) to an explicitly empirical Observation/Conjecture
   box**, physically separated, never adjacent-borrowing the theorem's authority; name both proof
   gaps (masked-operator spectral scaling; linear→nonlinear bifurcation). (Attack 3.)
4. **(SERIOUS) Fix the `\beta` hole.** Either prove `\beta\ge\beta_0>0` for the target graph family,
   or restate the extremizer as the leading singular mode of the `P_E`-restricted perturbation map
   (matching what the code computes) so `\beta` is replaced by a constructively-positive `\sigma_1`.
   Add the `N`-independence of `\beta` as an explicit hypothesis if you keep `\beta`. (Attack 4.)
5. **(SERIOUS) Fix N1 (convexity word) and N2 (split (iii)).** Both one-line corrections; both are
   referee-checkable false statements as written.
6. **(MODERATE) State `\gW`/`C` are a-posteriori audited, not a-priori bounded by training**; and
   tighten the headline so "`\ecrit` understates by `C`" cannot be read as a practical-robustness
   gain (the boundary is at `\|\delta\Ahat\|\approx\|\Ahat\|`). (Attacks 5, 6.)
7. **(STRATEGIC) Re-aim the contribution to C4 unification + exact gap attribution.** Makes the
   novelty objection (Attack 7) go away and reframes the bracket as a quantitative corollary.

---

## Strongest defensible version of the theorem (exact restatement)

> **Theorem (Constant-factor two-sided bracket for the IGNN contraction boundary).**
> Under Assumption (A1)–(A3), with `\ecrit>0` and `\Ahat` symmetric, the
> structural-robustness (well-posedness/contraction) boundary `\ebreak` satisfies:
>
> **(i) Lower side — sound certificate (any 1-Lipschitz `\phi`, fully nonlinear).** For every
> feasible `\delta\Ahat` with `\|\delta\Ahat\|_F\le\varepsilon<\ecrit`, the perturbed operator is a
> spectral-norm contraction with a unique equilibrium; hence `\ebreak\ge\ecrit`.
>
> **(ii) Upper side — all-active matching construction (requires A4: `\phi'\equiv1`, so
> `J_z=\Ahat\otimes W`).** There is a feasible rank-one `\delta\Ahat^\star`, with
> `\|\delta\Ahat^\star\|_F=\espec/\beta`, driving the all-active spectral radius to 1; hence for the
> all-active operator, `\ebreak^{\mathrm{all\text{-}act}}\le\espec/\beta`. Here
> `\beta=\langle u_1,P_E(u_1u_1^\top)u_1\rangle\in(0,1]` (assumed `>0` for the threat model), and in
> the unconstrained-symmetric model (`\beta=1`) the construction is exact: `\ebreak^{\mathrm{all\text{-}
> act}}=\espec`, extremizer `\espec\,u_1u_1^\top`.
>
> **(iii) Constant-factor enclosure.** `\espec\le C\,\ecrit` with the explicit, dimension-free,
> a-posteriori-computable constant `C=\gW\frac{1+\kappa}{1-\kappa}`; hence, in the all-active model,
> `\ecrit\le\ebreak^{\mathrm{all\text{-}act}}\le (C/\beta)\,\ecrit`.
>
> **(iv) Exact two-sided boundary (rate-sharp regime).** `\ecrit=\espec` iff `W` is normal
> (`\gW=1`); additionally `\ebreak^{\mathrm{all\text{-}act}}` equals this common value iff also
> `\beta=1`. In that regime `\ecrit` is the exact contraction boundary (matching upper/lower bound).
> The enclosure constant `C/\beta` additionally equals 1 iff also `\kappa\to0^+`.

**Honest descriptor:** *"a constant-factor (dimension-free, `O(1)`) two-sided characterization of the
all-active IGNN contraction boundary, rate-sharp in the normal/edge-aligned regime, with the slack
constant `C` equal to the trained weight's nonnormality `\gW` up to the contraction-margin factor
`\frac{1+\kappa}{1-\kappa}`."* Not "tight."

**Demote to empirical (separate Observation/Conjecture box, NOT the theorem):**
- the `1/a` active-fraction inflation `\ereach\approx\espec/a`;
- the nonlinear break budget `\ereach` and all `\ereach/\ecrit\in[2.3,9.5]` claims;
- the linear→nonlinear bifurcation (50/50 prediction flip at the crossing);
- any numeric `\gW\in[1.19,2.47]`, `\beta\approx0.62`, `\eta` envelope (report as audited constants,
  not as theorem hypotheses).

---

## Acceptance-risk verdict for the flagship

**As written: HIGH RISK / likely-reject framing** — "tight" + surrogate-as-deployed + curve-fit-
adjacent-to-theorem + a false "concavity" justification is exactly the cluster a hostile COLT/ICLR
referee converts into a reject.
**After the seven fixes: MODERATE-to-LOW RISK** — the corrected statement is a correct, honest,
constant-factor two-sided result with a genuinely novel unification (C4) and an exact attribution of
the norm-vs-radius gap to `W`-nonnormality. That is a defensible flagship. The result is sound; the
exposure is entirely in the framing and two one-line slips.
