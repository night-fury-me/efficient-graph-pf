# R2 — Theory Review (GNN / Certified-Robustness Theorist)

**Submission:** AEGIS: A Matrix-Free Operator to Audit, Certify, and Defend Graph Neural Networks (AAAI-2026, anonymous)
**Reviewer role:** Peer Reviewer 2 — theory / certified robustness. Independent, line-by-line proof audit. Read-only.
**Files audited:** `sections/{abstract,introduction,background,framework,theory}.tex`; `sections/appendix/{A_preliminaries,B_sensitivity,C_rankings,D_boundary,E_conformal}.tex`; `aegis.bib`.
**Verification:** numerical (rng-seeded; seeds 20260605 / 3141 / 2718 / 101) on every load-bearing inequality. Results are quoted inline.

---

## Summary

AEGIS reads three audit diagnostics off one operator, the constrained sensitivity matrix `S_c=(I-J_z)^{-1} J_A P_c`, and packages them with (i) a contraction phase-transition characterisation, (ii) a constant-factor two-sided bracket on the break budget, (iii) an "AEGIS-Conformal" robust-coverage certificate, and a per-node radius / discrete-transfer / explicit-GNN trio. The core sensitivity calculus (IFT + Neumann resolvent) is textbook-correct, and the authors are unusually honest in scoping: the two-budget split (`eglob ≤ ecrit`), the explicit "where rigor stops, honestly" boxes, and the labelled conjecture (O1) are genuine strengths and close several gaps that would otherwise be fatal.

The theory is **sound in its proved core but over-claimed in its headline framing in two specific, fixable places**: (1) the main-text conformal guarantee Eq. (conformal) is printed as an unconditional worst-case statement, whereas the only thing proved (Thm `robust-cov`) is conditional on exchangeability (C1), and the white-box structural threat model the paper itself defines does **not** preserve C1; (2) the novelty of the phase-transition / bracket results over textbook resolvent-perturbation theory is thin — the new content is essentially a one-line symmetric rank-one extremiser plus an honest non-normality bookkeeping, not a new mechanism. The bracket is non-vacuous on small subgraphs but its upper side degrades as `O(1/sqrt(N))` for sparse graphs, which the "non-vacuous whenever β>0" wording hides.

---

## Overall recommendation

**Major Revision**, confidence **4/5**.

The defects are framing/scope overclaims and a novelty-attribution gap, not arithmetic errors — every inequality I could check numerically holds *as scoped in the appendix*. But two of them (the unconditional conformal claim; the bracket's hidden N-dependence) are load-bearing for the abstract's selling points ("distribution-free certificate … holds … under the attack it certifies"; "non-vacuous"), so they cannot ship as written. After the fixes below the paper is plausibly accept-grade; as written a careful referee can exhibit a counterexample to Eq. (conformal).

## Scores (0–10)

| Axis | Score | One-line justification |
|---|---|---|
| Novelty | 4 | Resolvent gain `1/(1-κ)` is the standard Neumann bound; new content is a rank-one extremiser + non-normality bookkeeping + the `S_c` unification, which is engineering, not new theory. |
| Soundness | 6 | Proved core is correct and well-scoped; **−** main-text Eq. (conformal) overstates the conditional appendix theorem, and the threat model breaks the hypothesis it needs. |
| Clarity | 7 | Exceptionally clear and honest appendices; **−** main-text statements (conformal, bracket non-vacuity) are looser than the appendix they point to. |
| Significance | 6 | The unified-operator framing is genuinely useful to practitioners; the formal results mostly re-package known perturbation theory. |
| Reproducibility | 7 | Constants are a-posteriori computable; algorithm is explicit; numbers I re-derived matched. Seeds/suite well specified. |

---

## Strengths

1. **Honest two-budget repair is correct.** `eglob = max(0, 1/‖W‖₂ − ‖Â‖₂) ≤ ecrit = (1−κ)/‖W‖₂` (Eq. budgets / B_sensitivity Step 2). The mask-free argument — a ReLU mask is a 0/1 projection so `‖J_z'‖₂ = ‖diag(φ')(Â'⊗W)‖₂ ≤ ‖Â'‖₂‖W‖₂ ≤ (‖Â‖₂+ε)‖W‖₂` for **every** activation pattern — is airtight and correctly attributed to A1+A2+threat-bound only (no A3/A4). This is a sound *global* contraction radius. Verified: 0/4000 adversarial-mask contraction failures inside `eglob`. The earlier overclaim (safety on `(eglob, ecrit]`) is now correctly demoted to the labelled empirical regularity of Rem `obs_o1`.
2. **One-sidedness of `ecrit` is stated honestly.** Step 3 of the phase proof uses `M = diag(−s,0)`: `‖M‖₂ = s` yet `‖(I−M)⁻¹‖₂ = 1`. Verified for s∈{0.5,0.9,0.99}: resolvent ≡ 1.0000, `min|1−λ| ≡ 1`. So `ecrit` lower-bounds the divergence threshold; the paper does not claim a two-sided sharp transition (it correctly retired that — good).
3. **β=σ_E identity and the convex tangent are correct.** `β := ⟨u₁, B u₁⟩ = σ_E` (D_boundary Eq. beta-eq) verified 0/3000; the lower bound `ρ(Â+tB) ≥ ρ(Â)+βt` from **convexity** of `λ_max` (not concavity) verified 0/15000. The N1 "concavity" wording slip from earlier drafts is fixed in the current text (line 129–130 correctly says "pointwise maximum of affine functions, hence convex").
4. **The `‖z*‖ ≤ ‖X_proj‖/(1−κ)` bound uses the correct (local) κ.** C_rankings:60 / E_conformal:54 carry the right denominator. Verified 0/3968 at ReLU fixed points, including 263 cases where the old global `1/(1−‖Â‖‖W‖)` form would be vacuous/negative. The historical L_J defect is genuinely fixed; `C_v` is therefore not under-estimated.
5. **Genuine honesty boxes.** Rem `exchange-honesty`, Rem `conf-caveats`, and Rem `obs_o1` explicitly flag the load-bearing hypotheses and the two open gaps. This is the right scientific posture and should be preserved verbatim.

---

## Per-result verdict

| Result | Correct? | Novel? | Non-vacuous? | Verdict |
|---|---|---|---|---|
| Thm `phase_transition` | Yes (as scoped) | Marginal | Yes | **OK with caveat** — proved core correct; novelty over Neumann/resolvent theory is thin (see MAJOR-2). |
| Thm `cf2s` (bracket) | Yes (all-active, as scoped) | Marginal | Conditionally (β-dependent) | **Issue** — upper side hides `O(1/√N)` β decay (MAJOR-3); soundness OK when κ,ecrit,C evaluated consistently. |
| `robust-cov` / Eq. (conformal) | Appendix Yes; main-text overstated | Low (reuses Zargarbashi 2023) | Yes | **Issue (CRITICAL-1)** — Eq. (conformal) printed unconditionally; only conditional-on-C1 is proved, and the threat model breaks C1. |
| Lem `score-shift` | Yes | Low–moderate | Yes | **OK** — margin form sound; one soft APS wording spot (MINOR). |
| Prop `attack` | Yes (constrained `S_c` form) | Low (standard σ₁ variational) | Yes | **OK** — feasible maximiser is right-singular vector of `S_c`; `reshape(v₁)` infeasibility correctly noted. |
| Prop `radius` | Yes | Low–moderate | Yes | **OK** — min-over-all-competitors Cauchy–Schwarz is the correct robust-to-runner-up form. |
| Prop `transfer` | Yes (as scoped) | Moderate (the real empirical contribution) | Yes for small `w_k` | **OK with caveat** — proves magnitude + sufficient *pairwise* order, not the global τ headline (MAJOR-4). |
| Prop `explicit` | Yes | Low | Yes | **OK** — chain-rule unroll + weight-tied geometric limit correct; excludes hard-mask aggregators (MINOR). |

---

## Weaknesses

### CRITICAL-1 — Eq. (conformal) is printed as an unconditional worst-case guarantee, but only a *conditional* (on exchangeability) statement is proved, and the white-box threat model does not preserve that hypothesis.

- **What.** Main text Eq. (conformal) reads `Pr[y_v ∈ C_ε(v)] ≥ 1−α  for all ‖δÂ‖_F ≤ ε`, with surrounding prose "turning split conformal into a distribution-free … certificate … that holds at the nominal level under the attack it certifies" (abstract) and "distribution-free certificate … with coverage ≥1−α over the ε-ball, holding under worst-case attack" (intro contribution 2). The actual theorem (Thm `robust-cov`, E_conformal) proves this **only under (C1) exchangeability** of clean calibration+test true-label scores.
- **Where.** `theory.tex` Eq. (conformal) line 70–72 + abstract + intro contribution (2); proof `E_conformal.tex` Thm `robust-cov` (C1), Step 2 ("the lowered scores stay exchangeable") and Step 4 ("transfer to the whole ball"). Rem `exchange-honesty` itself concedes "(C1) … does not hold for free on a single fixed transductive graph."
- **Why it is a problem.** The threat model (`background.tex` Eq. threat) is a **white-box** adversary perturbing the **test** graph. Two independent ways the adversary breaks C1: (a) under transductivity calibration and test share one realized adjacency, and the IGNN output at `v` depends on the whole graph, so perturbing test-incident edges shifts the test score relative to the (clean-graph) calibration scores; (b) a white-box adversary may *select* the target subgraph / test node, inducing covariate shift. I verified the mechanical robust-coverage claim is **correct given C1** (coverage = 0.8988 ≈ 1−α=0.90 over the worst in-ball attack, with the per-node deterministic lowering and the one-sided construction), but that **once C1 is broken by exactly the kind of test-distribution shift this threat model permits, coverage collapses to 0.13** (target 0.90). So the score-shift envelope (Lem `score-shift`) is sound and uniform — that part is real and is a faithful instantiation of Zargarbashi et al. 2023 — but the *coverage validity* rests entirely on C1, which is an exchangeability/i.i.d.-design assumption the adversary is not bound by. The main-text quantifier "for all ‖δÂ‖≤ε" reads as robustness *to the adversary*, while the proof only delivers robustness *of an exchangeable-by-design pipeline*. These are different statements; the abstract conflates them.
- **Severity.** CRITICAL — this is the paper's headline "certify" pillar and the abstract's central guarantee; a referee can hand back the D2 covariate-shift counterexample.
- **Concrete fix (no new experiments needed).** (1) In the main text, restate Eq. (conformal) as conditional: `Pr[y_v ∈ C_ε(v)] ≥ 1−α` **under (C1)**, and move the hypothesis onto the displayed equation, not a downstream remark. (2) In the abstract/intro, replace "holds … under the attack it certifies" with "holds at the nominal level under the certified attack **given exchangeability (C1)**, which we impose by inductive calibration." (3) State explicitly which experimental protocol enforces C1 (inductive split vs. the transductive-exchangeability argument of Zargarbashi et al.) and confirm the gate numbers (0.95–1.00 at ε=0.05) are measured under that protocol — otherwise the gate is evidence for C1 holding empirically, not a proof. (4) Distinguish, in one sentence, "the score-shift bound is adversary-uniform (proved)" from "coverage validity is exchangeability-conditional (assumed)". The honesty box `exchange-honesty` already says most of this; the fix is to lift it into the *statement* and the *abstract*.

### MAJOR-2 — Novelty of `phase_transition` over textbook resolvent / perturbation theory is thin and is not delimited.

- **What.** The resolvent gain `1/(1−κ)` (Eq. neumann-bound) is exactly the standard Neumann bound for a contraction: `‖(I−J_z)⁻¹‖₂ ≤ Σ κⁱ = 1/(1−κ)`. The subcritical shift `‖Δz*‖ ≤ σ₁(S)ε + O(ε²)` is the first-order IFT response with that bound substituted. The critical resolvent-divergence `‖(I−M)⁻¹‖₂ ≥ 1/min_i|1−λ_i(M)|` is the spectral-radius lower bound on a resolvent norm. None of these is new: they are Horn–Johnson / Trefethen–Embree material.
- **Where.** `theory.tex` Thm `phase_transition` (a)/(b); `B_sensitivity.tex` Eqs. neumann-bound, resolvent-lb.
- **Why it is a problem.** The theorem is *advertised* as a "Vulnerability Characterization" / "phase transition" — language that implies a new dynamical phenomenon. The genuinely AEGIS-specific content is narrow: (i) instantiating `J_z = diag(φ')(Â⊗W)` so `κ ≤ ‖Â‖₂‖W‖₂`; (ii) the `eglob` mask-free radius (this *is* a clean, slightly novel observation worth keeping); (iii) the graph-independent non-normality bound `η ≤ κ(V_W)` (Obs `eta_bound`, all-active only). A referee will say "(a) and (b) are textbook applied to a Kronecker Jacobian." `ecrit` is "anything beyond the standard Neumann bound" only in that it reads the threshold off the trained κ rather than the worst-case `‖Â‖‖W‖` — a relabelling, not a new bound.
- **Severity.** MAJOR for the Novelty score; not a soundness issue.
- **Concrete fix.** State plainly which steps are standard (resolvent/Neumann/Weyl) and which are the contribution (the `eglob` mask-agnostic radius; the `η ≤ κ(V_W)` graph-independence; the `S_c` unification). Retitle Thm 1 to something like "Contraction-budget characterisation for IGNNs" and drop "phase transition" from the theorem (keep it as intuition). Frame the contribution as *the operator that makes all three diagnostics one SVD*, not as new perturbation theory.

### MAJOR-3 — The bracket's upper side is non-vacuous on small subgraphs only; β decays as `O(1/√N)` for sparse graphs, and the "non-vacuous whenever β>0" wording hides this.

- **What.** Thm `cf2s` (iii) gives `ecrit ≤ ebreak^all ≤ (C/β) ecrit`, "non-vacuous whenever β>0", with "on the suite β≈0.62, giving C/β ≲ 16". β = σ_E = edge-supported Frobenius mass of the Perron mode.
- **Where.** `theory.tex` Thm `cf2s` line 49 + line 52; `D_boundary.tex` Thm `cf2s_full` (iii) line 96.
- **Why it is a problem.** β being *positive* for connected graphs is true (verified, β∈[0.76,0.99] on dense randoms), but for **sparse graphs at fixed average degree** β shrinks with N: I measured (Erdős–Rényi, avg-degree 6, sym-norm) β mean = 0.671 (N=20), 0.443 (N=50), 0.319 (N=100), 0.227 (N=200), 0.161 (N=400) — i.e. β ≈ Θ(1/√N). So `C/β` grows like `√N`; the upper side does not stay an O(1) constant on the very sparse graphs the paper targets (and on which it reports `N=7650`). The "β≈0.62 → C/β≲16" number is an artifact of the 50-node ego-subgraph scale, not a property of the full graph. "Non-vacuous whenever β>0" is literally true but operationally misleading.
- **Severity.** MAJOR — it directly undercuts the abstract/intro framing of the bracket as a "constant-factor two-sided characterisation" and pairs with the abstract claim that the certificate "stays non-vacuous".
- **Concrete fix.** (1) State the β scaling explicitly: report β as a function of N on the actual graphs (not only the 50-node subgraph) and acknowledge `C/β = Θ(√N)` for fixed-degree sparse families. (2) Either (a) restate the extremiser as the **leading singular mode of the `P_E`-restricted perturbation map** — which is what the code computes, has σ₁>0 by construction, and makes β disappear from the bound — or (b) prove `β ≥ β₀(family) > 0` for the specific graph families used, or (c) explicitly scope the "≲16" claim to the subgraph regime and drop "constant-factor" for the full-graph regime. The soundness of the inequality is intact (verified 0/5000 with κ,ecrit,C consistent); this is a vacuity/scaling-honesty fix.

### MAJOR-4 — Prop `transfer` proves magnitude + a *sufficient pairwise* order condition, but the abstract's `τ=0.98` global rank statistic exceeds what is proved.

- **What.** Prop `transfer` gives `d_k = w_k v_k + R_k`, `|R_k| ≤ L_J w_k²/(2(1−κ)²)`, and a *pairwise* order-preservation condition Eq. (ranking): a pair `(k₁,k₂)` keeps its order iff `w_{k₁} < (v_{k₁}−v_{k₂})/(L_J/(1−κ)²)`. The abstract/experiments report Kendall `τ=0.98`/`+0.996` against brute-force removal as if predicted by the proposition.
- **Where.** `theory.tex` Prop `transfer`; `C_rankings.tex` Eq. (ranking) + Step 3; abstract ("edge-weighted τ=0.98"); the proof itself (C_rankings:86–88) quotes the empirical τ inside the proof environment.
- **Why it is a problem.** A per-pair sufficient condition that (per the paper's own experiments) holds for a fraction of pairs does not entail a global Kendall-τ value; τ=0.98 is an empirical outcome, not a theorem prediction. Reporting it adjacent to the proposition (and inside the proof) reads as derived. The constant itself is now correct (L_J carries the `‖z*‖` factor; the curvature bound is the *complete* Hessian by bilinearity, F_zz=F_AA=0 verified analytically), so the *magnitude* claim (a) is sound; only the *ranking* over-read is the issue.
- **Severity.** MAJOR for honest claim-vs-proof alignment; not a soundness error.
- **Concrete fix.** Label τ=0.98/0.996 as empirical everywhere (it is, in the experiments), and state in the proposition's neighbourhood: "Prop `transfer` proves (a) the O(w_k) magnitude with O(w_k²) remainder and (b) a sufficient pairwise order condition; the global τ is empirical and exceeds the proposition." Move the empirical τ out of the proof environment.

---

## Questions for authors

1. **Conformal / C1.** Which calibration protocol enforces exchangeability in your reported coverage experiments — inductive (fresh-graph) calibration, or the transductive-exchangeability argument of Zargarbashi et al. 2023? Under a white-box adversary that perturbs test-incident edges (your threat model), what prevents the test true-label score from leaving the calibration distribution, and hence breaking C1? Can you give coverage under an adversary that *selects* the worst test subgraph rather than a fixed one?
2. **Bracket / β.** What is β on the *full* graphs (N up to 7650), not the 50-node ego-subgraph? Given β ≈ Θ(1/√N) at fixed degree, in what sense is `C/β` a "constant factor" for your largest graphs? Would restating the extremiser as the leading singular mode of the `P_E`-restricted map (σ₁>0 by construction) let you drop β entirely?
3. **Novelty.** Beyond instantiating the resolvent bound on a Kronecker Jacobian, what in Thm `phase_transition` (a)/(b) is not standard Neumann/spectral-radius perturbation theory? Is the intended contribution the `eglob` mask-free radius and the `η ≤ κ(V_W)` graph-independence, rather than the three-regime picture itself?
4. **`ecrit` consistency.** `C/β ≲ 16` is computed with partial κ∈[0.14,0.59], but (iii) derives `C` using `‖Â‖₂ = κ/‖W‖₂`, i.e. the all-active κ ≈ ‖Â‖‖W‖ ≈ 0.85–0.95. With consistent all-active κ, `espec/ecrit` reaches ~50–137× in my sweep. Which κ should a reader use to reproduce "≲16", and is the bound you sell the partial-κ number or the all-active-consistent one? (The inequality is sound either way when κ,ecrit,C are mutually consistent — 0 violations both ways — but the headline number mixes the two.)
5. **APS worst-case.** Step 3 of Lem `score-shift` claims the "all-lowered corner" dominates APS because `ρ_r` is maximised when every margin is at its budget. Since `ρ_r` sums over a *rank-dependent* set whose membership jumps discontinuously as a competitor crosses, please confirm the worst-case is computed by exact softmax recomputation at that corner (sound, an over-estimate) rather than via a Lipschitz-in-decrement constant (which the discontinuous rank-set does not satisfy).

---

## MINOR issues (kept in file)

- **M1 — APS "monotone map Ψ_r" wording.** Lem `score-shift` Step 3 is sound as an over-estimate (worst-case softmax recomputation is exact), but describing Ψ_r purely as "monotone" glosses the rank-set discontinuity in `ρ_r`. Add one clause: "computed by exact softmax recomputation at the all-lowered corner, not via a Lipschitz constant." (Recurring soft spot across drafts.)
- **M2 — Prop `explicit` differentiability premise.** `S_K` chain-rule silently assumes differentiable per-layer Jacobians, excluding hard-mask GAT / max-pool aggregators; conceded only in experiments (GAT†). Add a one-line scope note in the proposition.
- **M3 — `gW` not a-priori bounded.** D_boundary line 41 correctly notes spectral-norm normalisation caps `‖W‖₂` but not `ρ(W)`, so `gW=‖W‖₂/ρ(W)` could blow up; `gW∈[1.19,2.47]` is an *audited* observation, not a controlled quantity. The text says this — keep it, and make sure no abstract-level claim reads `C` as training-controlled.
- **M4 — `tab:cross_domain` "certified".** A first-order radius `r_v` is labelled "certified" in the cross-domain table; `r_v` is a first-order no-flip threshold, breachable past it (the conformal certificate is the actual certificate). Use "first-order radius" in that table to avoid over-reading.
- **M5 — `η` used as both a bound and an observation.** `η∈[1.19,2.47]` is proved (`η ≤ κ(V_W)`) only for all-active; general ReLU is empirical (Rem `eta_relu`). Thm `phase_transition` (b)'s "slack governed by η" leans on the all-active bound — keep the empirical/proved split visible.
- **M6 — `J_z` eigenvalue-location hypothesis in (b).** The `Ω(1/(ecrit−ε))` rate needs the dominant eigenvalue real-positive (Perron); the text states this ("dominant real-positive (Perron)"), which is load-bearing and correctly flagged. The `diag(−s,0)` counterexample is the right guard. No change needed; noting for completeness.

---

## What I verified numerically (so the editor can spot-check)

| Claim | Location | Result |
|---|---|---|
| `β = σ_E = ⟨u₁,Bu₁⟩` | D_boundary Eq. beta-eq | 0/3000 violations |
| convex tangent `ρ(Â+tB) ≥ ρ(Â)+βt` | D_boundary (ii) | 0/15000 violations |
| β decays `~1/√N` (sparse, fixed degree) | (vacuity check) | β: 0.67→0.16 for N=20→400 |
| `espec ≤ C·ecrit` (κ,ecrit,C consistent) | D_boundary (iii) | 0/5000 both partial & all-active κ |
| `espec/ecrit` range (all-active κ→1) | — | up to 137× (not "≲16") |
| robust coverage *given C1* | Thm robust-cov | 0.8988 ≈ 1−α=0.90 |
| robust coverage *C1 broken* (covariate shift) | (threat-model counterexample) | 0.13 (collapses) |
| resolvent one-sided `diag(−s,0)` | B_sensitivity Step 3 | resolvent ≡ 1 for s∈{.5,.9,.99} |
| `‖z*‖ ≤ ‖X_proj‖/(1−κ)` local κ | C_rankings Eq. LJ-bound | 0/3968; 263 cases where global form vacuous |
| bilinearity `F_zz=F_AA=0` | C_rankings Eq. bilinear | analytic, correct |

---

*Reviewer 2 (Theory). Independent audit; no other reviewers consulted.*
