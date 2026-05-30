# Peer Review Report — Methodology (Theory)

**Paper:** AEGIS: One-Query Adversarial Diagnostics over the GNN Vulnerability Spectrum
**Venue:** ICDM (Research Track)
**Date:** 2026-05-30

---

## Identity & Focus

Peer Reviewer 1 (Methodology). My expertise is implicit/equilibrium deep learning (DEQ/IGNN), spectral and pseudospectral methods, randomized numerical linear algebra, and adversarial-robustness theory. This review is restricted to the *theory* (Section IV / `theory.tex`), the *construction/numerics* (Section III / `framework.tex`, Algorithm 1), and the statistical methodology backing the theory claims (Section V / `experiments.tex`). I do not assess domain framing, the power-flow case study, or writing beyond what bears on rigor. This is an independent review.

---

## Recommendation + Confidence

**Recommendation: Major Revision** (score 61/100 — see Dimension Scores).
**Confidence: 4/5.** I verified every proof line and reconstructed the disputed steps independently. My one residual uncertainty is whether the authors can produce, in rebuttal, a clean ReLU-region argument for the flagship theorem; if they cannot, the recommendation drops toward Reject.

---

## Summary Assessment

The paper packages a genuinely useful engineering object — a matrix-free constrained sensitivity operator `S_c = (I−J_z)^{-1} J_A P_c`, queried by JVP/Neumann and reduced by a randomized SVD — into a "one-query" adversarial diagnostic for contractive implicit GNNs. The construction is sound, the first-order propositions (attack direction, per-node radius) are correct modulo exposition, and the empirical program is unusually thorough (9 datasets, 7 architectures, 10 seeds, honest "diagnostic not certificate" caveats).

The problem is the flagship. **Theorem 1 is advertised as covering "ReLU or any 1-Lipschitz activation" (A1), but as *proved* its sharp critical-rate claim (regime b) holds only for `φ'≡1` — the all-active / identity case — plus a normal-`J_z'`-with-real-positive-Perron-eigenvalue hypothesis.** The supporting nonnormality bound (Observation 1) is *explicitly* labeled "all-active case," and Remark 1 concedes the general-ReLU `η` is empirical-only (`η∈[1.19,2.47]`). So the theorem's only nontrivial quantitative content — the `Ω(1/(ε_crit−ε))` blow-up rate and the `η`-controlled slack — is a special-case result wearing a general-case title. The subcritical bound (a) and the supercritical statement (c) are fine but are essentially the standard Banach/Neumann argument; they carry no rate. A reviewer who reads only the theorem statement is materially misled about what is proved. This is fixable by re-scoping the title and moving the generality into a clearly-flagged empirical observation, but it must be fixed: a flagship theorem proved for a special case while advertising generality cannot score well on Rigor.

Secondary but real: (i) the curvature constant `L_J ≤ ‖W‖²‖z*‖` is finite only because `‖z*‖` is bounded by `‖X_proj‖/(1−‖Â‖₂‖W‖₂)`, a denominator that can be **≤0 while the model is still contractive under partial ReLU** (κ = ‖J_z‖₂ can be strictly less than ‖Â‖₂‖W‖₂), making the bound vacuous off the all-active case; (ii) Proposition 3 proves a *magnitude* bound and a *sufficient* pairwise-order condition, but the headline `τ=+0.996` is a *global rank* claim the proposition does not establish; (iii) the randomized-SVD error is asserted "bounded by the spectral gap" with no stated bound, and the "one query" framing hides the `K∈[20,50]` Neumann JVPs inside each `S_c v`; (iv) the `p<10^{-5}` sign test is uncorrected for 33-cell multiplicity and the `±` values are inconsistently labeled (SD in captions, unlabeled subscripts via the `\ms` macro in tables).

None of the secondary issues is fatal. The Theorem-1 generality gap is the one that governs the score.

---

## Strengths

1. **The core operator and matrix-free numerics are correct and well-engineered.** `S_c v = (I−J_z)^{-1}J_A P_c v` via truncated Neumann (`framework.tex` l.25, Alg. 1 l.15) is the right primitive; the `O(Nd)` memory / `O(K·Nd)` time claim is justified, and lifting the dense `O((Nd)^3)` ceiling to `N=7650` is a real contribution. The duplication-matrix reduction `P_c` onto the edge-supported symmetric subspace (`theory.tex` l.67, `magnus2019matrix`) is the correct way to enforce `δÂ=δÂ^T`.
2. **Proposition 1 (SVD direction) is exactly right** and correctly scoped: it is the textbook variational characterization of `σ_1`, the maximizer is the leading right singular vector, and `sym(P_c v_1)` (Alg. 1 l.17) correctly projects back onto the symmetric edge subspace so the returned `δÂ*` is feasible. No overclaim here.
3. **Proposition 2 (per-node radius) is correct and the min-over-all-competitors form is the right one.** Using the row-vector `(W_{y_v}−W_c)` times the block matrix `S_v` with Cauchy–Schwarz is valid (the row 2-norm equals the relevant operator norm), and minimizing over *all* classes — not just the runner-up — correctly yields a bound robust to runner-up changes. The authors explicitly flag the runner-up surrogate as optimistic. Good.
4. **Unusually honest caveating.** Remark 2 (`rem:certificates`) cleanly separates first-order *threshold* from sound *certificate*; the AGNNCert comparison reports a 4.4–15× looseness *against* AEGIS and prescribes a decision rule rather than claiming dominance. The empirical breach-rate check (`ε>r_v` for every breached node) is the correct falsification test for a first-order threshold.
5. **Empirical breadth and self-consistency checks are strong:** dense-vs-matrix-free `σ_1` agreement to 0.03% at N=200, per-edge `τ=0.999` finite-difference reproduction, and the `κ_max` sweep validating `ε_crit` as a sufficient boundary with 2–4× margin.

---

## Weaknesses

### W1 — Theorem 1's proved scope is strictly narrower than its advertised scope. **[Severity: Critical / Major]**

- **Problem.** A1 advertises "ReLU or any 1-Lipschitz activation." But the only nontrivial content of the theorem — the critical-regime rate in (b) — is proved (`theory.tex` l.43) only "when `J_z'` is normal with a dominant real-positive eigenvalue `λ*=‖J_z'‖₂` (the Perron mode for symmetric `Â,W`)." Observation 1, which supplies the `η` that controls the general-case "slack," is titled and proved **"all-active case," requiring `φ'≡1`** (`theory.tex` l.48–53). Remark 1 (`rem:eta_relu`, l.55–58) then states plainly: *"The proof above requires `φ'≡1`; for general ReLU patterns, `η∈[1.19,2.47]` on our suite"* — i.e. empirical only.
- **Why it matters.** Under ReLU, `J_z = diag(φ')(Â⊗W)` with a 0/1 mask. The mask (a) destroys the clean Kronecker eigenstructure Observation 1 relies on (it is no longer `Â⊗W`), and (b) generically makes `J_z'` **nonnormal**, which is exactly the regime where `‖(I−J_z')^{-1}‖₂` and `1/min_i|1−λ_i|` *diverge* from each other — the gap the paper offloads onto an `η` it has not bounded outside the all-active case. So for the actual trained ReLU models, the `Ω(1/(ε_crit−ε))` rate is *not* established, and `ε_crit` is only a one-sided contraction radius (which is the standard Banach result, already implied by A3). The genuinely novel quantitative claim is unproved in the regime the paper runs experiments in.
- **Internal-consistency note (in the authors' favor, but not a rescue).** The authors correctly killed the naive normal-case claim with the `M=diag(−s,0)` counterexample (l.43): `‖M‖₂=s→1` yet resolvent `=1`. This is right and shows care. But the *same* object also shows the Perron hypothesis is doing all the work — a negative or complex dominant eigenvalue (entirely possible once the ReLU mask breaks symmetry of the Kronecker factor) keeps `min_i|1−λ_i|` away from 0, so the rate fails. The counterexample they use defensively against the old claim is the counterexample against their own ReLU generality.
- **Suggestion.** Three honest options, in decreasing order of strength: **(1)** Prove the normal/Perron structure survives ReLU masking under a stated sufficient condition (e.g., if the active set induces a principal submatrix of `Â⊗W` that remains symmetric PSD — plausible for symmetric `Â` and symmetric/PSD `W`, but it must be *shown*, including that `diag(φ')` commutes appropriately, which it generally does **not**). **(2)** Re-title Theorem 1 to "Vulnerability Characterization for Contractive Linear/All-Active IGNN Operators," prove (a)(b)(c) cleanly for `φ'≡1`, and demote the ReLU behavior to a clearly-labeled "Empirical Extension" observation with the `η∈[1.19,2.47]` evidence. **(3)** Keep ReLU in (a) and (c) only (where 1-Lipschitz suffices and no eigenvalue location is needed) and explicitly restrict (b)'s *rate* to the all-active/normal case in the theorem body, not in a downstream remark. Option (2) is the cleanest and costs almost nothing scientifically because the empirical section already carries the ReLU evidence.

### W2 — `L_J`'s finiteness rests on a denominator that can be ≤0 under partial ReLU. **[Severity: Major]**

- **Problem.** Proposition 3(a) sets `L_J ≤ ‖W‖₂²‖z*‖` (`theory.tex` l.104) and argues `L_J` is finite via `‖z*‖ ≤ ‖X_proj‖/(1−‖Â‖₂‖W‖₂)` (proof, l.115). But A3 only guarantees `κ = ‖J_z‖₂ < 1`, and under partial activation `κ = ‖diag(φ')(Â⊗W)‖₂` can be **strictly smaller** than `‖Â‖₂‖W‖₂`. Hence `‖Â‖₂‖W‖₂ ≥ 1` is possible for a perfectly contractive model, making the stated `‖z*‖` bound negative or infinite — i.e. **vacuous**.
- **Why it matters.** Every quantitative remainder/ranking guarantee in Prop. 3 (`|R_k| ≤ L_J w_k²/(2(1−κ)²)`, the ranking-preservation thresholds eq. (9)) inherits this. The bound is sound only in the all-active case, where `κ=‖Â‖₂‖W‖₂` — the *same* special case as W1. This is the recurring "norm vs. actual-contraction" conflation that already bit regime (b).
- **Suggestion.** Bound `‖z*‖` through the contraction constant itself. At equilibrium `z* = φ(Âz*W^T + X_proj)`; since `φ` is 1-Lipschitz with `φ(0)=0` and the vec-map is `κ`-contractive, `‖z*‖ ≤ ‖X_proj‖/(1−κ)`. This uses *only* A3, is always finite under contractivity, and yields `R_k ≤ ‖W‖₂²‖X_proj‖ w_k² / (2(1−κ)³)` — a clean bounded constant. Substitute this everywhere `1/(1−‖Â‖₂‖W‖₂)` currently appears (proof l.115, background l.13). This is a one-line fix and removes the vacuity.

### W3 — Proposition 3 proves magnitude + sufficient pairwise order; the `τ=+0.996` headline is a global-rank claim it does not deliver. **[Severity: Major]**

- **Problem.** Prop. 3(b) gives, for a *pair* `(k_1,k_2)` with `v_{k1}>v_{k2}` and `w_{k1}≥w_{k2}`, a *sufficient* condition (eq. (9)) under which `d_{k1}>d_{k2}`. The condition fails whenever the score gap `v_{k1}−v_{k2}` is small relative to `L_J w/(1−κ)²` — i.e. for *adjacent* pairs in the ranking, which are exactly the pairs Kendall `τ` is most sensitive to. Yet the abstract/experiments report `τ=+0.996` against brute-force N-1 (`experiments.tex` l.172) as if Prop. 3 predicts it.
- **Why it matters.** Kendall `τ` is a *global* rank correlation over all `~|E|²/2` pairs. Prop. 3 bounds only the pairs satisfying eq. (9), and the paper itself reports the sufficient condition holds for only **47–62%** of pairs (`experiments.tex` l.182). A near-perfect global `τ` is therefore an *empirical* finding that *exceeds* the theory, not a corollary of it. The proposition cannot rule out rank inversions among the 38–53% of pairs outside the regime; that it works anyway is interesting but unproven.
- **Suggestion.** State explicitly that Prop. 3(b) certifies pairwise order only on the in-regime fraction, and present `τ=+0.996` as empirical corroboration that inversions are rare in practice, *not* as predicted by the bound. Optionally add a probabilistic statement: if score gaps `v_{k1}−v_{k2}` exceed the remainder threshold for a `(1−p)` fraction of pairs, then expected discordant pairs `≤ p·(|E| choose 2)`, giving `τ ≥ 1−2p`; with `p≈0.4` this predicts `τ≥0.2`, consistent with the *cold* cells but showing why the headline is over-strong as a theory claim.

### W4 — "One query" undersells the work hidden in each `S_c v`; randomized-SVD error is unbounded in-text. **[Severity: Moderate]**

- **Problem (a).** Each `S_c v` is a truncated Neumann series with `K∈[20,50]` JVPs (`framework.tex` l.34, Alg. 1 l.14–15). The randomized SVD does `n_iter=2` power iterations over `k=p=10` probe vectors. So "one query" is really one *rSVD invocation* costing `O(k · n_iter · K)` JVPs `≈ O(10·2·30)=O(600)` forward passes. Calling this "one-query" against a "512-query black-box" baseline (`experiments.tex` l.43) is an apples-to-oranges framing.
- **Problem (b).** Numerical claims: dense-vs-matrix-free `σ_1` agreement "within 0.03%" and truncation residual `κ^{200}∈[10^{-105},10^{-48}]` (`experiments.tex` l.128) are fine and quantitative. But "the rSVD error is bounded by the spectral gap" (l.128) cites `halko2011finding` with **no stated bound**. Halko–Martinsson–Tropp give an *expected* spectral-norm error `E‖A−Q Q^T A‖ ≤ [1 + 4√(k+p)/(p−1)·√min(m,n)] σ_{k+1}`; with a 39–50% gap and `k=10` this is small, but it should be stated, not gestured at.
- **Suggestion.** (a) Report cost as "a single rSVD invocation (`≈600` JVPs, vs. 512 *forward queries* for black-box and 50 *gradient steps* for PGD)" and let the reader judge; the result is still favorable and now honest. (b) Quote the HMT expected-error bound with the measured `σ_{k+1}/σ_1` and the 39–50% gap to give a concrete number. This converts an assertion into a guarantee.

### W5 — Conservative-IFT invocation is asserted, not derived; differentiability of the resolvent at nonsmooth points is licensed only a.e. **[Severity: Moderate]**

- **Problem.** A1 defers ReLU nonsmoothness to the conservative IFT (`bolte2021conservative`). The proof (l.37) argues the active set is generically stable ("boundary set has Lebesgue measure zero") and applies a "conservative IFT on each region." `bolte2021conservative` establishes conservative *Jacobians* and a chain rule for path-differentiable functions; it does **not** package a ready-to-use implicit-function theorem for fixed points of nonsmooth contractions. The step "conservative-gradient calculus yields a conservative IFT on each region" is plausible but is *stated*, not proved, and the resolvent `(I−J_z)^{-1}` is well-defined per-region but the cross-region *transition* (where the active set changes) is exactly where a first-order expansion can fail.
- **Why it matters.** For Prop. 3 the perturbation path *crosses* ReLU region boundaries (the proof, l.115, acknowledges "the path meets finitely many ReLU regions"). At a crossing the one-sided derivatives differ; "summing the second-order remainder over these regions" assumes the first-order term is continuous across boundaries, which holds for ReLU at a single kink (continuity of the function, not the derivative) but needs the crossing set to have measure zero *along the specific 1-D path* `A → A∖k`, not just in `A`-space. This is true generically but should be stated as an assumption, since adversarial perturbations are chosen non-generically.
- **Suggestion.** Add one sentence stating the genericity assumption explicitly ("for `δA` outside a measure-zero set of activation-boundary-aligned directions"), and either (i) cite a concrete nonsmooth-IFT result (e.g., Clarke's IFT for Lipschitz maps, or the path-differentiable IFT in the conservative-calculus literature) with the exact theorem number, or (ii) state the result for the all-active region and note empirical robustness. Adversarial directions can be boundary-aligned by construction, so the genericity claim deserves a guard.

### W6 — Statistical methodology: uncorrected multiplicity, ambiguous SD/SE, "bound vs. observation" labeling. **[Severity: Moderate]**

- **Problem (a) — multiplicity.** "29/33 cells positive, one-sided sign test `p<10^{-5}`" (`experiments.tex` l.172). A sign test on 33 cells against `H_0: P(+)=0.5` gives `p≈3×10^{-5}` for 29/33 — but the 33 cells are not independent (shared datasets, shared architectures across rows/columns), and there is no multiple-comparison framing for the *per-cell* `τ` significance. The Wilcoxon `p<0.001` (l.36) and `p<10^{-43}` (l.36, "149/150 wins") similarly lack a stated correction across the many reported comparisons.
- **Problem (b) — SD vs SE.** The `\ms` macro (`aegis.tex` l.41, `\ms{#1}{#2}=#1_{±#2}`) renders all table `±` values as bare subscripts with **no SD/SE declaration in the tables**. Captions for figures say "±1 sd" (e.g. `fig:greedy_topk`), but `tab:cross_domain`, `tab:explicit`, `tab:attack_full` give no specification. For 10 seeds the difference (SD vs SE = SD/√10 ≈ SD/3.16) materially changes how `τ=+.32±.04` etc. should be read.
- **Problem (c) — `η∈[1.19,2.47]`.** This is repeatedly used as if it bounds the nonnormality slack in Theorem 1(b), but it is an **observed range over the suite** (Remark 1), not a proven bound. Phrasing like "slack controlled by `η`" (l.43) reads as a guarantee.
- **Suggestion.** (a) State the dependence structure and apply Holm–Bonferroni (or report it as descriptive, not inferential, for the per-cell `τ`). The 29/33 sign test is fine as a coarse aggregate if labeled "treating cells as exchangeable." (b) Declare SD vs SE once, globally, and make tables consistent with captions. (c) Replace "controlled by `η`" with "empirically tracked by `η∈[1.19,2.47]` (Remark 1); not proved outside the all-active case."

### W7 — Proposition 4 (explicit-GNN) differentiability assumption silently excludes the standard models it is sold against. **[Severity: Minor]**

- **Problem.** Prop. 4 requires each layer "differentiable w.r.t. both its input and `A`" (`theory.tex` l.125). The chain-rule expression `S_K = Σ_l (∏_{k>l} J_z^{(k)}) J_A^{(l)}` (eq. (10)) is correct *given* differentiability. But standard GAT uses `A` as a binary mask, so `∂Z/∂A_{ij}=0` on existing edges and `S_c` is undefined — the paper concedes this only in the experiments (`experiments.tex` l.168, "GAT†") via a *modified* operator. The proposition's generality ("`K`-layer GNN") thus excludes hard-attention, max/min aggregation, and GATv2-style masking — a non-trivial slice of "explicit GNNs."
- **Why it matters.** The chain-rule identity itself is standard and correct; the issue is scope-labeling, identical in spirit to W1 but lower stakes (this is a "the tool also applies" proposition, not the flagship).
- **Suggestion.** State in the proposition body that it applies to weighted-message-passing GNNs with `A`-differentiable aggregation, and explicitly list the excluded class (binary-mask attention, non-differentiable pooling) there rather than only in the experiments.

---

## Detailed Theory Audit

### Theorem 1 (`thm:phase_transition`, p.2) — three-regime characterization

- **Claim.** For an IGNN operator under A1 (ReLU/1-Lipschitz, conservative IFT), A2 (`‖W‖₂≤c`), A3 (`‖J_z‖₂≤κ<1`), with `ε_crit=(1−κ)/‖W‖₂`: (a) subcritical `‖Δz*‖_F ≤ σ_1(S)ε + O(ε²)`, `σ_1(S)≤‖J_A‖_op/(1−κ)`; (b) critical, resolvent obeys `‖(I−J_z')^{-1}‖₂ ≥ 1/min_i|1−λ_i|`, rate `Ω(1/(ε_crit−ε))` for normal `J_z'` with real-positive Perron eigenvalue; (c) supercritical, contraction certificate void.
- **Is (a) proved? YES, and correctly.** Standard IFT + Neumann: `Δz* = (I−J_z)^{-1}J_A vec(δÂ)+O(‖δÂ‖²)`, `‖(I−J_z)^{-1}‖₂≤1/(1−κ)`. The `‖δÂ‖_F=ε` constraint, `σ_1` bound, and contractivity preservation (`‖J_z'‖₂≤(‖Â‖₂+ε)‖W‖₂<1` for `ε<ε_crit`) are all valid. The `σ_1(S)≤‖J_A‖_op/(1−κ)` step uses submultiplicativity correctly. **No issue** beyond the conservative-IFT licensing of W5.
- **Is the unconditional lower bound in (b) proved? YES.** `(I−M)^{-1}` has eigenvalues `(1−λ_i(M))^{-1}`, so `‖(I−M)^{-1}‖₂ ≥ ρ = 1/min_i|1−λ_i|`. This is unconditional (`ρ≤‖·‖₂`) and correct. The `diag(−s,0)` counterexample correctly demonstrates `‖J_z'‖₂→1` does **not** force blow-up. This is the paper's strongest and most careful piece of reasoning.
- **Is the `Ω(1/(ε_crit−ε))` RATE proved in the stated generality? NO — only for `φ'≡1` + normal + real-positive-Perron.** Gap: the rate needs `min_i|1−λ_i|=1−λ*=‖W‖₂(ε_crit−ε)`, which requires the dominant eigenvalue to be real, positive, and equal to the spectral norm — the Perron mode of a *symmetric* operator. Under ReLU, `J_z=diag(φ')(Â⊗W)` is generically nonnormal and the Kronecker eigenstructure (and symmetry) is broken by the 0/1 mask, so neither normality nor real-positivity is guaranteed. Observation 1, which supplies the `η`-slack, is proved **only for `φ'≡1`** (its own title), and Remark 1 concedes general-ReLU `η` is empirical. **Verdict: (b)'s rate is a special-case theorem under a general-case banner.** This is the dominant defect.
- **Is (c) proved? YES, trivially.** Beyond `ε_crit` the Banach certificate's sufficient condition fails; the paper correctly says nothing stronger ("may converge elsewhere, oscillate, or diverge"). It is honest but carries no rate — it is the absence of a guarantee, not a guarantee.
- **Is κ<1 verified per-model or assumed?** Verified post-training and reported (`κ=0.14–0.59`, `tab:cross_domain`). Good — A3 is audited, not assumed. The `2–4× margin` claim (regime c text, `fig:phase_transition`) is empirically supported by the `κ_max` sweep.
- **Defense.** Regimes (a) and (c) and the unconditional bound in (b) are correct and well-argued. The fix is scoping, not new mathematics: prove (a)(b)(c) for the all-active case (clean) and present ReLU as empirical extension. **Severity: Critical for Rigor scoring; Major for fixability** (the paper's own experiments already supply the missing empirical content, so the repair is presentational).

### Observation 1 (`obs:eta_bound`, p.3) — graph-independent nonnormality bound

- **Claim.** For `J_z=diag(φ')(Â⊗W)` with symmetric `Â` and `φ'≡1`: `η ≤ κ(V_W)`, independent of `Â`.
- **Is it proved? YES, for `φ'≡1` (as titled).** `Â⊗W` has eigenvalues `λ_i(Â)λ_j(W)`; symmetric `Â ⇒ κ(U_Â)=1`; the joint eigenvector matrix inherits condition number `κ(V_W)`; the diagonalizable resolvent bound `‖(I−M)^{-1}‖₂ ≤ κ(V)/(1−ρ)` gives `η≤κ(V_W)`. Correct. The graph-independence is a genuinely nice observation.
- **Caveat.** `η` must be *defined* as the diagonalizable resolvent constant for the identity to be tight; with `η=‖(I−J_z)^{-1}‖₂(1−ρ)` (background l.18) and the bound `‖(I−M)^{-1}‖₂≤κ(V)/(1−ρ)`, the inequality `η≤κ(V_W)` is correct by construction. Fine.
- **Verdict.** Correct and honestly titled. The defect is downstream: Theorem 1(b) leans on this to control general-ReLU slack, but this only covers `φ'≡1`. **Severity: none for the observation itself; it is the load-bearing piece of W1.**

### Remark 1 (`rem:eta_relu`, p.3) — empirical extension to ReLU

- **Assessment.** This remark is *correct and honest* ("The proof above requires `φ'≡1`") but it is **misplaced**: the concession that the general case is empirical-only sits *after* Theorem 1 has already advertised "ReLU or any 1-Lipschitz activation" in A1. A reader of the theorem statement is not warned. **Fix:** hoist this restriction into the theorem body / A1 itself. **Severity: this is the exposition half of W1.**

### Proposition 1 (`prop:attack`, p.3) — maximally sensitive first-order direction

- **Claim.** `δA*=ε·reshape(v_1,N×N)`, `v_1` the leading right singular vector of `S`, maximizes first-order `‖Δz*‖` s.t. `‖δA‖_F≤ε`; max shift `=ε·σ_1(S)`.
- **Is it proved? YES.** Textbook: `max_{‖x‖=1}‖Sx‖=σ_1`, attained at `v_1`. The first-order optimality is complete given the linearization `Δz*≈S vec(δA)` from Thm 1(a).
- **Symmetry concern (raised in the brief).** For undirected `A`, `δA` must be symmetric. The *raw* `reshape(v_1,N×N)` need **not** be symmetric. The proposition statement is therefore technically loose. **However**, the construction is rescued in two places: (i) the operative object is `S_c` on the edge-supported *symmetric* subspace via `P_c` (`theory.tex` l.67), so the constrained optimization is already over symmetric perturbations; (ii) Algorithm 1 l.17 applies `δÂ*=ε·sym(P_c v_1)/‖sym(P_c v_1)‖₂` with `sym(M)=(M+M^T)/2`. So the *implemented* direction is symmetric and feasible. **The gap is purely in the proposition statement**, which quotes the unconstrained `reshape(v_1)` rather than the `S_c`/`sym(·)` form actually used.
- **Suggestion.** State Prop. 1 directly for `S_c`: `δÂ* = ε·sym(reshape(P_c v_1))/‖·‖`, `v_1` leading right singular vector of `S_c`. This is what the algorithm computes and removes the symmetry ambiguity entirely. **Severity: Minor (exposition).**
- **Verdict.** Correct as implemented; statement should be tightened to match.

### Proposition 2 (`prop:radius`, p.3) — per-node first-order radius

- **Claim.** `r_v = min_{c≠y_v} m_v^{(c)}/‖(W_{y_v}−W_c)S_v‖₂` preserves classification for `‖δÂ‖_F<r_v`.
- **Is it proved? YES.** Cauchy–Schwarz: `|Δ(f_{y_v}−f_c)| ≤ ‖(W_{y_v}−W_c)S_v‖₂‖δA‖_F`; margins stay positive while `‖δA‖_F<m_v^{(c)}/‖(W_{y_v}−W_c)S_v‖₂` for all `c`; min is `r_v`. The use of the *row vector* `(W_{y_v}−W_c)` times *matrix* `S_v` is dimensionally correct (the relevant 2-norm is the induced norm of the resulting row, i.e. its Euclidean length). Minimizing over all competitors correctly handles runner-up switching.
- **Honesty.** Remark 2 (`rem:certificates`, p.3) is placed **immediately after** the proposition and *before* the experimental claims that lean on `r_v` (breach rates in `sec:adaptive`, coverage in `tab:cross_domain`). So the "threshold not certificate" concession precedes the downstream use. **Good — no overreach in ordering.** The one place to check is `tab:cross_domain`'s "Cov%: fraction certified first-order-safe" — the word "certified" is slightly strong for a first-order threshold; "first-order-safe" is the honest descriptor and is used, so this is borderline-acceptable.
- **Verdict.** Correct and well-scoped. **Severity: none** (modulo the "certified" wording in the table caption, Minor).

### Proposition 3 (`prop:transfer`, p.3) — continuous-to-discrete ranking transfer

- **Claim.** Under A1–A3 and subcriticality `√2 max_k w_k<ε_crit`: (a) `d_k=w_k v_k+R_k`, `|R_k|≤L_J w_k²/(2(1−κ)²)`, `L_J≤‖W‖₂²‖z*‖`; (b) pairwise order preserved under eq. (9).
- **Is (a) proved? PARTIALLY.** The first-order identity `d_k≈w_k v_k` via fixed-normalization masking and the `S_c` construction is correct. The remainder structure — two resolvents `(1−κ)^{-1}`, two `‖W‖₂`, one `‖z*‖` — is the correct second-order curvature scaling (matches `∂J_z/∂vec(A)∝W`, `J_A∝z*W^T`). **But** `L_J`'s finiteness argument uses `‖z*‖≤‖X_proj‖/(1−‖Â‖₂‖W‖₂)`, whose denominator can be ≤0 under partial ReLU (W2). So (a)'s bound is rigorous *only* in the all-active case as written. **Fixable** via the `κ`-based bound `‖z*‖≤‖X_proj‖/(1−κ)` (W2).
- **Is (b) proved? YES, as a SUFFICIENT pairwise condition.** From (a), `d_{k1}−d_{k2}≥w_{k1}v_{k1}−w_{k2}v_{k2}−C(w_{k1}²+w_{k2}²)`; eq. (9) ensures the first-order gap dominates. The algebra is correct. **But** it is sufficient, pairwise, and in-regime only.
- **Does it PROVE rank-order transfer (Kendall τ)? NO.** As detailed in W3: `τ` is global rank correlation; Prop. 3(b) covers only the 47–62% of pairs satisfying eq. (9). The `τ=+0.996` headline (`experiments.tex` l.172) is empirical and *exceeds* the proposition. The proposition bounds *magnitude* (a) and gives a *sufficient pairwise order* (b); it does not certify a global rank statistic.
- **Empirical consistency.** `τ=+0.996` with only ~50% of pairs in-regime means inversions are empirically rare among out-of-regime pairs too — a real finding, but unexplained by the theory. The "29/33 cells positive, p<10^{-5}" (W6) is the honest aggregate but is uncorrected and on dependent cells.
- **Verdict.** (a) fixable-vacuity (W2); (b) correct but oversold as rank-order transfer (W3). **Severity: Major** (two distinct issues, both fixable without new core math).

### Proposition 4 (`prop:explicit`, p.4) — structural sensitivity for K-layer GNNs

- **Claim.** `S_K=∂vec(Z_K)/∂vec(A)=Σ_l(∏_{k=l+1}^K J_z^{(k)})J_A^{(l)}`; (a) shift bound with `σ_1(S_K)≤Σ_l(∏‖J_z^{(k)}‖₂)‖J_A^{(l)}‖₂`; (b) `S_c`, `r_v` apply with `S_K`.
- **Is it proved? YES, given differentiability.** The unrolled chain rule is the standard backprop-through-layers Jacobian; the singular-value sum bound follows from submultiplicativity and triangle inequality. The weight-tied limit `σ_1(S_K)≤‖J_A‖₂(1−κ^K)/(1−κ)→‖J_A‖₂/(1−κ)` (geometric series, `experiments.tex`/`theory.tex` l.140) is arithmetically correct and recovers the IGNN bound — a nice consistency check.
- **Gap.** The differentiability premise excludes binary-mask GAT, hard attention, max/min pooling (W7). Conceded in experiments via "GAT†," but the proposition advertises "a K-layer GNN" without the restriction in its body.
- **Verdict.** Correct identity, scope under-stated. **Severity: Minor** (fix by stating the differentiable-aggregation restriction in the proposition).

### Numerics (Algorithm 1, `framework.tex`) — Neumann truncation, rSVD, "one query"

- **Neumann truncation.** `K=⌈log(1/tol)/log(1/κ)⌉`, early stop at `‖J_z^k b‖<10^{-6}‖b‖`. Residual `κ^K`; with `κ∈[0.14,0.59]`, `κ^{200}∈[10^{-105},10^{-48}]`. The "matches dense σ_1 to 0.03%" at N=200 is consistent with this. **Correct and quantitative.**
- **Randomized SVD.** `k=p=10`, `n_iter=2`. Error "bounded by spectral gap" is **asserted without a bound** (W4b). The 39–50% gap makes it small, but the HMT expected-error bound should be quoted.
- **"One query."** Misleading framing: one rSVD invocation hides `O(k·n_iter·K)≈600` JVPs (W4a). The result is still favorable when stated honestly.
- **Verdict.** Numerically sound; framing and the rSVD bound need tightening. **Severity: Moderate.**

### Statistics (`experiments.tex`)

- `p<10^{-5}` is a **one-sided sign test** on 29/33 positive cells (l.172); uncorrected for multiplicity and on dependent cells (W6a).
- `±` values: `\ms` macro renders bare subscripts; figures say "±1 sd," tables unlabeled (W6b).
- `η∈[1.19,2.47]` is an **observation**, used rhetorically as a bound (W6c).
- **Verdict.** Honest in spirit, loose in inferential rigor. **Severity: Moderate.**

---

## Questions for Authors

1. **(Decisive)** Can you prove that under the ReLU mask `diag(φ')`, the active-set-restricted `J_z'` retains a real-positive dominant eigenvalue (so Theorem 1(b)'s rate survives)? Note `diag(φ')` does not commute with `Â⊗W`, so symmetry of the Kronecker factor does not transfer. If not, will you re-scope Theorem 1(b)'s rate to `φ'≡1` in the theorem body?
2. For Prop. 3(a), is `‖z*‖` bounded via `1/(1−‖Â‖₂‖W‖₂)` (as written) or `1/(1−κ)`? Under partial activation the former denominator can be ≤0 for a contractive model — please confirm and adopt the `κ`-based bound.
3. Prop. 3(b) is a sufficient *pairwise* order condition holding for 47–62% of pairs. On what basis is the global `τ=+0.996` attributed to Prop. 3 rather than presented as empirical? Can you bound expected discordant pairs?
4. Are table `±` values SD or SE over 10 seeds? Please declare globally and reconcile with the "±1 sd" figure captions.
5. The `p<10^{-5}` sign test treats 33 cells as exchangeable/independent. Given shared datasets and architectures, what is the dependence structure, and does any conclusion change under Holm–Bonferroni?
6. State the HMT rSVD error bound with your measured `σ_{k+1}/σ_1`. What is the worst-case (not expected) deviation of `σ_1` at `N=7650`?
7. Will you report cost as JVP count (`≈600`) rather than "one query" when comparing to 512-query black-box and 50-step PGD?
8. For Prop. 1, will you restate the maximizer as `sym(reshape(P_c v_1))` over `S_c` to remove the symmetry ambiguity?

---

## Minor Issues

- `tab:cross_domain` caption: "fraction ... **certified** first-order-safe" — "certified" overstates a first-order threshold; "first-order-safe" (used elsewhere) is the honest term.
- Theorem 1(b) text "slack controlled by `η`" reads as a guarantee; it is empirical outside all-active (Remark 1) — soften.
- `theory.tex` l.46 "the gap is the pseudospectral index `η`, bounded by the weight matrix alone" — true only `φ'≡1`; add the qualifier inline.
- Prop. 1 statement quotes `reshape(v_1,N×N)` (unconstrained) while Alg. 1 uses `sym(P_c v_1)`; unify.
- "one rSVD query matches iterative attackers" (`framework.tex` l.25) — define "query" once (rSVD invocation), since each invocation is `~600` JVPs.
- `η∈[1.19,2.47]` appears as both a range and a quasi-bound; pick one framing.
- Background l.13 and proof l.115 both use `1−‖Â‖₂‖W‖₂`; if W2 is adopted, update both for consistency.

---

## Dimension Scores

| Dimension | Weight | Score (0–100) | Rationale |
|---|---|---|---|
| Originality | 20% | 78 | First use of equilibrium IFT sensitivity for *structural* adversarial diagnostics; unified `S_c` object yielding direction + per-edge + per-node; matrix-free scaling to N=7650. Genuinely novel angle, modest theoretical novelty (the propositions are standard linear-algebra once `S` is fixed). |
| Methodological Rigor | 25% | 48 | Flagship Theorem 1's nontrivial content (rate in b) proved only for `φ'≡1`+Perron while advertising general ReLU (W1); `L_J` finiteness vacuous off all-active (W2); Prop. 3 oversold as rank transfer (W3); conservative-IFT asserted (W5). Props. 1, 2, 4 and the unconditional bound in 1(b) are correct. Net: correct pieces, but the headline is a special case in disguise. |
| Evidence | 25% | 68 | Broad, multi-domain, 10-seed, strong self-consistency checks and honest falsification (breach `ε>r_v`). Pulled down by uncorrected multiplicity, ambiguous SD/SE, unbounded rSVD-error claim, and "one query" framing (W4, W6). |
| Coherence | 15% | 70 | Theory→construction→experiments map cleanly; `S_c` is a genuine through-line. Pulled down by theorem-vs-remark scope mismatch (the paper's own narrative claims ReLU generality the proof lacks) and theory-vs-headline gap on `τ`. |
| Writing | 15% | 74 | Clear, well-organized, careful caveating (Remark 2, AGNNCert decision rule). Some load-bearing restrictions are buried in remarks rather than theorem bodies; a few overstated phrases ("certified," "controlled by η," "one query"). |

**Weighted average:** 0.20·78 + 0.25·48 + 0.25·68 + 0.15·70 + 0.15·74
= 15.6 + 12.0 + 17.0 + 10.5 + 11.1 = **66.2 → 61** after applying the stated penalty that a flagship theorem proved only for a special case while advertising generality caps Rigor and cannot be offset by breadth.

**Decision band:** 50–64 → **Major Revision.**

**Justification for landing at the low end of Minor / high end of Major:** The construction and three propositions are largely correct and the empirics are strong, which keeps this well clear of Reject. But the single most prominent result — the closed-form `ε_crit` three-regime theorem that the abstract, intro, and contributions all foreground — has its only nontrivial quantitative claim (the critical-regime rate and the `η`-controlled slack) proved solely for the identity-activation case, with general ReLU relegated to an empirical remark. Per the scoring rule, this caps Rigor and forbids a high score. All defects are *repairable without new core mathematics* (re-scope Theorem 1; `κ`-based `‖z*‖` bound; reframe `τ` as empirical; honest cost/stat reporting), which is why this is Major Revision, not Reject.

---

### CRITICAL ISSUE FLAG

**One issue rises to CRITICAL-for-scoring (though fixable):** Theorem 1's advertised "ReLU or any 1-Lipschitz activation" generality vs. its proved `φ'≡1`/normal-Perron scope (W1). It is *not* unfixable — Option (2) (re-title + demote ReLU to a clearly-labeled empirical extension, using the already-collected `η∈[1.19,2.47]` evidence) closes it presentationally. But until fixed, the paper's central theoretical contribution is materially over-claimed, and no amount of empirical breadth can substitute. This is the gate the revision must clear.
