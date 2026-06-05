# Reviewer 3 (Perspective) — Numerical Linear Algebra / Dynamical Systems / Safety-Critical Deployment

**Submission:** AEGIS: A Matrix-Free Operator to Audit, Certify, and Defend Graph Neural Networks (AAAI-2026, anonymous)
**Lens:** pseudospectra & nonnormality, matrix-free scaling & truncation control, cost-comparison fairness, deployment evidence, dual-use.
**Mode:** Independent. I did not consult any prior-round material under `paper/review/`.

---

## Summary

AEGIS builds a single constrained sensitivity operator `S_c = (I - J_z)^{-1} J_A P_c` from equilibrium implicit-function-theorem (IFT) sensitivity of an implicit GNN, restricted to structural (edge) perturbations by a projector `P_c`. From one matrix-free query it reads (i) an SVD-optimal attack direction `v_1`, (ii) a per-edge vulnerability ranking via column norms, and (iii) per-node first-order safe radii `r_v`. The same operator yields a distribution-free conformal certificate (a closed-form worst-case conformity-score shift) and a defense (penalizing `σ_1(S_c)`). The theory contributes a three-regime phase transition with a closed-form critical budget `ε_crit = (1-κ)/‖W‖_2` (Thm `thm:phase_transition`) and a constant-factor two-sided bracket `ε_crit ≤ ε_break ≤ (C/β) ε_crit` (App. D, `D_boundary.tex`). Evaluation spans 6 datasets, 7 architectures, 420 runs.

The core construction is genuinely useful and the matrix-free assembly is clean. My concerns from a numerical/dynamical-systems standpoint center on **one regime where the method's own convergence guarantee is void yet is used as the scaling headline (κ≈1.00 at N=7,650)**, and on **framing of the cost comparison and the grid motivation** that, while honestly caveated deep in the appendix, oversells in the abstract/intro. The pseudospectral machinery is used correctly but is more decoration than load-bearing.

---

## Overall recommendation

**Major Revision**, confidence **4/5**.

The contribution is real and several scope notes are admirably honest. But the flagship scaling claim rests on a numerical regime (κ≈1) where the truncated Neumann resolvent does not converge and no computational protocol or error is reported there; this must be closed before the headline can stand. The cost-comparison and grid framing need to be brought into line with what the appendix already concedes. None of these is fatal; all are addressable with text and a modest additional experiment.

---

## Scores (0-10)

| Axis | Score | One-line justification |
|---|---|---|
| **Novelty** | 7 | Unifying a single operator for audit+certify+defend on *structural* sensitivity (vs. input sensitivity) is a fresh, well-positioned synthesis; the matrix-free `P_c`-restricted resolvent reading is new. Components (IFT influence, conformal, smoothing) are individually standard. |
| **Soundness** | 5 | Closed-form `ε_crit`, the two-sided bracket (App. D), and the all-active η bound are correctly derived. But the N=7,650 scaling result sits at κ≈1.00 where the Neumann truncation residual `κ^K → 1` and `1/(1-κ)` diverges, with no reported computation protocol or truncation error there — the headline rests on an uncontrolled regime. |
| **Clarity** | 7 | Generally precise and well-cross-referenced; assumptions (A1)-(A3) explicit. Costs the reader because the most important caveats (smoothing "favors smoothing twice"; κ≈1 is "past the contractive regime") live in the appendix while the abstract states the optimistic version. |
| **Significance** | 6 | Edge-ranking audit (median τ=0.98, 42/42 positive) and the coupled defense are compelling for any GNN deployer. The deterministic certificate is first-order only; the grid story is now (correctly) rescoped to near-zero; fraud evidence is one cluster. |
| **Reproducibility** | 7 | Seeds, rSVD params (`k=p=10`, `n_iter=2`), early-stop tolerance, GPU all stated; code release gated by disclosure. Missing: the per-dataset κ table and the actual S_c-computation protocol at κ≈1 (number of Neumann terms / residual / dense fallback), without which the N=7,650 number is not reproducible. |

---

## Strengths (genuine)

1. **The matrix-free construction is the right object and is done correctly.** Reading `S_c v = (I-J_z)^{-1} J_A P_c v` as a sequence of forward-mode JVPs with an implicit `P_c`, then a randomized SVD (`halko2011finding`) for `(σ_1, v_1)`, is a sound O(Nd)-memory assembly (`framework.tex`). One query returning all four artifacts at 0.24 s (`tab:compute`) is a real engineering result.
2. **The two-sided bracket (App. D, `D_boundary.tex`) is a clean, honest theorem.** `ε_crit ≤ ε_break ≤ (C/β) ε_crit` with `C` dimension-free, `β` an *observable* alignment (`β = ⟨u_1, B u_1⟩`, positive for any connected graph), the upper side attained by an explicit rank-one critical-driving attack, and crisp equality conditions (collapses iff `g_W=1`, `β=1`). On the suite `β≈0.62 ⇒ C/β ≲ 16`. This is the strongest part of the paper and is appropriately scoped to the contractive, symmetric, all-active case.
3. **Several scope notes are unusually honest.** `experiments.tex:74` explicitly flags Amazon Photo's κ≈1.00 as "past the contractive regime `prop:transfer` assumes, so we read it as an empirical regularity, not theory." `F_experiments.tex:182-188` concedes the smoothing comparison "favors smoothing twice (larger ball, best matching)." The conclusion concedes "a contractive surrogate cannot model voltage collapse." This intellectual honesty is worth crediting and is rare.
4. **Assumption (A3) is reported, not assumed away.** Trained `κ = 0.14-0.59` is tabulated (`tab:cross_domain`), and the theorem text states AEGIS still returns rankings/attacks if (A3) fails, with guarantees scoped to the contractive regime. Good practice for a safety claim.

---

## Weaknesses

### CRITICAL

#### C1. The scaling headline (N=7,650) sits at κ≈1.00, where the truncated-Neumann S_c computation has no convergence guarantee, and no computational protocol or error is reported there.
- **What's wrong.** The entire matrix-free pipeline computes `(I-J_z)^{-1}` as a truncated Neumann series, with the convergence guarantee `κ^K → 0` and the resolvent gain `1/(1-κ)`. The paper's truncation-error evidence is `κ^200 ∈ [10^-105, 10^-48]` and `σ_1` agreement within 0.03% — but these hold **at N=200, across a suite where κ<0.8** (`F_experiments.tex:221-223`). The single full-graph result, Amazon Photo at N=7,650, is at **κ≈1.00** (`experiments.tex:74`, `C_rankings.tex:87`). At κ≈1: `κ^K` does not shrink (`0.999^200 ≈ 0.82`), `1/(1-κ)` diverges, and the stated early-stop criterion `‖J_z^k b‖ < 10^-6 ‖b‖` (`F_experiments.tex:371`) would essentially never trigger within the 50-step fixed-point budget. So `S_c v`, `σ_1(S_c)`, and the ranking at N=7,650 are computed in a regime where the method's own error control is void, yet this is the number quoted in the abstract ("scales to N=7,650") and as Contribution (1).
- **Where.** Abstract; `introduction.tex:24` (Contribution 1); `framework.tex` (matrix-free + "lifts the ceiling to N=7,650"); `experiments.tex:74`; `appendix/F_experiments.tex:215, 221-225, 371`.
- **Why it matters.** The headline trades on a tension the paper never resolves: the method *requires* contractivity (κ<1, the smaller κ the faster), but the *only* large-N demonstration is at the boundary where contractivity effectively fails. The honest note at `experiments.tex:74` covers only the *ranking-transfer theory*; it says nothing about whether the *numerical object* `S_c` was computed accurately. `F_experiments.tex:223-224` addresses only the rSVD error ("bounded by the spectral gap"), not the Neumann truncation residual at κ≈1 — the dominant error source there. A safety-critical reader cannot trust `σ_1(S_c)` or the τ=0.996 ranking if the resolvent was never converged.
- **Concrete fix.** (a) Report the **actual computation protocol at N=7,650**: how many Neumann terms were used, the realized residual `‖J_z^K b‖/‖b‖`, and whether the series even met tolerance. (b) Since κ≈1, the truncated Neumann series is the wrong solver — use a Krylov resolvent solve (GMRES/CG on `(I-J_z)x = J_A P_c v`, which converges by spectral *clustering* not by κ<1) and report its residual. (c) Add a κ-vs-truncation-error curve at fixed large N to show where the truncated series breaks. (d) State the per-dataset trained κ in a table; if Amazon Photo's κ is genuinely ≈1.00, either re-train under the spectral cap to a verified κ<1 *or* present N=7,650 explicitly as a Krylov-solved (not Neumann-truncated) result and soften "scales to N=7,650" to reflect the solver actually used.

### MAJOR

#### M1. The "10^4× cheaper than smoothing" headline compares a first-order deterministic bound to a sampling certificate on a ball where both are weak, and the abstract states the favorable half while the appendix states the honest half.
- **What's wrong.** On the *matched* Frobenius ball `σ = ε/√(2|E|)`, randomized smoothing abstains (certifies nothing; `tab:smoothing` Cert=0.00), and AEGIS is "non-vacuous" only as a **first-order linearized** radius — not validated against the true nonlinear breaking point on that ball. The abstract states "stays non-vacuous on the matched Frobenius ball where smoothing abstains, at 10^4× lower cost"; the appendix (`F_experiments.tex:182-188`) is far more candid: smoothing only certifies on the strictly larger per-coordinate ball, and "the comparison favors smoothing twice." A cost ratio is not a fair figure of merit when, on the matched comparison, the cheap method delivers a *linearized* certificate and the baseline a *probabilistic* one — they are different objects, so "cheaper" conflates accuracy with price.
- **Where.** Abstract; `theory.tex:68` (conformal vs ~10^4-sample smoothing); `appendix/F_experiments.tex:180-189` (`app:smoothing`).
- **Why it matters.** A deployer reading only the abstract concludes AEGIS dominates smoothing by 4 orders of magnitude. The truthful statement is: AEGIS gives a *cheap first-order/conformal* certificate; smoothing gives an *expensive probabilistic* one; on the matched ball smoothing happens to abstain because σ is tiny. The win is real but narrow, and the framing is advantageous.
- **Concrete fix.** Move the "favors smoothing twice" concession into the abstract/intro in one clause. State plainly that the matched-ball comparison pits a first-order/conformal certificate against a sampling certificate, and that the cost win does not imply a tighter or stronger guarantee. Also disambiguate **which** AEGIS certificate is non-vacuous on the matched ball — the deterministic `S_c`/`r_v` bound (per `tab:smoothing`) or AEGIS-Conformal — since the abstract attributes it to "AEGIS-Conformal" but the appendix evidence is the deterministic certificate.

#### M2. The pseudospectral / nonnormality apparatus is correct but largely decorative: the operative η numbers (1.19-2.47, "tightens 7-14×") are measured, not derived, and η changes no bound or conclusion.
- **What's wrong.** Exactly one η result is *derived*: `obs:eta_bound` (η ≤ κ(V_W), the eigenvector conditioning of W) holds **only in the all-active case** (φ'≡1), where `J_z = diag(φ')(Â⊗W)` and Â contributes nothing to the joint conditioning (`B_sensitivity.tex:121-137`). For general ReLU patterns, `rem:eta_relu` openly states "the diagonalization argument no longer applies exactly," so the headline numbers η∈[1.19,2.47] and "tracking κ(V_W) closely" are **empirical** (`B_sensitivity.tex:139-145`). The "tightens ε_crit 7-14×" (`theory.tex:37`) is a property of `S_c` being *data-dependent*, not of the nonnormality analysis: η is named as the *slack* between the spectral-radius rate and the resolvent norm, not the mechanism that tightens anything. Trefethen-Embree (`trefethen2005spectra`) is cited but the only thing actually used is the standard diagonalizable resolvent bound `‖(I-M)^{-1}‖ ≤ κ(V)/(1-ρ(M))`.
- **Where.** `theory.tex:30, 37`; `appendix/B_sensitivity.tex:94-145` (`app:proof_eta`, `obs:eta_bound`, `rem:eta_relu`); `related_work.tex:10` ("matrix perturbation theory ... supplies the pseudospectral toolkit").
- **Why it matters.** As written, "η ≤ 2.47 for ReLU; η ∈ [1.19, 2.47]; tightens 7-14×" reads as a derived quantitative refinement, but it is a measured slack plus one all-active inequality. The pseudospectral framing dresses up a standard resolvent estimate. This is a presentation/rigor issue, not an error.
- **Concrete fix.** State explicitly that `obs:eta_bound` is the *only* derived nonnormality result and holds in the all-active case; relabel the [1.19,2.47] range and the 7-14× factor as *empirical measurements* (which they are) and point to the figure/table they come from. Either keep the pseudospectral language but tie it to a bound it actually controls, or trim it to "standard resolvent bound for diagonalizable operators" and drop the implied novelty.

#### M3. Deployment evidence is thin where the paper leans hardest: one fraud cluster (n=1, τ=1.0) and a grid motivation now rescoped to near-zero, leaving the abstract/intro over-promising relative to delivery.
- **What's wrong.** (a) The case study reports τ=1.0 on **one** Amazon Fraud cluster (`case_study.tex:13`, "τ=1.0 *here*"). A single hub is an anecdote; with n=1 there is no variance, no distribution of τ, no failure case. The aggregate evidence (median τ=0.98 over 42 cells, Δτ up to +0.90) is far stronger and should carry the deployment claim. (b) The introduction still opens with three deployment domains including power grids "can miss contingencies when their graph is structurally fragile" (`introduction.tex:6`), parenthetically caveated ("we audit the model, not the grid's physics"), while the conclusion concedes "a contractive surrogate cannot model voltage collapse, so it complements rather than replaces power-flow contingency screening" (`conclusion.tex`). The grid contribution is therefore essentially nil, yet it is one-third of the motivating triad and the title's "safety-critical" framing rides on it.
- **Where.** `introduction.tex:6`; `case_study.tex:13`; `conclusion.tex` (Limitations).
- **Why it matters.** A deployer who is **not** running an IGNN (the vast majority) gets value from the *audit/edge-ranking* path (which transfers to 7 explicit architectures) and from the coupled defense — not from the grid story or a single fraud hub. Leading with grids and a τ=1.0 anecdote risks the reviewer (and reader) discounting the genuinely strong general result.
- **Concrete fix.** (a) Either add 5-10 fraud clusters with a τ distribution (mean ± std, worst case) or demote the single-cluster figure to an *illustration* and let `tab:da_decomp` / `fig:tau_heatmap` carry the evidentiary weight. (b) Reframe the introduction so the grid is named as *motivation for structural fragility in general*, not a delivered capability; given the conclusion's concession, consider dropping the grid contingency clause from the opening triad and replacing it with a domain AEGIS actually serves.

#### M4. The disclosure window is asserted, not justified, given the attack is the most damaging the model admits.
- **What's wrong.** `prop:attack` yields the *optimal* (most damaging) `δA*` — by construction the worst rank-one structural attack the model admits. The mitigation is a "90-day coordinated-notification window" with attack-direction reconstruction (Alg. 1, steps 3-4) gated behind ethics review (`conclusion.tex`; `F_experiments.tex:374`). No rationale is given for *why 90 days*, who the notified parties are, what an "ethics review" gate concretely is, or why releasing the diagnostic path (`r_v`, `v_{ij}`) "unconditionally" is safe when an adversary can plausibly reconstruct a near-optimal direction from the per-edge ranking alone (the ranking *is* the column-norm proxy for `v_1`, per `prop:transfer`).
- **Where.** `conclusion.tex` (Disclosure); `appendix/F_experiments.tex:372-374`.
- **Why it matters.** For a safety-critical, dual-use method, "90 days + ethics review" is a slogan without an operationalization. The claim that the diagnostic path is harmless is in tension with the paper's own finding that the edge-weighted ranking transfers to single-edge-removal damage with τ=0.98 — i.e., the "defender-only" ranking is itself a high-quality attack prioritization.
- **Concrete fix.** Justify the 90 days (or cite the convention you follow), name the notification target (model owners? a CERT-style body?), define the ethics-review gate concretely, and address the leakage from the diagnostic path: quantify how much attack damage an adversary achieves using *only* the released `v_{ij}` ranking vs. the gated `v_1` reconstruction. If the gap is small, the gating is largely cosmetic and should be acknowledged.

### MINOR

- **m1. Per-dataset κ is never tabulated as a standalone column.** The range κ=0.14-0.59 is stated for the contractive suite, but Amazon Photo's κ≈1.00 implies a much wider true range. Add a κ column to `tab:cross_domain`/`tab:da_decomp`. (`theory.tex`, `appendix/F_experiments.tex`)
- **m2. "Spectrum well separated, one query suffices" is shown on one Cora ego-graph (σ_1 43% above σ_2).** The gap "0.39-0.50 across the suite" is asserted; show the gap distribution or note that at κ≈1 the leading-gap argument also degrades. (`framework.tex`)
- **m3. Cost-ratio figures are internally consistent but spread across abstract (10^4×), `F_experiments` (11,700-16,700× matched; 23,000-57,000× larger ball).** State the matched-ball ratio in the abstract to avoid a reader inferring the larger number applies to the matched comparison. (Abstract; `appendix/F_experiments.tex:184-189`)
- **m4. The Neumann band "K∈[20,50] for κ<0.8" (framework) vs. "50-step fixed-point budget at κ_max≥0.85" (F_exp) vs. "κ^200" truncation evidence.** Three different K's appear; reconcile which K is actually used and reported. (`framework.tex`; `appendix/F_experiments.tex:198, 222`)
- **m5. `tab:da_decomp` `D_{Av}/D_v` is ≈1.00 on Pubmed.** The sensitivity adds nothing over edge-weight on at least one dataset; the abstract's "+0.16 to +0.90" range hides that the low end is essentially zero marginal value. Note the spread honestly. (`appendix/F_experiments.tex:145-150`)
- **m6. rSVD with `n_iter=2` and `k=p=10` is aggressive for a near-degenerate spectrum at κ≈1.** Two subspace iterations may not resolve `v_1` if the gap collapses; report the rSVD residual at N=7,650. (`appendix/F_experiments.tex:371`)

---

## Cross-disciplinary opportunities

1. **Krylov / shift-invert resolvent solves.** The κ≈1 regime is exactly where numerical linear algebra abandons stationary (Neumann) iteration for Krylov methods. A GMRES/MINRES solve of `(I-J_z)x = b` converges by eigenvalue *clustering*, not by κ<1, and would make the N=7,650 (and Pubmed N=19,717) results rigorous. This also gives a cheap residual-based error certificate per query — directly strengthening the safety claim.
2. **True pseudospectra, used quantitatively.** If the authors want the nonnormality story to be load-bearing rather than decorative, compute the actual ε-pseudospectral abscissa of `J_z` (e.g., via `eigtool`-style sweeps or the criss-cross algorithm) and show it predicts the *measured* breaking point better than `ε_crit`. That would convert η from a citation flourish into a sharper boundary than the spectral-radius bound — a genuine cross-disciplinary contribution.
3. **Transient (non-asymptotic) amplification.** Nonnormal operators exhibit large transient growth even when contractive. For a *deployed* model evaluated at a finite fixed-point budget (50 steps), the relevant quantity is the numerical abscissa / transient envelope, not the asymptotic resolvent. Connecting `r_v` to transient growth bounds would tighten the deployment story.
4. **Conditioning of the equilibrium under structural perturbation.** `S_c` is essentially a structured derivative of an implicit map; backward-error / structured-conditioning analysis (structured pseudospectra for the `Â⊗W` Kronecker structure) could give the per-edge radius a backward-stable interpretation.

---

## Questions for authors

1. At Amazon Photo (κ≈1.00, N=7,650): how many Neumann terms were used, what was the realized residual `‖J_z^K b‖/‖b‖`, and did the early-stop tolerance `10^-6` ever trigger? If not, how is `σ_1(S_c)`/the τ=0.996 ranking trustworthy?
2. What is the exact trained κ for Amazon Photo? Is it >1, =1±ε, or a borderline value where the spectral cap was relaxed?
3. On the matched Frobenius ball, is the "non-vacuous" certificate the deterministic `S_c`/`r_v` first-order radius or AEGIS-Conformal? If the former, why does the abstract attribute it to AEGIS-Conformal?
4. Is the first-order radius `r_v` ever *validated against the true nonlinear breaking point* on the matched ball, or only stated as non-vacuous? A vacuous-vs-correct distinction matters for a safety certificate.
5. How much single-edge-removal damage can an adversary achieve using **only** the unconditionally released `v_{ij}` ranking, vs. the gated `v_1` reconstruction? (This tests whether the disclosure gate is meaningful.)
6. Why 90 days specifically, and what concretely constitutes the "ethics review" gate?
7. For the η claims: is there any derived bound for the general ReLU pattern, or is η∈[1.19,2.47] purely measured? If measured, will you relabel it as such?
8. Does any conclusion in the paper *change* if the nonnormality analysis is removed and only the spectral-radius resolvent bound `1/(1-κ)` is used? If not, what does η add?

---

*Reviewer 3 — numerical linear algebra / dynamical systems / safety-critical deployment. Independent review; no prior-round material consulted.*
