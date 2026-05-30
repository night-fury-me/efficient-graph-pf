# Editorial Decision

## Manuscript Information
- **Title:** AEGIS: One-Query Adversarial Diagnostics over the GNN Vulnerability Spectrum
- **Target venue:** ICDM-class (IEEE conference format, 10-page limit)
- **Decision Date:** 2026-05-30
- **Review Round:** Full panel (5 reviewers), simulated

---

## Decision

### **MAJOR REVISION**

A genuinely useful, technically sound, and unusually honest paper whose **abstract, title, and Theorem 1 statement overclaim relative to its own body**. No reviewer found a fatal flaw; both CRITICAL findings are *over-scoping/over-claiming* defects that the body's existing evidence already supports fixing. But re-scoping a flagship theorem, restructuring the headline attacker comparison, fixing a finiteness bound, and adding corrected statistics are structural changes that require **re-review** — hence Major, not Minor.

---

## Reviewer Summary

| Reviewer | Identity | Recommendation | Confidence |
|----------|----------|---------------|------------|
| EIC | Senior AC, data-mining venue | Major Revision | 4/5 |
| R1 Methodology | Implicit-DL / spectral / rNLA theorist | Major Revision | 4/5 |
| R2 Domain | Adversarial graph-ML specialist | Minor Revision | 4/5 |
| R3 Perspective | Power systems + safety-critical ML | Minor Revision | 4/5 |
| Devil's Advocate | Core-thesis challenger | Survives *Partially* (1 Critical, 3 Major) | — |

**Synthesized dimension picture** (mean of the 4 scoring reviewers): Originality ~71 · Rigor ~66 (R1 alone: **48**, gated by the theorem scope) · Evidence ~71 · Coherence ~73 · Writing ~77 → numeric mean **≈69** (Minor/Major boundary). Conservative arbitration (below) places the decision at **Major**.

---

## Consensus Analysis

### Points of Agreement

**[CONSENSUS-5] (all five reviewers) — The body is honest; the abstract/intro/title are not calibrated to it.**
The single dominant finding. EIC (W2), R1 (W3), R2 (W1), R3 (W1/W4), and DA (C1, M3) independently converge: the paper *internally* concedes the right caveats (Shift-PGD is "solver validation, not an independent baseline"; `r_v` "is not a certificate"; GR-BCD decorrelation on Cora) — but the abstract and intro recombine the best-case numbers into claims the body does not support.

**[CONSENSUS-4] (EIC, R1, R2, DA) — The headline attacker claim ("matches a 50-step PGD") is partly true-by-construction.**
The 50-step comparison leans on Shift-PGD, which uses AEGIS's *own* IFT gradients on the first-order objective the leading singular vector maximizes by definition. The only *independent* gradient baseline (Cls-PGD) is beaten chiefly in the small-ε regime; DA documents that an independent *scalable* attacker (GR-BCD, Cora k=5) actually beats AEGIS (1.207 vs 0.643, τ=+0.16), and that prediction *flips* are 0–1.8% for every method — i.e., the win is on the equilibrium-shift proxy, not on label changes.

**[CONSENSUS-3] (R1, DA, EIC) — Theory↔headline mismatch on Theorem 1.**
The rigorous result holds only for contractive spectral-norm IGNN, and (per Remark `eta_relu`) its non-trivial content is proved for `φ'≡1`; the abstract leads with "we prove a closed-form three-regime characterisation" while foregrounding the *general* GCN/SAGE/GIN/APPNP/GAT extension, which carries no regime guarantee.

**[CONSENSUS-3] (EIC, DA; R1 partial) — Originality is consolidation/operationalization, not invention.**
The three diagnostics are three reads (leading singular vector / column norms / per-node margins) of one IFT Jacobian the paper says "specialises equilibrium IFT sensitivity via the standard duplication-matrix `P_c`." (R1 scored Originality higher, 78, crediting the matrix-free operationalization — see Disagreement 2.)

### Points of Disagreement

**Disagreement 1 — Severity of the two CRITICAL findings (fatal vs fixable).**
- **DA / R1 view:** CRITICAL. The attacker headline (DA C1) and the theorem scope (R1 W1) are the paper's two flagship claims.
- **EIC / R2 / R3 view:** Major at most; "no Critical issues," "fixes are predominantly reframing."
- **Type:** Severity disagreement.
- **Editor's resolution:** Treat both as **CRITICAL-but-non-fatal**. Both flaggers explicitly downgrade from fatal (R1: "repairable without new core mathematics"; DA: "downgraded by the genuine transfer/black-box tests, 99%/44%"). The defects are real (they touch the title and abstract) but the body already contains the honest evidence needed to fix them. → Required, gating, but salvageable in revision.
- **Rationale:** Conservative principle + the IRON rule that a DA-CRITICAL finding forbids Accept; the existence of correct underlying experiments forbids Reject. Major Revision is the calibrated midpoint.

**Disagreement 2 — Overall decision: Major (EIC, R1) vs Minor (R2, R3).**
- **Minor camp (R2 76, R3 73):** reviewers focused on literature and cross-domain impact, where the paper is strong (Lit Integration 86; genuine grid result).
- **Major camp (EIC 67, R1 61):** the EIC (decision role) and the methodology expert *most relevant to the central claim* both land on Major; R1's Rigor = 48 because the flagship theorem advertises more than it proves.
- **Type:** Decision/severity disagreement.
- **Editor's resolution:** **Major Revision.** The changes needed (re-scope a theorem, restructure the headline experiment's framing and metrics, fix a math bound, add multiplicity-corrected statistics) materially change tables and claims and warrant re-review — the definitional test for Major over Minor. The methodology expert's CRITICAL on the flagship theorem outweighs the literature/impact reviewers' optimism on the central claim.

**Disagreement 3 — Originality score (R1 78 vs EIC 58 / DA "thin").**
- **Type:** Perspective difference (operational novelty vs conceptual novelty).
- **Editor's resolution:** Both are right about different axes. Conceptually thin (it is known IFT + rSVD), operationally non-trivial (matrix-free, N=7,650, matches dense σ₁ to 0.03%). The revision should *claim the axis it earns* — operationalization/consolidation — and drop conceptual-novelty language ("spectrum", "we prove ... characterisation" framing).

---

## Decision Rationale

AEGIS is a well-engineered, well-cited, and refreshingly self-critical paper. R1 confirms the core operator `S_c=(I−J_z)⁻¹J_A P_c`, its Neumann/rSVD numerics, Propositions 1/2/4, and the unconditional resolvent bound are **correct** and genuinely scale; R3 confirms a real, non-obvious cross-domain grid result (binary-beats-admittance, P@10 0.81 vs 0.27 on case118); R2 credits a 91-reference literature net and decisive independence tests (transfer 99% vs black-box 44%). The science is largely there.

The problem is calibration. Five reviewers independently found that the **abstract, title, and Theorem 1 statement promise more than the body delivers** (CONSENSUS-5). Two of those over-claims are CRITICAL because they sit on the paper's two flagship results: (i) the headline "one-query SVD matches a 50-step PGD" rests on a self-validation baseline and an equilibrium-shift proxy rather than prediction flips, and an independent scalable attacker beats AEGIS where the comparison is fair (CONSENSUS-4); (ii) "we prove a closed-form three-regime characterisation" advertises ReLU generality while the non-trivial content is proved for `φ'≡1` (CONSENSUS-3).

I chose Major over Minor because fixing these requires re-scoping a theorem, restructuring the headline experiment's primary baseline and metric, correcting a finiteness bound (R1 W2) and the multiplicity of the `p<10⁻⁵` test (R1 W6), and reframing "competitive with industry LODF" (R3 W1) — structural edits that change tables and claims and must be re-reviewed. I chose Major over Reject because *no new core science is required*: every fix is supported by evidence already in the body. This is the encouraging kind of Major Revision.

---

## Required Revisions (Must Fix)

| # | Revision Item | Source | Severity | Section |
|---|--------------|--------|----------|---------|
| R1 | Re-scope Theorem 1 to the proved case; demote general-ReLU to a labeled empirical extension; align abstract verb | R1, DA, EIC | **Critical** | theory.tex, abstract |
| R2 | Restructure headline attacker claim around the *independent* baseline + report prediction flips, not only shift | DA, R2, EIC | **Critical** | experiments.tex (§Four-Quadrant), abstract |
| R3 | Fix the `L_J` finiteness bound via `1/(1−κ)` (A3 only); current denominator can be ≤0 | R1 | Major | theory.tex (Thm 1 proof) |
| R4 | Reframe Prop. `transfer`: it proves magnitude + a 47–62%-of-pairs sufficient order, not global rank; move `τ=+0.996` (Amazon-only) out of the abstract as "typical"; disclose failing cells | R1, DA | Major | theory.tex, experiments.tex, abstract |
| R5 | Reframe "competitive with industry LODF" (LODF is exact, ~150× faster); foreground the binary-beats-admittance result instead | R3 | Major | case_study.tex, abstract |
| R6 | Close the motivational↔evaluated gap: drug-interaction/fraud are invoked but never tested | R3 | Major | introduction.tex, abstract |
| R7 | Statistics hygiene: name the test behind `p<10⁻⁵` + apply Holm/BH multiplicity correction across 33 cells; label all `±` as SD or SE; state the rSVD error as bound or empirical; make "one query ≈ 600 JVPs" definition consistent | R1, R2 | Major | experiments.tex, framework.tex |

### Required Item Details

**R1 — Re-scope Theorem 1 (CRITICAL).**
- *Problem:* A1 advertises "ReLU or any 1-Lipschitz," but the regime-(b) rate and η-slack are proved only for `φ'≡1` ("all-active case"); Remark `eta_relu` concedes general-ReLU η∈[1.19,2.47] is empirical. The abstract says "we prove a closed-form three-regime characterisation."
- *Requirement:* State Theorem 1 for the case actually proved (linear/all-active), present the general-ReLU behaviour as a clearly-labeled **empirical extension**, and change the abstract verb (e.g., "we characterise ... and prove it for the all-active regime; trained ReLU models satisfy it empirically with η∈[1.19,2.47]").
- *Acceptance:* Theorem assumptions match the proof's scope; no abstract/intro sentence claims a proof broader than what is shown.

**R2 — Restructure the headline attacker claim (CRITICAL).**
- *Problem:* "One-query SVD matches a 50-step PGD" leans on Shift-PGD (AEGIS's own IFT gradients — the paper itself calls it "solver validation, not an independent baseline") and is scored on equilibrium shift; independent GR-BCD beats AEGIS at Cora k=5 (1.207 vs 0.643), and flips are 0–1.8% for all methods.
- *Requirement:* Make **Cls-PGD** (independent) the primary comparator in the abstract; report **prediction-flip rates** beside equilibrium shift; state explicitly the regime (small-ε, first-order) where the SVD direction is optimal *by construction*, and concede that scalable budget-heavy attackers (GR-BCD) can exceed AEGIS at larger budgets.
- *Acceptance:* Every abstract attacker claim is traceable to an independent baseline and a label-level metric, with the operating regime named.

**R3 — Fix the finiteness bound (Major).** Replace the `‖z*‖ ≤ ‖X_proj‖/(1−‖Â‖₂‖W‖₂)` step (denominator can be ≤0 under partial ReLU) with a bound through `1/(1−κ)` using A3 only. *Acceptance:* denominator provably positive for every contractive model.

**R4 — Reframe the transfer claim (Major).** Report `τ` as an empirical quantity; replace the abstract's single best-case `τ=+0.996` (Amazon Photo only) with a representative statistic (median/IQR across the 33 cells) and name the non-positive cells (e.g., GCN-2 τ=−0.04; 4/33 non-positive). *Acceptance:* abstract reports typical-not-best, failures disclosed.

**R5 — Reframe the LODF claim (Major).** Position `S_c` as a *learned-surrogate* diagnostic, not a rival to exact LODF; report the honest "competitive but not dominant" (P@10 0.60) on LODF's fair target and foreground the genuine contribution (binary-edge ranking beating admittance, P@10 0.81 vs 0.27). *Acceptance:* no "competitive with industry LODF" appears without the exactness/speed caveat.

**R6 — Close the motivation gap (Major).** Either soften the abstract/intro framing to the domains actually tested (citation / co-purchase / IEEE power) or add a safety-relevant dataset (e.g., a molecular or fraud graph). *Acceptance:* every domain named in abstract/intro is either evaluated or explicitly flagged as motivation only.

**R7 — Statistics hygiene (Major).** As tabulated. *Acceptance:* corrected p-values with the test named; all dispersion labeled SD/SE; rSVD error qualified; "one-query" definition consistent across abstract/intro/framework.

---

## Suggested Revisions (Should Fix)

| # | Revision Item | Source | Priority |
|---|--------------|--------|----------|
| S1 | Validate the AGNNCert "complementary, not interchangeable" rule with a per-quadrant breach-rate experiment, or demote `r_v` to an explicit "first-order screen" | R2 | P2 |
| S2 | Evaluate GR-BCD/PR-BCD in their intended large-budget regime; add the missing PR-BCD numeric row | R2 | P2 |
| S3 | Reframe Contribution 1 as consolidation/operationalization; reconsider "spectrum" in the title | EIC, DA | P2 |
| S4 | Relieve the 10-page over-pack: demote either the explicit-GNN extension or the power case study to make room for legible proofs + the new independent-baseline comparison | EIC, R1 | P2 |
| S5 | Reconsider the disclosure protocol: it releases the `v_ij` target list unconditionally while gating the (weaker) SVD-reconstruction artifact | R3 | P3 |

---

## Revision Roadmap

### Priority 1 — Structural / claim-calibration (the gating fixes)
- [ ] **R1** Re-scope Theorem 1 + align abstract verb (theory.tex, abstract)
- [ ] **R2** Restructure §Four-Quadrant around an independent baseline + report flips (experiments.tex, abstract)
- [ ] **R3** Fix `L_J` finiteness bound (theory.tex)
- [ ] **R4** Demote `τ=+0.996` to empirical/representative; disclose failing cells (abstract, experiments.tex)
- [ ] **R5** Reframe LODF positioning (abstract, case_study.tex)
- [ ] **R6** Resolve motivation↔evaluation gap (abstract, introduction.tex)

### Priority 2 — Evidence supplementation
- [ ] **R7** Multiplicity-corrected statistics + SD/SE labels + rSVD error qualifier (experiments.tex, framework.tex)
- [ ] **S1** AGNNCert complementarity validation or demotion
- [ ] **S2** GR-BCD/PR-BCD in fair budget regime + PR-BCD row
- [ ] **S3/S4** Originality reframing + page-budget relief

### Priority 3 — Consistency & formatting
- [ ] Reconcile numeric inconsistencies: AGNNCert cell 0.187 vs 0.163; `r_cert/r_v` ∈ [4.4,15.0] vs footnote [4.9,10.2]; weighted-vs-binary transfer inconsistency; the two distinct "N-1 τ" numbers co-located in the abstract (Amazon τ=+0.996 vs power τ=0.37–0.62 — disambiguate that they measure different things)
- [ ] **S5** Disclosure-protocol asymmetry

### Total Estimated Effort
- **Major Revision:** ~3–5 weeks (most effort is reframing + R7 statistics + R3 proof patch; no new core science required).

---

## Closing

We encourage the authors to carefully consider the reviewers' comments and submit a substantially revised manuscript. The work is technically sound and the underlying evidence is strong; the required changes are concentrated in **aligning the abstract, title, and Theorem 1 statement with the paper's own (commendably honest) body**, plus a small number of concrete technical patches. The revised manuscript will undergo another round of review.

---

## Appendix: Reviewer Reports
- `01_eic_review.md` · `02_methodology_review.md` · `03_domain_review.md` · `04_perspective_review.md` · `05_devils_advocate.md`
