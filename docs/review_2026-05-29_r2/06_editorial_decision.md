# Phase 2 — Editorial Synthesis & Decision (R2)

**Manuscript.** *AEGIS: Closed-Form Adversarial Diagnostics over the GNN Vulnerability Spectrum.*
**Round.** R2 (post-Major-Revision; R1 panel report at `docs/review_2026-05-29/`).
**Panel.** EIC + R1 Methodology + R2 Domain + R3 Perspective + Devil's Advocate. Reviewers were paper-blind to each other during Phase 1.

---

## Aggregate verdict

| Reviewer | Verdict | Confidence | Headline judgment |
|---|---|---|---|
| EIC | Minor Revision | 4 | Rescoped title earns; contribution (2) over-sells the deflated theorem; acceptable at ICDM, borderline at NeurIPS |
| R1 Methodology | Minor Revision | 4 | Theorems mostly defensible after R2 rewrite; κ²⁰⁰ band number is **wrong** (13/50 rows above the claimed upper bound); salvaged CSV columns empty |
| R2 Domain | **Major Revision** | 4 | PR-BCD claimed but never run; AGNNCert "complementary" framing leans on rank-noise; **full-graph τ on Amazon Photo (G8) has no CSV backing**; defense G6 unclosed despite script existing; Mettack run at its weak budget |
| R3 Perspective | Minor Revision | 4 | Case-study framing now restrained; PI numbers exist but not in `tab:ieee`; LODF-thermal P@10 = 0.60 buried in baselines paragraph |
| Devil's Advocate | **Major Revision floor** (Reject if DA-C1 / DA-C2 / DA-M3 unaddressed) | 4 | 4 CRITICAL + 9 MAJOR. Phase-transition empirically inactive; "closed-form" is a misnomer; "spectrum" is a motte-and-bailey; defense ablation auto-evaluates |

Vote split: **3 Minor / 1 Major / 1 Major-or-Reject (DA)**.

### Iron Rule Checkpoint #4 applied

Devil's Advocate raises 4 CRITICAL issues. **Accept is therefore off the table.** The minimum decision is Minor Revision; the actual decision is set by whether the CRITICAL items can be cleared by mechanical text edits or require compute.

DA-C1 (phase-transition framing), DA-C3 (κ-sweep anchors nothing about (b)/(c)), and DA-C4 (title rename motte-and-bailey) are clearable by text edits + a contribution-block demotion. **DA-C2 ("closed-form" misnomer) is a global terminology fix.** DA-M3 (adaptive attacker for defense ablation) requires compute. R2 Domain's MAJOR findings (PR-BCD on Pubmed + Amazon Photo; full-graph τ on Amazon Photo; Mettack at competitive budget; adaptive defense column) **also require compute, not text edits.**

**Decision floor is Major Revision, not Minor.**

---

## Consensus matrix (≥2 reviewers agree)

| # | Issue | EIC | R1 | R2 | R3 | DA | Source citations |
|---|---|:--:|:--:|:--:|:--:|:--:|---|
| C1 | Phase-transition theorem hollowed out; contribution (2) over-sells the κ-sweep deflation | ✓ | implicit (W1) | implicit | – | ✓ (DA-C1, DA-C3) | EIC §Significance; R1 §3.1(c); DA-C1 |
| C2 | "Closed-form" misnomer — Neumann truncation + rSVD is iterative | implicit | – | – | – | ✓ (DA-C2) | EIC §Originality; DA-C2 |
| C3 | "Vulnerability spectrum" framing exceeds IGNN-only formal coverage | ✓ | – | ✓ | – | ✓ (DA-C4, DA-M1) | EIC §Scope; R2 §Recommendation; DA-C4 |
| C4 | AGNNCert "complementary" framing leans on rank-noise (τ = 0.02–0.14 across seeds) | ✓ (Q3) | – | ✓ | – | ✓ (DA-M2) | EIC Q3; R2 §AGNNCert; DA-M2 |
| C5 | **PR-BCD listed as baseline but never run** | – | – | ✓ | – | – | R2 §PR-BCD |
| C6 | **Full-graph τ on Amazon Photo (G8) has no CSV** | – | – | ✓ | – | – | R2 §G8 |
| C7 | **Defense ablation lacks adaptive attacker (G6)** | – | – | ✓ | – | ✓ (DA-M3 → Reject) | R2 §G6; DA-M3 |
| C8 | Mettack 149/150 win at its weak budget regime (k ∈ {1,…,5}) — category error | – | – | ✓ | – | ✓ (DA-M5) | R2 §Mettack; DA-M5 |
| C9 | **κ²⁰⁰ band incorrect** ($[10^{-105},10^{-48}]$ vs. actual $[10^{-177},10^{-12}]$; 13/50 rows above claimed upper bound) | – | ✓ (W1) | – | – | implicit (DA-M7) | R1 W1 |
| C10 | Halko bound + `sigma_dense` columns empty in salvaged R2_04 CSV — claims unverifiable | – | ✓ (W2, W3) | – | – | ✓ (DA-M7) | R1 W2, W3; DA-M7 |
| C11 | Obs. 3.3(b) mathematically meaningless for general ReLU (cond(diag(φ')⊗I) = ∞ on any inactive node) | – | ✓ (W4) | – | – | – | R1 W4 |
| C12 | 29/33 sign-test treats correlated architecture-dataset cells as exchangeable | – | ✓ (Q5) | – | – | ✓ (DA-M4) | R1 Q5; DA-M4 |
| C13 | PI baseline numbers exist but not in `tab:ieee`; LODF-thermal forward-ref missing in case study | – | – | – | ✓ | – | R3 §Weaknesses |
| C14 | Edge-insertion excluded from threat model; not foregrounded in abstract/related-work | ✓ (Q4) | – | ✓ | – | ✓ (DA-m3) | EIC Q4; R2 §Threat Model; DA-m3 |
| C15 | Standard GAT silently elided from abstract; only edge-weighted GAT† tested | ✓ (W3) | – | – | – | ✓ (DA-m2) | EIC W3; DA-m2 |
| C16 | Ethics statement performative — no named gatekeeper, stakeholder, or review criterion | – | – | ✓ | – | – | R2 §Ethics |
| C17 | t-CIs in `tab:breach` correct but inappropriate for right-skewed Pubmed; BCa bootstrap recommended | – | ✓ (W7) | – | – | – | R1 W7 |
| C18 | tab:baselines Cora r_v = 0.187 ≠ agnncert_comparison.csv median 0.4075 | – | – | ✓ | – | – | R2 §AGNNCert |
| C19 | "Closed-form" SVD direction reaches 0.72–0.92× of IFT-gradient PGD — i.e. PGD beats AEGIS, but abstract frames as parity | – | – | – | – | ✓ (DA-M9) | DA-M9 |

---

## Disagreements requiring arbitration

### D1 — Severity of Theorem 3.1(c) defensive rewrite
- **R1:** "Defensible. The Stewart–Sun lower bound holds; the certificate-fails caveat is meaningful, not vacuous."
- **DA:** "Non-falsifiable. Theorem 3.1(c) is now a non-prediction, and no experiment crosses ε_crit. Headline theorem hollowed out."
- **Arbitration:** R1 is right on the proof's mathematical content; DA is right on its load-bearing role. The rewrite is honest as Theorem 3.1(c). What is dishonest is keeping (2) as a top-line contribution in §1 and "spectrum" in the title when the empirical regime never crosses the boundary. **The proof stays; the framing demotes.** This becomes gating items G1 (contribution-block rewrite) and G4 (title/abstract scope).

### D2 — Severity of "closed-form" terminology
- **DA-C2:** Reject if not addressed.
- **EIC + R1:** flagged but not gating.
- **Arbitration:** DA is right on the terminology. The word "closed-form" appears in the title, abstract, and contribution (1) — and the underlying primitives (Neumann + rSVD) are iterative. **The fix is mechanical:** replace "closed-form" with "single-pass matrix-free" (or "analytical-form with numerical truncation" where formally accurate), keeping it ONLY where it describes a closed-form *expression* (e.g., the formula for $r_v$ in Prop. 3.5, the SVD-optimal direction's analytical form in Prop. 3.4). This is ~30 lines of grep/replace plus careful verification. **Gating.**

### D3 — R2_04 salvaged-CSV trustworthiness
- **R1:** "30-minute re-run with `top_k_svd(k=7)` + repopulate `sigma_dense` is the fix."
- **DA:** "Bug story raises the question of what else hasn't been audited."
- **Arbitration:** R1 has the operational recommendation. The principled fix is re-running R2_04 cleanly; the salvaged CSV is a partial substitute. **Gating with R1's specification.**

### D4 — Decision floor
- **EIC, R1, R3:** Minor Revision (text edits sufficient)
- **R2, DA:** Major Revision (compute required)
- **Arbitration:** Compute is required for (a) PR-BCD on Pubmed + Amazon Photo, (b) Amazon Photo full-graph τ, (c) adaptive defense column, (d) Mettack at competitive budget, (e) R2_04 re-run. None of these are achievable in a Minor-Revision turnaround. **Major Revision.**

### D5 — Defense ablation severity
- **R2:** "Methodological hole — adaptive script exists, just not run."
- **DA-M3:** Reject if not addressed.
- **EIC, R1, R3:** silent.
- **Arbitration:** R2 is correct that the script (`iem/examples/exp_adaptive_attack.py`) is in the tree, so this is hours of compute, not days. **Gating.**

---

## Decision

**Major Revision.**

Justification:
1. Devil's Advocate raises 4 CRITICAL issues (Checkpoint Rule #4 → no Accept).
2. R2 Domain identifies multiple claim-vs-evidence gaps requiring compute (PR-BCD, G8, G6, Mettack) — not Minor-fixable.
3. R1 Methodology identifies a factual numerical error (κ²⁰⁰ band) and an unverifiable claim (σ₁ dense agreement) requiring a CSV re-run.
4. EIC and R3 lean Minor on the grounds that the paper's contribution is real and the rescoping is honest — both of which the synthesizer agrees with. Their lean is reflected in the *kind* of Major Revision required: this is **rescope + close-the-evidence-gaps**, not "redo the paper."

---

## Revision Roadmap

### Gating items (must address — sufficient to clear the gate)

| Gate | Description | Source | Type | Estimated work |
|---|---|---|---|---|
| **G1** | **Demote contribution (2) in §1 from headline phase-transition theorem to "regime characterisation for the IGNN subclass, with empirical regime confined to the subcritical band ($\kappa\!=\!0.14$–$0.59$); see Theorem 3.1 and §sec:phase_transition for the full statement."** Keep Theorem 3.1 in §3 as currently rewritten. | EIC §Significance; R1 §3.1(c); DA-C1 | Writing | 1 day |
| **G2** | **Remove "Closed-Form" from the title and from contribution (1)**, replacing with "Single-Pass Matrix-Free" or equivalent. Replace all instances of "closed-form" in the abstract and intro EXCEPT where it describes a true closed-form expression (the formula for $r_v$ in Prop. 3.5; the SVD-optimal $\delta A^* = \varepsilon \cdot \mathrm{reshape}(v_1, N\!\times\!N)$ in Prop. 3.4). Document the convention in §1.1 or a footnote on the title page. | DA-C2; EIC §Originality | Writing | 0.5 day |
| **G3** | **Soften "Vulnerability Spectrum" framing.** Either (a) replace "spectrum" with "rankings + radii + direction" in title, or (b) add explicit scope clause to abstract: "Formal three-regime characterisation applies to the IGNN subclass (Theorem 3.1); $S_c$ extends as a computational tool to GCN/SAGE/GIN/APPNP/edge-weighted GAT$^\dagger$ without the regime guarantees." Move the AGNNCert sound-vs-first-order distinction into the abstract. | EIC §Scope; R2 §Recommendation; DA-C4 | Writing | 0.5 day |
| **G4** | **Run PR-BCD head-to-head** on Pubmed and Amazon Photo. Add row to `tab:baselines`. Without this, the abstract/intro claim of "head-to-head vs. PR-BCD" must be retracted. | R2 §PR-BCD | Compute | 1–2 days |
| **G5** | **Run Amazon Photo full-graph τ to convergence** ($N{=}7{,}650$, ≥3 seeds). Land the result in `results/revision_R2/fullgraph_repro.csv` (currently Cora + Citeseer only) and in `tab:tau_heatmap` or its successor. Without this, the conclusion.tex limitation-(iii) "AtkAdv amplifies on the full graph" claim is unsupported for the cold cell. | R2 §G8 | Compute | 1 day |
| **G6** | **Add adaptive-attacker column to defense ablation.** The infrastructure (`iem/examples/exp_adaptive_attack.py`) is in the tree. Report damage reduction under (i) non-adaptive top-$k$ $v_{ij}$ masking (current 42±8%, 61±7%), and (ii) adaptive attacker that recomputes $S_c$ after each mask step. Even if the gain erodes, this is required to engage with the Mujkanović 2022 critique that the paper cites. | R2 §G6; DA-M3 (→ Reject) | Compute | 1–2 days |
| **G7** | **Re-run R2_04 with `top_k_svd(k=7)` and `sigma_dense` populated.** Correct the κ²⁰⁰ band number in `experiments.tex` scalability paragraph to the actual range from `matfree_error_bounds_corrected.csv` ($[10^{-177},10^{-12}]$) and report the worst-case row (Amazon seed 2718, $1.87\times 10^{-12}$) explicitly. Repopulate `halko_bound_estimate` and `sigma_dense` columns. | R1 W1, W2, W3 | Compute (30 min) + Writing | 0.5 day |
| **G8** | **Demote Observation 3.3(b) to "Empirical Remark 3.3(b)"** or restrict the cond(diag(φ')⊗I) bound to the active-mask sub-block. As stated, the bound is $\infty$ on any node with $\phi'\!=\!0$ — mathematically meaningless for general ReLU. | R1 W4 | Writing | 0.5 day |
| **G9** | **Either (i) extend Mettack budget to $k\in\{50,100,250\}$** on Cora/Citeseer and report the result, **or (ii) downscale the 149/150 claim** in the abstract to "AEGIS dominates Mettack on equilibrium-shift damage at small $\ell_0$ budgets ($k\!\leq\!5$), the regime relevant to early-warning diagnostics." | R2 §Mettack; DA-M5 | Compute (option i) or Writing (option ii) | 1 day or 1 hour |
| **G10** | **Land the PI baseline in `tab:ieee`** (case57 τ=+0.335, P@10=0.50; case118 τ=+0.101, P@10=0.30 from `pi_baseline.csv`) and **forward-reference the LODF-thermal P@10=0.60 retarget result** in `case_study.tex` baselines paragraph. | R3 §Weaknesses | Writing | 0.5 day |
| **G11** | **Correct or reconcile the AGNNCert numerical mismatch** (tab:baselines Cora $r_v\!=\!0.187$ vs `agnncert_comparison.csv` median 0.4075). Document the decision rule for "complementary" — if the practitioner sees AGNNCert says safe and AEGIS says unsafe, which to trust? | R2 §AGNNCert; EIC Q3; DA-M2 | Writing + verification | 0.5 day |
| **G12** | **Foreground the edge-insertion scope leak** in §II (Threat Model) and in the abstract. The motivating domains in §1 (fraud, drug-interaction) are inherently insertion-dominated; the current limitation-(vi) burial is misleading. Either add Nettack-style insertion to $S_c$ via a candidate-edge basis $\bar E$ extension, or scope the abstract claim. | EIC Q4; R2 §Threat Model; DA-m3 | Writing | 0.5 day |
| **G13** | **Standard-GAT elision: add four words to the abstract** — "GCN, SAGE, GIN, APPNP, IGNN, edge-weighted GAT variant (standard GAT requires continuous edge weights and is out of scope)." | EIC W3; DA-m2 | Writing | 5 minutes |

### Recommended (non-gating)

- **R1 — Multiple-testing correction across 30 breach cells.** Bonferroni or BH. (R1 W6.)
- **R2 — BCa bootstrap for Pubmed `tab:breach` CIs** to handle right-skew. (R1 W7.)
- **R3 — Ethics statement upgrade.** Name a gatekeeper (e.g. an institutional review board or co-author affiliation), define the review criterion, and identify the stakeholders being notified. (R2 §Ethics.) NeurIPS-tier ethics, not ICDM-tier.
- **R4 — IFT/influence-function lineage paragraph in §Related Work.** Position against Koh–Liang and recent 2023–2025 influence-function-for-GNN work. (EIC §Originality.)
- **R5 — PTDF as standalone baseline row** in `tab:ieee`. PTDF will lose to AEGIS more clearly than LODF does — strengthens the case. (R3 §G9.)
- **R6 — Tightness reporting at $\varepsilon\!=\!0.20$**, not $\varepsilon\!=\!0.01$. First-order Taylor is exact at zero — the headline number should be where the bound is non-trivial. (DA-m4.)
- **R7 — Reframe SVD vs PGD ratio.** $0.72\!-\!0.92\times$ PGD means PGD wins; current abstract framing reads as parity. Either reframe as "AEGIS recovers $72\!-\!92\%$ of PGD damage *label-free, single-pass*" or report PGD as the ceiling. (DA-M9.)

---

## Estimated total revision cost

- **Compute (gating):** G4 (1–2 days) + G5 (1 day) + G6 (1–2 days) + G7 (30 min) + G9 option i (1 day) = **4–6 days of GPU wall-clock.**
- **Writing (gating):** G1 + G2 + G3 + G8 + G10 + G11 + G12 + G13 ≈ **4–5 days of focused writing.**
- **Total:** ~**1.5–2 weeks** of focused author effort for one author.

If G9 option ii (downscale the Mettack claim) is taken, the compute budget drops to 3–4 days.

---

## Re-review trigger

Re-submit when:
1. All G1–G13 gating items are addressed (in revised manuscript + R&R response letter using the R→A→C format).
2. The R&R response letter cites the new CSV rows for G4, G5, G6, G7, G9 (option i) by file path.
3. The "closed-form" / "vulnerability spectrum" rephrasing is consistent across title, abstract, intro, conclusion (G2, G3).

Re-review will be conducted using the panel's `re-review` mode (R&R traceability matrix per `references/re_review_mode_protocol.md`).

---

## Devil's Advocate CRITICAL fence (Iron Rule #4)

The following DA-CRITICAL items are conditions that, if unaddressed in the next round, force the re-review decision to Reject regardless of other improvements:

- **DA-C1** — Phase-transition empirical anchor. Addressed by G1 (contribution demotion) + G3 (title/abstract softening). The author may either (a) keep the IGNN-only formal track and demote, OR (b) run an actual $\varepsilon > \varepsilon_{\rm crit}$ breach experiment on a non-spectral-normalised IGNN where $\kappa \to 1$ from above. Option (a) is what the panel expects.
- **DA-C2** — "Closed-form" misnomer. Addressed by G2.
- **DA-M3** — Adaptive attacker. Addressed by G6.

---

## Open questions the author must answer in the response letter

1. (EIC Q1) After the κ_max sweep, what is the *predictive* claim of Theorem 3.1 that a practitioner should design around? If the answer is "none — it is a sufficient condition that never empirically fires," demote (2) from headline contribution to remark.
2. (EIC Q3 / R2 §AGNNCert / DA-M2) For a node where AEGIS says "fragile" ($r_v < \varepsilon$) and AGNNCert says "certified safe," which should the practitioner trust? Without a decision rule, "complementary" is rhetorical.
3. (R2 §G8) Where is the Amazon Photo full-graph τ CSV? If it does not exist, the conclusion.tex (iii) claim must be retracted; if it does, point to the artefact.
4. (R1 Q2) Why was R2_04 not re-run with the fixed `top_k_svd(k=7)` script? The stated 8-hour cost is GPU wall-clock; a 50-row partial re-run is a fraction of that.
5. (DA "So what?") Strip "closed-form" and "spectrum" and Theorem 3.1(c). What remains as the scientific contribution? Is that contribution NeurIPS/ICDM-grade — and at which of the two venues?

---

## Reviewer report files (this round)

- `docs/review_2026-05-29_r2/00_reviewer_config.md` — Phase 0 persona configuration
- `docs/review_2026-05-29_r2/01_eic_review.md`
- `docs/review_2026-05-29_r2/02_methodology_review.md`
- `docs/review_2026-05-29_r2/03_domain_review.md`
- `docs/review_2026-05-29_r2/04_perspective_review.md`
- `docs/review_2026-05-29_r2/05_devils_advocate.md`
- `docs/review_2026-05-29_r2/06_editorial_decision.md` (this file)
