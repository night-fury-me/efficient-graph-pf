# Editorial Decision — AEGIS

**Paper**: *AEGIS: Mining Graph Structure for Adversarial Vulnerability Analysis of GNNs*
**Decision**: **Minor Revision**
**Date**: 2026-05-27

---

## Panel Summary

| Reviewer | Role | Score | Recommendation | Confidence |
|----------|------|-------|----------------|------------|
| EIC (Günnemann-style) | Editor-in-Chief | 80/100 | Minor Revision | 4/5 |
| R1 (Kolter-style) | Methodology | 83/100 | Minor Revision | 4/5 |
| R2 (Bojchevski-style) | Domain | 65/100 | Minor Revision | 4/5 |
| R3 (Roald-style) | Cross-Disciplinary | 72/100 | Minor Revision | 4/5 |
| DA | Devil's Advocate | — | — | 4/5 |

**Aggregate**: 75/100 (weighted by confidence). All reviewers recommend Minor Revision; no reviewer recommends Reject.

---

## Consensus Strengths (All or Near-All Reviewers Agree)

### S1. The $S_c$ Constrained Projection Is Genuinely Novel
**Agreed by**: EIC, R1, R2, DA (all 5)

The reduction from $N^2$ to $|E|$ dimensions with enforced symmetry is the paper's core insight. The unconstrained-vs-constrained tightness gap (0.31 → 1.00) validates that this is not a minor refinement but essential for realistic perturbation models. DA explicitly acknowledges: "The S_c constrained projection is genuinely novel... the tightness improvement is well-demonstrated."

### S2. Comprehensive and Honest Experimental Evaluation
**Agreed by**: EIC, R1, R2, DA (4/5)

330 runs across 7 architectures × 5 datasets × 10 seeds, with a 4-quadrant attack taxonomy (gradient-based/free × same/different objective). The transparent reporting of failures (Amazon Photo negative τ, GCN-2 anti-correlation, Pubmed breach rate skewness) exceeds venue norms. R1: "10-seed evaluation with diverse seeds is above the norm for this literature."

### S3. Unified Three-Output Design
**Agreed by**: EIC, R1 (2/5 — others note specific outputs)

Attacks, rankings, and radii from a single $S_c$ computation is elegant. Each output is independently validated: attack advantage 2–8× (Table I), positive Kendall τ in 29/33 settings (Table VII), 0% breach rate at ε = 0.01 (Table III).

### S4. Matrix-Free Pipeline Is Well-Engineered
**Agreed by**: R1, EIC (2/5)

Neumann series + autograd JVPs + randomized SVD is a correct and scalable approach. Full-graph analysis validated up to N = 7,650 (Amazon Photo).

### S5. Cross-Domain Case Study Opens New Direction
**Agreed by**: EIC, R3 (2/5 — R3 with significant caveats)

The structural isomorphism between adversarial edge perturbation and N-1 contingency is genuinely novel. P@10 = 0.66–0.87 demonstrates practical relevance.

---

## Consensus Weaknesses (Multiple Reviewers Converge)

### W1. "Architecture-Agnostic" Framing Is Overstated
**Raised by**: EIC (W1), R2 (W2), DA (Strongest Counter-Argument)

Formal guarantees (ε_crit, three regimes, convergence) apply only to contractive implicit GNNs (IGNN). For explicit GNNs — which practitioners actually deploy — AEGIS provides only the computational tool (chain rule + SVD). The paper's abstract and introduction lead with formal guarantees, creating a bait-and-switch: the best-performing architectures for τ (GAT†: +0.54 on Cora, SAGE-2: +0.60 on Amazon Photo) receive no formal guarantee, while the one architecture with guarantees (IGNN) fails on the densest dataset (τ = −0.15 on Amazon Photo).

**Arbitration**: This is the paper's most significant framing issue. The *technical content* is correct — the paper does distinguish computational tool vs. formal guarantee in the theory sections. But the abstract/introduction/title framing overpromises. **Must address in revision.**

### W2. Continuous-to-Discrete Gap Limits Practical Utility
**Raised by**: R1 (W6), R2 (W4), DA (C3)

Real adversarial attacks are discrete (add/remove edges). The continuous perturbation model (Section II-B) is disconnected from practical threats. While Proposition 3 provides a theoretical bridge, 4 of 33 architecture-dataset combinations show *negative* τ (Table VII), meaning continuous rankings actively mislead about discrete vulnerability in 12% of settings. DA notes the irony: degree-proportional continuous perturbation is within 6–8% of AEGIS (Table I), but degree-ranked discrete removal is *worse than random* on Cora (Table IV).

**Arbitration**: The paper already discusses this honestly (Section V-G, conclusion limitations), but the abstract's "first-order optimal attack direction" framing is misleading when 12% of settings show negative transfer. **Must temper claims and add practitioner guidance on when to trust continuous rankings.**

### W3. IGNN Accuracy Penalty Undermines Practical Relevance
**Raised by**: R2 (W1), DA (M1), R3 (implicit)

IGNN achieves 77.5% on Cora vs. APPNP at 82.2%. A practitioner choosing IGNN for formal guarantees pays a 5-point accuracy penalty. On safety-critical applications (the paper's motivating use case), this accuracy loss may itself be the greater risk. The paper frames this as "the cost of formal vulnerability guarantees," but the guarantees are first-order approximations that degrade at operationally relevant perturbation magnitudes (15% error at ε = 0.10).

**Arbitration**: The paper already includes an accuracy-vs-guarantee discussion (Section V-G), but it should be formalized as a quantitative decision framework. **Recommend adding a practitioner decision table.**

### W4. Scalability Limitations Need Clearer Scoping
**Raised by**: DA (C2), R1 (W6), R3 (implicit)

Dense Jacobian limits subgraph analysis to N ≈ 200–300. The BFS ego-subgraph extraction fundamentally changes what is being analyzed — 50-node subgraphs from Cora (2,708 nodes) miss global structure. The paper's justification ("Localization is justified by the locality of per-node vulnerability") is circular. The matrix-free pipeline scales to N = 7,650 but τ validation at this scale is limited.

**Arbitration**: The paper should clearly delineate which results use dense vs. matrix-free paths and provide a scalability roadmap. The matrix-free pipeline partially addresses this concern (Cora full-graph in 78s). **Recommend clearer scoping, not new experiments.**

### W5. Power Grid Case Study Needs Deeper Domain Engagement
**Raised by**: R3 (primary — 5 specific concerns), DA (M2)

Binary adjacency discards essential electrical information (impedance, thermal ratings). Training data is narrow (uniform load scaling only). Case300 angle RMSE (0.394 p.u.) is physically unrealistic. Missing comparison with PTDF and thermal/voltage limit metrics. LODF comparison lacks disagreement analysis.

**Arbitration**: R3's concerns are technically valid but the paper already positions this as "a proof-of-concept demonstration" with explicit operational caveats. **Recommend addressing R3's top 2 concerns (impedance and model quality) while rescoping claims.**

---

## Divergent Opinions

### Score Spread: R1 (83) vs. R2 (65)
R1 evaluates the *mathematical framework* as rigorous and well-executed. R2 evaluates the *domain novelty* as incremental — Theorem 1 is standard IFT/contraction theory repackaged, and the claimed literature gap is narrower than presented (localized smoothing already provides per-node certificates).

**EIC Assessment**: Both perspectives are valid. The paper's novelty is concentrated in the $S_c$ construction and its empirical validation, not in the individual mathematical tools. R2's lower score reflects legitimate concern about overselling theoretical novelty. The revision should clarify that Theorem 1 is a *synthesis* of known results applied to a new problem (structural sensitivity), not a fundamental mathematical advance.

### Power Grid Framing: R3 (Concerned) vs. EIC (Positive)
R3 sees the case study as underdeveloped from a power systems perspective. EIC sees it as a compelling cross-domain validation. Both are right — the case study succeeds as an ML demonstration but would not convince a power systems reviewer.

**EIC Assessment**: The paper should scope the power grid results as "proof-of-concept cross-domain transfer" rather than "contingency analysis tool," and add a forward-looking paragraph on what impedance-weighted adjacency would look like.

---

## Devil's Advocate CRITICAL Issues — EIC Adjudication

### DA-C1: The Gap AEGIS Fills Is Narrower Than Claimed
**Claim**: Localized smoothing (Schuchardt et al.) provides per-node certificates; meta-gradients and Nettack provide per-edge attack information.

**Adjudication**: Partially valid. The paper should acknowledge these existing capabilities more explicitly. However, none of these provides all three outputs (optimal attack direction, per-edge ranking, per-node radii) from a single computation with closed-form SVD optimality. The gap is narrower than the introduction implies but still real. **Requires reframing in introduction, not retraction of claims.**

### DA-C2: First-Order Optimality Optimizes the Wrong Objective
**Claim**: AEGIS optimizes equilibrium shift, not prediction flipping. The SVD-optimal direction is optimal for hidden-state damage, not classification accuracy.

**Adjudication**: Technically valid but the paper already addresses this empirically (Table VI: SVD achieves comparable or higher flip rates than classification-loss PGD across all datasets/budgets). The equilibrium-shift objective is a meaningful proxy for prediction damage, as validated by the experiments. **Recommend adding a paragraph explicitly connecting equilibrium shift to classification impact, citing Table VI.**

### DA-C3: Continuous Perturbation Model Disconnected from Real Threats
**Claim**: 12% of settings show negative continuous-to-discrete transfer.

**Adjudication**: This is the most substantive CRITICAL issue. The paper's honesty about negative τ cases is commendable, but the abstract does not signal this limitation. **Requires: (1) Abstract caveat about transfer variability, (2) Practitioner guidance on when continuous rankings are reliable (depth ≥ 4, sparse graphs), (3) Clear labeling of settings with negative transfer.**

**Verdict on DA-CRITICAL issues**: None rises to the level of preventing acceptance. All three can be addressed through honest reframing and added discussion. No CRITICAL issue requires new experiments or fundamental restructuring.

---

## Editorial Decision: MINOR REVISION

### Rationale

AEGIS presents a genuinely novel contribution to the adversarial GNN literature through the constrained sensitivity matrix $S_c$. The core technical idea — projecting the $N^2$-dimensional perturbation space onto the $|E|$-dimensional space of realistic graph perturbations, then extracting attacks, rankings, and radii via SVD — is elegant, correct, and practically useful. The experimental evaluation (330 runs, 7 architectures, 9 datasets) is among the most thorough in this area.

The weaknesses identified by the panel are significant but uniformly addressable through reframing, added discussion, and minor additional analysis — none require new large-scale experiments or fundamental restructuring. The paper is above the acceptance threshold in its current form; the revision will bring it to a confident accept.

---

## Revision Roadmap

### Priority 1 — ESSENTIAL (Must Address)

| # | Issue | Source | Action Required |
|---|-------|--------|-----------------|
| R1 | Architecture-agnostic framing | W1 (EIC, R2, DA) | Revise abstract/intro to clearly distinguish "computational tool (all GNNs)" from "formal guarantees (IGNN-class only)." Add a sentence to the abstract: "For contractive implicit models, supplementary formal guarantees..." already in conclusion — mirror in abstract. |
| R2 | Continuous-to-discrete caveat | W2 (R1, R2, DA) | Add to abstract: "Positive transfer observed in 29/33 architecture-dataset settings; practitioners should verify transfer for their specific architecture-graph combination." Add practitioner guidance table in Section V-G. |
| R3 | Equilibrium shift vs. classification | DA-C2 | Add a paragraph in Section V-D connecting equilibrium shift to classification impact, explicitly citing Table VI flip rates as empirical validation. |
| R4 | Deterministic certification comparison | EIC (W4) | Add quantitative comparison with AGNNCert or Geisler et al. (2021) to Section V-B. Even if only qualitative (different threat models), explain why $r_v$ and smoothing radii are incomparable. |

### Priority 2 — STRONGLY RECOMMENDED

| # | Issue | Source | Action Required |
|---|-------|--------|-----------------|
| R5 | Theorem 1 novelty clarification | R2, DA | Add a remark after Theorem 1 explicitly stating it is a synthesis of classical tools (IFT, Banach, Neumann) applied to the new problem of structural sensitivity, with novelty in the application and the $S_c$ construction. |
| R6 | Accuracy-vs-guarantee decision framework | W3 (R2, DA) | Add a practitioner decision table: "Use IGNN when formal ε_crit is required; use APPNP/SAGE when only vulnerability rankings needed." |
| R7 | Power grid case study scoping | W5 (R3) | Rescope Section VII intro: "proof-of-concept cross-domain transfer" not "contingency analysis tool." Add forward-looking paragraph on impedance-weighted adjacency. Address case300 θ RMSE = 0.394 as a model quality limitation. |
| R8 | Missing literature | R2 | Add citations: Kenlay et al. (spectral stability), Topping et al. / Di Giovanni et al. (curvature/over-squashing connection), recent 2024 certified robustness extensions. Discuss relationship to S_c. |
| R9 | Scalability scoping | W4 (DA, R1) | Add a clear table distinguishing dense path (N ≤ 200, formal S matrix) vs. matrix-free path (N ≤ 7,650, operator-only) capabilities and limitations. |

### Priority 3 — OPTIONAL BUT STRENGTHENING

| # | Issue | Source | Action Required |
|---|-------|--------|-----------------|
| R10 | Thm 1(b) directional clarification | R1 (W1) | Clarify that the Ω(1/(ε_crit − ε)) divergence rate is worst-case over directions; add a sentence on direction-specific behavior. |
| R11 | N-2 results rescoping | DA (m4) | Present N-2 pair-level overlap (7–18%) as a limitation/negative finding, not as evidence of multi-edge detection. |
| R12 | Amazon Photo controlled experiment | EIC (W2) | Run density-controlled experiment (downsample Amazon Photo edges to citation-network density) to isolate density vs. architecture effects. |
| R13 | Ethical considerations expansion | R3 | Add dual-use discussion and tiered release protocol details (currently in conclusion but brief). |
| R14 | Bibliography deduplication | EIC (W7) | Check for and remove duplicate entries. |

### Estimated Revision Effort
- Priority 1: ~2–3 days (rewriting, no new experiments)
- Priority 2: ~3–5 days (literature additions, discussion sections, minor analysis)
- Priority 3: ~2–5 days (one new experiment + text refinements)
- **Total: 1–2 weeks**

---

## Summary for Authors

Your paper makes a strong contribution through the $S_c$ constrained sensitivity framework. The core idea is novel, the experiments are thorough, and the cross-domain case study opens an interesting research direction. The primary revision need is *framing*: the paper's claims — particularly "architecture-agnostic" and "first-order optimal" — outrun its evidence in specific respects that all five reviewers independently identified. Tightening the framing to match the (already strong) evidence will resolve the panel's main concerns. No fundamental restructuring or major new experiments are required.

We look forward to the revised manuscript.

---

*Editorial Decision Report synthesized from 5 independent reviewer reports.*
*Phase 2 synthesis follows IRON RULE: every point traces to a specific Phase 1 reviewer report.*
