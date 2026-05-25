# IEM Paper Positioning — Revised After Critical Review

**Date:** 2026-05-25
**Context:** Adversarial pre-review identified 3 fatal flaws. This doc captures the revised positioning.

---

## Title (revised)

**"Implicit Equilibrium Mining: First-Order Sensitivity Analysis and Certified Bounds for Deep Equilibrium Graph Models"**

(Dropped: "Exact Shapley" — replaced with honest "first-order sensitivity")

## Framing (revised)

### OLD (overclaimed)
> "Four theorems: convergence, causal sensitivity, exact Shapley, certified bounds"

### NEW (defensible)
> "A practical framework packaging known mathematical tools (IFT, Neumann bounds, gradient attribution) into a domain-agnostic mining toolkit for DEQ graph models, with novel applications in power system contingency analysis"

## Contributions (revised)

| # | OLD claim | Problem identified | NEW claim |
|---|---|---|---|
| 1 | PE_DEQ_PF architecture | No issue — genuine contribution | **Same** |
| 2 | "Exact Shapley via IFT" | `ift_attribution` computes gradient norms, NOT Shapley values. Violates efficiency axiom. | **IFT-based node attribution** — O(n) gradient importance, empirically validated against exact Shapley on small graphs |
| 3 | "IFT = structural causal effect" | First-order gradient ≠ Pearl's do-calculus intervention | **First-order sensitivity analysis via IFT** — local perturbation analysis, not causal inference |
| 4 | "Certified sensitivity bound" (Theorem 4) | Standard Neumann series norm bound | **Property:** Certified bound from contractivity — standard result applied to novel domain |

## Propositions (not theorems)

### Property 1: Convergence guarantee (from Banach contraction mapping theorem)
For contractive F_θ with ρ(∂F/∂z) < 1, the fixed-point iteration converges geometrically: ||z_k - z*|| ≤ ρ^k ||z_0 - z*||. **Citation:** Banach 1922; Bai, Kolter & Koltun (NeurIPS 2019) for DEQ application.

### Proposition 1: First-order sensitivity via IFT
At the fixed point z* = F(z*, p), the first-order sensitivity is ∂z*/∂p = (I - ∂F/∂z)⁻¹ · ∂F/∂p. This provides a LOCAL LINEAR approximation to the effect of parameter perturbation. **NOT a causal estimand in the do-calculus sense.** **Citation:** Implicit Function Theorem; Lorraine+ 2020 (iMAML); Gould+ 2021 (implicit layers tutorial).

### Proposition 2: IFT attribution as fast proxy for node importance
The per-node attribution φ_i = ||∂z*/∂x_i|| is an O(n) gradient-based importance score. Empirically correlates with exact Shapley values (rank correlation validated on small graphs). Does NOT satisfy Shapley axioms (efficiency, symmetry) in general.

### Property 2: Certified sensitivity bound (from Neumann series)
When ρ < 1: ||∂z*/∂p|| ≤ ||∂F/∂p|| / (1 - ρ). Standard result from functional analysis. **Citation:** Kato, Perturbation Theory for Linear Operators.

## Key disclosures (pre-empt reviewer attacks)

1. **PE_DEQ_PF has ρ ≈ 1.00** — Properties 1 and 2 require ρ < 1, achieved only by the contractive variant (ContractiveGCN-PF with spectral norm). Disclosed in Section X.

2. **IFT attribution ≠ Shapley** — gradient norms approximate node importance but don't satisfy coalitional axioms. Exact Shapley available via coalition enumeration for n ≤ 20. Rank correlation between IFT attribution and exact Shapley reported in Section X.

3. **First-order ≠ finite perturbation** — IFT captures infinitesimal sensitivity. For large perturbations (full edge removal in N-1), second-order effects are not captured. τ = 0.73 on power flow reflects this gap honestly.

4. **Citation network N-1 ranking is weak** — BF variance is near-zero (all edges equally unimportant). This is a DOMAIN PROPERTY, not a method failure. Discussed in Section X.

## What IS novel (defensible)

1. **First application of IFT-based sensitivity analysis to DEQ graph models for power system contingency mining** — no prior work does this.

2. **Domain-agnostic framework with 5-line API** — same code works on power flow, citations, e-commerce, encyclopedia. Engineering contribution.

3. **Matched-capacity comparison on 7 PF datasets** — PE_DEQ wins physics 7/7 (2-40×), supervised 4/7. Most comprehensive DEQ-vs-explicit comparison in PF literature.

4. **N-1 contingency ranking via one-pass IFT** — τ = 0.73 with 70% top-5 agreement on power flow. Novel application.

5. **ContractiveGCN-PF for power flow** — IGNN-style architecture achieving ρ < 1 on PF, enabling certified bounds. Novel instantiation.

## Anticipated reviewer objections (with responses)

| Objection | Response |
|---|---|
| "Propositions are textbook" | Yes — we cite origins explicitly. The contribution is the novel APPLICATION to DEQ graph models on power systems, not the math itself. |
| "IFT attribution isn't Shapley" | Correct — we renamed it and validate rank correlation against exact Shapley (ρ = X.XX on small graphs). |
| "PE_DEQ_PF violates ρ < 1" | Disclosed. ContractiveGCN-PF achieves ρ < 1 and is presented as the model for certified analysis. |
| "N-1 only works on PF" | Domain-dependent by design. BF variance analysis explains why citation networks have flat edge criticality. |
| "Missing baselines" | [TODO: add GNNExplainer, Integrated Gradients, DC-PF sensitivity comparisons] |
