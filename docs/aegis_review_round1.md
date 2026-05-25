# AEGIS Paper Review — Round 1 (Pre-Submission Simulated Review)

**Date:** 2026-05-25
**Venue:** IEEE ICDM 2026 (Research Track)
**Decision:** Major Revision

---

## Score Summary

| Dimension | EIC | R1 (Method) | R2 (Domain) | R3 (PF) | DA |
|---|---|---|---|---|---|
| Originality | 82 | — | 58 | — | — |
| Significance | 78 | — | — | 38 | — |
| Soundness | 70 | 62 | 62 | 52 | — |
| Reproducibility | 60 | 71 | — | — | — |
| Overall | 71 | ~66 | ~57 | ~45 | — |

---

## EIC Review (Overall: 71, Major Revision)

### Scores
| Dimension | Score |
|---|---|
| Originality | 82 |
| Significance | 78 |
| Clarity | 65 |
| Soundness | 70 |
| Reproducibility | 60 |

### Top 3 Strengths
1. **Novel theoretical framing.** The phase-transition theorem connecting DEQ fixed-point stability to adversarial vulnerability is genuinely original. A sharp threshold (ε_crit) that delineates three regimes gives practitioners an actionable diagnostic.
2. **Breadth of empirical validation.** 30/30 wins vs Mettack across 9 datasets, 10 seeds. Constrained tightness 1.00±0.01 is striking. Power-flow case study adds real-world grounding.
3. **Deterministic certificates 2-5x larger than randomized smoothing.** Practically significant for safety-critical applications.

### Top 3 Weaknesses
1. **Soundness gap in phase-transition theorem.** Proof sketch at 5 pages needs explicit assumptions on activation function, graph spectrum, DEQ convergence. IFT step requires Lipschitz continuity verification.
2. **Constrained tightness of 1.00±0.01 is suspiciously perfect.** Must rule out circular tightness (certificate and attack sharing optimization pathway). Need ablation on certificate vs attack independence.
3. **Reproducibility and page budget.** At 5 pages, critical details missing — hyperparameter sensitivity, DEQ convergence diagnostics, wall-clock cost of certification vs smoothing.

---

## R1: Methodology Expert (Rigor: 62, Reproducibility: 71)

### Proof Correctness Issues
- **Theorem 1(a):** The step claiming `ρ = ||A||₂ · ||W||₂` holds "in the worst case" is hand-waved. True Jacobian is `J_z = diag(σ') · (A ⊗ W)`, so `ρ(J_z) ≤ ||A||₂ · ||W||₂ · sup(σ')`. Must explicitly state `sup(σ') = 1` for ReLU. Transition from `||·||₂` to `||·||_F` means ε_crit is conservative, not "sharp."
- **Parts (b) and (c)** are qualitative observations, not formal proofs. Part (b) resolvent lower bound is for normal operators only; non-normality (η up to 1.4) means actual divergence rate could differ.
- **Proposition 2 (Per-Node Radius):** Chain rule tacitly assumes linear classifier head. If nonlinear (softmax + MLP), local Lipschitz constant needed.

### Top 3 Methodological Strengths
1. **Constrained sensitivity matrix S_c** is genuinely novel. Encoding symmetry/sparsity constraints yields tightness 1.00±0.01.
2. **Phase transition characterization** provides interpretable threshold ε_crit. 83x amplification scan is convincing.
3. **Reproducibility hygiene** above average: 10 named seeds, explicit hyperparameters, hardware specified, wall-clock times reported.

### Top 3 Methodological Weaknesses
1. **Unfair smoothing baseline.** σ=0.01 / 200 samples is orders of magnitude below standard practice (typically σ∈{0.25, 0.5, 1.0}, 10,000+ samples). Constant median radius 0.015 confirms degenerate regime. "2-5x larger" claim is inflated.
2. **Mettack is surrogate-transfer attack evaluated on different architecture.** White-box PGD on IGNN adjacency required for fair comparison. At minimum, Mettack with IGNN surrogate.
3. **Locality assumption unvalidated.** N=50 BFS truncation critical to scalability and certificate validity. No experiment varies subgraph size to show certificate stability.

---

## R2: Domain Expert (Novelty: 58, Lit Coverage: 52, Theory: 62)

### Top 3 Strengths
1. **Connecting DEQ fixed-point structure to adversarial robustness** is timely and useful framing.
2. **Constrained sensitivity matrix S_c** goes beyond El Ghaoui+2021 and Revay+2020 by exploiting graph topology.
3. **Empirical protocol is sound** — evaluating against Mettack and smoothing with error bars.

### Top 3 Weaknesses
1. **Phase transition theorem overstates novelty.** Result that robustness degrades when ρ crosses 1 is direct consequence of Banach fixed-point theorem applied to IGNN (Gu+2020, Theorem 1). Reframing as "phase boundary" is descriptive language, not new mathematical object.
2. **Critical missing references.** (a) Winston & Kolter (2020) monotone-operator certificates undiscussed. (b) Bojchevski & Günnemann (2019) certifiable GNN robustness absent. (c) Pabbaraju+2021 directly bounds DEQ sensitivity — omitting it weakens related work.
3. **IGNN-specific assumptions limit generality.** Theory assumes well-split W, A structure. Most deployed GNNs (GAT, GIN, GraphSAGE) don't factor this way. Should discuss or acknowledge limitation.

**Recommendation:** Borderline accept. S_c is real contribution but phase-transition claim needs deflation and missing references must be incorporated.

---

## R3: Power Systems Perspective (Impact: 38, Cross-Domain: 52)

### Top 3 Strengths
1. **Novel framing** of contingency as graph perturbation. ML-to-power-systems correspondence table is useful.
2. **Speed advantage is real** — ~100x over brute-force on IEEE 118-bus.
3. **IFT-based sensitivity is principled** — parallels how PTDF/LODF are derived analytically.

### Top 3 Weaknesses
1. **τ = 0.42-0.67 is insufficient for operational use.** ~30% pairwise inversions. Grid operators wouldn't trust this over DC-approximation screening (PTDF/LODF). Must report **top-k recall** (precision@10).
2. **2000 PandaPower samples is woefully inadequate.** Real grid conditions span seasonal loads, renewable intermittency, outage combinations. 118-bus is a toy benchmark.
3. **AC vs DC approximation unaddressed.** Paper never clarifies whether model trained on full AC or DC approximation. If DC, voltage-collapse dimension missing.

### Additional Concerns
- Correspondence table maps "edge weight perturbation" to "line impedance change" — but impedance is fixed; what changes is topology/loading.
- No comparison against PTDF/LODF-based linear screening (actual industry baseline).

---

## Devil's Advocate (CRITICAL issue found)

### Issue 1: "Exact Prediction" — MAJOR
Tightness 1.00 measured only under constrained (symmetric, edge-weight-only) perturbations. Edge additions change sparsity pattern, breaking smoothness assumptions. The "exact" claim should be scoped to "exact under affine perturbation models."

### Issue 2: Phase Transition "Theorem" — MINOR
Parts (a)+(b) follow directly from IFT + contraction mapping. Part (c) is negation of hypothesis. Calling this a "phase transition" borrows statistical physics language without universality or critical-exponent analysis.

### Issue 3: 30/30 vs Mettack — CRITICAL
Comparing white-box certificate against black-box transfer attack is not meaningful. The correct comparison is an **adaptive attack**: compute adversarial perturbations using the IGNN fixed-point and its Jacobian (the same information AEGIS has). Without this, the 30/30 rate proves only that transfer attacks are weak against architectural mismatch.

### Issue 4: Certificate Size vs Smoothing — MAJOR
Smoothing certificates hold for arbitrary perturbation magnitudes. AEGIS certificates are first-order: tight only for small ε. For perturbations removing ≥5% of edges, first-order diverges. The 2-5x advantage is measured where first-order works; at large budgets, smoothing dominates.

### Issue 5: "Domain-Agnostic" — MINOR
All experiments use IGNN (single DEQ variant). Other DEQ architectures have different fixed-point structures. Claim should read "architecture-specific, applicable to contractive implicit GNNs."

### Strongest Counter-Argument
The entire empirical validation conflates *transfer-attack weakness* with *certificate strength*. Mettack's 0/30 bypass rate is not evidence that AEGIS certificates are tight — it is evidence that GCN-surrogate perturbations don't transfer to implicit architectures. To demonstrate AEGIS adds value beyond inherent DEQ robustness, must evaluate against an adaptive attacker that differentiates through the fixed-point iteration via implicit differentiation and optimizes perturbations directly against IGNN's loss surface. The missing adaptive-attack baseline is the single largest gap.

---

## Revision Roadmap (Prioritized)

| Priority | Issue | Source | Action |
|---|---|---|---|
| **P0** | Adaptive attack baseline | DA, R1 | Implement white-box PGD on IGNN adjacency via IFT gradients |
| **P0** | Fix smoothing σ | R1 | Rerun with σ∈{0.1, 0.25} and ≥1000 samples |
| **P1** | Scope "exact" language | R2, DA | "Exact under constrained perturbations" throughout |
| **P1** | Subgraph size ablation | R1 | N=50/100/200 certificate stability table |
| **P2** | Top-k precision for PF | R3 | Report precision@5, precision@10 for N-1 |
| **P2** | Phase transition language | R2, DA | "Corollary of contraction mapping" or "characterization" |
| **P3** | Missing references | R2 | Add Bojchevski 2019, discuss Winston 2020, Pabbaraju 2021 |
| **P3** | Expand to 8 pages | EIC | Full proofs, hyperparameters, convergence diagnostics |
