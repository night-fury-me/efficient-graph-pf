# Editorial Decision — AEGIS

**Paper**: AEGIS: Mining Graph Structure for Adversarial Vulnerability Analysis of Graph Neural Networks

**Date**: 2026-05-25

---

## Panel Summary

| Reviewer | Role | Overall Score | Recommendation |
|----------|------|:------------:|----------------|
| EIC | Editor-in-Chief | 74 | Minor Revision |
| R1 | Methodology | 77 | Minor Revision |
| R2 | Domain Expert | 72 | Minor Revision |
| R3 | Perspective (Power Systems) | 63 | Minor Revision (leaning Major) |
| DA | Devil's Advocate | — | 3 CRITICAL / 5 MAJOR / 5 MINOR |

---

## Decision: Minor Revision

**Rationale**: The panel unanimously agrees that AEGIS introduces a genuinely useful contribution — the constrained sensitivity matrix $S_c$ — that fills a clear gap between adversarial attack methods and certified robustness defenses. The constrained-vs-unconstrained tightness gap (1.00 vs 0.31) demonstrates the projection's practical necessity, the 10-seed evaluation protocol is above community norms, and the adaptive attacker evaluation is methodologically sound.

However, the paper's rhetorical ambition exceeds its demonstrated scope. The Devil's Advocate identified three CRITICAL issues (overstated generality, scalability ceiling, continuous/discrete mismatch) that, while addressable through reframing and targeted experiments, preclude acceptance in the current form. All four peer reviewers independently converged on Minor Revision, and the CRITICAL issues do not require fundamental rework — they require honest scoping.

---

## Cross-Reviewer Consensus Matrix

### Issues Where 4+ Reviewers Agree

| Issue | EIC | R1 | R2 | R3 | DA | Severity |
|-------|:---:|:--:|:--:|:--:|:--:|----------|
| "Architecture-agnostic" claim overstated | ✓ | — | ✓ | — | ✓ CRITICAL | **HIGH** |
| Scalability N≈300 limit vs safety-critical framing | ✓ | ✓ | ✓ | ✓ | ✓ CRITICAL | **HIGH** |
| Continuous vs discrete perturbation gap | ✓ | ✓ | ✓ | — | ✓ CRITICAL | **HIGH** |
| κ vs ρ inconsistency in formal guarantees | ✓ | ✓ | ✓ | — | — | **MEDIUM** |
| IGNN accuracy gap (77.5% vs ~82-85%) | — | — | ✓ | — | ✓ MAJOR | **MEDIUM** |

### Issues Where Reviewers Disagree

| Issue | For | Against | Resolution |
|-------|-----|---------|------------|
| Tightness at ε=0.01 informativeness | EIC, R2 (S_c projection is the contribution) | R1, DA (trivially expected from Taylor) | **Both sides are right**: the tightness itself is expected; the novelty is that S_c makes it achievable under *constrained* perturbations. Reframe emphasis. |
| Power grid case study value | EIC, R2 (creative cross-domain demo) | R3 (too simplified for domain experts), DA (moderate τ on tiny grids) | **Scope down claims**: position as proof-of-concept, not practical tool. Address R3's engineering concerns. |
| "Implicit physics" observation | R2 (interesting) | DA (known DEQ property), R3 (needs explicit GNN comparison) | **Downgrade**: present as observation, not contribution. Add comparison to explicit GNNs per R3. |

---

## Required Revisions (Must Address)

### R1. Scope the generality claim [CRITICAL — DA C1, EIC W3, R2 W1]

**Problem**: The abstract claims S_c "applies to any differentiable GNN" while the theoretical guarantees (Theorem 1: ε_crit, phase transition) require contractive implicit models (A1-A3). For explicit GNNs, the contribution reduces to standard chain-rule Jacobian computation (Proposition 3).

**Required action**:
- Revise the abstract to clearly distinguish: "S_c as a computational diagnostic for any differentiable GNN" vs "formal regime characterization for contractive implicit GNNs"
- In contributions list, separate the two levels explicitly
- Acknowledge that GAT requires architectural modification (edge-weighted variant) and that this limits the "any GNN" claim
- Add a paragraph in Section IV-C explicitly discussing what is lost for explicit GNNs (no ε_crit, no convergence regimes, no formal phase transition)

### R2. Address the scalability ceiling honestly [CRITICAL — DA C2, All reviewers]

**Problem**: N≈300 subgraph limit (24 GB at N=400) contradicts the "safety-critical deployment" framing, since real-world graphs in the motivating domains (fraud detection, drug interaction, power grids) have thousands to millions of nodes. The BFS ego-subgraph extraction is a workaround whose validity is assumed, not proven.

**Required action**:
- Add a subgraph-to-full-graph validation experiment on a small graph (e.g., case14 or case30) where full-graph analysis is feasible: compare full-graph S_c rankings against subgraph-extracted rankings
- Discuss when the locality assumption holds and when it breaks (e.g., long-range dependencies in power grids)
- Reframe the deployment narrative from "general-purpose pre-deployment diagnostic" to "local vulnerability diagnostic with subgraph extraction"
- Mention sparse/randomized SVD as a concrete scalability path (not just "future work")

### R3. Quantify the continuous-discrete gap [CRITICAL — DA C3, EIC W1, R1 T4]

**Problem**: Real graph attacks add/remove edges (discrete), but AEGIS operates on continuous edge weights. The N-1 case study makes this contradiction explicit (complete line removal vs continuous perturbation). The gap is acknowledged but never quantified.

**Required action**:
- Add an experiment comparing S_c vulnerability rankings against discrete edge-removal ground truth (which already exists as the brute-force baseline in the Mettack comparison)
- Report the Kendall τ between continuous S_c rankings and discrete removal rankings explicitly
- Discuss under what conditions continuous approximation faithfully predicts discrete vulnerability (small perturbation regime, high-degree nodes, etc.)

### R4. Resolve the κ vs ρ reporting inconsistency [MEDIUM — EIC W4, R1 W2]

**Problem**: Theorem 1 formally requires the operator-norm contraction constant κ = ||J_z||_2, but all tables report only the spectral radius ρ(J_z). The paper notes "at most 28% optimistic" but never reports formal κ-based ε_crit values.

**Required action**:
- Report both κ and ρ in Table I (or at least for representative datasets)
- Compute and report the formal κ-based ε_crit alongside the ρ-based approximation
- If κ computation is prohibitively expensive, state this explicitly and provide the η (non-normality) values in the main text, not just the appendix

---

## Recommended Revisions (Strongly Suggested)

### S1. Add dual-use / ethical considerations [R3 — Required by R3]

Add a brief ethics paragraph addressing that AEGIS provides optimal attack directions for safety-critical systems. Discuss responsible disclosure practices and how the framework's diagnostic value outweighs the attack information it reveals.

### S2. Address the IGNN accuracy gap [R2 W3, DA M4]

Cora 77.5% vs ~82% (APPNP) or ~85% (state-of-art). Either:
- (a) Discuss accuracy-guarantee tradeoffs (implicit models sacrifice accuracy for formal guarantees)
- (b) Benchmark on a more competitive implicit architecture (EIGNN)
- (c) Frame the accuracy gap as a limitation of the IGNN architecture, not of AEGIS

### S3. Fix the Citeseer accuracy discrepancy [EIC Minor 3, DA m5]

Main text Table I reports Citeseer accuracy as 66.0%; Appendix early-stopping table reports 0.421 (42.1%). These cannot both be correct. Investigate whether one uses early stopping and the other does not, and reconcile.

### S4. Expand literature coverage [R2 — 10 missing references identified]

Key missing references to address:
- AGNNCert (Li et al., 2025) — deterministic certification comparison
- EIGNN (Liu et al., 2021) — faster implicit GNN
- Monotone operator networks (Winston & Kolter, 2020) — built-in Lipschitz comparison
- Topology Attack (Xu et al., 2019) — continuous relaxation connection
- GCNII (Chen et al., 2020) — deep GCN accuracy baseline

### S5. Downgrade the "implicit physics" observation [DA M5, R3]

Present as an interesting side observation, not a contribution. Add comparison against explicit GNN residuals on the same power flow task. Cite prior work on implicit physics in equilibrium models.

### S6. Strengthen statistical reporting [R1 W5, DA m3]

- Replace the single-seed Mettack comparison (Appendix Table VII) with 10-seed results, or clearly label it as illustrative
- Add confidence intervals or significance tests for the key comparisons (attack advantage, τ rankings)

---

## Strengths Acknowledged by the Panel

These aspects were praised by multiple reviewers and should be preserved in revision:

1. **S_c constrained projection** (all 5): The N² → |E| projection is the paper's core insight. The constrained-vs-unconstrained gap (1.00 vs 0.31) is the strongest single piece of evidence. *Preserve and emphasize.*

2. **Adaptive attacker evaluation** (EIC, R1, DA): Using the same IFT gradients for the adaptive attacker ensures a fair comparison. 0% breach rate at ε=0.01 provides genuine empirical validation. *This is methodologically exemplary.*

3. **Transparent limitation disclosure** (EIC, R1, DA): Five explicit limitations in the conclusion, operational caveats on power grid results. *Rare and commendable — expand this transparency to the abstract and introduction.*

4. **10-seed evaluation protocol** (R1, DA): Non-sequential seeds, consistent mean±std reporting, multiple ablations. *Above community norms.*

5. **Cross-architecture breadth** (EIC, R1, R2): 7 architectures, 9 datasets, 4 domains in a single paper. *Impressive coverage.*

6. **Power grid cross-domain transfer** (EIC, R2): Creative application that connects structural sensitivity to N-1 contingency. *Novel framing even if engineering details need work.*

---

## Revision Roadmap (Prioritized)

| Priority | Item | Effort | Impact |
|:--------:|------|:------:|:------:|
| 🔴 P1 | Scope generality claim (abstract, intro, contributions) | Low | High |
| 🔴 P2 | Subgraph-to-full-graph validation experiment | Medium | High |
| 🔴 P3 | Continuous-vs-discrete ranking comparison experiment | Medium | High |
| 🔴 P4 | Report κ alongside ρ in main tables | Low | Medium |
| 🟡 P5 | Add ethics/dual-use paragraph | Low | Medium |
| 🟡 P6 | Fix Citeseer accuracy discrepancy | Low | Medium |
| 🟡 P7 | Expand literature (5 key missing refs) | Low | Medium |
| 🟡 P8 | Downgrade "implicit physics" framing | Low | Low |
| 🟢 P9 | Replace single-seed Mettack with 10-seed | Medium | Low |
| 🟢 P10 | Discuss IGNN accuracy-guarantee tradeoff | Low | Low |

**Estimated revision effort**: 2-3 weeks (P1-P4 are the critical path; P2-P3 require new experiments)

---

## Individual Review Reports

| Report | File |
|--------|------|
| EIC Review | [`01_eic_review.md`](01_eic_review.md) |
| Methodology Review | [`02_methodology_review.md`](02_methodology_review.md) |
| Domain Review | [`03_domain_review.md`](03_domain_review.md) |
| Perspective Review | [`04_perspective_review.md`](04_perspective_review.md) |
| Devil's Advocate | [`05_devils_advocate.md`](05_devils_advocate.md) |
