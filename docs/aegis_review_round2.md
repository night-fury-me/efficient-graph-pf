# AEGIS Simulated Review — Round 2

**Paper:** AEGIS: Equilibrium Structure Mining for Certifiable Graph Robustness
**Venue:** ICDM 2026 Research Track (triple-blind)
**Format:** IEEE 2-column, 10 pages including references
**Citations:** 56
**Date:** 2026-05-25

---

## Meta-Review (Area Chair Summary)

The paper received mixed-to-positive reviews. All reviewers agree the core idea (IFT-based structural sensitivity for implicit GNNs) is interesting and the experimental discipline (10 seeds, adaptive attack, cross-domain) is above average. The main debate centers on whether the narrow scope (IGNN-class only) and the mathematical expectedness of first-order tightness at small epsilon constitute sufficient novelty for ICDM. The power flow case study is seen as creative but underdeveloped. With minor revisions addressing Reviewer 3's concerns about claim calibration, this paper is suitable for acceptance.

**Recommendation: Weak Accept (3 accept, 1 borderline, 1 weak reject)**

---

## Reviewer 1 — Graph Mining / Adversarial ML

**Overall Score: 6/10 (Weak Accept)**
**Confidence: 4/5**

### Summary
The paper introduces AEGIS, a framework for analyzing adversarial vulnerability of contractive implicit GNNs using the implicit function theorem. The key contribution is the constrained sensitivity matrix S_c, which restricts the IFT analysis to symmetric edge-only perturbations and achieves first-order tightness of 1.00 across 9 datasets. The paper also includes an SVD-optimal attack, per-node robustness radii, and a power grid case study.

### Strengths
- **S1.** The constrained sensitivity matrix S_c is genuinely novel and technically clean. The reduction from N^2 to |E| dimensions with enforced symmetry is a nice construction that enables tight predictions where unconstrained IFT would be loose. This is the real contribution.
- **S2.** Experimental rigor is above average for the venue: 10 seeds, adaptive white-box attacker (not just surrogate-based), honest acknowledgment of limitations (Mettack unfairness, smoothing radius comparison, power flow tau).
- **S3.** The paper is well-written with smooth transitions between sections. The Preliminaries and Threat Model section is clearly structured with an explicit problem statement. Notation is consistent.
- **S4.** The 56 references provide comprehensive coverage of the relevant literature across 7 threads. Positioning against each thread is clear.

### Weaknesses
- **W1.** The scope is narrow: only IGNN-class models. The paper acknowledges this, but GAT, GIN, GraphSAGE dominate practical deployments. The impact argument would be stronger with evidence that IGNNs are gaining adoption, or with a roadmap for extension to monotone operator networks (mentioned in future work but undeveloped).
- **W2.** The Mettack comparison (Table II) is acknowledged as unfair due to surrogate mismatch but occupies a full subsection and table. This space would be better used for additional ablations or a clean accuracy table.
- **W3.** No clean accuracy or robust accuracy metrics reported. The primary attack metric is fixed-point shift, not prediction change. Adversarial robustness papers typically report attack success rate, robust accuracy, and certified accuracy.

### Questions for Authors
- Q1. Could you report clean accuracy and attack success rate (fraction of correctly classified nodes whose prediction flips) alongside fixed-point shift?
- Q2. What is the relationship between the subgraph-level analysis and full-graph robustness? If a node's 50-hop neighborhood changes, does the radius computed on the BFS subgraph still hold?

### Minor Issues
- The pipeline figure (Fig. 1) is a text box, not a proper diagram. A visual architecture diagram would significantly improve readability.
- "Cert%" column in tables is not defined in the table caption (only in the preceding text).

---

## Reviewer 2 — Implicit/Equilibrium Networks

**Overall Score: 7/10 (Accept)**
**Confidence: 4/5**

### Summary
AEGIS applies the implicit function theorem to contractive implicit GNNs for structural adversarial analysis. The main novelty over standard IFT-for-DEQ work is the application to graph structure perturbations (changing A, not x) and the constrained sensitivity construction S_c.

### Strengths
- **S1.** The paper correctly uses the operator-norm contraction constant kappa = ||J_z||_2 rather than the spectral radius, which is a common mistake in the DEQ robustness literature. The Remark after Theorem 1 clearly explains the distinction and why kappa is preferred.
- **S2.** The threat model is explicitly stated and the limitations are honestly discussed (normalized adjacency perturbation, continuous only, no edge additions). This is refreshing compared to papers that sweep threat model assumptions under the rug.
- **S3.** The constrained sensitivity construction S_c is the right abstraction. Projecting the full N^2-dimensional sensitivity into the |E|-dimensional edge-constrained space is what makes tightness achievable. This observation, while simple, is the technical heart of the paper.
- **S4.** The Jacobian regularization reference [6] and JFB reference [18] in the background show awareness of the DEQ training literature, which strengthens the positioning.
- **S5.** The adaptive attack using the same IFT gradients (Table IV) is the correct experimental design for validating robustness claims. The 0% breach at eps=0.01 and <1% at eps=0.10 is convincing.

### Weaknesses
- **W1.** The radii are explicitly first-order. The paper calls them "deterministic first-order per-node robustness radii" which is technically correct, but reviewers and readers will inevitably compare against true certificates. The paper should more prominently state that these are NOT certificates in the formal sense (the conclusion does this but the abstract and introduction could be clearer).
- **W2.** The Proposition 2 (per-node radius) assumes a linear classification head. While IGNN uses a linear head, this limits generalization. The paper mentions "local Lipschitz constant" for nonlinear heads but does not develop this.
- **W3.** The pseudospectral index eta = 1.02-1.28 is reported but not deeply analyzed. When does eta matter? Is there a synthetic example where eta > 2 and the spectral-radius-based bound fails while the kappa-based bound holds? This would strengthen the motivation for using kappa.

### Questions for Authors
- Q1. Can you bound the second-order remainder ||O(||delta A||^2)|| empirically? Even a plot of tightness vs. epsilon (not just eps=0.01) would show where first-order breaks down.
- Q2. For the power grid case study, have you tried using the actual line reactance values instead of binary adjacency? This would make the LODF comparison fairer.

---

## Reviewer 3 — Theory / Certified Robustness

**Overall Score: 4/10 (Borderline Reject)**
**Confidence: 5/5**

### Summary
The paper applies the IFT to implicit GNN fixed points for adversarial structural sensitivity analysis. The main theoretical contribution is Theorem 1 (three-regime characterization) and Proposition 2 (per-node radius). The constrained sensitivity S_c is presented as the central technical contribution.

### Strengths
- **S1.** The problem is well-motivated: structural vulnerability analysis for implicit GNNs is genuinely underexplored.
- **S2.** The experimental design is solid: 10 seeds, adaptive attack, multiple domains.

### Weaknesses
- **W1. Theorem 1 is mathematically routine.** Part (a) is the standard IFT + Neumann series bound. Part (b) is the standard resolvent divergence. Part (c) is the negation of contractivity. The three-regime framing adds presentation value but no mathematical depth. The proof is 12 lines and uses only undergraduate-level operator theory. For a paper that includes "Certifiable" in the title, I expect deeper theoretical contributions.
- **W2. The tightness claim is misleading.** At eps=0.01, first-order tightness is *expected* for any smooth function. The Taylor expansion of z*(A + delta A) around delta A = 0 is exact to first order by definition. Reporting tightness=1.00 at eps=0.01 is confirming that the implementation correctly computes the Jacobian, not that the method is tight. A meaningful tightness result would hold at eps=0.05 or eps=0.10, where higher-order terms matter.
- **W3. S_c is a simple projection, not a deep construction.** Taking columns S_{:,iN+j} + S_{:,jN+i} for each edge is the obvious way to enforce symmetry. Calling this the "central technical contribution" overstates its novelty.
- **W4. The paper title says "Certifiable" but the text explicitly says the radii are NOT certificates.** The conclusion states: "Without bounding the second-order remainder, they are local sensitivity radii rather than global robustness certificates." This contradiction between title and content is a serious framing problem.
- **W5. No comparison against Bojchevski & Gunnemann [10] certificates.** The related work states their convex relaxation is "incompatible with IGNN" but does not substantiate this claim. If one trains a GCN and IGNN on the same dataset, one could compare the Bojchevski certificates (for GCN) against AEGIS radii (for IGNN) to show what the IGNN equilibrium structure buys.

### Questions for Authors
- Q1. Can you remove "Certifiable" from the title, or provide a genuine second-order bound that makes the radii into actual certificates?
- Q2. What is the tightness at eps=0.05 and eps=0.10? Table IV shows breach rates but not tightness ratios at these budgets.
- Q3. The S_c construction enforces symmetry by summing two columns. Have you considered the weighted version where the two columns are weighted by degree or edge weight? This would more faithfully model the effect of perturbing raw A (where the normalization couples edge perturbations to degree changes).

### Minor Issues
- The abstract says "deterministic first-order per-node robustness radii" — this is technically correct but the word "robustness" implies a guarantee that the first-order approximation does not provide.

---

## Reviewer 4 — Applied Data Mining / Power Systems

**Overall Score: 6/10 (Weak Accept)**
**Confidence: 3/5**

### Summary
The paper proposes AEGIS for vulnerability analysis of implicit GNNs and includes a case study on IEEE power grids where the vulnerability spectrum is compared against N-1 contingency rankings.

### Strengths
- **S1.** The power grid case study is creative and well-positioned for ICDM's cross-domain audience. The ML-to-power-systems analogy table is effective.
- **S2.** The paper honestly reports limitations of the power flow results (moderate tau, simplified LODF baseline, limited training diversity).
- **S3.** P@10 = 0.81 on case118 is a strong result for a domain-agnostic method.

### Weaknesses
- **W1.** The effective-resistance baseline is weak. A proper LODF baseline with actual line reactances would be the fair comparison. The paper acknowledges this but the acknowledgment does not remove the weakness.
- **W2.** Only 4 IEEE test cases (14-118 buses). Adding IEEE 300 or PEGASE 1354 would strengthen the scalability argument for power systems.
- **W3.** No comparison against Nakiganda et al. [42] (GNN contingency screening) despite citing them. Even a qualitative comparison would help.
- **W4.** The training data (2000 samples, uniform load scaling) is unrealistic for power systems applications. Real grids have seasonal, diurnal, and stochastic renewable generation patterns.

### Questions for Authors
- Q1. Have you considered using weighted adjacency (admittance values) instead of binary adjacency for the power flow GNN?
- Q2. What is the GNN prediction error (MAE/RMSE) on the power flow task itself? AEGIS vulnerability analysis is only as good as the underlying model.

---

## Reviewer 5 — Scalability / Systems

**Overall Score: 6/10 (Weak Accept)**
**Confidence: 3/5**

### Summary
AEGIS provides a pipeline for vulnerability analysis of contractive implicit GNNs with a clear 4-stage architecture. The paper includes scalability analysis and subgraph ablation.

### Strengths
- **S1.** The AEGIS Construction section (Section IV) is well-organized with a clear pipeline figure and stage-by-stage description. The computational cost analysis is honest.
- **S2.** The subgraph ablation (Table V) demonstrates that N=50 is a good default with 66x speedup over N=200 at minimal tightness cost.
- **S3.** Wall-clock times (0.5s to 8.1s) are practical for offline vulnerability analysis.

### Weaknesses
- **W1.** The dense Jacobian computation makes AEGIS fundamentally O(D^3) in the linear solve. This is acceptable for N<=200 but prevents application to graphs with thousands of nodes without subgraph extraction. The paper does not discuss how multiple subgraph analyses would be composed for full-graph vulnerability ranking.
- **W2.** No GPU memory analysis. For N=200 with d=64, J_z is 12800x12800 (1.3 GB in float32) and J_A is 12800x40000 (4.1 GB). This approaches the memory limit of consumer GPUs.
- **W3.** The pipeline figure is a text mockup. A proper diagram with data flow arrows and tensor shapes would be much more informative.

### Questions for Authors
- Q1. What is the GPU memory consumption at N=200? Does this fit on a 24GB RTX 4090?
- Q2. Could iterative solvers (e.g., conjugate gradient) replace LU factorization for the (I-J_z)S = J_A solve, reducing the O(D^3) bottleneck?

---

## Summary of Scores

| Reviewer | Score | Confidence | Recommendation |
|----------|-------|------------|---------------|
| R1 (Adversarial ML) | 6/10 | 4/5 | Weak Accept |
| R2 (Implicit Networks) | 7/10 | 4/5 | Accept |
| R3 (Theory) | 4/10 | 5/5 | Borderline Reject |
| R4 (Applied/Power) | 6/10 | 3/5 | Weak Accept |
| R5 (Systems) | 6/10 | 3/5 | Weak Accept |

**Aggregate: Weak Accept — likely accepted on a non-competitive cycle, borderline on a competitive one.**

---

## Priority Revision Items

### P0 — Must fix before submission
1. **Title change:** Remove "Certifiable" — the paper explicitly disclaims certificates. Consider: "AEGIS: Equilibrium Structure Mining for Graph Adversarial Vulnerability Analysis" or "AEGIS: Structural Sensitivity Analysis for Implicit Graph Neural Networks."
2. **Tightness at larger epsilon:** Report tightness ratio at eps=0.05 and eps=0.10 (not just breach rate). This addresses R3-W2 head-on.
3. **Add clean accuracy:** Report clean accuracy per dataset. This is standard and its absence is conspicuous (R1-W3).

### P1 — Strongly recommended
4. **Report attack success rate:** Fraction of nodes whose prediction changes (not just fixed-point shift). This is the standard adversarial metric (R1-W3).
5. **Trim or remove Mettack table:** The comparison is acknowledged as unfair. Consider moving to supplementary or reducing to one sentence with the 30/30 number (R1-W2).
6. **Replace text-box figure with proper diagram:** The pipeline "figure" is currently text in a framebox. A proper architecture diagram would significantly help (R1, R5).

### P2 — Nice to have
7. **Tightness vs. epsilon plot:** A curve showing tightness degradation from eps=0.01 to eps=0.20 would address R2-Q1 and R3-W2 simultaneously.
8. **Weighted adjacency for power flow:** Use actual admittance values instead of binary adjacency (R4-Q1).
9. **GPU memory analysis at N=200:** One line noting memory consumption (R5-W2).
10. **Degree-weighted S_c:** Address R3-Q3 about whether weighting S_c columns by degree would better model raw-A perturbations.
