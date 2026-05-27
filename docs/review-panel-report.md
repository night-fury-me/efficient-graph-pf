# AEGIS — Simulated Peer-Review Panel Report

**Paper:** *AEGIS: Mining Graph Structure for Adversarial Vulnerability Analysis of GNNs*
**Venue assumption:** IEEE top venue (TNNLS / TIFS / TDSC-class)
**Date:** 2026-05-27
**Review mode:** Full (5 independent reviewers)

---

## Panel Summary

| Reviewer | Role | Recommendation | Confidence |
|----------|------|---------------|------------|
| R0 (EIC) | Editor-in-Chief / Meta-reviewer | Minor Revision | 4/5 |
| R1 | Theory & Methods | Minor Revision | 4/5 |
| R2 | Empirical & Experimental Design | Minor Revision | 4/5 |
| R3 | Domain Specialist (Power Systems / Applied ML) | Minor Revision | 3/5 |
| R4 | Devil's Advocate | Minor Revision | 4/5 |

**Consensus:** Minor Revision. Core contribution (constrained sensitivity matrix $S_c$) is sound, novel, and well-validated. No fatal flaws identified. Revision items are addressable without re-running experiments.

---

## R0 — Editor-in-Chief Meta-Review

### Overall Assessment

AEGIS presents a structurally-grounded vulnerability analysis framework for GNNs built on the implicit function theorem. The central contribution — the constrained sensitivity matrix $S_c$ that projects perturbations from $N^2$ to $|E|$ dimensions while enforcing symmetry and edge-only support — is a genuine methodological advance. The constrained-vs-unconstrained tightness gap (1.00 vs 0.31) demonstrates this projection is not cosmetic but essential.

The paper is unusually thorough: 7 architectures, 9 datasets across 4 domains, 10 seeds with standard deviations, honest baseline treatment (the Mettack disclaimer is exemplary), and a transparent limitations section. The power flow case study adds real cross-domain value.

### Decision: Minor Revision

The paper should be accepted after addressing the items below. No fundamental re-design is needed.

### Required Revisions (4 items)

1. **[R1-C1] Norm conflation in Theorem 1(a) proof.** The proof bounds $\|\hat{A}' Z W^\top\|$ using $\|\delta A\|_F \cdot \|W\|_2$, but the Frobenius norm of the adjacency perturbation does not directly bound the operator norm of the composed map. State $\varepsilon_{\text{crit}}$ as a Frobenius-norm sufficient condition, or add the missing norm inequality step.

2. **[R2-C1] Report $\kappa$ directly or explain omission.** The formal bounds require $\kappa = \|J_z\|_2$ but tables only report $\rho(J_z)$. The $\eta = 1.02$–$1.28$ values suggest $\kappa/\rho \leq 1.28$, but $\kappa$ is never tabulated. Either report $\kappa$ alongside $\rho$ or add a sentence explaining why $\rho$ suffices given the mild non-normality.

3. **[R2-C2] Mettack comparison needs 10 seeds or removal.** If Table VII uses fewer than 10 seeds, either re-run with 10 seeds for consistency or remove it and rely on the adaptive PGD comparison (which is the fairer baseline anyway).

4. **[R3-C1] Clarify "Converged" column in Appendix Table IV.** Does "Converged = No" mean the IGNN fixed-point iteration failed during AEGIS analysis, or that training at high $\rho$ failed? These have very different implications for the phase transition claim.

### Recommended Improvements (5 items)

5. **[R1-R1] Use $\varepsilon = 0.10$ tightness as headline.** Tightness 1.00 at $\varepsilon = 0.01$ is mathematically expected (any first-order approximation is good at small $\varepsilon$). The more impressive and informative result is tightness within 15% at $\varepsilon = 0.10$ — lead with that.

6. **[R2-R1] Add significance tests for defense ablation.** The spectral pruning defense (Sec. V-F) should include paired significance tests (Wilcoxon signed-rank or paired t-test) comparing protected vs. unprotected accuracy drops.

7. **[R3-R1] Discuss admittance-weighted adjacency for power grids.** The binary adjacency simplification is reasonable for a first study, but the paper should acknowledge that admittance-weighted edges would better capture electrical coupling and discuss this as future work.

8. **[R3-R2] Validate subgraph representativeness on larger graphs.** Report rank correlation between subgraph vulnerability rankings from different BFS roots on the same graph. On small IEEE cases (case14, case30), compare subgraph top-$k$ against full-graph brute-force rankings.

9. **[R4-R1] Reconcile Citeseer appendix vs. main-text accuracy.** Appendix B reports 42–47% accuracy for Citeseer, but Table I reports 66.0%. Explain the discrepancy or update the appendix.

---

## R1 — Theory & Methods Reviewer

**Expertise:** Spectral graph theory, adversarial ML, implicit differentiation
**Recommendation:** Minor Revision
**Confidence:** 4/5

### Strengths

1. **$S_c$ is the right abstraction.** The projection from $N^2$ to $|E|$ (Eq. 7) enforces symmetry and edge-only support by construction. The 3.2–3.3× tightness ratio over unconstrained analysis (Appendix Table) proves this is not optional — unconstrained SVD wastes attack budget on non-edges, asymmetric entries, and self-loops. This is the paper's most important insight.

2. **Three-regime phase transition is well-characterized.** Theorem 1 cleanly identifies subcritical (first-order tight), critical (resolvent blowup), and supercritical (contractivity loss) regimes. The critical budget $\varepsilon_{\text{crit}} = (1 - \kappa)/\|W\|_2$ gives practitioners an actionable threshold.

3. **Explicit GNN extension is non-trivial.** Proposition 2's unrolled Jacobian computation $S_K = (\sum_{l=0}^{K-1} J_z^l) J_A$ correctly handles the finite-depth case without requiring convergence. The Cayley-Hamilton truncation remark shows awareness of numerical issues.

4. **Transparent limitation handling.** Five explicit limitations in the conclusion, the rho-vs-kappa caveat, the Mettack disclaimer, and the "local sensitivity ≠ global certificate" distinction are all commendable.

### Weaknesses

1. **Norm conflation in Theorem 1(a).** The proof uses $\|(\hat{A} + \delta A) Z W^\top\| \leq (\|\hat{A}\|_2 + \|\delta A\|_F) \|Z\|_F \|W\|_2$, but $\|\delta A\|_F$ does not bound the operator norm $\|\delta A\|_2$ in general (though $\|\cdot\|_2 \leq \|\cdot\|_F$, so this direction is actually fine — the issue is that the bound is looser than necessary). Clarify which norm the budget $\varepsilon$ refers to in the theorem statement. **Severity: Medium.** The bound is correct but potentially conservative; the proof should be explicit about this.

2. **Proposition 3 uses unconstrained $S_v$ instead of constrained.** The per-node radius $r_v = \gamma_v / \sigma_1(S_v)$ uses the full $S_v \in \mathbb{R}^{d \times N^2}$, not the constrained rows of $S_c$. Since the threat model restricts to symmetric, edge-only perturbations, using $S_c$ rows would give strictly larger (less conservative) radii. This is a missed opportunity, not a bug. **Severity: Low.** Mention the constrained variant and note it gives tighter certificates.

3. **Assumption A3 is verified post-hoc, not enforced.** A3 requires $\|J_z\|_2 < 1$ at the fixed point, but spectral-norm constraining $W$ during training does not guarantee this (it depends on $\hat{A}$ and activation patterns). The paper checks A3 empirically ($\rho < 1$ on all datasets) but does not discuss what happens if training produces $\rho \geq 1$. **Severity: Low.** Add a remark about what AEGIS reports when A3 is violated (presumably: a warning, not a crash).

4. **The 28% optimism claim ($\kappa$ vs $\rho$) lacks formal bounds.** The paper states $\eta = 1.02$–$1.28$, meaning operator-norm contractivity could be up to 28% worse than spectral-radius contractivity. But no formal bound on $\eta$ is provided — it's purely empirical. For a theory paper, this gap should at least be acknowledged as a limitation of the spectral-radius diagnostic. **Severity: Low.**

### Questions for Authors

1. Have you computed $\kappa = \|J_z\|_2$ directly (via power iteration on $J_z$) rather than relying on $\rho$ and $\eta$? If so, please tabulate it.
2. Could you derive a bound on $\eta$ in terms of the graph structure (e.g., graph regularity, degree distribution)?
3. Does the unrolled Jacobian $S_K$ for explicit GNNs converge to $S$ as $K \to \infty$ when the model is contractive? If so, this would unify the implicit and explicit cases.

### Minor Issues

- Eq. (3): The notation $J_A = \partial F / \partial \text{vec}(A)$ should specify whether this is the Jacobian w.r.t. the full $A$ or only the upper triangle (given symmetry). This matters for the dimension of $S$.
- Observation 1 (nonnormality index): The threshold $\eta < 2$ for "normal" is ad hoc. Consider citing Trefethen & Embree's pseudospectral theory for a principled threshold.
- The paper uses "IGNN-class" but should cite Gu et al. (2020) at the first use of this term, not later.

### Scores

| Criterion | Score |
|-----------|-------|
| Novelty | 7/10 |
| Technical Soundness | 7/10 |
| Clarity | 8/10 |
| Significance | 7/10 |

---

## R2 — Empirical & Experimental Design Reviewer

**Expertise:** GNN benchmarking, adversarial attacks on graphs, reproducibility
**Recommendation:** Minor Revision
**Confidence:** 4/5

### Strengths

1. **Exceptional experimental breadth.** 7 architectures (IGNN, GCN-2, GCN-4, GIN-2, GAT†-2, SAGE-2, APPNP), 9 datasets across 4 domains (citation, e-commerce, encyclopedia, power grid), 10 fixed seeds reported upfront. This is substantially more thorough than typical adversarial GNN papers (which often evaluate on 1–2 architectures and 3 citation datasets).

2. **Honest adaptive attack evaluation.** The Mettack disclaimer ("this gap largely reflects surrogate-to-IGNN architectural mismatch") is rare and commendable. The white-box adaptive PGD attacker using the same IFT gradients (Sec. V-C) is the correct control: it confirms SVD is 20–50% more damaging than iterative PGD, and the 0% breach rate at $\varepsilon = 0.01$ validates the certificates.

3. **Comprehensive statistical protocol.** Standard deviations on all aggregated metrics. Effect sizes (attack advantage ratios) rather than just p-values. The 10-seed protocol with explicitly listed seeds (42, 137, 271, …, 9999) enables exact replication.

4. **Hyperparameter sensitivity analysis.** Sec. V-E showing tightness stability across hidden dimensions $d \in \{16, 32, 64, 128\}$ and subgraph sizes $N \in \{30, 50, 100, 200\}$ addresses a common reviewer concern preemptively.

5. **Structured baselines are informative.** The degree-proportional, spectral, and edge-betweenness baselines (Table II) provide structural context that random-only comparisons miss. AEGIS outperforming degree-proportional by only 6% on Cora while dominating spectral by 47% reveals that high-degree edges are genuinely important.

### Weaknesses

1. **Baseline GNN accuracies are below published results.** IGNN achieves 77.5% on Cora (vs. 83.5% reported by Gu et al. 2020) and 66.0% on Citeseer (vs. 72.4%). The spectral-norm constraint likely explains this gap, but the paper should explicitly state: "Our IGNN uses spectral-norm constrained weights to satisfy A2, which reduces accuracy by X% relative to unconstrained IGNN." Without this, readers may question the implementation. **Severity: Medium.**

2. **Subgraph-to-full-graph gap is not fully addressed.** AEGIS analyzes 50-node BFS subgraphs, but real attacks target the full graph. The paper shows tightness is stable across subgraph sizes (Sec. V-D) but does not validate whether the most vulnerable edges in the subgraph are also globally vulnerable. The Amazon Photo case ($\tau = -0.15$ on subgraphs, partially recovering to $+0.03$ on full graph) reveals this is a real issue. **Severity: Medium.**

3. **Defense ablation lacks significance testing.** Sec. V-F reports accuracy-drop reductions from spectral pruning, but no statistical tests accompany the comparison. With only 10 seeds and potentially high variance, the defense improvements may not be statistically significant. **Severity: Low-Medium.**

4. **Cross-dataset $\tau$ table should report confidence intervals.** Table IV reports $\tau \pm \text{std}$, but with 10 seeds, a 95% CI would be more informative. Several $\tau$ values have large standard deviations (e.g., GIN-2 on Amazon Photo: $+0.18 \pm 0.20$), making it unclear whether the positive correlation is reliable. **Severity: Low.**

5. **Missing code availability statement.** Reproducibility requires code. The paper promises code but does not provide a URL or anonymous repository link. **Severity: Low** (common for anonymous submission, but should be addressed in camera-ready).

### Questions for Authors

1. What is the accuracy gap between spectral-norm-constrained and unconstrained IGNN on each dataset? This contexualizes the baseline numbers.
2. On Amazon Photo (avg. degree ~31), what fraction of the graph's edges fall within the 50-node BFS subgraph? If it's <5%, the subgraph may not be representative.
3. For the defense ablation: did you run a paired Wilcoxon test? If so, what are the p-values?
4. Is the code repository ready for camera-ready? Will it include pre-trained models and analysis scripts?

### Minor Issues

- Table I: Report both constrained and unconstrained tightness in the main table (not just appendix) to emphasize the key insight.
- Table IV: Bold the highest $\tau$ per dataset for visual clarity.
- The 330-run cross-architecture table is impressive but dense. Consider a heatmap visualization.
- Sec. V-A: "149/150 wins" against Mettack — report the exact loss case details in the appendix (seed 5772, WikiCS, $k=1$).

### Scores

| Criterion | Score |
|-----------|-------|
| Experimental Design | 8/10 |
| Statistical Rigor | 7/10 |
| Reproducibility | 7/10 |
| Baselines & Comparisons | 8/10 |

---

## R3 — Domain Specialist (Power Systems / Applied ML)

**Expertise:** Power flow computation, GNNs for grid applications, N-1 contingency analysis
**Recommendation:** Minor Revision
**Confidence:** 3/5

### Strengths

1. **Conceptually compelling N-1 analogy.** The mapping — edge perturbation → line trip, vulnerability spectrum → contingency severity, $\varepsilon_{\text{crit}}$ → stability margin — is elegant and well-motivated. This is a genuine insight connecting adversarial ML to classical power systems analysis.

2. **Comprehensive IEEE test cases.** Evaluation on case14, case30, case57, and case118 covers the standard hierarchy. The model quality metrics (per-unit RMSE for $|V|$, $\theta$, $\Delta S$) are the right diagnostics for power flow.

3. **Positive $\tau$ across all cases.** Kendall $\tau = +0.37$ to $+0.67$ with P@10 = 0.66–0.81 across 10 seeds is a meaningful result. The correlation is not perfect, but recovering 66–81% of the top-10 critical lines from a single IFT analysis (vs. $|E|$ full power flow solves) has practical value.

4. **Honest operational caveat.** The paper explicitly states that $\tau = 0.37$–$0.67$ "requires verification for operational use" — this is the correct framing for a proof-of-concept cross-domain application.

### Weaknesses

1. **Binary adjacency is a significant simplification.** Real power grids have admittance-weighted edges where line impedance determines electrical coupling. A binary adjacency matrix treats a 10 MW tie line the same as a 500 MW backbone transmission line. This limits the operational relevance of vulnerability rankings. **Severity: Medium.** The paper should discuss admittance-weighted adjacency as future work and explain why binary adjacency was chosen (presumably: compatibility with the classification-benchmark pipeline).

2. **Uniform load scaling training data.** The training data is generated by uniformly scaling all loads, which does not capture realistic operating conditions (correlated renewable generation, localized demand patterns, N-k contingencies). This limits the GNN's ability to learn realistic power flow behavior, which in turn limits the relevance of the vulnerability analysis. **Severity: Medium.**

3. **No comparison to fast contingency screening methods.** The paper positions AEGIS as a faster alternative to brute-force N-1, but does not compare against established fast screening methods (DC approximation, linear sensitivity factors like PTDFs/LODFs, or recent ML-based screening). Without this comparison, the speed-accuracy tradeoff is uncontextualized. **Severity: Medium.**

4. **case300 is missing.** The IEEE case300 is a standard benchmark for scalability. Its absence is notable, especially since the paper claims "the vulnerability spectrum captures domain-specific structure." Does AEGIS scale to 300+ buses? **Severity: Low-Medium.**

5. **Continuous perturbation ≠ line trip.** The vulnerability spectrum computes the sensitivity of the GNN's output to continuous edge-weight changes, not to binary line removals. While the correlation is positive, the physical interpretation ("edge perturbation maps to line trips") overstates the correspondence. A line trip is a discrete, complete removal of an edge — not a small continuous weight change. **Severity: Low** (acknowledged in limitations, but the prose in Sec. VII could be more careful).

### Questions for Authors

1. Have you tried admittance-weighted adjacency? Even a simple experiment on case14 would contextualize the binary adjacency choice.
2. What is the GNN's prediction accuracy on realistic (non-uniform-scaling) operating conditions? Would the vulnerability rankings change substantially?
3. How does AEGIS's computation time compare to DC power flow (which gives approximate N-1 rankings in milliseconds)?
4. What happens on case300? Is the bottleneck memory, computation, or model quality?

### Minor Issues

- Table V: Add a column for computation time (AEGIS vs. brute-force N-1) to quantify the speed advantage.
- The power-balance residual $\Delta S$ for case14 (0.106 p.u.) is relatively high — is this per-bus or total? If per-bus, this suggests the GNN's predictions violate Kirchhoff's laws by ~10%, which would concern power systems reviewers.
- Cite Ronellenfitsch et al. (2017) on spectral approaches to grid vulnerability for additional context.

### Scores

| Criterion | Score |
|-----------|-------|
| Domain Relevance | 7/10 |
| Technical Correctness (domain) | 6/10 |
| Practical Impact | 6/10 |
| Novelty for Domain | 8/10 |

---

## R4 — Devil's Advocate

**Expertise:** Adversarial ML, robustness certification, stress-testing claims
**Recommendation:** Minor Revision
**Confidence:** 4/5

### Mandate

My role is to identify the strongest possible attacks on this paper's claims, methodology, and framing. The following are not necessarily fatal — they are the arguments a hostile reviewer would make.

### Attack 1: The Tightness = 1.00 Claim is Tautological at $\varepsilon = 0.01$

The headline result — "constrained first-order tightness is $1.00 \pm 0.01$ at $\varepsilon = 0.01$" — is mathematically expected for any smooth function. A first-order Taylor approximation is good to $O(\varepsilon^2)$ by definition. At $\varepsilon = 0.01$, the $O(\varepsilon^2)$ term is $O(10^{-4})$, so tightness $\approx 1$ is guaranteed without any analysis.

**The real test** is $\varepsilon = 0.10$ (where the paper reports "within 15%"), and this should be the headline result. The paper does report this, but buries the more informative number in favor of the impressive-sounding 1.00.

**Defense available to authors:** Reframe. The $\varepsilon = 0.01$ result validates the implementation (if tightness ≠ 1.00, something is wrong). The $\varepsilon = 0.10$ result demonstrates utility.

### Attack 2: Continuous Edge-Weight Perturbations Have No Real-World Attack Analogue

No known real-world adversary perturbs edge weights continuously. Real graph attacks are discrete: add/remove edges (Nettack, Mettack), inject nodes, or modify features. The entire threat model is a mathematical convenience, not a realistic attack scenario.

The paper's defense ($\tau = +0.22$ to $+0.54$ continuous-to-discrete transfer) partially addresses this, but: (a) these $\tau$ values are moderate at best, (b) the transfer is dataset-dependent (negative $\tau$ on Amazon Photo subgraphs), and (c) no formal guarantee connects continuous sensitivity to discrete vulnerability.

**Defense available to authors:** Position AEGIS as a diagnostic/analysis tool, not an attack tool. The value is understanding *which edges matter*, not executing attacks. The continuous framework gives tractable, differentiable analysis that transfers to discrete settings in most cases.

### Attack 3: The Subgraph Analysis May Not Generalize

AEGIS analyzes 50-node BFS subgraphs, but:
- On Amazon Photo (7,650 nodes, avg. degree 31), the subgraph contains <1% of edges
- The BFS root determines which subgraph is analyzed, introducing root-selection bias
- The paper does not report inter-root variance (do different BFS roots give different vulnerability rankings?)
- The full-graph "matrix-free" analysis partially recovers ($\tau: -0.14 \to +0.03$) but $+0.03$ is essentially zero correlation

**Defense available to authors:** (a) Report inter-root $\tau$ variance. (b) The subgraph analysis is explicitly positioned for local vulnerability assessment, not global ranking. (c) Explicit GNNs (GCN-4, SAGE-2) show strong $\tau$ even on dense graphs, suggesting the issue is IGNN-specific, not framework-specific.

### Attack 4: The Defense Ablation is Weak

The defense experiment (Sec. V-F) masks the top-$k$ vulnerable edges and shows reduced attack damage. But:
- This is a trivially expected result: removing the SVD attack's preferred edges reduces SVD attack damage
- The paper does not test whether the defense helps against *other* attacks (adaptive PGD, Mettack, random)
- No comparison to existing defense methods (adversarial training, GNNGuard, robust aggregation)

**Defense available to authors:** The defense is presented as an illustrative application, not a state-of-the-art defense method. The point is that $S_c$ rankings inform defense design, not that spectral pruning is optimal.

### Attack 5: $\kappa$ vs $\rho$ — The 28% Gap Undermines Formal Guarantees

The formal bounds use $\kappa = \|J_z\|_2$, but the paper only reports $\rho(J_z)$. If $\kappa$ is 28% larger than $\rho$ (as $\eta$ values suggest), then:
- $\varepsilon_{\text{crit}}$ computed from $\rho$ is 28% optimistic
- The certified shift bound is 28% loose
- Some edges certified as "safe" at $\varepsilon$ near $\varepsilon_{\text{crit}}$ may actually be in the critical regime

The paper acknowledges this ("at most 28% optimistic"), but a hostile reviewer would ask: why not just compute $\kappa$ directly? It's a single SVD of $J_z$, which is $O(D^2)$ — dominated by the $O(D^3)$ linear solve already in the pipeline.

**Defense available to authors:** Just compute and report $\kappa$. This is a revision-level fix, not a fundamental problem.

### Attack 6: Citeseer Appendix Anomaly

Appendix B reports Citeseer accuracy of 42–47%, while Table I reports 66.0%. This is a 20-percentage-point discrepancy that is unexplained. If the appendix uses different experimental conditions, it should say so. If not, one of the numbers is wrong.

**Defense available to authors:** Explain or fix the discrepancy. Likely the appendix is from an earlier experimental run.

### Attack 7: No Comparison to Existing GNN Robustness Certificates

The paper's per-node radius (Proposition 3) is a first-order sensitivity radius, not a robustness certificate in the sense of Zügner & Günnemann (2019), Bojchevski et al. (2020), or Schuchardt et al. (2023). The paper carefully avoids calling it a "certificate" in the formal sense, but uses the term "per-node robust radius" which invites comparison. The paper should either:
- Compare against existing certification methods (and explain why first-order radii are preferable), or
- Rename to "per-node sensitivity radius" to avoid the comparison entirely

**Defense available to authors:** Rename and add a paragraph in related work distinguishing first-order sensitivity radii from global robustness certificates.

### Overall Assessment

None of these attacks are fatal. The paper's core contribution ($S_c$) is sound, the experimental breadth is impressive, and the limitations are honestly disclosed. The main risks are: (a) hostile reviewers fixating on the continuous-perturbation threat model, and (b) the $\varepsilon = 0.01$ tightness being dismissed as trivial. Both are addressable through reframing in the introduction.

---

## Consolidated Revision Checklist

### Required (address before resubmission)

| # | Source | Item | Effort |
|---|--------|------|--------|
| 1 | R1-C1 | Fix norm conflation in Thm 1(a) proof; clarify $\varepsilon$ refers to $\|\cdot\|_F$ | Low (editorial) |
| 2 | R2-C1 | Report $\kappa$ alongside $\rho$ in convergence table, or justify omission | Low (1 SVD per run) |
| 3 | R2-C2 | Run Mettack with 10 seeds or remove Table VII | Low–Med |
| 4 | R0-C4 | Clarify "Converged" column semantics in Appendix Table IV | Low (editorial) |

### Strongly Recommended

| # | Source | Item | Effort |
|---|--------|------|--------|
| 5 | R4-A1 | Lead with $\varepsilon = 0.10$ tightness as headline; use $\varepsilon = 0.01$ as validation | Low (editorial) |
| 6 | R2-W3 | Add paired significance tests to defense ablation | Low |
| 7 | R3-W1 | Discuss admittance-weighted adjacency as future work | Low (editorial) |
| 8 | R0-R4 | Validate subgraph representativeness: inter-root $\tau$ variance | Med |
| 9 | R4-A6 | Reconcile Citeseer appendix vs main-text accuracy | Low |
| 10 | R4-A7 | Rename "per-node robust radius" → "per-node sensitivity radius" or add certificate comparison | Low (editorial) |

### Nice-to-Have

| # | Source | Item | Effort |
|---|--------|------|--------|
| 11 | R1-W2 | Mention constrained Proposition 3 variant | Low (editorial) |
| 12 | R3-W3 | Compare AEGIS timing vs DC power flow for N-1 | Med |
| 13 | R3-W4 | Add case300 results | Med–High |
| 14 | R2-M1 | Heatmap visualization for cross-architecture $\tau$ table | Low |
| 15 | R1-M3 | Bound on $\eta$ in terms of graph structure | High (new theory) |

---

## Score Summary

| Dimension | R1 | R2 | R3 | R4 | Avg |
|-----------|----|----|----|----|-----|
| Novelty / Originality | 7 | — | 8 | — | 7.5 |
| Technical Soundness | 7 | — | 6 | — | 6.5 |
| Experimental Design | — | 8 | — | — | 8.0 |
| Statistical Rigor | — | 7 | — | — | 7.0 |
| Reproducibility | — | 7 | — | — | 7.0 |
| Domain Relevance | — | — | 7 | — | 7.0 |
| Clarity / Writing | 8 | — | — | — | 8.0 |
| Significance / Impact | 7 | — | 6 | — | 6.5 |
| **Overall** | **7.3** | **7.3** | **6.8** | **7.0** | **7.1** |

**Verdict: Accept with Minor Revision.** The constrained sensitivity matrix $S_c$ is a genuine contribution to the GNN adversarial robustness literature. The revision items are tractable and do not require new experiments (except possibly items 8 and 13). The paper's unusual breadth and honesty set it above the median submission at a top IEEE venue.
