# EIC Review: AEGIS -- Mining Graph Structure for Adversarial Vulnerability Analysis of GNNs

**Venue**: IEEE International Conference on Data Mining (ICDM)  
**Track**: Graph Mining  
**Role**: Senior Area Chair  
**Date**: 2026-05-26

---

## 1. Summary

This paper introduces AEGIS (Adversarial Evaluation of Graph Integrity via Sensitivity), a framework for pre-deployment adversarial vulnerability analysis of graph neural networks. The core idea is to construct a constrained sensitivity matrix $S_c$ that maps continuous edge-weight perturbations to hidden-state shifts, derived via the Implicit Function Theorem for contractive implicit GNNs (IGNN-class) and via unrolled Jacobians for explicit architectures. From $S_c$, AEGIS extracts three artifacts: (1) SVD-optimal attack directions, (2) per-edge vulnerability rankings, and (3) per-node first-order sensitivity radii. A matrix-free pipeline using Neumann-series resolvent iteration, autograd JVPs, and randomized SVD enables full-graph analysis (Cora $N$=2,708 in 78s, 1 GB). The paper proves a three-regime vulnerability characterization (Theorem 1) with a critical perturbation budget $\varepsilon_{\text{crit}}$ for contractive models, and generalizes the $S_c$ construction to 7 GNN architectures (Proposition 2). Experiments span 9 datasets across 4 domains, including a power grid case study where AEGIS recovers N-1 contingency rankings (P@10 = 0.66--0.81). Tightness is consistently $1.00 \pm 0.01$ at $\varepsilon = 0.01$, and SVD-optimal attacks inflict 2--8x more damage than random perturbation.

---

## 2. Strengths

- **S1. Novel mining perspective on adversarial vulnerability.** Rather than proposing a new attack or defense, AEGIS frames adversarial analysis as a *graph structure mining* problem -- extracting vulnerability spectra from the sensitivity matrix. This "diagnostic before deployment" framing is fresh and practically motivated. The three-output design (ranking, attack, radius) from a single $S_c$ computation is elegant (Section V, Figure 1).

- **S2. Strong theoretical grounding with practical reach.** Theorem 1 (Section IV) provides a rigorous three-regime characterization of vulnerability (subcritical, critical, supercritical) with a closed-form critical budget $\varepsilon_{\text{crit}} = (1-\kappa)/\|W\|_2$. Proposition 2 then extends $S_c$ to arbitrary differentiable $K$-layer GNNs, broadening applicability beyond implicit models. The theory-to-practice gap is well managed: the paper is upfront that explicit GNNs inherit the computational tool but not the formal convergence guarantees.

- **S3. Comprehensive experimental validation.** 9 datasets across 4 domains (citation, co-purchase, Wikipedia, power grids), 7 GNN architectures (IGNN, GCN-2, GCN-4, GIN-2, GIN-3, APPNP, GAT), 10 random seeds each. This breadth is substantially above the ICDM average. The tightness ratio of $1.00 \pm 0.01$ at $\varepsilon = 0.01$ across all datasets (Table I) is a strong empirical result.

- **S4. Scalable matrix-free computation.** The dense pipeline OOMs at $N > 200$ on a 24 GB GPU, while the matrix-free pipeline handles $N = 2,708$ in 78s using only 1.1 GB (Table IV). This is a genuine engineering contribution that makes the theoretical framework practically deployable, not just an analytical curiosity.

- **S5. Compelling cross-domain transfer (power grid case study).** Applying AEGIS to IEEE power flow benchmarks and recovering N-1 contingency rankings without domain-specific inputs (P@10 = 0.66--0.81, Table VI) demonstrates that structural sensitivity analysis captures domain-meaningful vulnerability. This is the kind of mining-to-application transfer that ICDM values.

- **S6. Thorough ablation and diagnostic experiments.** Subgraph size ablation (Section VI-F), tightness degradation across $\varepsilon$ (Table II), convergence analysis with pseudospectral index $\eta$ (Section VI-G), and subgraph-vs-full-graph ranking validation on IEEE cases -- these systematically address potential concerns rather than leaving them implicit.

- **S7. Honest self-assessment.** The paper explicitly identifies six limitations in Section VIII (first-order only, continuous perturbations, GAT modification, power flow verification, Neumann convergence requirement, IGNN accuracy gap) and acknowledges the Mettack comparison is unfair due to architectural mismatch. This intellectual honesty strengthens the paper.

---

## 3. Weaknesses

- **W1. Attack baselines are weak, making the "2--8x advantage" less informative.** The primary attack advantage metric (AtkAdv in Table I) compares AEGIS against *random* perturbation -- the weakest possible baseline. The Mettack comparison uses a GCN surrogate against IGNN targets, and the authors correctly note this reflects architectural mismatch. The adaptive attacker (Table III) uses the *same* IFT gradients as AEGIS, making the comparison partly circular (SVD is the exact solution to the linearized problem; PGD is an approximate solver for the same problem).  
  **Suggested fix**: Include at least one structured attack baseline that operates directly on the target model, such as topology attack (Xu et al. 2019) or RL-based attack (Dai et al. 2018) applied with white-box access to the IGNN. Alternatively, compare against gradient-based edge-importance methods (e.g., integrated gradients on edge weights) as an attack heuristic.

- **W2. Continuous perturbation model limits practical relevance.** Real-world graph attacks involve discrete edge insertions/deletions, not continuous weight perturbations. While the paper shows that continuous $S_c$ rankings transfer to discrete removal ($\tau = +0.22$ to $+0.54$ in Table V), the $\tau$ values are moderate and one architecture (GCN-2) produces $\tau = -0.04$. The formal guarantees (Theorem 1) strictly apply only to continuous perturbations.  
  **Suggested fix**: Add an experiment that directly evaluates AEGIS vulnerability rankings as predictors of discrete edge-removal impact across all 7 architectures and multiple datasets (not just Cora). Report precision@k for discrete removal alongside the existing Kendall $\tau$.

- **W3. IGNN accuracy gap raises questions about practical deployment.** IGNN achieves 77.5% on Cora vs. ~82% for APPNP (Table V). The formal theoretical guarantees (Theorem 1, three regimes, $\varepsilon_{\text{crit}}$) apply only to IGNN-class models. If practitioners must use weaker models to get formal guarantees, the practical value proposition weakens. The paper acknowledges this (Limitation 6 in Section VIII) but does not quantify the tradeoff.  
  **Suggested fix**: Present a table or figure explicitly showing the accuracy-vs-guarantee frontier: for each architecture, show what formal guarantees are available and at what accuracy cost. This would help practitioners make informed decisions.

- **W4. Scalability ceiling at $N \approx 5,000$ is not stress-tested.** The largest graph analyzed is Cora ($N = 2,708$). Pubmed ($N = 19,717$) appears in Table I but likely uses subgraph extraction (50-node BFS). The future work mentions "extending full-graph analysis beyond $N \approx 5,000$." For many ICDM applications (social networks, web graphs), graphs have millions of nodes. The matrix-free pipeline's $O(D \cdot N^2)$ cost for the full sensitivity matrix remains quadratic in $N$.  
  **Suggested fix**: Report wall-clock time and memory for the matrix-free pipeline on Pubmed (full-graph, not subgraph) and Amazon Photo ($N = 7,650$). If full-graph analysis is infeasible, clearly delineate the scalability boundary and discuss whether subgraph-level analysis is sufficient for practical vulnerability assessment.

- **W5. Defense application is mentioned but not fully developed.** The conclusion references a defense ablation (Section VI, "\cref{sec:defense_ablation}") where per-edge rankings inform defense design, but the defense experiment details were not fully visible in the indexed content. If AEGIS's vulnerability map can guide targeted defense (e.g., rewiring or hardening the top-$k$ vulnerable edges), this would be a major practical selling point -- but it appears underexplored.  
  **Suggested fix**: Expand the defense ablation into a standalone subsection with a table showing: baseline accuracy under attack, accuracy after AEGIS-guided defense (top-$k$ rewiring), and accuracy after random defense. Compare against GNNGuard or Pro-GNN as defense baselines.

- **W6. The constrained projection (Eq. 7) assumes perturbations only on existing edges.** This is a restrictive threat model: adversarial edge *insertion* (adding edges between distant nodes) is often more damaging than edge weight perturbation, as demonstrated by Nettack and Mettack. The $S_c$ construction projects $S$ onto the existing edge set, fundamentally excluding insertion attacks.  
  **Suggested fix**: Discuss the edge-insertion case formally. Even if the current framework cannot handle it, a theoretical analysis of what $S_c$ would look like under an expanded perturbation set ($|E| \to |E| + k$ candidate edges) would strengthen the paper.

---

## 4. Questions for Authors

1. **On the adaptive attacker (Table III)**: Since both AEGIS and the adaptive attacker use the same IFT gradients, the comparison validates that SVD is the optimal linear solution -- which is mathematically expected. Can you provide a comparison against an attacker that uses *different* information (e.g., a black-box attack, a spectral attack, or a gradient-free evolutionary approach) to demonstrate that the $S_c$-identified vulnerabilities are robust to the attack methodology?

2. **On explicit GNN tightness**: For explicit architectures (Table V), tightness is reported only on Cora. Do the tightness values hold across all 9 datasets? If tightness degrades on larger or denser graphs (e.g., Amazon Photo, WikiCS), this would affect the practical reliability of the $S_c$ framework for explicit models.

3. **On the power flow case study**: The training data uses uniform load scaling (70--130% of nominal). Real power grids experience highly non-uniform, correlated load patterns and generator outages. Have you tested whether the vulnerability rankings remain stable under more realistic load distributions? Would the rankings change qualitatively if the training data included contingency scenarios (e.g., generator trips)?

4. **On the GAT edge-weighted variant**: The modification to GAT (multiplying attention coefficients by $\hat{A}_{ij}$) changes the model semantics -- standard GAT learns attention weights independently, while the modified version couples them to adjacency weights. Does this modification affect GAT's downstream classification accuracy? Is the modified GAT's vulnerability profile representative of standard GAT?

---

## 5. Minor Issues

- **Notation inconsistency**: The paper uses both $\kappa$ and $\rho$ for spectral quantities. Section VI-A introduces $\kappa = \|J_z\|_2$ as the contraction constant, while Table IV (subgraph ablation) reports $\rho$ without defining it in context. Clarify whether $\rho$ refers to the spectral radius $\rho(J_z)$ or the pseudospectral index $\eta$.

- **Table density**: Tables I--VI contain a large amount of information. Consider moving the tightness degradation table (Table II) or the subgraph ablation table (Table IV) to supplementary material to give the remaining tables more visual breathing room.

- **Section ordering**: Related work (Section VII) appears after experiments and the case study. While this is a valid choice, ICDM readers typically expect related work before the technical sections. Consider whether moving it to Section III (after background) would improve flow.

- **Missing standard deviations in Table VI**: The power flow model quality columns (RMSE for $|V|$, $\theta$, $\Delta S$) appear to report single values rather than mean $\pm$ std across seeds. Adding variance estimates would be consistent with the rest of the paper.

- **Acronym overload**: The abstract introduces AEGIS, $S_c$, SVD, IFT, IGNN, BFS, and PGD within a few sentences. Consider defining fewer acronyms in the abstract and introducing them in the body.

---

## 6. Venue Fit Assessment

**AEGIS is a strong fit for IEEE ICDM.** The paper sits at the intersection of graph mining and adversarial ML, which maps directly to ICDM's core tracks. Specific venue-fit considerations:

- **(a) Novel mining algorithm**: AEGIS frames adversarial vulnerability analysis as a structure-mining problem -- extracting vulnerability spectra from a sensitivity matrix. This is a genuine mining perspective, not merely an attack or defense paper. The per-edge vulnerability ranking is a graph-level structural descriptor derived from the model's sensitivity landscape.

- **(b) Graph analytics**: The $S_c$ construction, constrained projection, and SVD-based analysis are fundamentally graph-analytic operations. The cross-domain validation (citation, commerce, Wikipedia, power grids) demonstrates the generality that ICDM values.

- **(c) Practical application**: The power grid case study (Section VII) demonstrates that AEGIS-mined vulnerability spectra have real-world engineering interpretation (N-1 contingency), which aligns with ICDM's emphasis on practical impact.

- **(d) Experimental rigor**: 9 datasets, 7 architectures, 10 seeds, multiple ablations -- this exceeds ICDM's experimental expectations. The matrix-free scalability engineering makes the contribution deployable, not just theoretical.

- **(e) Potential concern for venue fit**: The theoretical core (Theorem 1, IFT-based analysis) leans toward ICML/NeurIPS-style analysis. However, the mining framing, cross-domain experiments, and case study anchor the paper firmly in the ICDM tradition.

**Verdict**: This paper would be among the stronger submissions in a typical ICDM graph mining track. The mining perspective is distinctive, the experiments are thorough, and the application is compelling.

---

## 7. Overall Assessment

AEGIS presents a well-executed framework that reframes GNN adversarial vulnerability analysis as a graph structure mining problem. The theoretical contributions (Theorem 1, Proposition 2) are sound and the experimental validation is comprehensive across 9 datasets and 7 architectures. The main limitations are the reliance on weak attack baselines (random perturbation), the restriction to continuous edge-weight perturbations, and the underexplored defense application. Despite these, the paper makes a clear, novel contribution to graph mining with strong practical potential. With revisions addressing the attack baselines and the continuous-vs-discrete gap, this would be a strong ICDM paper.

---

## 8. Scores

| Dimension | Score (0--100) |
|-----------|:--------------:|
| Novelty | 75 |
| Technical Soundness | 82 |
| Significance | 70 |
| Clarity | 78 |
| Reproducibility | 80 |

**Detailed justification**:

- **Novelty (75)**: The mining perspective on adversarial vulnerability is genuinely new -- existing work either attacks, defends, or certifies. AEGIS provides a structural diagnostic tool. However, the underlying technical machinery (IFT, sensitivity analysis, SVD) is well-established; the contribution is in their novel combination and application to graph structure. Score reflects "clearly novel framing, incremental technical novelty."

- **Technical Soundness (82)**: Theorem 1 is correct and well-proved (contraction mapping + IFT is standard but carefully applied). Proposition 2's extension to explicit GNNs is clean. The constrained projection (Eq. 7) is simple but effective. Tightness validation at $1.00 \pm 0.01$ is strong. Deducted points for: the circular adaptive attacker comparison, and the gap between continuous theory and discrete practice ($\tau$ as low as $-0.04$ for GCN-2).

- **Significance (70)**: High practical potential for pre-deployment vulnerability assessment. The power grid case study is compelling. However, significance is limited by: (a) the continuous perturbation restriction, which excludes the most common attack models (discrete insertion/deletion); (b) the scalability ceiling at $N \approx 5,000$; (c) the accuracy gap of IGNN (the only architecture with full formal guarantees). Score reflects "solid contribution with clear limitations on practical reach."

- **Clarity (78)**: The paper is well-organized with clear notation, a helpful pipeline figure (Figure 1), and honest limitation discussion. The theory section (Section IV) is dense but readable. Minor issues: notation inconsistency ($\kappa$ vs $\rho$), heavy acronym load in abstract, and related work placement after experiments is unusual. Score reflects "above average clarity with minor issues."

- **Reproducibility (80)**: 10 random seeds with specific values listed, detailed hyperparameter specification, standard datasets, PyTorch implementation. Code release promised upon publication. Deducted points for: missing details on the defense ablation experiment, and the GAT edge-weighted variant's exact implementation is described only in prose.

---

**Overall Recommendation: Weak Accept**

The paper is above the acceptance threshold for ICDM. It presents a novel mining perspective, sound theory, and thorough experiments. The weaknesses (attack baselines, continuous perturbation model, scalability) are addressable in revision and do not undermine the core contribution. I would advocate for acceptance contingent on the authors addressing the attack baseline concern (W1) and providing explicit GNN tightness validation beyond Cora (Q2).
