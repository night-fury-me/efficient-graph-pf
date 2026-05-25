# EIC Review Report — AEGIS

## Reviewer Profile
- **Role**: Editor-in-Chief
- **Expertise**: Adversarial ML, Graph Learning, Neural Network Theory
- **Venue**: IEEE TNNLS-class conference

## Summary Assessment

AEGIS introduces a constrained sensitivity matrix $S_c$ that projects the full $N^2$-dimensional adjacency perturbation space onto the $|E|$-dimensional space of realistic (symmetric, edge-only) perturbations, enabling closed-form extraction of SVD-optimal attack directions, per-edge vulnerability rankings, and per-node first-order sensitivity radii from a single forward-backward pass. The framework is grounded in the implicit function theorem for contractive implicit GNNs (where it additionally provides a critical perturbation budget $\varepsilon_{\text{crit}}$ and three vulnerability regimes) and extends computationally to any differentiable $K$-layer GNN via unrolled Jacobians. Experiments span 7 architectures, 9 datasets (citation networks, e-commerce, encyclopedias, IEEE power grids), and 10 random seeds, with a power-flow case study showing the vulnerability spectrum recovers N-1 contingency rankings (P@10 = 0.66--0.81).

The paper addresses a genuinely underserved niche: the gap between adversarial attack methods (which find damaging perturbations but do not characterize the full attack surface) and certified defenses (which provide uniform guarantees but do not differentiate per-node or per-edge vulnerability). The constrained projection from $S$ to $S_c$ is the key technical insight --- it is what converts theoretically expected first-order tightness into a practically useful tool under realistic perturbation constraints. The cross-architecture generalization and the power-grid case study elevate the work beyond a single-model analysis paper.

However, several substantive concerns temper my enthusiasm. The continuous edge-weight threat model, while mathematically elegant, sidesteps the discrete topology changes (edge insertions/deletions) that dominate real-world graph attacks. The $O(D^3)$ scalability wall at $N \approx 300$ nodes limits applicability to the subgraph regime, and the paper does not adequately address how subgraph-level vulnerability relates to full-graph adversarial risk. The IGNN baseline accuracies (Cora 77.5%, Citeseer 66.0%) fall below published GCN/GAT baselines, raising questions about whether the vulnerability analysis is evaluated on competitive models. Finally, the first-order sensitivity radii are explicitly not global certificates, yet the paper's framing occasionally blurs this distinction.

## Scores (0-100)
| Dimension | Score | Justification |
|-----------|-------|---------------|
| Originality | 78 | The $S_c$ constrained projection and its triple-output extraction (SVD attack, edge ranking, node radii) from a single object is novel. The IFT-based structural sensitivity for GNNs is new; prior IFT work addresses input sensitivity only. However, the individual ingredients (IFT, SVD for worst-case perturbation, Jacobian sensitivity) are well-established. |
| Significance | 72 | Fills a real gap between attack methods and certified defenses. The power-grid case study demonstrates cross-domain transfer. Impact is moderated by the continuous-perturbation restriction and the $N \leq 300$ scalability ceiling, which limits deployment on large real-world graphs. |
| Clarity | 82 | Well-structured with clear notation, logical flow from theory to framework to experiments. The threat model limitations are honestly stated. Minor issues: some notation overloading ($S$ vs. $S_c$ vs. $S_K$ vs. $S_v$) and the paper is dense in places. The pipeline figure (Fig. 1) is helpful. |
| Technical Soundness | 75 | The IFT derivation and constrained projection are correct. Proofs are complete for the claims made. However: (1) the Neumann-series bound uses operator norm $\kappa$ but tables report spectral radius $\rho$, creating a 28% gap the paper acknowledges but does not fully resolve; (2) first-order radii are validated empirically but lack formal second-order remainder bounds; (3) the phase-transition scan (Appendix Table) shows non-convergence at $\rho = 0.85$, which is well below the theoretical critical threshold for most datasets. |
| Reproducibility | 70 | 10 seeds are reported with standard deviations. Implementation details are provided in the appendix. Code is promised but not yet available. The IGNN training procedure (spectral normalization, early stopping over 200 epochs) is described but key hyperparameters (learning rate schedule, BFS seed selection strategy) require the code release for full reproduction. |
| Overall | 74 | A solid contribution that introduces a useful analytical tool with genuine novelty in the constrained sensitivity construction. The cross-architecture and cross-domain validation is commendable. The paper would benefit from addressing the scalability limitations, strengthening the connection between subgraph and full-graph vulnerability, and providing tighter formal guarantees. |

## Strengths

1. **Novel constrained sensitivity construction.** The projection $S \to S_c$ (Eq. in Sec. IV-C, $[S_c]_{:,k} = S_{:,iN+j} + S_{:,jN+i}$) is the key contribution that makes first-order sensitivity practically useful. The constrained-vs-unconstrained tightness gap (1.00 vs. 0.31, Appendix Table) demonstrates that this projection is not a minor refinement but essential for realistic perturbation models. Without it, the SVD attack targets non-edges and asymmetric entries, achieving only 31% tightness.

2. **Unified triple output from a single object.** Extracting SVD-optimal attacks, per-edge vulnerability rankings, and per-node sensitivity radii all from $S_c$ is elegant and practically useful. Each output is independently validated: attack advantage 2--8x over random (Table I), positive Kendall $\tau$ vs. brute-force ground truth (Table IV), and 0% breach rate at $\varepsilon = 0.01$ (Table III).

3. **Broad empirical validation.** 7 architectures (IGNN, GCN-2, GCN-4, GIN-2, GAT-2, SAGE-2, APPNP), 9 datasets across 4 domains, 10 seeds each. This is substantially more thorough than typical adversarial GNN papers that evaluate on 1--2 architectures and 3 citation datasets.

4. **Honest adaptive attack evaluation.** The authors acknowledge that the Mettack comparison (Sec. V-A) is confounded by surrogate mismatch and implement a white-box PGD attacker using the same IFT gradients as AEGIS (Sec. V-C). This is methodologically rigorous --- the adaptive attacker confirms that AEGIS's SVD direction is 20--50% more damaging than PGD, and the 0% breach rate at $\varepsilon = 0.01$ provides empirical validation of the radii.

5. **Power-grid case study with domain-relevant baselines.** Comparing against LODF (industry-standard DC approximation) and demonstrating P@10 = 0.66--0.81 without domain-specific inputs is a compelling cross-domain validation. The observation that the equilibrium architecture implicitly enforces power-balance consistency ($\Delta S = 0.03$--$0.11$ p.u. without explicit Kirchhoff penalty) is an interesting structural insight (Sec. VI-C).

6. **Principled limitation disclosure.** The paper explicitly states that first-order radii are not global certificates (Sec. III, after Prop. 3), that continuous perturbations exclude discrete topology changes (Sec. II-B), that GAT requires modification (Sec. V-G), and that power-flow $\tau$ is insufficient for operational use (Sec. VI-C). This level of intellectual honesty is commendable.

## Weaknesses

1. **Continuous-perturbation threat model is restrictive (Sec. II-B).** Real adversarial attacks on graphs involve discrete edge insertions/deletions (Nettack, Mettack), not continuous weight perturbations. The paper acknowledges this ("Discrete edge insertions or deletions...are outside the formal guarantee") but does not quantify how well continuous-$\varepsilon$ vulnerability rankings predict discrete attack vulnerability. The defense ablation (Sec. V-F) masks edges from continuous SVD attack but does not validate against discrete attackers. **Suggested fix:** Add an experiment comparing $S_c$ edge rankings against discrete edge-removal impact, even if only as a Kendall $\tau$ correlation.

2. **Subgraph-to-full-graph gap is not addressed (Sec. IV-A, V-D).** AEGIS analyzes 50-node BFS subgraphs, but adversarial attacks target the full graph. The paper validates that tightness is stable across subgraph sizes (Sec. V-D) but does not address: (a) whether the most vulnerable edges in the subgraph are also the most vulnerable globally, (b) how subgraph boundary effects distort the sensitivity matrix, or (c) whether different BFS roots produce different vulnerability rankings. **Suggested fix:** Report rank correlation between subgraph vulnerability rankings from different BFS roots, and compare subgraph top-$k$ edges against full-graph brute-force rankings on small graphs (case14, case30).

3. **Baseline GNN accuracies are below published results (Tables I, IV).** IGNN achieves 77.5% on Cora and 66.0% on Citeseer; published GCN baselines are ~81% and ~71% respectively. GCN-2 achieves 78.9% vs. the standard 81.5%. The vulnerability analysis of an underperforming model may not transfer to well-tuned production models. **Suggested fix:** Report results on at least one architecture achieving near-SOTA accuracy (e.g., APPNP at 82.2% is close), and discuss whether vulnerability patterns change with model quality.

4. **$\kappa$ vs. $\rho$ inconsistency undermines formal guarantees (Sec. V, notation paragraph).** Theorem 1 uses the operator-norm contraction constant $\kappa = \|J_z\|_2$, but all tables report the spectral radius $\rho(J_z)$. The paper states the $\varepsilon_{\text{crit}}$ values computed with $\rho$ are "at most 28% optimistic relative to the $\kappa$-based formal threshold." This means the reported critical budgets are not formal guarantees as stated in Theorem 1 --- they are approximations. **Suggested fix:** Report $\kappa$ directly alongside $\rho$ in Table I, or compute and report the formal $\varepsilon_{\text{crit}}$ using $\kappa$.

5. **Phase-transition evidence is incomplete (Sec. V-E, Appendix Table).** The phase-transition scan shows non-convergence at $\rho = 0.85$, but all five benchmark datasets have $\rho \leq 0.59$ (Table I). The critical regime ($\rho \to 1$) is never reached in the actual experiments, making the three-regime characterization (Theorem 1) largely untested on real data. The $83\times$ amplification claim at $\rho = 0.99$ comes from a synthetic $\rho$-scaled experiment, not natural data. **Suggested fix:** Discuss why trained models never approach the critical regime (spectral normalization prevents it by design) and acknowledge that the three-regime theory is primarily of analytical rather than empirical interest for well-trained models.

6. **Scalability is limited to small subgraphs (Sec. IV-E).** The dense Jacobian $J_A \in \mathbb{R}^{D \times N^2}$ requires $O(D \cdot N^2)$ computation and $O(D^2)$ memory for the linear solve, limiting analysis to $N \leq 300$. For large graphs (millions of nodes), even BFS subgraphs of 300 nodes may miss long-range vulnerability pathways. The paper mentions "sparse approximations would extend this" but provides no concrete proposal or analysis. **Suggested fix:** Provide at least a preliminary sparse-Jacobian experiment or a concrete algorithmic sketch for extending beyond $N = 300$.

7. **GAT modification changes the architecture under analysis (Sec. V-G).** Standard GAT uses binary attention masking, making $\partial Z / \partial A_{ij} = 0$ for existing edges. The paper introduces an "edge-weighted variant" that multiplies attention coefficients by $\hat{A}_{ij}$. This is a different architecture from standard GAT, so the claim of "7 architectures" is slightly misleading --- it is 6 standard architectures plus a modified GAT. **Suggested fix:** Clearly label GAT results as "AEGIS-compatible GAT variant" throughout, not just with a dagger footnote in one table.

## Minor Issues

1. **Notation density.** Four variants of the sensitivity matrix ($S$, $S_c$, $S_K$, $S_v$) are introduced across Sections III--IV. A notation table would help readers track these.

2. **Duplicate labels.** Section V-E carries both `\label{sec:phase_transition}` and `\label{sec:scalability}` on the same subsection, and Section V-H carries both `\label{sec:hyperparams}` and `\label{sec:convergence}`. This is not visible to readers but suggests the sections were merged without cleanup.

3. **Appendix accuracy discrepancy.** The appendix reports Citeseer accuracy as $0.467 \pm 0.056$ (without early stopping) and $0.421 \pm 0.028$ (with early stopping), but Table I reports $66.0 \pm 0.7$. The appendix appears to show an older result that was not updated after the Citeseer data-loader fix referenced in the git history.

4. **Missing Pubmed and Amazon in tightness degradation table.** Table II reports tightness at increasing $\varepsilon$ for only Cora, Citeseer, and WikiCS. Pubmed and Amazon Photo are omitted without explanation.

5. **"Code will be released upon publication" (Abstract).** For a venue emphasizing reproducibility, a code availability statement without a repository link or anonymous supplement weakens the reproducibility posture.

6. **Power-flow model quality varies.** Table V shows $|V|$ RMSE ranging from 0.007 (case14) to 0.033 (case57) per-unit, and $\Delta S$ from 0.033 to 0.106 p.u. The case14 power-balance residual (0.106 p.u.) is large --- 10.6% error in power balance is not negligible for engineering applications. This is not discussed.

7. **Defense ablation (Sec. V-F) uses only Cora/IGNN.** A single dataset/architecture combination is insufficient to establish the generality of vulnerability-aware edge protection.

## Questions for Authors

1. How sensitive are the vulnerability rankings to the BFS root selection? If different roots are chosen for subgraph extraction, do the top-10 most vulnerable edges remain consistent?

2. The paper claims first-order radii are "deterministic, not probabilistic" (Sec. III). But they are also local approximations that degrade at larger $\varepsilon$ (Table II shows 36--39% overshoot at $\varepsilon = 0.20$). At what $\varepsilon$ do the radii become unreliable enough that the deterministic label is misleading?

3. For the power-grid case study, have you validated the ContractiveGCN-PF model against standard AC power flow benchmarks (e.g., MATPOWER)? The per-unit RMSEs in Table V suggest non-trivial approximation error. How does model accuracy affect the vulnerability ranking quality?

4. The Mettack comparison uses IGNN pseudo-labels to train the GCN surrogate (Appendix). This is an unusual choice --- standard Mettack trains on ground-truth labels. Can you justify this design decision and discuss how it affects the comparison?

5. WikiCS shows notably tighter first-order predictions than Cora/Citeseer at all $\varepsilon$ levels (Table II: tightness 1.05 at $\varepsilon = 0.20$ vs. 1.36--1.39). What structural property of WikiCS explains this? Is it graph density, homophily, or the lower spectral radius?

6. The defense ablation shows that masking 5 vulnerable edges reduces SVD attack damage by 42%. But does this transfer to defense against discrete attacks (Nettack, Mettack)? If the protected edges are different from what discrete attackers would target, the defense value is limited.

7. Can you provide the actual $\kappa = \|J_z\|_2$ values for all datasets, not just $\rho$? This would allow readers to assess the formal guarantees of Theorem 1 directly.

## Recommendation

**Minor Revision.** The paper presents a genuinely novel analytical framework (the constrained sensitivity matrix $S_c$) that fills a meaningful gap between attack methods and certified defenses. The cross-architecture and cross-domain validation is thorough, the adaptive attack evaluation is methodologically sound, and the limitation disclosure is commendable. The core theoretical contribution --- projecting unconstrained sensitivity onto the realistic perturbation space and extracting a triple output --- is clean and useful.

The weaknesses are real but addressable: the $\kappa$/$\rho$ discrepancy can be resolved by reporting both (Weakness 4), the subgraph-to-full-graph question can be addressed with targeted experiments on small graphs (Weakness 2), and the Appendix accuracy discrepancy (Minor Issue 3) is likely a leftover from an earlier revision. The continuous-perturbation restriction (Weakness 1) is a genuine limitation but one that the authors acknowledge honestly, and it does not invalidate the contribution --- it scopes it.

The paper would benefit most from: (1) reporting formal $\kappa$-based $\varepsilon_{\text{crit}}$ values alongside the approximate $\rho$-based ones, (2) a brief experiment correlating $S_c$ rankings with discrete edge-removal impact, (3) fixing the Appendix accuracy discrepancy, and (4) discussing why trained models never approach the critical regime. These are all minor-revision-scale changes that would substantially strengthen the paper without altering its core contribution.
