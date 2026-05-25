# Domain Review Report -- AEGIS

## Reviewer Profile
- **Role**: Peer Reviewer 2 (Domain Expert)
- **Expertise**: Graph representation learning, implicit/equilibrium neural networks, GNN robustness
- **Review Focus**: Literature coverage, theoretical framework, domain contribution

## Summary Assessment

AEGIS introduces the constrained sensitivity matrix S_c as a unified diagnostic object for adversarial vulnerability analysis of GNNs. The paper makes a well-structured argument that existing attack methods (Nettack, Mettack) find adversarial perturbations but do not produce per-node vulnerability maps, while certified defenses (randomized smoothing, IBP) provide uniform guarantees without structural differentiation. S_c fills this gap by projecting the full N^2-dimensional adjacency sensitivity down to |E| realistic perturbation dimensions, yielding SVD-optimal attacks, per-edge vulnerability rankings, and per-node first-order sensitivity radii from a single computation.

The theoretical contribution is strongest in the implicit GNN (IGNN) regime, where Theorem 1 characterizes three vulnerability regimes via the critical budget e_crit = (1 - kappa) / ||W||_2. The application of the IFT to structural (rather than input) sensitivity is novel in this domain: prior work on implicit network robustness (El Ghaoui et al., Revay et al., Pabbaraju et al.) analyzed dz*/dx, not dZ/dA. The extension to explicit GNNs via Proposition 4 is mathematically straightforward (unrolled chain-rule Jacobian) but practically useful, achieving near-perfect tightness across six architectures. The power flow case study is an effective demonstration of domain transfer, though it remains a proof of concept rather than a validated engineering tool.

The paper's primary weakness is a tension between its two identities: it positions IGNN as the architecture with the strongest guarantees, yet IGNN's limited adoption and accuracy gap (77.5% on Cora vs. 81-83% for GCN/GAT/APPNP) undermine the practical relevance of those guarantees. The "architecture-agnostic" framing is partially compromised by the GAT modification required for continuous differentiability and by the loss of formal guarantees (e_crit, convergence regimes) for all non-implicit architectures. Several important references in the GNN robustness and equilibrium model literature are missing.

## Scores (0-100)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Literature Coverage | 68 | Comprehensive on attacks/defenses; gaps in equilibrium model robustness, graph Lipschitz analysis, and recent certified methods |
| Theoretical Framework | 75 | Sound IFT application with novel structural sensitivity angle; e_crit is a sufficient condition and the three-regime characterization, while clean, is a direct consequence of standard contraction theory |
| Novelty vs. Prior Art | 70 | The S_c construction and its SVD decomposition are genuinely useful; the underlying IFT machinery is standard; the novelty is in the problem formulation rather than the mathematical tools |
| Domain Contribution | 72 | Fills a real gap (structural vulnerability maps), but practical impact is limited by subgraph size cap (~300 nodes), continuous-only perturbation model, and IGNN accuracy lag |
| Positioning Clarity | 78 | Clear articulation of what S_c provides vs. attacks/defenses/explainability; the computational-tool vs. formal-guarantee distinction is well-handled |
| Overall | 72 | Solid contribution with a novel angle on an important problem; needs stronger justification for the IGNN focus and broader literature coverage |

## Strengths

1. **Novel problem formulation.** The paper identifies a genuine gap between attack methods (which find bad perturbations) and certified defenses (which give uniform bounds). The constrained sensitivity matrix S_c is a clean mathematical object that addresses this gap. The N^2 -> |E| projection enforcing symmetry and edge-only constraints is the key insight that makes first-order predictions tight in practice (Section IV-C, Eq. 8).

2. **Strong empirical validation of tightness.** Tightness of 1.00 +/- 0.01 at epsilon = 0.01 across 9 datasets and 7 architectures (Tables I and IV) is convincing evidence that the first-order approximation is practical. The tightness degradation analysis at larger epsilon (Table II) is honest and informative.

3. **Principled adaptive attack evaluation.** The inclusion of a white-box PGD attacker using the same IFT gradients (Section V-D, Table III) is methodologically rigorous. This addresses the common criticism that sensitivity-based methods may not reflect true adversarial risk. The finding that SVD-optimal attacks outperform PGD by 20-50% is notable.

4. **Clear theoretical architecture.** The progression from IFT at equilibrium (Eq. 3) to full sensitivity matrix S to constrained S_c to SVD decomposition is logically tight. The three-regime characterization (Theorem 1) provides intuitive physical analogy (restoring force weakening).

5. **Honest limitation reporting.** The paper explicitly acknowledges that first-order radii are not global certificates (Section III, after Proposition 3), that the continuous perturbation model excludes discrete edge insertions/deletions (Section II-B), that GAT requires modification (Section V-G), and that power flow tau = 0.37-0.67 is insufficient for direct operational use (Section VI-C). This candor strengthens credibility.

6. **Cross-architecture generalization.** Proposition 4 and the experiments in Table IV demonstrate that S_c works for GCN, GIN, GAT, GraphSAGE, and APPNP, not only IGNN. This broadens the paper's relevance considerably.

## Weaknesses

1. **IGNN accuracy gap undermines practical relevance.** IGNN achieves 77.5% on Cora (Table I), while GCN achieves 78.9%, APPNP 82.2%, and published GCN baselines reach ~81-83% (Kipf & Welling 2017). The paper positions IGNN as the primary architecture because it offers the strongest guarantees (e_crit, convergence regimes), but a practitioner choosing a model for deployment would likely select a higher-accuracy architecture -- at which point the formal guarantees vanish, and AEGIS becomes "only" a computational tool. The paper should explicitly discuss whether the ~4% accuracy gap on Cora is fundamental to contractivity constraints or an artifact of hyperparameter choices, and whether recent implicit GNN improvements (EIGNN, cited but not benchmarked) close this gap.

   *Suggested fix:* Add an accuracy-vs-guarantee tradeoff discussion in Section V or the conclusion. Consider benchmarking EIGNN to show whether implicit models can close the accuracy gap while retaining formal guarantees.

2. **"Architecture-agnostic" claim is overstated.** The abstract states "the computation applies to any differentiable GNN," and the introduction claims "architecture-agnostic" (implicitly). However: (a) GAT requires a custom edge-weighted variant because standard GAT's binary masking yields dZ/dA_ij = 0 for all existing edges (Section V-G, line 150); (b) all formal guarantees (e_crit, three regimes) are specific to contractive implicit GNNs; (c) the paper does not test on heterogeneous GNNs, transformers (e.g., Graphormer), or message-passing networks with edge features. The claim should be scoped to "any GNN whose message passing is smoothly differentiable with respect to edge weights" -- which the paper does say in Section V-G but not consistently elsewhere.

   *Suggested fix:* Replace "any differentiable GNN" in the abstract and introduction with "any GNN with continuously differentiable message passing w.r.t. edge weights" and explicitly list the class of excluded architectures (binary-mask attention, discrete message passing).

3. **Theoretical novelty is application-level, not method-level.** The IFT (Eq. 3) is a standard result. The Neumann series bound ||( I - J_z)^{-1}||_2 <= 1/(1-kappa) is classical (Stewart 1990). The SVD characterization of optimal linear perturbation (Proposition 2) is a direct application of the variational characterization of singular values. The S_c projection (Eq. 8) is a linear restriction. Theorem 1's three regimes follow immediately from the contraction mapping theorem applied to the perturbed operator. The paper's novelty lies in assembling these tools for the structural sensitivity problem and demonstrating their practical utility -- this is a valid contribution, but the paper sometimes presents standard results with a level of emphasis that may overstate their novelty (e.g., Theorem 1 is presented as a major result, but its proof is a direct chain of standard inequalities).

   *Suggested fix:* Acknowledge in Section III that the individual mathematical ingredients are classical, and frame the contribution as the problem formulation and the S_c construction that makes these tools practically useful under realistic constraints.

4. **Continuous perturbation model limits practical attack relevance.** The threat model (Section II-B) restricts perturbations to continuous edge-weight modifications of existing edges. Real graph attacks involve discrete edge insertions/deletions (Nettack, Mettack) that change the graph topology. The paper acknowledges this limitation but does not quantify how well continuous S_c rankings predict discrete-attack vulnerability. The Mettack comparison (Section V-A) uses a different architecture (GCN surrogate), making it impossible to assess whether S_c-based rankings correlate with discrete Nettack/Mettack vulnerability on the same model.

   *Suggested fix:* Add an experiment correlating S_c edge-vulnerability rankings with brute-force single-edge-removal damage (which is discrete but does not require re-normalization for edge removal). The paper partially does this (tau in Table IV) but only reports it for the explicit GNN extension, not for the main IGNN results.

5. **Subgraph localization may miss global attack patterns.** Stage 1 extracts a 50-node BFS ego-subgraph (Section IV-A). Global attacks like Mettack operate on the full graph and may target edges far from the victim node. The paper's threat model implicitly restricts to local attacks, but this restriction is not stated in Section II-B. The subgraph ablation (Section V-E) shows tightness is stable, but tightness measures prediction accuracy, not attack completeness -- a globally optimal attack may not lie within the 50-node subgraph.

   *Suggested fix:* Add a discussion in Section IV-A or Section II-B explicitly scoping the threat model to local perturbations within k hops of the target node, and discuss how global attack patterns interact with the BFS extraction.

6. **Missing comparison with GNN Lipschitz analysis.** The paper compares AEGIS radii against randomized smoothing (Section V-C) but not against deterministic Lipschitz-based certificates. Recent work on GNN Lipschitz bounds (Dasoulas et al., "Lipschitz Normalization for Self-Attention Layers," ICML 2021; Thorpe et al., "Grand++: Graph Neural Diffusion with a Source Term," ICML 2022; Zhao et al., "From Stars to Subgraphs: Uplifting Any GNN with Local Structure Awareness," ICLR 2022) provides per-node deterministic bounds that could serve as a more apples-to-apples comparison than smoothing.

   *Suggested fix:* Add a brief discussion comparing AEGIS first-order radii with Lipschitz-based deterministic certificates, noting the tradeoffs (tighter local prediction vs. global guarantee).

7. **Phase transition experiment (Section V-D) lacks critical detail.** The phase transition scan (Appendix Table, also referenced in Section V-D) shows convergence failures at rho >= 0.85, but the text (Section V-D) claims "scaling rho from 0.3 to 0.99 amplifies the fixed-point shift by 83x," which includes non-converged runs. Using non-converged shift values to validate the 1/(1-kappa) prediction is methodologically questionable -- if the iteration did not converge, the "shift" is not a meaningful equilibrium quantity.

   *Suggested fix:* Report the phase transition analysis only for converged runs, or explicitly separate converged and non-converged regimes in the main text.

8. **Kendall tau for vulnerability ranking is moderate.** On IGNN (Table IV), tau = +0.32, meaning the S_c ranking agrees with brute-force only weakly. GCN-2 has tau = -0.04 (essentially random). The paper emphasizes GAT (tau = +0.54) and GCN-4 (tau = +0.49) as strong results, but these are selective. The negative tau for GCN-2 and low tau for SAGE-2 (+0.22) suggest that the first-order approximation does not reliably identify the most vulnerable edges for all architectures at finite perturbation budgets.

   *Suggested fix:* Discuss why tau varies so widely across architectures (is it related to depth, nonlinearity, or the discrete-vs-continuous gap?) and provide guidance on when S_c rankings are trustworthy.

## Missing References / Literature Gaps

1. **Xu et al., "Optimization-Based Adversarial Perturbation on Graphs" (NeurIPS 2019, "Topology Attack").** Cited but not adequately discussed. This work formulates structural attack as bilevel optimization with continuous relaxation of edge additions -- directly relevant to the continuous perturbation model in AEGIS. The relationship between their gradient-based attack and AEGIS's SVD-optimal attack should be compared.

2. **Geisler et al., "Robustness of Graph Neural Networks at Scale" (NeurIPS 2021).** Cited for scaled certified defenses but their PPRGo-based certification approach provides per-node certificates that are structurally differentiated (unlike the uniform smoothing baseline used in Section V-C). This is a more relevant baseline for AEGIS's per-node radii.

3. **Li et al., "AGNNCert: Certified Robustness via Adversarial Graph Neural Network Certification" (2025).** Cited once in the related work but not benchmarked. Since AGNNCert provides deterministic (not probabilistic) per-node certificates, it is the closest existing method to AEGIS's first-order radii and deserves direct comparison.

4. **Maron et al., "Provably Powerful Graph Networks" (NeurIPS 2019) and higher-order GNNs.** AEGIS's sensitivity analysis assumes standard message passing (1-WL). Higher-order GNNs have different sensitivity structures. A brief discussion of applicability limits would strengthen the framework.

5. **Zhu et al., "Graph Neural Networks with Adaptive Residual" (NeurIPS 2021, GCNII).** Deep GCN with initial residual connections achieves 85%+ on Cora. Relevant because AEGIS tests GCN-2 and GCN-4 but not deep GCN variants, and the accuracy gap between IGNN and modern deep GCNs is even larger than with vanilla GCN.

6. **Fung et al., "JFB: Jacobian-Free Backpropagation for Implicit Models" (CVPR 2022).** Cited for training but not discussed in the context of sensitivity computation. JFB's Jacobian-free approach could potentially accelerate AEGIS's Stage 2 for implicit models -- this is worth noting as a scalability path.

7. **Winston & Kolter, "Monotone Operator Equilibrium Networks" (NeurIPS 2020).** Cited but the relationship between monotone operator Lipschitz bounds and AEGIS's e_crit is not discussed. Monotone DEQs have tighter contraction guarantees that would yield different e_crit values -- this comparison would strengthen the theoretical framework.

8. **Mu~noz-Gonzalez et al., "Towards Poisoning of Deep Learning Algorithms with Back-gradient Optimization" (AISec 2017).** Early work on back-gradient poisoning that is conceptually similar to AEGIS's IFT-based gradient computation. Not cited.

9. **Chen et al., "Adversarial Robustness of Graph Neural Networks via Spectral Methods" (NeurIPS 2023, or similar recent spectral-robustness works).** The connection between S_c's SVD and the spectral observations of Entezari et al. is mentioned but not developed. Recent spectral-robustness methods provide a more detailed comparison point.

10. **Liu et al., "EIGNN: Efficient Infinite-Depth Graph Neural Networks" (NeurIPS 2021).** Cited once in related work. Since EIGNN achieves faster convergence and potentially better accuracy than IGNN, benchmarking EIGNN with AEGIS would address the accuracy gap concern and demonstrate that the framework works with improved implicit architectures.

## Questions for Authors

1. How does IGNN accuracy change if you use EIGNN's eigendecomposition-based solver instead of fixed-point iteration? Does the accuracy gap with GCN/APPNP close, and do the formal guarantees (e_crit, Theorem 1) still hold?

2. For the GCN-2 result with tau = -0.04 (Table IV): this means S_c rankings are anti-correlated with brute-force ground truth. What causes this? Is it that 2-layer GCNs have such short receptive fields that the first-order and discrete perturbation effects diverge?

3. The threat model restricts to continuous perturbations of existing edges (Section II-B). Have you measured the correlation between S_c vulnerability rankings and discrete single-edge-removal impact? This would directly test whether the continuous first-order analysis transfers to the more practical discrete setting.

4. The subgraph size is capped at N ~ 300 due to dense Jacobian memory (Section IV-F). Have you explored Jacobian-vector product (JVP) or vector-Jacobian product (VJP) approaches that avoid materializing the full Jacobian? Randomized SVD (Halko et al., 2011) on S_c via JVP queries could potentially scale to much larger subgraphs.

5. The Mettack comparison (Section V-A) uses a GCN surrogate, and you correctly note the architectural mismatch. Have you tried running Mettack with an IGNN surrogate (using IFT for the meta-gradient) to provide a fair comparison?

6. For the power flow case study, the model is trained on uniform load scaling only (Section VI-B limitation). How sensitive are the vulnerability rankings to the training distribution? If the model were trained on more realistic load profiles, would P@10 improve or change character?

7. The paper claims S_c reveals the "mathematically optimal attack direction" (abstract). This is true to first order, but the tightness at epsilon = 0.10 is 1.15 on Cora (Table II), meaning the actual shift exceeds the prediction by 15%. At what epsilon does the SVD-optimal direction cease to be the actually optimal direction (i.e., when does a PGD-style iterative attack find a better direction than the SVD one)?

8. Theorem 1's proof uses ||delta A||_F <= epsilon, but the constrained S_c operates on |E|-dimensional vectors with ||delta_E||_2 <= epsilon. Are these norms equivalent under the symmetry constraint, or does the projection introduce a constant factor? The paper should clarify this.

9. What is the wall-clock time for the adaptive PGD attack (50 steps, Section V-D) vs. the single-pass AEGIS computation? If AEGIS is significantly faster, this strengthens the practical case.

10. The pseudospectral index eta = 1.02-1.28 (Section V-F) suggests mild non-normality. Have you tested on graphs where the IGNN Jacobian is more non-normal (e.g., directed graphs, or graphs with highly skewed degree distributions)? The kappa-based bound could become much more conservative in those cases.

## Recommendation

**Minor Revision.**

The paper presents a genuinely useful contribution: the constrained sensitivity matrix S_c provides a unified, computationally efficient diagnostic for GNN structural vulnerability that fills a clear gap between attack methods and certified defenses. The experimental validation is thorough (9 datasets, 7 architectures, 10 seeds), the adaptive attack evaluation is methodologically sound, and the power flow case study demonstrates cross-domain transfer.

The primary concerns that require revision are:

1. **Scope the "architecture-agnostic" claim** to architectures with continuously differentiable message passing w.r.t. edge weights. Acknowledge the GAT limitation and the loss of formal guarantees for explicit GNNs more prominently in the abstract and introduction.

2. **Address the IGNN accuracy gap** with either (a) a discussion of accuracy-guarantee tradeoffs, (b) benchmarking EIGNN or other improved implicit architectures, or (c) an argument that the 4% gap is acceptable given the formal guarantees.

3. **Expand the literature coverage** with at least the key missing references identified above (AGNNCert deterministic comparison, monotone operator relationship, Topology Attack continuous relaxation connection, EIGNN benchmarking).

4. **Clarify the phase transition experiment** by separating converged and non-converged regimes.

5. **Discuss the tau variability** across architectures (Table IV) -- the wide range from -0.04 to +0.54 needs explanation and guidance for practitioners.

None of these issues invalidate the contribution, but they would significantly strengthen the paper's credibility and positioning. The mathematical framework is sound, the experiments are reproducible (10 seeds, code promised), and the practical utility is demonstrated. With the above revisions, this paper would be a solid contribution to the GNN robustness literature.
