# EIC Review Report -- AEGIS

**Reviewer**: Prof. Stephan Guennemann, Technical University of Munich
**Expertise**: Graph mining, adversarial ML on graphs, scalable GNN methods, spectral methods for graph learning
**Confidence**: 5 (expert -- I have published extensively on adversarial robustness of GNNs, certified defenses, and graph perturbation theory)

## Summary (200--300 words)

This paper introduces AEGIS, a framework for pre-deployment adversarial vulnerability analysis of graph neural networks. The central technical object is a constrained sensitivity matrix $S_c \in \mathbb{R}^{Nd \times |E|}$ that projects the full $N^2$-dimensional adjacency perturbation space onto the $|E|$-dimensional space of realistic (symmetric, edge-only) perturbations. From $S_c$, three outputs are extracted: (i) the SVD-optimal first-order attack direction, (ii) per-edge vulnerability rankings, and (iii) per-node first-order sensitivity radii. For contractive implicit GNNs (IGNN-class), the authors additionally derive a critical perturbation budget $\varepsilon_{\mathrm{crit}}$ and a three-regime vulnerability characterization via the implicit function theorem. A matrix-free pipeline using Neumann-series resolvent iteration and randomized SVD enables full-graph analysis up to $N = 7{,}650$ nodes on a single GPU.

The framework is validated on 9 datasets across 4 domains (citation networks, e-commerce, encyclopedias, power grids), 7 GNN architectures, and 10 random seeds per configuration. Continuous-to-discrete transfer is formalized in Proposition 3, with positive Kendall $\tau$ in 29 of 33 architecture--dataset combinations. A power grid case study demonstrates that the vulnerability spectrum recovers N-1 contingency rankings (P@10 = 0.66--0.87) without line-impedance data.

The paper occupies a genuinely novel position in the landscape: it is neither an attack method nor a defense, but a diagnostic tool that maps the structural attack surface of a trained GNN. The theoretical grounding is solid for the implicit case, and the empirical extension to explicit architectures is thorough.

## Strengths

1. **Novel problem framing and unified object.** The constrained sensitivity matrix $S_c$ is the paper's central contribution, and it is well-motivated. The reduction from $N^2$ to $|E|$ dimensions (Section IV, Eq. 5) is what transforms a vacuous unconstrained bound into a tight practical tool. The three-output design (attacks, rankings, radii from a single computation) is elegant and practically useful. No prior work in the GNN robustness literature provides this combination.

2. **Rigorous theoretical development.** Theorem 1 (Section III) is carefully stated with explicit assumptions (A1--A3), and the three-regime characterization is physically intuitive. The proof handles the ReLU non-differentiability issue correctly via the nonsmooth IFT for piecewise-linear maps (Bolte & Pauwels, 2021). Observation 1 (graph-independent nonnormality bound) is a clean result: all nonnormality originates from the weight matrix $W$, not from graph topology, because symmetric $\hat{A}$ has orthogonal eigenvectors. This is a useful structural insight.

3. **Honest treatment of the continuous-to-discrete gap.** Proposition 3 (Section III) formalizes the bridge between continuous sensitivity scores and discrete edge-removal damage, with an explicit remainder bound $|R_k| \leq L_J w_k^2 / (2(1-\kappa)^2)$. The sufficient condition for ranking preservation (Eq. 9) is stated as conservative, and the empirical validation (29/33 positive $\tau$, Table VII) is thorough. The failure case (GCN-2, shallow depth) is diagnosed convincingly.

4. **Comprehensive experimental design.** The attack evaluation taxonomy (Section V-C) spans four quadrants: gradient-based vs. gradient-free, same-objective vs. different-objective. This is methodologically sound and avoids the circularity pitfall common in adversarial ML papers. The SVD optimality verification (1,000 random directions, best reaches only 48% of $\sigma_1$; singular value gap 0.39--0.50) is convincing. The structured baselines (Table III: degree, spectral, betweenness) contextualize the advantage honestly, noting that degree-proportional continuous perturbation is within 6--8% of AEGIS.

5. **Scalability engineering.** The matrix-free pipeline (Section IV-B) is well-designed: Neumann-series resolvent with adaptive depth, autograd JVPs, and randomized SVD. The scalability table (Table VI) demonstrates practical utility: Cora ($N = 2{,}708$) in 78s / 1 GB, Amazon Photo ($N = 7{,}650$) in 363s / 6.5 GB. The $\sigma_1$ accuracy within 0.03% of the dense reference at $N = 200$ validates the approximation.

6. **Cross-domain case study with engineering relevance.** The power grid case study (Section VI) is a genuine application, not a toy example. Recovering N-1 contingency rankings (P@10 = 0.66--0.87 across IEEE case14--case300) without line-impedance data is practically meaningful. The comparison against LODF (industry standard) is fair: AEGIS achieves higher $\tau$ on case57/118 but is slower and requires training. The operational caveat ("screening layer, not standalone tool") is appropriately stated.

7. **Statistical rigor.** 10 random seeds throughout, with mean $\pm$ std reported consistently. Wilcoxon signed-rank tests for key comparisons (AEGIS vs. degree, $p < 0.001$; AEGIS vs. LODF on case57/118, $p < 0.01$). The Pubmed breach rate distribution is noted as right-skewed with IQR reported alongside the mean.

## Weaknesses

1. **The formal guarantees (Theorem 1) apply only to IGNN, which has the weakest accuracy.** IGNN achieves 77.5% on Cora vs. 82.2% for APPNP (Table V). The spectral-norm constraint reduces accuracy by ~6% (noted in Section V). This means the theoretical apparatus -- the paper's primary differentiator from pure empirical work -- applies to a model that practitioners would rarely choose for deployment. The extension to explicit GNNs (Observation 2) is acknowledged as "a direct application of the multivariate chain rule" with empirical-only validation. This creates an uncomfortable gap: the strongest theory applies to the weakest model.

   **Suggested fix:** Investigate whether the three-regime characterization can be extended (even approximately) to weight-tied deep GNNs like APPNP, which have a propagation structure closer to implicit models. Alternatively, provide a formal statement bounding the gap between IGNN's $\varepsilon_{\mathrm{crit}}$ and what an explicit model would exhibit, using the convergence result in the paragraph after Observation 2 ($\sigma_1(S_K) \to \sigma_1(S)$ as $K \to \infty$ for weight-tied models).

2. **Amazon Photo negative $\tau$ for IGNN is not fully resolved.** The paper identifies three contributing factors (Section V-F: subgraph artifact, weak signal from high average degree, architecture-specific) and notes that full-graph analysis shifts $\tau$ from $-0.14$ to $+0.03$. However, $+0.03$ is essentially zero, and only 1 of 3 full-graph seeds reaches $+0.24$. The practitioner guidance ("use explicit GNNs for dense graphs") is pragmatic but does not explain why the theoretical framework fails specifically for IGNN on dense graphs. Is this a fundamental limitation of the contraction-based analysis, or a training artifact?

   **Suggested fix:** Run a controlled experiment isolating the effect of graph density: subsample Amazon Photo to varying average degrees (e.g., 5, 10, 15, 20, 25, 31) and track $\tau$ as a function of density for both IGNN and a reference explicit GNN. This would clarify whether the failure is density-driven or architecture-driven.

3. **The continuous threat model (Eq. 4) limits practical applicability.** Real adversarial attacks on graphs are discrete (edge insertion/deletion), not continuous weight perturbation. The paper addresses this via Proposition 3 (continuous-to-discrete transfer), but the transfer is imperfect: on Cora, AEGIS reaches only 54% of greedy-optimal discrete damage (Table IV, $k = 5$). The "cascade effects that static rankings cannot capture" explanation is qualitative. For a paper emphasizing formal guarantees, this gap deserves more rigorous treatment.

   **Suggested fix:** Provide a second-order correction term or an iterative re-ranking scheme (remove top edge, recompute $S_c$ on the reduced graph, repeat) and measure how much of the greedy gap it closes. Even a proof-of-concept on the 50-node subgraphs would strengthen the claim.

4. **Missing comparison with recent deterministic certification methods.** The related work mentions AGNNCert (Li et al., ICLR 2025) and Geisler et al. (NeurIPS 2021) but does not compare against them experimentally. Randomized smoothing is the only certificate baseline (Section V-B), and the comparison is brief (one paragraph). For a paper claiming to provide "per-node sensitivity radii," a quantitative comparison of these radii against deterministic certificates on the same datasets would be essential.

   **Suggested fix:** Add a table comparing AEGIS $r_v$ radii against AGNNCert and/or the Geisler et al. deterministic certificates on Cora/Citeseer, reporting both the radius magnitude and the fraction of nodes certified. Discuss the computational cost comparison.

5. **Power grid model quality degrades at scale.** In Table VIII, case300 has $\theta$ RMSE of 0.394 p.u., which is an order of magnitude worse than case14--118 (0.020--0.076). The paper uses a 200-node BFS subgraph for case300 and only 1,000 training samples (vs. 2,000 for smaller cases). The high $\tau = +0.72$ and P@10 $= 0.87$ on case300 may reflect that the GNN has learned a coarse approximation whose sensitivity aligns with topology rather than physics, inflating the apparent agreement with N-1 rankings.

   **Suggested fix:** (i) Report $|V|$ and $\theta$ RMSE separately for the 200-node subgraph used in the analysis (not just full-graph). (ii) Train with 2,000 samples on case300 and report whether $\tau$ changes. (iii) Discuss whether the high $\tau$ is an artifact of the model learning topological rather than physical sensitivity.

6. **The defense ablation (Section V-E) is limited to a single dataset and a simple masking protocol.** Masking the top-$k$ edges from the perturbation space (42% damage reduction at $k = 5$) is a proof-of-concept, but real defense design requires retraining the model after edge masking, which could change the vulnerability landscape entirely. The claim that "AEGIS identifies edges whose removal disproportionately reduces vulnerability" is validated only statically.

   **Suggested fix:** Add a dynamic defense experiment: mask the top-$k$ edges, retrain the model, and recompute $S_c$. Report whether the vulnerability reduction persists or whether new vulnerable edges emerge (a "whack-a-mole" effect). Even on Cora alone, this would substantially strengthen the defense narrative.

7. **Duplicate BibTeX entries and minor bibliography issues.** The file `aegis.bib` contains duplicate entries for `trefethen2005spectra` (lines 12--17 and 177--182) and `pei2020geomgcn` (lines 19--24 and 628--633). While minor, this suggests the bibliography was assembled incrementally without deduplication.

   **Suggested fix:** Deduplicate the BibTeX file. Also verify that all 40+ references are cited in the text (a quick audit suggests they are, but the duplicates could cause LaTeX warnings).

## Scores (0--100 scale)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Originality | 82 | The $S_c$ construction and three-output design are genuinely novel. The individual mathematical ingredients (IFT, Neumann series, SVD) are classical, but the paper is transparent about this (Section III, "Relationship to existing sensitivity analysis"). The constrained projection is the key insight. Deducted for the fact that the explicit-GNN extension is mathematically straightforward (chain rule). |
| Significance | 78 | High potential impact as a diagnostic tool. The power grid case study demonstrates cross-domain value. However, the practical significance is tempered by (i) the accuracy penalty of IGNN, (ii) the continuous-only threat model, and (iii) the limited defense integration. The "screening layer" framing is honest but limits the significance claim. |
| Rigor | 85 | Strong theoretical development for the implicit case. Proofs are correct and carefully stated. The continuous-to-discrete bridge (Proposition 3) is a valuable formal contribution. The ReLU non-differentiability is handled correctly. Deducted for the gap between formal guarantees (IGNN-only) and empirical claims (7 architectures), and for the Amazon Photo anomaly. |
| Clarity | 80 | The paper is generally well-written with clear notation (Table I is helpful). The four-stage pipeline description (Section IV) is clean. However, the experiments section is dense: 7 subsections with many tables, and the narrative thread can be hard to follow. The distinction between what is formally guaranteed (implicit) and what is empirically validated (explicit) could be signposted more clearly throughout. |
| Reproducibility | 83 | 10 seeds, explicit hyperparameters, PyTorch implementation, promise of code release. The experimental setup (Section V, first paragraph) is detailed. Deducted because code is not yet available ("upon publication"), the matrix-free pipeline involves several implementation choices (adaptive Neumann depth, power iteration for $\kappa$) whose details are spread across sections, and the GAT modification is non-standard. |
| Literature Coverage | 86 | Comprehensive coverage of adversarial attacks, certified defenses, implicit networks, and GNN explainability. The positioning against randomized smoothing (Section V-B) and localized smoothing (Schuchardt et al., 2023) is fair. Missing: no discussion of graph spectral stability (Gama et al., 2020 is cited in related work but not compared experimentally), and the connection to Lipschitz-bounded equilibrium networks (Revay et al., 2020) deserves more than a passing mention given the shared focus on contraction. |
| Overall | 80 | A strong paper with a genuinely novel contribution ($S_c$ construction), solid theory for the implicit case, and thorough experiments. The main limitations are the theory-practice gap (formal guarantees for the weakest model), the continuous threat model restriction, and insufficient comparison with deterministic certification baselines. These are addressable through targeted revisions. |

## Venue Fit Assessment

This paper is appropriate for a top-tier IEEE data mining or machine learning conference (e.g., ICDM, KDD, AAAI). The contribution sits at the intersection of graph mining, adversarial ML, and applied mathematics, which is core to the IEEE data mining community. The power grid case study adds practical relevance that IEEE venues value.

**In favor of venue fit:** (i) Novel analytical framework for graph vulnerability, directly relevant to the graph mining community. (ii) Cross-domain validation including an engineering application. (iii) Scalability to thousands of nodes, which is necessary for practical graph mining. (iv) The paper addresses a gap between attack methods (offensive) and defense methods (defensive) by providing a diagnostic tool -- this is a new category that the community needs.

**Mild concerns about venue fit:** (i) The theoretical core (IFT on equilibrium equations) is more mathematical than typical data mining papers; the theory section may benefit from more intuitive exposition for the ICDM audience. (ii) The IGNN model, while theoretically elegant, is not widely used in the graph mining community compared to GCN/GAT/SAGE. The explicit extension partially addresses this.

Overall, the venue fit is strong. The paper would also be appropriate for NeurIPS or ICML, but the engineering case study and practical framing make it a natural fit for IEEE venues.

## Questions for Authors

1. **On the accuracy-guarantee tradeoff (Table V):** APPNP achieves 82.2% accuracy with $\tau = +0.35$ but no formal guarantees, while IGNN achieves 77.5% with full theoretical backing. Have you investigated whether APPNP's teleportation structure (which resembles a contraction with restart) could support a relaxed version of Theorem 1? The convergence of $\sigma_1(S_K) \to \sigma_1(S)$ for weight-tied models (paragraph after Observation 2) suggests this might be tractable.

2. **On the phase transition experiment (Section V-D):** You observe that the actual spectral radius $\rho(J_z)$ saturates at approximately 0.42 even at $\kappa_{\max} = 0.99$, due to the ReLU activation pattern. This is an important empirical finding. Can you characterize the activation pattern distribution (fraction of active neurons) as a function of $\kappa_{\max}$? If the activation pattern is sparse at high $\kappa$, this would provide a tighter effective contraction constant.

3. **On the N-2 contingency connection (Section VI-C):** The edge-level overlap (40--64%) between the SVD direction and brute-force N-2 critical pairs is promising. Have you considered using the top-$k$ singular vectors (not just $v_1$) to construct a rank-$k$ approximation of multi-line vulnerability? The subspace spanned by $v_1, \ldots, v_k$ should capture multi-directional vulnerability more effectively.

4. **On scalability beyond $N = 7{,}650$ (Section V-D, Table VI):** Pubmed ($N = 19{,}717$) exceeds 24 GB. You mention distributed JVP computation as future work. Have you estimated the communication overhead? For the Neumann series, each JVP is a local operation on the sparse graph, so the bottleneck is likely the randomized SVD's orthogonalization step. Can you estimate the scaling law (time vs. $N$) from the existing data points?

5. **On the GAT modification (Section V-F):** Standard GAT has zero sensitivity to $A_{ij}$ because attention acts as a binary mask. Your edge-weighted variant GAT-dagger restores differentiability. However, this changes the model's inductive bias: the original GAT learns to weight neighbors purely by content, while GAT-dagger also uses topology. Do you observe any accuracy or generalization difference between GAT and GAT-dagger beyond the 80.5% vs. 77.8% reported? Is the vulnerability ranking of GAT-dagger informative about standard GAT's actual vulnerability to discrete perturbations?

## Recommendation

**Minor Revision.**

The paper presents a genuinely novel contribution to the adversarial robustness landscape for GNNs. The $S_c$ construction, three-output design, and matrix-free pipeline are technically sound and practically useful. The experimental evaluation is among the most thorough I have seen in this area, with 10 seeds, multiple baselines across four quadrants, and honest discussion of limitations (Amazon Photo, GCN-2, continuous threat model).

The weaknesses identified above are all addressable without fundamental restructuring:

- **Essential revisions:** (W4) Add quantitative comparison with deterministic certification baselines. (W1) Provide formal or empirical connection between IGNN guarantees and explicit-GNN behavior for weight-tied models. (W7) Deduplicate bibliography.
- **Strongly recommended:** (W2) Controlled density experiment for Amazon Photo. (W3) Iterative re-ranking proof-of-concept for discrete edge removal. (W5) Subgraph-level model quality for case300.
- **Optional but strengthening:** (W6) Dynamic defense experiment with retraining.

The paper is above the acceptance threshold in its current form. With the essential revisions addressed, it would be a strong accept. I am recommending minor revision rather than accept because the missing deterministic certification comparison (W4) is a significant gap for a paper that introduces per-node sensitivity radii as a key output, and the accuracy-guarantee tradeoff (W1) deserves a formal remark even if the full extension is future work.
