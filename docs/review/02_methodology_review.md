# Methodology Review Report -- AEGIS

**Reviewer**: Dr. J. Zico Kolter, Carnegie Mellon University
**Expertise**: Implicit/equilibrium models, certified robustness, Lipschitz analysis, optimization theory, deep equilibrium networks, implicit function theorem
**Confidence**: 5/5 (core expertise directly overlaps with the paper's theoretical foundations)

## Summary (200--300 words)

This paper introduces AEGIS, a framework for structural vulnerability analysis of graph neural networks via a constrained sensitivity matrix $S_c = (I - J_z)^{-1} J_A P_c$, derived by applying the implicit function theorem (IFT) to the equilibrium equation of implicit GNNs (IGNN-class). The central theoretical contribution is Theorem 1, which characterizes three vulnerability regimes (subcritical, critical, supercritical) governed by a critical perturbation budget $\varepsilon_{\mathrm{crit}} = (1 - \kappa)/\|W\|_2$. From $S_c$, three outputs are extracted: SVD-optimal attack directions (Proposition 2), per-edge vulnerability rankings via column norms, and per-node first-order sensitivity radii (Proposition 3). A matrix-free computational pipeline using Neumann-series resolvent iteration, autograd JVPs, and randomized SVD enables scalability to $N = 7{,}650$ nodes. The framework generalizes to explicit $K$-layer GNNs via Observation 2 (unrolled sensitivity), though formal guarantees are restricted to the implicit case. Proposition 4 provides a continuous-to-discrete transfer bridge with second-order remainder bounds. Experiments span 9 datasets, 7 architectures, and 10 seeds, with a power grid case study validating against N-1 contingency rankings.

The methodology is well-structured: assumptions are clearly stated (A1--A3), the derivation chain from IFT to practical outputs is logically coherent, and the authors are commendably transparent about the conservativeness of their bounds (e.g., the $\sqrt{|E|}$ factor in $\varepsilon_{\mathrm{crit}}$, the accuracy--guarantee tradeoff). The experimental design is comprehensive, with multiple baseline types (gradient-based, gradient-free, same-objective, independent-objective) and appropriate statistical reporting. This is a methodologically mature paper that connects classical perturbation theory to a practical GNN analysis tool.

## Strengths

1. **Rigorous IFT derivation with explicit assumptions.** Assumptions A1--A3 (Sec. 4, lines 10--14 of theory.tex) are precisely stated, with A1 carefully invoking the nonsmooth IFT for piecewise-linear maps (citing Bolte & Pauwels 2021) to handle ReLU non-differentiability. The measure-zero caveat for activation-boundary crossings is mathematically correct and honest. The authors verify A3 empirically post-training ($\kappa = 0.14$--$0.59$), which is the right approach for a condition that depends on the trained model.

2. **Transparent conservativeness analysis of $\varepsilon_{\mathrm{crit}}$.** The Remark after Theorem 1 (theory.tex, line 50) explicitly quantifies the conservativeness: $\varepsilon_{\mathrm{crit}}$ is conservative by up to $\sqrt{|E|} \approx 7$--$14\times$ on 50-node subgraphs. This is a level of intellectual honesty rarely seen. The distinction between Frobenius-norm budgets and operator-norm contractivity is clearly articulated, and the choice to use $\kappa = \|J_z\|_2$ rather than $\rho(J_z)$ is correctly justified by the Neumann series convergence requirement.

3. **Complete proof of Theorem 1 with all three parts.** Part (a) correctly applies the IFT and Neumann series bound. Part (b) correctly identifies the divergence of the resolvent norm as $\kappa' \to 1$. Part (c) correctly states that the Banach guarantee becomes void without overstating the consequence (the system "may still be contractive depending on perturbation direction"). The proof handles the contractivity preservation argument via sub-multiplicativity and triangle inequality (theory.tex, line 43), which is correct.

4. **Observation 1 ($\eta$ bound) is a genuinely useful theoretical insight.** The proof that nonnormality originates entirely from the weight matrix $W$ and not from graph topology (because symmetric $\hat{A}$ has orthogonal eigenvectors, so $\kappa(U_{\hat{A}}) = 1$) is elegant and practically important (theory.tex, lines 53--63). The empirical validation ($\eta = 1.02$--$1.28$) confirms the theoretical prediction. This result means practitioners can diagnose nonnormality from $\kappa(V_W)$ alone, which is a useful post-hoc tool.

5. **Proposition 4 (continuous-to-discrete transfer) fills a critical gap.** The first-order bridge $d_k = w_k \cdot v_k + R_k$ with explicit remainder bound $|R_k| \leq L_J w_k^2 / (2(1-\kappa)^2)$ (Eq. 9, theory.tex) is the result that makes the continuous $S_c$ analysis practically relevant for discrete edge removal. The proof via Taylor expansion with the resolvent identity is correct. The ranking preservation condition (Eq. 10) is a useful sufficient condition, and the authors honestly report that it holds for only 47--62% of edge pairs, correctly characterizing it as "conservative sufficient."

6. **Comprehensive attack evaluation taxonomy (Sec. 5.3).** The four-quadrant design (gradient-based vs. gradient-free, same-objective vs. different-objective) in Table 4 is methodologically exemplary. The inclusion of an independent adaptive baseline (classification-loss PGD using autograd through unrolled iterations, not IFT) avoids the circularity trap. The 1,000 random direction test confirming best-random achieves only 45--49% of $\sigma_1$ is strong evidence for SVD optimality.

7. **Appropriate statistical design.** 10 seeds with explicit seed values (42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999) reported in Sec. 5. Mean $\pm$ std reported throughout. Wilcoxon signed-rank tests used for paired comparisons (e.g., Table 3: $p < 0.001$; defense ablation: $p < 0.002$). The Pubmed breach rate discussion (Sec. 5.3) correctly identifies right-skewed distributions and reports IQR alongside means.

8. **Matrix-free pipeline is well-designed.** The Neumann series with adaptive depth from power-iteration $\kappa$ estimate, early stopping at $\|J_z^k b\| < 10^{-6} \|b\|$, and the use of forward-mode AD for JVPs and reverse-mode for VJPs (framework.tex, Sec. 4.2) is computationally sound. The randomized SVD with $k=10$, oversampling $p=10$, and $n_{\mathrm{iter}}=2$ power iterations follows Halko et al. (2011) best practices.

## Weaknesses

1. **Theorem 1 Part (b): the divergence rate $\Omega(1/(\varepsilon_{\mathrm{crit}} - \varepsilon))$ is stated but the lower bound is only for the worst-case perturbation direction.** The proof (theory.tex, line 45) shows $\|(I - J_z')^{-1}\|_2 \geq 1/(1 - \|J_z'\|_2)$, but this requires $\|J_z'\|_2 \to 1$, which only happens if the perturbation $\delta\hat{A}$ is aligned with the direction that maximizes $\|\hat{A} + \delta\hat{A}\|_2$. For a generic perturbation direction, $\|J_z'\|_2$ may not approach 1 even as $\|\delta\hat{A}\|_F \to \varepsilon_{\mathrm{crit}}$. The authors should clarify that Part (b) describes worst-case behavior over perturbation directions, or provide a direction-dependent refinement.

   **Suggested fix**: Add a sentence in Part (b) clarifying that the $\Omega$ lower bound holds for perturbations aligned with the top singular vector of $\hat{A}$, and that generic directions may yield slower divergence. This does not invalidate the result but would improve precision.

2. **The $L_J$ constant in Proposition 4 is not empirically estimated or bounded.** Proposition 4 states $|R_k| \leq L_J w_k^2 / (2(1-\kappa)^2)$ and the interpretation section (theory.tex, line 139) claims $L_J \leq \|W\|_2^2$ for IGNN-class operators, but this bound is asserted without proof and never verified empirically. Since Proposition 4 is the theoretical foundation for continuous-to-discrete transfer (the paper's key practical claim), the $L_J$ bound deserves more rigorous treatment.

   **Suggested fix**: (i) Provide a short derivation of $L_J \leq \|W\|_2^2$ for the IGNN operator in Eq. 2 (this should follow from the chain rule on $J_z = \mathrm{diag}(\phi') \cdot (\hat{A} \otimes W)$ with respect to $\mathrm{vec}(A)$). (ii) Report the empirical ratio $|R_k| / (L_J w_k^2 / (2(1-\kappa)^2))$ across datasets to show the bound is not vacuous.

3. **The ReLU non-differentiability treatment in Proposition 4 is weaker than in Theorem 1.** For Theorem 1, the authors invoke the nonsmooth IFT (Bolte & Pauwels 2021) to handle ReLU, giving first-order derivatives within each linear region. But Proposition 4's Taylor remainder requires second-order smoothness ($g''(\xi)$ exists), which fails at activation boundaries. The Remark after Proposition 4 (theory.tex, lines starting at "Remark (ReLU non-differentiability)") states $L_J = 0$ within each linear region, which is correct, but the measure-zero argument for boundary crossings is insufficient for a Taylor remainder bound: the path $g(t)$ for $t \in [0,1]$ may cross activation boundaries even if the set of crossing directions has measure zero, because the path is fixed once the perturbation direction is chosen.

   **Suggested fix**: Strengthen the Remark by noting that for a fixed perturbation direction $\delta\hat{A}_k$ (corresponding to a single edge), the path $g(t)$ is piecewise linear (because the IGNN operator is piecewise affine in $A$ for fixed activation patterns). On each linear piece, the Taylor expansion is exact ($R_k = 0$). The bound $|R_k| \leq L_J w_k^2 / (2(1-\kappa)^2)$ then holds as a worst-case over the finitely many linear pieces, with $L_J$ interpreted as the maximum Lipschitz constant across adjacent activation regions (i.e., the jump in $J_z$ at boundaries, bounded by $\|W\|_2^2$).

4. **Observation 2 (Explicit GNN extension) understates the loss of guarantees.** The observation correctly computes $S_K$ via the chain rule (Eq. 7) and notes that explicit GNNs lack $\varepsilon_{\mathrm{crit}}$ and the three-regime characterization. However, the first-order shift bound $\|\Delta Z_K\|_F \leq \sigma_1(S_K) \cdot \|\delta A\|_F + O(\|\delta A\|^2)$ also requires the remainder to be controlled, which for explicit GNNs has no contraction-based bound. The $O(\|\delta A\|^2)$ term could be large if the explicit GNN has exploding gradients or operates near a bifurcation, and no diagnostic is provided to detect this.

   **Suggested fix**: Add a practical diagnostic for explicit GNNs: compute $\prod_{k=1}^K \|J_z^{(k)}\|_2$ and flag when this product exceeds a threshold (e.g., $> 10$), indicating that first-order predictions may be unreliable. The tightness ratio (reported in Table 7) already serves this purpose empirically, but a principled a priori bound would strengthen the claim.

5. **Per-node radius $r_v$ (Proposition 3) conflates two different $S_v$ objects.** Equation 6 uses $S_v$ (block-rows of $S$, the unconstrained sensitivity matrix), while the text notes that a tighter radius $r_v^{(c)}$ using $S_{c,v}$ (block-rows of $S_c$) is available. The experiments "report the conservative unconstrained $r_v$" (theory.tex, line 85). But for the matrix-free pipeline, the full $S$ is never formed -- only $S_c$ actions are available. How is $\|S_v\|_2$ computed in the matrix-free regime? If it requires materializing $S_v \in \mathbb{R}^{d \times N^2}$, this contradicts the scalability claim. If it uses $\|S_{c,v}\|_2$ instead, then the reported radii are actually $r_v^{(c)}$, not $r_v$, and the conservativeness claim is wrong.

   **Suggested fix**: Clarify in Sec. 4.4 (framework.tex) which radius is actually computed in the matrix-free pipeline. If $r_v^{(c)}$ is used, relabel it consistently. If $r_v$ requires the dense path, state this limitation explicitly.

6. **Subgraph extraction introduces uncontrolled approximation for large graphs.** The subgraph ablation (Sec. 5.5) shows that on Cora ($N = 2{,}708$), the 50-node BFS subgraph yields Kendall $\tau = 0.16 \pm 0.13$ vs. full-graph rankings, with P@10 $= 0.17 \pm 0.10$. This is essentially random. The paper acknowledges this (experiments.tex, line 201) but still uses 50-node subgraphs as the default for most experiments. This means the main results (Tables 1--6) are only valid for the local neighborhood around the highest-degree node, not for the graph as a whole.

   **Suggested fix**: (i) Clearly state in the experimental setup that subgraph-level results characterize local vulnerability around the BFS center, not global graph vulnerability. (ii) For the main cross-domain results (Table 1), add full-graph matrix-free results for all datasets where feasible (Cora, Citeseer, WikiCS, Amazon Photo) to validate that the conclusions hold at graph scale.

7. **The $\sigma_1$ accuracy claim for matrix-free vs. dense ("within 0.03%") is reported for a single $N$ value.** The scalability section (experiments.tex, line 175) states that the matrix-free path achieves $\sigma_1$ accuracy within 0.03% of the dense reference at $N = 200$, but no error analysis is provided for larger $N$ where the dense path is unavailable. The Neumann truncation error and randomized SVD approximation error both grow with $N$ (more Neumann terms needed as $\kappa$ increases; randomized SVD error depends on the singular value gap $\sigma_k - \sigma_{k+1}$).

   **Suggested fix**: Report the Neumann residual $\|J_z^{K+1} b\| / \|b\|$ and the randomized SVD error bound $\|(I - Q Q^T) S_c\|$ (from Halko et al. 2011, Theorem 10.5) for the largest graphs tested (Amazon Photo, $N = 7{,}650$). This would validate that truncation and approximation errors remain small at scale.

8. **10 seeds may be insufficient for high-variance settings.** Pubmed breach rate at $\varepsilon = 0.10$ shows mean $10.3\%$ with std $11.0\%$ and a right-skewed distribution (3 of 10 seeds show 0% breach). With only 10 seeds, the confidence interval on the mean is approximately $\pm 7\%$ (using $t$-distribution), making the mean estimate unreliable. Similarly, case14 ranking correlation ($\tau = +0.42 \pm 0.19$) has a 95% CI of approximately $[0.28, 0.56]$, which is wide.

   **Suggested fix**: For high-variance settings (Pubmed breach rates, small power grid cases), either increase to 30+ seeds or report bootstrap confidence intervals. Alternatively, report medians with IQR (as done for Pubmed breach rates) consistently across all tables.

## Detailed Methodology Assessment

### Mathematical Framework

The IFT derivation from equilibrium $G(z^*, A) = z^* - F(z^*, A) = 0$ to the sensitivity matrix $S = (I - J_z)^{-1} J_A$ is standard and correct (Eq. 3, background.tex). The key novelty is the constrained projection $S_c$ (theory.tex, Eq. 5), which reduces the $N^2$-dimensional perturbation space to $|E|$ dimensions by enforcing symmetry ($\delta A = \delta A^T$) and edge-only constraints. This construction is simple but impactful: the unconstrained $\varepsilon_{\mathrm{crit}}$ is conservative by $\sqrt{|E|}$, while the constrained first-order prediction achieves tightness $\approx 1.00$.

**Theorem 1** is correctly stated and proven. The three regimes are a natural consequence of the Neumann series convergence condition. The critical budget $\varepsilon_{\mathrm{crit}} = (1 - \kappa)/\|W\|_2$ is a sufficient condition, not a necessary one, which the authors clearly state. The phase transition experiment (Sec. 5.4) provides a useful empirical calibration: even at $\kappa_{\max} = 0.99$, the actual $\rho(J_z)$ saturates at $\approx 0.42$ due to ReLU activation patterns, so the theoretical divergence is never observed in practice. This is an important finding that contextualizes the formalism.

**Proposition 2** (SVD optimality) follows directly from the variational characterization of singular values and requires no additional proof. The claim is correctly scoped as "first-order optimal."

**Proposition 3** (per-node radius) is a straightforward application of the sensitivity bound to the linear classification head. The proof sketch is correct. The distinction between deterministic-local and probabilistic-global certificates (Remark 1) is well-articulated.

**Observation 1** ($\eta$ bound) is mathematically correct. The Kronecker product eigendecomposition argument is standard (Stewart 1990), and the conclusion that $\eta \leq \kappa(V_W)$ for fully-active networks is a clean result.

**Observation 2** (explicit GNN extension) is the multivariate chain rule applied to $K$-layer composition. The mathematical content is trivial, but the empirical validation across 6 architectures (Table 7) is valuable. The weight-tied simplification converging to the IGNN bound as $K \to \infty$ is a nice consistency check.

**Proposition 4** (continuous-to-discrete transfer) is the most technically involved result. The Taylor expansion via the resolvent identity is correct within each ReLU linear region. The ranking preservation condition (Eq. 10) is a useful sufficient condition. The main concern (Weakness 3 above) is the treatment of activation boundary crossings.

### Computational Pipeline

The Neumann-series resolvent with adaptive depth from $\kappa$ is the correct approach for implicit models. The convergence rate is geometric ($\|J_z^k b\| \leq \kappa^k \|b\|$), so $K \sim \log(10^{-6}) / \log(\kappa)$ terms suffice. For $\kappa = 0.59$ (Citeseer), this gives $K \approx 26$; for $\kappa = 0.33$ (Cora), $K \approx 12$. The reported range $K = 20$--$50$ is consistent.

The JVP/VJP computation via autograd is standard and correct. Using forward-mode AD for $J_A \cdot v$ and reverse-mode for $J_A^T u$ is the right choice.

The randomized SVD with oversampling $p = 10$ and $n_{\mathrm{iter}} = 2$ follows the Halko et al. (2011) recommendations. For a rank-10 approximation, the expected error is $O(\sigma_{11} / \sqrt{p})$, which is small if the singular value spectrum decays. The reported singular value gap $(\sigma_1 - \sigma_2)/\sigma_1 = 0.39$--$0.50$ suggests good spectral separation, making the rank-10 approximation reliable for the leading singular triplet.

The scalability results (Table 6) are impressive: Cora ($N = 2{,}708$) in 78s with 428 MB, Amazon Photo ($N = 7{,}650$) in 363s with 5.5 GB. The dense-to-matrix-free crossover at $N \approx 200$ is correctly identified.

### Experimental Design

**Statistical validity**: 10 seeds is adequate for most settings but marginal for high-variance ones (Pubmed, small power grids). The use of Wilcoxon signed-rank tests for paired comparisons is appropriate. Standard deviations are reported consistently.

**Baseline selection**: The four-quadrant attack taxonomy (Sec. 5.3) is thorough. The structured baselines (degree, spectral, betweenness in Table 3) provide useful context. The Mettack comparison is honestly contextualized as reflecting "surrogate-to-IGNN architectural mismatch." The classification-gradient comparison (Sec. 5.3) demonstrates that equilibrium sensitivity and classification sensitivity are genuinely different vulnerability surfaces.

**Ablation completeness**: Subgraph size (Sec. 5.5), hidden dimension $d$ (Sec. 5.6), spectral norm ceiling $c$ (Sec. 5.6), BFS center strategy (Sec. 5.5), and defense-informed masking (Sec. 5.7) are all ablated. The $\kappa_{\max}$ sweep (Sec. 5.4) is particularly informative.

**Cross-architecture validation**: 7 architectures (Table 7) with honest reporting of failures (GCN-2 negative $\tau$, IGNN negative $\tau$ on Amazon Photo). The GAT modification (GAT$^\dagger$) is clearly described and justified.

**Cross-domain validation**: 9 datasets in 4 domains (citation, e-commerce, encyclopedia, power grids) plus 3 heterophilic benchmarks. The power grid case study (Sec. 6) provides genuine cross-domain transfer evidence.

**Finite-difference validation**: The $\tau = 0.999$ agreement between $S_c$ column norms and finite-difference sensitivity (Sec. 5.3) confirms numerical correctness.

## Scores (0--100 scale)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Mathematical Rigor | 82 | IFT derivation and Theorem 1 are correct; Observation 1 is elegant. Proposition 4's ReLU treatment (Weakness 3) and the $L_J$ bound (Weakness 2) need strengthening. The $\Omega$ lower bound in Part (b) needs directional clarification (Weakness 1). |
| Proof Correctness | 85 | All proofs are correct within their stated scope. The main concerns are edge cases (activation boundaries in Prop. 4) and unstated directional conditions (Thm. 1(b)), not outright errors. |
| Statistical Validity | 78 | 10 seeds with proper reporting is solid for most settings. High-variance settings (Pubmed, small grids) would benefit from more seeds or bootstrap CIs (Weakness 8). The Wilcoxon tests are appropriate. |
| Reproducibility | 88 | All hyperparameters, seed values, and implementation details are reported. Architecture choices are justified. Code release is promised. The only gap is the matrix-free $r_v$ computation (Weakness 5). |
| Computational Soundness | 86 | Neumann series, JVP/VJP, and randomized SVD are correctly implemented. Adaptive depth is well-motivated. The missing error analysis at large $N$ (Weakness 7) is the main gap. |
| Overall Methodology | 83 | A methodologically strong paper with a clean theoretical framework, comprehensive experiments, and honest discussion of limitations. The weaknesses are primarily about tightening edge cases and clarifying technical details, not fundamental flaws. |

## Questions for Authors

1. **Radius computation in matrix-free regime.** Proposition 3 uses $\|S_v\|_2$ (unconstrained), but the matrix-free pipeline only computes $S_c$ actions. How exactly is $r_v$ computed for $N > 200$? If you use $\|S_{c,v}\|_2$ via randomized estimation, please describe the procedure and its approximation guarantees.

2. **Neumann truncation at high $\kappa$.** The text states Neumann depth can reach $\sim 340$ for $\kappa \approx 0.96$. At this depth, accumulated floating-point error may become significant (340 sequential JVPs). Have you verified numerical stability at high $\kappa$? Do you monitor the residual $\|z - J_z z - b\|$ after the Neumann solve?

3. **Phase transition observability.** The phase transition experiment (Sec. 5.4) shows that empirical $\rho(J_z)$ saturates at $\approx 0.42$ even at $\kappa_{\max} = 0.99$. This means $\varepsilon_{\mathrm{crit}}$ is highly conservative in practice. Is there a way to compute a tighter, direction-dependent $\varepsilon_{\mathrm{crit}}$ using the actual perturbation direction $v_1$ from the SVD, rather than the worst-case Frobenius bound?

4. **GCN-2 failure mode.** GCN-2 consistently shows weak or negative $\tau$ across datasets (Table 8). The paper attributes this to shallow depth, but could it also be an artifact of the finite-difference $S_K$ computation for explicit GNNs? For 2-layer GCN, $S_K$ has only two terms in the chain-rule sum (Eq. 7), and the Jacobians may be poorly conditioned. Have you checked the condition number of $S_K$ for GCN-2 vs. deeper models?

5. **Threat model scope for power grids.** The threat model assumes continuous edge-weight perturbations to $\hat{A}$, but power grid N-1 contingency involves complete line removal (discrete). The transfer result (Proposition 4) bridges this gap for generic graphs, but the power grid case uses binary adjacency (Sec. 6.2), where edge weights are approximately uniform after normalization. Could the strong P@10 results (0.66--0.87) be partly an artifact of near-uniform weights making the $w_k \cdot v_k$ factor trivially proportional to $v_k$? This would be consistent with your theory (Proposition 4 interpretation) but should be explicitly discussed as a favorable special case rather than general evidence.

## Recommendation

**Minor Revision.**

This is a methodologically strong paper with a clean and correct theoretical framework, a well-designed computational pipeline, and comprehensive experiments. The IFT-based derivation of $S_c$ is the right tool for this problem, the constrained projection is the key insight that makes first-order analysis tight under realistic perturbation constraints, and the experimental coverage (9 datasets, 7 architectures, 10 seeds, structured baselines, adaptive attackers, cross-domain validation) exceeds the standard for this venue.

The weaknesses are real but addressable:
- Weaknesses 1--3 require clarifying edge cases in the proofs (directional dependence in Thm. 1(b), $L_J$ derivation in Prop. 4, ReLU boundary treatment) -- these are precision issues, not correctness issues.
- Weakness 5 (radius computation ambiguity) requires a one-paragraph clarification.
- Weaknesses 6--8 (subgraph limitations, error analysis, seed count) are presentation and completeness issues.
- Weakness 4 (explicit GNN diagnostics) is a genuine extension that would strengthen the paper but is not strictly required.

None of the weaknesses undermine the core claims. The paper advances the state of the art in GNN vulnerability analysis by providing a principled, theoretically grounded, and practically scalable framework. It correctly scopes its formal guarantees (implicit models only) while demonstrating broad practical applicability (7 architectures). The honest treatment of limitations (conservativeness of $\varepsilon_{\mathrm{crit}}$, GCN-2 failure, Amazon Photo negative $\tau$, subgraph-vs-full-graph discrepancy) reflects scientific maturity.

I recommend acceptance contingent on addressing the proof clarifications (Weaknesses 1--3, 5) in a minor revision.
