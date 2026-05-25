# Methodology Review Report -- AEGIS

## Reviewer Profile
- **Role**: Peer Reviewer 1 (Methodology)
- **Expertise**: Certified robustness, perturbation theory, mathematical ML
- **Review Focus**: Research design, statistical validity, reproducibility

## Summary Assessment

AEGIS presents a structurally-grounded sensitivity analysis framework for GNNs, built on the implicit function theorem applied to the equilibrium equation of contractive implicit GNNs, then generalized to explicit architectures via unrolled Jacobian computation. The central mathematical contribution -- the constrained sensitivity matrix S_c that projects the full N^2-dimensional perturbation space onto |E| realistic (symmetric, edge-only) perturbations -- is sound and well-motivated. The paper's theoretical core (Theorem 1, Propositions 1-3) is correctly derived under the stated assumptions, and the constrained-vs-unconstrained distinction (tightness ~1.00 vs ~0.31) is the paper's most important methodological insight, demonstrating that unconstrained first-order analysis is essentially useless for graph perturbation while constrained analysis is quantitatively accurate.

However, the paper has several methodological concerns that require attention. The proof of Theorem 1 Part (a) conflates the Frobenius norm and operator norm of the perturbation matrix in a way that yields a loose bound on contractivity preservation. The "28% optimism" claim regarding kappa vs rho is empirically quantified but not formally bounded. The tightness = 1.00 at epsilon = 0.01 result, while correctly contextualized by the authors as "mathematically expected," still occupies disproportionate emphasis relative to its informativeness. The scalability ceiling of N ~ 300 is a practical limitation that the paper acknowledges but does not adequately address for the paper's stated motivation of "safety-critical deployment." Statistical reporting is generally strong (10 seeds, standard deviations throughout), though some experiments (Mettack comparison in Table VII) report only a single seed.

The experimental design is thorough in breadth (7 architectures, 9 datasets, 4 domains) and appropriately self-critical. The authors correctly flag the Mettack comparison as reflecting architectural mismatch and introduce the adaptive PGD attacker as a fairer baseline -- a commendable methodological decision. The power flow case study is a compelling cross-domain validation, though the operational caveats are correctly noted.

## Scores (0-100)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Methodological Rigor | 74 | IFT derivation is correct; norm conflation in Thm 1(a) proof is a real but non-fatal gap; assumptions are reasonable but A3 is stated as a condition on the fixed point yet checked only post-training |
| Statistical Validity | 82 | 10-seed experiments with std devs throughout; effect sizes reported; a few single-seed results (Table VII) lower the score |
| Reproducibility | 78 | Seeds, hyperparameters, and architecture details are provided; code promised but not yet available; Mettack baseline implementation details are important but relegated to appendix |
| Experimental Design | 80 | Breadth across architectures and domains is excellent; Mettack mismatch is honestly flagged; adaptive attacker is the right control; subgraph ablation is well-executed |
| Technical Soundness | 76 | Core theory is correct; the kappa-vs-rho gap is handled transparently; the first-order-only nature of all guarantees is properly caveated; the GAT modification is a workaround rather than a general solution |
| Overall | 77 | A solid contribution with genuine methodological insight (constrained S_c) but with identifiable gaps in proof tightness, scalability, and the strength of theoretical guarantees |

## Strengths

1. **The constrained sensitivity matrix S_c is the right abstraction.** The projection from N^2 to |E| dimensions (Eq. 7 in Sec. IV-C) enforces symmetry and edge-only support by construction, and the empirical tightness gap (1.00 constrained vs 0.31 unconstrained, Appendix Table) demonstrates that this is not merely convenient but necessary. This is the paper's most important methodological contribution.

2. **Honest treatment of the Mettack baseline.** The authors explicitly state (Sec. V-A, paragraph following Table II) that "this gap largely reflects surrogate-to-IGNN architectural mismatch rather than AEGIS's analytical superiority alone" and introduce the adaptive PGD attacker (Sec. V-C) as the fair comparison. This is methodologically commendable and rare in adversarial ML papers.

3. **Comprehensive statistical protocol.** Ten fixed seeds (42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999) reported upfront, standard deviations on all aggregated metrics, and multiple ablations (subgraph size, hidden dimension, spectral norm bound). The hyperparameter sensitivity analysis (Sec. V-E) showing tightness stability across d in {16, 32, 64, 128} is informative.

4. **Transparent limitation disclosure.** Five explicit limitations in the conclusion, the "operational caveat" in the power flow section, the remark on conservativeness after Theorem 1, and the clear distinction between "local sensitivity measures" and "global robustness certificates" (end of Sec. III, after Proposition 3). This level of self-assessment strengthens the paper.

5. **Proposition 4 bound tightness analysis.** Reporting the ratio of the norm-product upper bound to the actual sigma_1(S_K) across architectures (1.4x to 5.9x, Sec. V-G) provides useful calibration for practitioners and honestly shows where the bounds are loose.

6. **Phase transition experiment confirms theoretical prediction.** The 83x amplification of fixed-point shift when scaling rho from 0.3 to 0.99 (Sec. V-D) is consistent with the 1/(1-kappa) factor from Theorem 1(a), providing empirical support for the regime characterization.

## Weaknesses

1. **Norm conflation in Theorem 1(a) proof.** The proof states "||J_z'||_2 <= (||A||_2 + ||delta A||_F) * ||W||_2 (using ||.||_2 <= ||.||_F for the perturbation)" (Sec. III, proof of Part a). This is valid as an inequality but introduces unnecessary looseness: the Frobenius norm of a rank-1 perturbation can be sqrt(N) times larger than its operator norm. The contractivity preservation condition epsilon < (1-kappa)/||W||_2 is therefore more conservative than necessary. **Suggestion:** State explicitly that epsilon_crit is a Frobenius-norm sufficient condition and discuss how much tighter a spectral-norm budget would be. This matters for interpreting the practical significance of epsilon_crit values in Table I.

2. **The kappa vs rho gap is empirically bounded but not formally controlled.** The paper states the pseudospectral index eta = 1.02-1.28, meaning epsilon_crit computed with rho is "at most 28% optimistic" (Sec. V, Notation paragraph). However, eta is a post-hoc diagnostic -- there is no a priori bound on eta for IGNN-class operators. If a practitioner deploys AEGIS on a new dataset, they must compute eta themselves to know whether the rho-based epsilon_crit is safe. **Suggestion:** Either (a) prove a structural bound on eta for spectrally-normalized ReLU operators, or (b) always report the kappa-based epsilon_crit as the primary quantity and relegate rho to a diagnostic role. Currently Tables report rho but the formal bounds require kappa, which is confusing.

3. **Tightness = 1.00 at epsilon = 0.01 is over-emphasized relative to its informativeness.** The authors acknowledge this is "mathematically expected" (Sec. V-A), but the abstract, introduction, and conclusion all highlight it as a primary result. Any smooth function has tightness approaching 1.00 as the perturbation approaches zero; the informative question is the domain of validity. The tightness degradation data (Table III: 1.07 at epsilon = 0.05, 1.15 at epsilon = 0.10 on Cora) is more informative. **Suggestion:** Lead with the epsilon = 0.10 tightness (within 15%) as the headline result and frame epsilon = 0.01 as the expected baseline. This would better highlight what is genuinely surprising: that constrained first-order analysis remains accurate well beyond the infinitesimal regime.

4. **Single-seed Mettack comparison (Table VII, Appendix).** The Mettack baseline is reported for "single representative seed" while all other experiments use 10 seeds. Given that the Mettack comparison is already flagged as potentially misleading due to architectural mismatch, the single-seed reporting compounds the concern. **Suggestion:** Run 10-seed Mettack or remove the table. The adaptive attacker (Table IV, 10 seeds) is the methodologically sound comparison and should stand alone.

5. **Scalability ceiling undermines the safety-critical deployment narrative.** The paper motivates AEGIS for "financial fraud detection, drug interaction prediction, infrastructure monitoring" (abstract), but the practical limit is N ~ 300 nodes on a 24 GB GPU (Sec. V-D). Real financial transaction graphs have millions of nodes. The BFS subgraph extraction addresses this partially, but the paper does not validate whether 50-node ego-subgraphs capture meaningful vulnerability structure in graphs with >10K nodes (Pubmed has 19,717 nodes but subgraph analysis uses only 50). **Suggestion:** Add an experiment showing that vulnerability rankings from 50-node subgraphs on Pubmed or WikiCS are consistent with those from larger subgraphs (the subgraph ablation in Sec. V-E is only on Cora, N <= 200). Alternatively, temper the safety-critical framing.

6. **Phase transition data (Appendix Table IV) shows non-convergence at rho = 0.85.** The table shows "Converged = No" for rho_actual = 0.854, but epsilon_crit at this rho would be (1-0.854)/||W||_2, which for typical ||W||_2 ~ 1 is about 0.15 -- well above the epsilon = 0.01 used. This suggests the non-convergence is a property of the retrained model (higher rho destabilizes training), not of the perturbation analysis. The text says "scaling rho from 0.3 to 0.99 amplifies the shift by 83x" but does not clarify that models at rho > 0.85 did not converge. **Suggestion:** Clarify whether "converged" refers to the fixed-point iteration during analysis or during training. If training, this is a confound -- the phase transition in vulnerability is conflated with training instability.

7. **Proposition 3 (Per-Node Radius) uses the full unconstrained S_v, not the constrained S_c.** The radius formula (Eq. 5) uses ||S_v||_2, which includes sensitivity to non-edge, asymmetric perturbations. Since these perturbations are excluded by the threat model, the radius is more conservative than necessary. **Suggestion:** Define a constrained radius using the block-rows of S_c corresponding to node v, or explain why the unconstrained S_v is used despite the constrained threat model.

8. **Missing effect size reporting for the defense ablation (Sec. V-F).** The 42 +/- 8% damage reduction from masking top-5 edges vs 11 +/- 6% for random masking is reported without a formal significance test. With 10 seeds, the standard errors suggest significance, but a paired test (e.g., Wilcoxon signed-rank) would be appropriate. **Suggestion:** Add p-values or confidence intervals for the defense ablation comparison.

## Technical Issues

### Issue 1: The J_z Kronecker factorization assumes vec(Z) ordering

In the proof of Theorem 1(a), J_z is written as diag(sigma') * (A_hat tensor W). This Kronecker product form holds when Z is vectorized column-by-column (vec(Z) = [Z_1; Z_2; ...; Z_N] where Z_i is the d-dimensional representation of node i). The paper does not specify this convention. If the implementation uses row-major ordering, the Kronecker factors would be transposed. This does not affect correctness of the final result (the IFT derivation is coordinate-free), but the explicit Kronecker form in the proof should specify the vectorization convention.

### Issue 2: The shift bound (Eq. 3) mixes first-order and higher-order terms

Equation 3 states ||Delta z*||_F <= sigma_1(S) * epsilon + O(epsilon^2). The sigma_1(S) term is the unconstrained maximum sensitivity; the constrained maximum is sigma_1(S_c) <= sigma_1(S). The constrained tightness of ~1.00 means sigma_1(S_c) * epsilon matches the actual shift almost exactly, but the unconstrained bound in Eq. 3 overestimates by a factor of ~3x (Table in Appendix). The theorem statement should clarify that Eq. 3 applies to unconstrained perturbations, and the constrained version uses sigma_1(S_c).

### Issue 3: Proposition 4 bound looseness grows with depth

The paper reports bound tightness ratios of 1.4x (SAGE-2) to 5.9x (APPNP), with deeper models yielding looser bounds. The text attributes this to "direction-dependent cancellations that the norm product overestimates" (Sec. V-G). This is correct but understated: for a K-layer GNN with per-layer contraction factor lambda, the norm-product bound grows as K * lambda^K while the actual sensitivity may be O(lambda^K) if Jacobians are aligned. For deep architectures, the bound becomes vacuous. This limits the practical utility of Proposition 4 as a robustness certificate for deep explicit GNNs.

### Issue 4: Continuous perturbation model vs discrete graph attacks

The threat model (Sec. II-B) restricts to continuous edge-weight perturbations on existing edges. Real graph attacks (Nettack, Mettack) add or remove edges discretely. The paper acknowledges this (Conclusion, limitation 2) but does not quantify the gap. How well does the continuous vulnerability ranking predict discrete attack success? The Mettack comparison (Table VII) partially addresses this by comparing against discrete edge removal, but the single-seed reporting and architectural mismatch limit its informativeness.

### Issue 5: Assumption A3 is verified post-hoc, not enforced architecturally

Assumption A3 requires ||J_z||_2 <= kappa < 1 at the fixed point. The IGNN architecture enforces ||W||_2 < 1 via spectral normalization, and the bound kappa <= ||A_hat||_2 * ||W||_2 provides a sufficient condition. However, ||A_hat||_2 depends on graph structure and can exceed 1 for non-regular graphs. The paper reports kappa (via rho) in the range 0.14-0.59 (Table I), confirming A3 holds in practice, but does not discuss what happens if a deployment graph has ||A_hat||_2 * ||W||_2 >= 1. The spectral normalization constraint should be set as ||W||_2 < 1/||A_hat||_2, which is graph-dependent.

## Questions for Authors

1. **On kappa vs rho:** Have you computed kappa (operator norm of J_z) directly, rather than only rho (spectral radius)? The eta values 1.02-1.28 suggest they differ by up to 28%, but the actual kappa values are never reported in any table. Computing kappa is O(D^2) (power iteration on J_z), which is dominated by the O(D^3) linear solve in Stage 3 -- why not report it?

2. **On the constrained radius:** Why does Proposition 3 use the unconstrained S_v rather than the constrained S_c rows? The constrained radius would be strictly larger (less conservative), which seems preferable given that the threat model explicitly restricts to symmetric, edge-only perturbations.

3. **On phase transition convergence:** In Appendix Table IV, does "Converged = No" refer to the fixed-point iteration failing to converge during AEGIS analysis, or to the IGNN training failing to converge at high rho? These are very different phenomena with different implications for the phase transition claim.

4. **On subgraph representativeness:** The subgraph ablation (Sec. V-E) shows tightness is stable across N in {30, 50, 100, 200} on Cora. But Cora has only 2,708 nodes. Does the 50-node BFS subgraph capture meaningful vulnerability structure on Pubmed (19,717 nodes) or WikiCS (11,701 nodes)? What fraction of the graph's edges fall within the subgraph?

5. **On the GAT modification:** The edge-weighted GAT variant multiplies attention coefficients by A_hat_{ij}. Does this change the model's representational capacity or accuracy? Table V shows GAT-dagger at 77.8% vs standard GAT's typical ~83% on Cora -- is the 5% accuracy gap a consequence of the modification?

6. **On second-order bounds:** The paper repeatedly states that first-order radii are "not global certificates without second-order bounds" (Conclusion, limitation 1). Have you estimated the magnitude of the second-order remainder term? The tightness degradation data (Table III) suggests the remainder is ~7% at epsilon = 0.05 and ~15% at epsilon = 0.10 -- could these empirical values serve as practical corrections?

7. **On the power flow case study:** Binary adjacency outperforms admittance-weighted adjacency (P@10 = 0.81 vs 0.27, Sec. VI-C). This is counter-intuitive -- why would discarding electrical parameters improve contingency ranking? Is this an artifact of the continuous perturbation model (perturbing admittance values is physically meaningful but changes the contingency semantics)?

8. **On reproducibility:** The paper states "Code will be released upon publication." Given the importance of implementation details (sign correction for Mettack gradients, BFS subgraph extraction strategy, spectral normalization implementation), can you provide the code for review?

## Recommendation

**Minor Revision.**

The paper presents a sound and well-executed contribution. The constrained sensitivity matrix S_c is a genuine methodological advance that makes first-order structural sensitivity analysis practical and accurate for GNNs. The experimental protocol (10 seeds, 7 architectures, 9 datasets, honest baseline treatment) meets the standards expected at a top venue. The theoretical framework is correctly derived, and the limitations are transparently disclosed.

The issues identified above are addressable without fundamental changes to the paper:

- **Required fixes:** (a) Clarify the norm conflation in the Theorem 1(a) proof and state epsilon_crit as a Frobenius-norm sufficient condition. (b) Report kappa directly or explain why only rho is tabulated when the formal bounds require kappa. (c) Run 10-seed Mettack or remove Table VII. (d) Clarify "Converged" column in Appendix Table IV.

- **Recommended improvements:** (e) Use epsilon = 0.10 tightness as the headline result rather than epsilon = 0.01. (f) Discuss the constrained radius variant of Proposition 3. (g) Add subgraph representativeness validation on larger graphs. (h) Add significance tests for the defense ablation.

None of these issues undermine the core contribution. The paper's methodological honesty (particularly the Mettack disclaimer and the exhaustive limitation list) gives confidence that the authors can address these points effectively in revision.
