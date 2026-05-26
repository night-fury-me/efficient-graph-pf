# Review Report: AEGIS -- Mining Graph Structure for Adversarial Vulnerability Analysis of GNNs

**Reviewer 2 (Domain Expert -- GNN Architecture, Implicit/Equilibrium Models, Graph Robustness)**
**Venue**: IEEE ICDM
**Date**: 2026-05-26

---

## 1. Summary (Domain Contribution and Positioning)

AEGIS introduces a structural sensitivity framework for GNN vulnerability analysis, centered on a constrained sensitivity matrix $S_c$ that maps edge perturbations to hidden-state shifts. The paper positions this as a "third pillar" between adversarial attack and certified defense: rather than finding attacks or providing certificates, it produces a vulnerability map. The core theoretical machinery applies the implicit function theorem (IFT) to contractive implicit GNNs (IGNN-class), yielding a critical perturbation budget $\varepsilon_{\text{crit}}$ and a three-regime characterization (subcritical/critical/supercritical). A constrained projection ($N^2 \to |E|$) enforces symmetric, edge-only perturbations, and a matrix-free Neumann-series/randomized-SVD pipeline scales the computation to ~2,700 nodes. Proposition 4 extends $S_c$ to arbitrary $K$-layer explicit GNNs (GCN, GIN, GAT, SAGE, APPNP), though without the convergence guarantees. Experiments span 7 architectures, 9 datasets (5 citation/co-purchase + 4 IEEE power grids), and a power-flow case study where AEGIS recovers N-1 contingency rankings (P@10 = 0.66--0.81). The paper is ambitious in scope, combining equilibrium model theory, scalable computation, and cross-domain application.

---

## 2. Strengths

**S1. Genuine gap identification.** The paper correctly identifies that existing attack methods (Nettack, Mettack) find damaging perturbations without optimality guarantees or per-node differentiation, while certified defenses (randomized smoothing, IBP) provide uniform certificates without identifying *which* edges are vulnerable. The "vulnerability map" framing fills a real diagnostic need (Sec. I, para. 2). This is not merely rebranding attack/defense -- the $S_c$ object produces qualitatively different outputs (per-edge rankings, per-node radii, optimal direction simultaneously).

**S2. Strong constrained-perturbation formulation.** The $N^2 \to |E|$ projection via $S_c$ (Sec. IV, Eq. for $[S_c]_{:,k}$) is the paper's most practically important contribution. Unconstrained sensitivity analysis in $\mathbb{R}^{N^2}$ produces unrealistically loose bounds; constraining to symmetric, edge-only perturbations yields tightness $1.00 \pm 0.01$ at $\varepsilon = 0.01$ (Table I). This is a meaningful advance over unconstrained IFT-based sensitivity, which the implicit-model literature (El Ghaoui et al., Revay et al.) has explored only for input perturbations.

**S3. Breadth of architectural coverage.** Testing $S_c$ on 7 architectures (Table V) is unusually thorough for a vulnerability paper. The fact that tightness holds across IGNN, GCN-2/4, GIN-2, GAT, SAGE, and APPNP -- with attack advantages of 1.9--7.6x -- demonstrates that the framework is not IGNN-specific. The explicit-GNN extension (Proposition 4) is cleanly stated and empirically validated.

**S4. Honest adaptive-attack evaluation.** The paper commendably implements a white-box PGD attacker using the same IFT gradients as AEGIS (Sec. V-D, Table III), rather than relying solely on the Mettack comparison (which the authors correctly flag as reflecting surrogate mismatch). The 0% breach rate at $\varepsilon = 0.01$ and < 1% at $\varepsilon = 0.10$ provides genuine evidence that first-order radii are meaningful, not just artifacts of weak baselines.

**S5. Cross-domain case study with quantitative transfer.** The power-flow case study (Sec. VI) goes beyond a toy demonstration: P@10 = 0.66--0.81 on IEEE standard test cases, with comparison against LODF (the industry-standard DC approximation). The observation that AEGIS outperforms LODF on larger grids ($\tau = 0.62$--$0.67$ vs. $0.44$--$0.58$ on case57/118) by capturing AC nonlinearity is a concrete domain contribution.

**S6. Principled scalability analysis.** The matrix-free pipeline (Neumann + randomized SVD) is well-engineered: Table IV shows concrete wall-clock times and memory, with the dense-to-matrix-free crossover clearly identified at $N \approx 200$. The 78s / 1 GB figure for full Cora is practically relevant.

**S7. Thoughtful limitations section.** The conclusion explicitly acknowledges the IGNN accuracy gap (77.5% vs. ~82% APPNP), continuous vs. discrete perturbation mismatch, GAT modification requirement, and the screening-layer caveat for power flow. This level of self-awareness is unusual and strengthens credibility.

---

## 3. Weaknesses

**W1. The "vulnerability analysis" framing, while distinct, overstates novelty relative to Jacobian-based sensitivity analysis.**
The core mathematical operation -- computing $\partial Z / \partial A$ via $(I - J_z)^{-1} J_A$ and extracting the leading SVD -- is a standard application of the IFT combined with principal sensitivity analysis from matrix perturbation theory (Stewart 1990, Trefethen & Embree 2005). The constrained projection $S_c$ is the genuinely new piece, but the framing suggests a more radical departure from existing sensitivity analysis than is warranted. The paper would benefit from a paragraph in Sec. III or IV explicitly comparing $S_c$ to existing Jacobian-based sensitivity analyses in dynamical systems (e.g., Chen et al. 2018, Neural ODE sensitivity; Bai et al. 2021, Jacobian regularization for DEQs) and stating precisely what is new (the constrained projection + SVD-optimal attack extraction) vs. what is standard (IFT application, Neumann series, resolvent bounds).
*Suggested fix:* Add a "relationship to existing sensitivity analysis" paragraph in Sec. III, clearly delineating the novel constrained projection from standard IFT machinery.

**W2. IGNN accuracy gap undermines the implicit-model theory as a practical contribution.**
IGNN achieves 77.5% on Cora (Table I) vs. 82.2% for APPNP (Table V). The formal guarantees ($\varepsilon_{\text{crit}}$, three regimes) are IGNN-specific, but the model is 5 points behind the best explicit baseline tested. The paper argues this is an "accuracy-guarantee tradeoff inherent in contractive architectures" (Sec. VII), but this raises the question: who would deploy IGNN when APPNP gives better accuracy *and* still benefits from $S_c$ rankings? The theoretical contribution (Theorem 1) is elegant but applies to a model class that practitioners may not choose. EIGNN (Liu et al. 2021) achieves better accuracy among implicit models but is not tested.
*Suggested fix:* (a) Test EIGNN to show whether implicit-model accuracy can be improved while retaining contractivity. (b) Alternatively, frame Theorem 1 as providing *understanding* of vulnerability phase transitions rather than *practical deployment guidance*, and emphasize Proposition 4 as the practically relevant result.

**W3. Continuous perturbation model limits practical attack relevance.**
The threat model (Sec. II-B) restricts perturbations to continuous edge-weight modifications of existing edges only. Real graph attacks (Nettack, Mettack) add or remove edges discretely. The paper acknowledges this (Conclusion, limitation 2) and shows positive $\tau$ for continuous-to-discrete transfer (Table V, $\tau = +0.22$ to $+0.54$), but these correlations are moderate -- a $\tau$ of $+0.32$ means the ranking is better than chance but far from reliable. The GCN-2 case ($\tau = -0.04$) shows the transfer can fail entirely for shallow models. The paper should be more explicit about *when* this transfer breaks down and *why* (is it depth? degree distribution? spectral gap?).
*Suggested fix:* Add a short analysis of what graph/model properties predict good continuous-to-discrete transfer. At minimum, correlate $\tau$ with model depth, graph density, and spectral gap across the 7 architectures.

**W4. Missing comparison with GNNExplainer/PGExplainer for structural importance ranking.**
The paper compares against gradient attribution (Sec. V-G, one paragraph) and shows negative $\tau$, but does not compare against GNNExplainer or PGExplainer, which are the standard methods for identifying important edges. These methods optimize for different objectives (prediction fidelity vs. adversarial vulnerability), but the paper claims AEGIS produces "per-edge vulnerability rankings" -- a reader would naturally ask how these compare to explanation-based edge importance. Ying et al. (2019) and Luo et al. (2020, PGExplainer) should be compared quantitatively, at least on the $\tau$ metric against brute-force edge removal.
*Suggested fix:* Add GNNExplainer and PGExplainer to Table V as baselines for the $\tau$ column. Even if they optimize for fidelity rather than vulnerability, showing the comparison clarifies AEGIS's positioning.

**W5. The Mettack comparison (Sec. V-A, "Surrogate baseline") is methodologically problematic.**
The paper reports "3--10x more damage" and "149/150 wins" but then immediately notes this "largely reflects surrogate-to-IGNN architectural mismatch." This is correct, but the comparison should not be featured prominently if the authors themselves consider it uninformative. A GCN surrogate attacking an IGNN is expected to perform poorly. The adaptive attack (Sec. V-D) is the meaningful comparison, and the Mettack result should be downweighted or moved to a supplementary section.
*Suggested fix:* Either (a) run Mettack with an IGNN surrogate (if feasible) or (b) reduce the Mettack comparison to a single sentence and make the adaptive attack the primary comparison.

**W6. Limited analysis of non-normal Jacobians and pseudospectral effects.**
The paper introduces the pseudospectral index $\eta$ (Sec. II-A) and reports $\eta = 1.02$--$1.28$ across datasets (Sec. V-F), but does not explore what happens when $\eta$ is large. For highly non-normal $J_z$ (common in deep or poorly conditioned networks), the operator-norm bound $1/(1-\kappa)$ can be much tighter than $1/(1-\rho)$, but also much more conservative. The paper claims this is "mild non-normality" but does not test on graphs or architectures where non-normality is expected to be severe (e.g., directed graphs, heterogeneous graphs, deep weight-tied models with $K > 50$).
*Suggested fix:* Either (a) construct a synthetic experiment with controlled $\eta$ values (e.g., by varying $K$ or using non-symmetric $W$) or (b) add a discussion paragraph explaining what range of $\eta$ the framework can tolerate and what would happen at $\eta > 2$.

**W7. Power flow model quality raises transferability concerns.**
The ContractiveGCN-PF achieves $\Delta S = 0.106$ p.u. on case14 (Table VI), meaning power-balance violations of ~10%. While the paper notes this is without a physics-informed loss, a 10% Kirchhoff violation means the model's internal representation only loosely corresponds to physical power flow. The N-1 correlation ($\tau = 0.42$ on case14) is the weakest of the four cases -- precisely where model quality is worst. This suggests that AEGIS's N-1 recovery quality is bounded by model quality, which is not surprising but should be analyzed more carefully.
*Suggested fix:* Plot $\tau$ vs. $\Delta S$ across the four cases and seeds to quantify the model-quality-to-ranking-quality relationship. If the correlation is strong, state it as a diagnostic: "AEGIS N-1 quality is upper-bounded by model fidelity."

---

## 4. Literature Assessment

### 4.1 Key Related Works: Coverage

The paper cites the core adversarial GNN literature competently:
- **Attack**: Nettack, Mettack, Dai et al. RL, Bojchevski & Gunnemann spectral, Xu et al. bilevel -- **adequate**.
- **Defense**: GNNGuard, Pro-GNN, RobustGCN, randomized smoothing (Bojchevski, Scholten), IBP (Zugner, Bojchevski), AGNNCert -- **adequate**.
- **Implicit models**: DEQ (Bai 2019/2020), IGNN (Gu 2020), EIGNN (Liu 2021), monotone operators (Winston & Kolter 2020), El Ghaoui et al. well-posedness -- **adequate**.

### 4.2 Missing References

- **GNNExplainer** (Ying et al., 2019) is cited but not compared quantitatively. **PGExplainer** (Luo et al., 2020) is not cited at all -- a significant omission for a paper claiming per-edge importance rankings.
- **SubgraphX** (Yuan et al., 2021) -- subgraph-level explanation, relevant to the BFS subgraph extraction methodology.
- **GRAND/GRAND++** (Chamberlain et al., 2021; Thorpe et al., 2022) -- continuous-depth GNNs via neural ODEs, directly relevant to the implicit/equilibrium discussion. The paper treats DEQ and IGNN but ignores the ODE-based equilibrium perspective.
- **Geisler et al. (2021), "Robustness of GNNs at Scale"** -- cited for deterministic certificates (AGNNCert is cited as Li et al. 2025) but the earlier Geisler et al. work on scalable robustness certificates is also relevant.
- **Entezari et al. (2020), "All You Need Is Low (Rank)"** -- low-rank perturbation defense, relevant to SVD-based attack analysis.
- **Mujkanovic et al. (2022), "Are Defenses for GNNs Robust?"** -- meta-analysis of GNN defense evaluation, relevant to the adaptive attack discussion.

### 4.3 Positioning Against Existing Methods

**Nettack/Mettack positioning**: Accurate. The paper correctly states these use surrogate gradients without optimality guarantees and do not produce per-node differentiation. The Mettack comparison is appropriately caveated as reflecting surrogate mismatch.

**Randomized smoothing/IBP positioning**: Mostly accurate but slightly unfair. The paper claims these provide "uniform" certificates with "zero per-node differentiation" (Sec. V-B). This is true for smoothing at low $\sigma$ (where all samples predict correctly), but at higher $\sigma$ values, smoothing certificates *do* differentiate between nodes (some nodes abstain while others certify). The comparison at $\sigma \leq 0.10$ is valid but selective. IBP-based methods (Zugner 2019, Bojchevski 2019) also produce per-node certificates by construction, not uniform ones. The distinction should be sharpened: AEGIS provides per-*edge* differentiation (which edges matter), while certified defenses provide per-*node* certificates (which nodes are safe). These are complementary, not competing.

**GNNExplainer/PGExplainer positioning**: The paper's claim that explanation methods "optimize for prediction fidelity, not adversarial vulnerability" (Sec. III, Sec. V-G) is correct but requires quantitative support. A negative $\tau$ for gradient attribution is shown, but GNNExplainer and PGExplainer use more sophisticated optimization and may perform better. The distinction between "important for prediction" and "vulnerable to perturbation" is conceptually valid but empirically unsubstantiated without direct comparison.

**Implicit model literature**: Well-situated overall. The paper correctly distinguishes input sensitivity ($\partial Z^*/\partial x$, studied by El Ghaoui et al., Revay et al.) from structural sensitivity ($\partial Z^*/\partial A$, AEGIS's contribution). However, the connection to Jacobian regularization (Bai et al. 2021) is underexplored: if $\|J_z\|$ is regularized during training, this directly affects $\varepsilon_{\text{crit}} = (1-\kappa)/\|W\|_2$. The paper should discuss whether AEGIS analysis can inform the regularization strength.

---

## 5. Theoretical Contribution Assessment

### 5.1 Genuinely Novel vs. Standard Application

| Component | Assessment |
|---|---|
| IFT applied to $Z^* = F(Z^*, A)$ to obtain $\partial Z^*/\partial A$ | **Standard**. Direct application of IFT to implicit models, done by Gould et al. 2021, Bai et al. 2019 for other parameters. |
| Neumann series for $(I - J_z)^{-1}$ | **Standard**. Textbook technique, used in DEQ training (Bai et al. 2019). |
| Resolvent norm bound $1/(1-\kappa)$ | **Standard**. Direct consequence of Neumann series convergence. |
| SVD of sensitivity matrix for optimal attack direction | **Standard** in matrix perturbation theory (Stewart 1990). Novel *application* to GNN vulnerability. |
| Randomized SVD for scalability | **Standard** technique (Halko et al. 2011). Good engineering but not a contribution. |
| **Constrained projection $S_c$ ($N^2 \to |E|$)** | **Novel and significant.** This is the paper's key contribution. Projecting unconstrained sensitivity onto the feasible perturbation set (symmetric, edge-only) and showing this yields tightness $\approx 1.00$ is non-trivial and practically impactful. |
| **Three-regime characterization (Theorem 1)** | **Moderately novel.** The subcritical regime (part a) follows from standard IFT + Neumann. The critical divergence (part b) is a direct consequence of $\|J_z'\| \to 1$. The supercritical loss of guarantees (part c) is definitional. The *packaging* as a three-regime vulnerability characterization is new; the individual bounds are not. |
| **Proposition 4 (explicit GNN extension)** | **Moderately novel.** The unrolled sensitivity $S_K$ via chain rule is straightforward, but stating it cleanly and validating across 6 architectures is valuable. |
| **Per-node sensitivity radius $r_v$ (Proposition 3)** | **Incremental.** Dividing margin by sensitivity norm is standard in robustness analysis. The contribution is applying it to structural (not input) perturbation. |

### 5.2 Significance of $S_c$ Construction

The $S_c$ construction is genuinely significant because:
1. It reduces the perturbation space dimensionality from $O(N^2)$ to $O(|E|)$, which is the difference between vacuous and tight first-order bounds.
2. It enforces symmetry by construction ($\delta A_{ij} = \delta A_{ji}$), which is mandatory for undirected graphs but ignored by unconstrained analysis.
3. It enables meaningful comparison against discrete perturbation outcomes (positive $\tau$ across 6/7 architectures in Table V).

The construction itself is simple (summing two columns of $S$ per edge), but its impact on tightness is dramatic. This is a case where a conceptually simple modification has outsized practical value.

### 5.3 Is "Structural Sensitivity" ($\partial Z/\partial A$ vs. $\partial Z/\partial X$) a Meaningful Distinction?

**Yes, with caveats.** The paper correctly argues (Sec. II-A, final paragraph) that structural perturbation changes the operator itself (through both $J_z$ and $J_A$), while input perturbation does not. For implicit models, this means structural perturbation can push the system through the critical threshold $\varepsilon_{\text{crit}}$, potentially destroying the fixed point -- something input perturbation cannot do (the contraction constant $\kappa$ depends on $A$ and $W$, not on $X$). This is a qualitative difference that justifies separate treatment.

The caveat is that for explicit GNNs, the $\partial Z/\partial A$ vs. $\partial Z/\partial X$ distinction is less sharp: both are standard Jacobians of a finite computation graph, and neither involves fixed-point dynamics. The "structural sensitivity" framing is most compelling for the IGNN case and somewhat less so for explicit architectures, where it reduces to "we compute a different Jacobian."

---

## 6. Questions for Authors

**Q1. Continuous-to-discrete transfer failure modes.** GCN-2 achieves $\tau = -0.04$ (Table V), meaning the continuous $S_c$ ranking is essentially uncorrelated with discrete edge-removal impact. What explains this? Is it purely a depth issue (2 layers provide insufficient sensitivity differentiation), or does the ReLU activation pattern change fundamentally between continuous perturbation and full edge removal? Can you predict when $\tau$ will be negative without running the brute-force baseline?

**Q2. Why not use EIGNN?** EIGNN (Liu et al. 2021) achieves higher accuracy than IGNN on several benchmarks while maintaining implicit structure. It satisfies the contractivity assumption differently (via eigenvalue decomposition rather than spectral normalization). Can AEGIS be applied to EIGNN? If so, does $\varepsilon_{\text{crit}}$ and Theorem 1 still hold, and what accuracy improvement results?

**Q3. Jacobian regularization interaction.** Bai et al. (2021) regularize $\|J_z\|$ during DEQ training to stabilize convergence. Since $\varepsilon_{\text{crit}} = (1-\kappa)/\|W\|_2$ depends directly on $\kappa = \|J_z\|_2$, does Jacobian regularization strength during training predictably affect AEGIS vulnerability scores post-training? Could AEGIS be used *during* training to set the regularization target?

**Q4. Feature perturbation interaction.** The threat model restricts perturbation to structure only. In practice, an attacker may perturb both $A$ and $X$ simultaneously. Can $S_c$ be extended to a joint structural-feature sensitivity matrix $S_{c,X}$? Would the constrained projection still yield tight bounds, or does the joint perturbation space reintroduce looseness?

**Q5. Defense-informed protection saturation.** Section V-H shows that masking top-5 edges reduces attack damage by 42%. Does this saturate? What happens at top-20, top-50? Is there a principled way to determine the minimum number of edges to protect for a target vulnerability reduction?

---

## 7. Minor Issues

- **Sec. II-A**: The pseudospectral index $\eta$ is defined but never used quantitatively in bounds. Either integrate it into the theory (e.g., tighter bounds using $\eta$) or move it to the experiments section where it is reported empirically.

- **Table V, GAT$^\dagger$**: The edge-weighted GAT modification is described in a text paragraph (Sec. V-G) but not formally defined. Since this is a non-standard variant, a 2-line equation would clarify exactly what $\Ahat_{ij}$ multiplies.

- **Sec. V-A**: "149/150 wins across $k=1,\ldots,5$ edge removals, 3 datasets, 10 seeds" -- this is $5 \times 3 \times 10 = 150$ trials. Stating it as "149/150" makes it seem like a single exceptional result; clarify the decomposition (which dataset/seed/k lost?).

- **Eq. (6) and Eq. (8)**: The paper switches between $S$ (unconstrained) and $S_c$ (constrained) without always being explicit. In Proposition 3 (Eq. 8), $S_v$ refers to block-rows of $S$ (unconstrained), but the text says AEGIS uses $S_c$. Clarify whether $r_v$ uses $S$ or $S_c$ in practice, and how this affects the radius.

- **Sec. VI-C**: "Binary adjacency outperforms admittance-weighted (P@10 = 0.81 vs. 0.27)" -- this is a surprising and strong result that deserves more discussion. Why would ignoring line impedances *improve* N-1 ranking? Is it because N-1 is a topological event (line exists or not) rather than a weighted one?

- **Reproducibility**: The paper states "Code will be released upon publication" but does not provide a supplementary implementation or anonymized repository for review. This is standard but limits verification.

- **Notation**: $\sigma$ is used for both the activation function (Eq. 2) and singular values ($\sigma_1(S)$ in Eq. 4). Consider $\phi$ for the activation to avoid ambiguity.

---

## 8. Scores

| Dimension | Score (0--100) | Justification |
|---|---|---|
| **Literature Coverage** | 72 | Core attack/defense/implicit literature is well-covered. Missing PGExplainer, GRAND/GRAND++, Entezari et al. GNNExplainer cited but not compared quantitatively. IBP positioning slightly unfair. |
| **Theoretical Novelty** | 58 | $S_c$ construction is genuinely novel and impactful. Theorem 1 is a clean packaging of standard results. IFT, Neumann, SVD components are all standard. The novelty is in the *combination and constrained projection*, not individual pieces. |
| **Domain Contribution** | 74 | Fills a real gap between attack and defense. Seven-architecture validation is thorough. Power-flow case study is a genuine cross-domain contribution. IGNN accuracy gap limits practical impact of the implicit-model theory. |
| **Positioning Accuracy** | 70 | Attack/defense positioning is accurate. Smoothing/IBP comparison is slightly selective. GNNExplainer comparison is insufficiently substantiated. Implicit-model positioning is good but misses GRAND and Jacobian regularization interaction. |

### Overall Recommendation: **Weak Accept**

**Rationale**: AEGIS makes a genuine contribution to the GNN robustness literature through the constrained sensitivity matrix $S_c$, which is a simple but effective projection that transforms loose unconstrained bounds into tight practical predictions. The seven-architecture validation and power-flow case study demonstrate breadth uncommon in adversarial GNN papers. The adaptive attack evaluation (Sec. V-D) is methodologically sound and builds confidence. However, the theoretical novelty is moderate (the individual techniques are standard; the contribution is their combination), the IGNN accuracy gap limits the practical value of the implicit-model theory, and the missing GNNExplainer/PGExplainer comparison leaves a gap in positioning. With the suggested fixes -- particularly W2 (EIGNN test or reframing), W4 (explainer comparison), and W1 (explicit novelty delineation) -- the paper would clear the acceptance threshold comfortably. In its current form, it is a solid applied contribution with moderate theoretical novelty, appropriate for ICDM.
