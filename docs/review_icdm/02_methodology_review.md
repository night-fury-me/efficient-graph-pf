# Reviewer 1 (Methodology): AEGIS -- Mining Graph Structure for Adversarial Vulnerability Analysis of GNNs

**Venue**: IEEE ICDM  
**Review type**: Methodology-focused  
**Reviewer expertise**: Adversarial certificates for GNNs, randomized smoothing, interval bound propagation, IFT, spectral perturbation theory, matrix analysis

---

## 1. Summary (Methodology Focus)

AEGIS introduces a structural sensitivity framework for GNNs built on the Implicit Function Theorem (IFT). The central object is a *constrained sensitivity matrix* $S_c \in \mathbb{R}^{Nd \times |E|}$ that projects the full $N^2$-dimensional adjacency perturbation space onto the $|E|$-dimensional space of realistic (symmetric, edge-only) perturbations. For contractive implicit GNNs (IGNN-class satisfying operator-norm contractivity $\kappa < 1$), the authors derive a critical perturbation budget $\varepsilon_{\text{crit}} = (1-\kappa)/\|W\|_2$ and a three-regime characterization (subcritical/critical/supercritical). From $S_c$, three outputs are extracted: (i) SVD-optimal attack directions, (ii) per-edge vulnerability rankings, and (iii) per-node first-order sensitivity radii. A matrix-free pipeline (Neumann-series resolvent + autograd JVPs + randomized SVD) avoids materializing the $Nd \times N^2$ matrix. The framework is generalized to explicit $K$-layer GNNs via Proposition 3 (unrolled sensitivity), though without the convergence guarantees. Experiments span 7 architectures, 9 datasets, 10 seeds, including a power grid N-1 contingency case study.

---

## 2. Strengths

**S1. Well-structured IFT derivation with explicit assumptions.**  
Theorem 1 cleanly states three assumptions (A1--A3) and derives consequences in three regimes. The proof (Section IV) correctly identifies $J_z = \text{diag}(\sigma') \cdot (\hat{A} \otimes W)$ and applies the Neumann-series bound $\|(I-J_z)^{-1}\|_2 \leq 1/(1-\kappa)$. The choice to use operator-norm $\kappa$ rather than spectral radius $\rho$ is mathematically rigorous and explicitly justified (the Neumann series requires operator-norm contractivity, not merely $\rho < 1$). The pseudospectral index $\eta = 1.02$--$1.28$ (Section V-F) empirically validates that the conservativeness cost is modest.

**S2. The constrained sensitivity matrix $S_c$ is the paper's strongest contribution.**  
The $N^2 \to |E|$ projection enforcing symmetry and edge-only perturbation (Eq. 8) is a simple but powerful insight. The paper demonstrates that unconstrained $S$ gives vacuous predictions while $S_c$ achieves tightness $1.00 \pm 0.01$ at $\varepsilon = 0.01$ (Table I). This concretely shows that threat-model-aware sensitivity is both theoretically cleaner and empirically tighter. The construction generalizes cleanly to explicit GNNs (Proposition 3).

**S3. Comprehensive statistical protocol.**  
Ten seeds (explicitly listed: 42, 137, 271, ..., 9999), with mean $\pm$ standard deviation reported throughout. This is above the community standard for ICDM papers. The tightness degradation table (Table II) systematically varies $\varepsilon \in \{0.01, 0.05, 0.10, 0.20\}$, showing how first-order accuracy degrades. Flip rates are reported alongside tightness, giving an operationally meaningful metric.

**S4. Honest treatment of the adaptive attacker (Section V-C).**  
The authors correctly identify that the Mettack comparison is confounded by surrogate-to-IGNN mismatch, and implement a PGD attacker using the *same IFT gradients* as AEGIS. This is methodologically sound: it tests whether the first-order radii hold against the strongest possible first-order attacker. The 0% breach rate at $\varepsilon = 0.01$ and $< 1\%$ at $\varepsilon = 0.10$ (Table III) is compelling.

**S5. Matrix-free pipeline is a genuine engineering contribution.**  
The Neumann-series resolvent with adaptive depth, autograd JVPs, and randomized SVD reduces memory from $>24$ GB (dense, $N = 500$) to 1.1 GB (matrix-free, $N = 2{,}708$) with $\sigma_1$ accuracy within 0.03% (Table IV). The $O(K \cdot Nd)$ cost per matvec is correctly stated and the scalability table provides concrete wall-clock numbers.

**S6. Per-node sensitivity radii are conceptually novel in the GNN robustness space.**  
Unlike randomized smoothing (uniform radii at small $\sigma$, Section V-B) or IBP (architecture-specific), AEGIS radii are structurally differentiated: dense-region nodes get $r_v \approx 0.09$, boundary nodes $r_v \approx 0.01$. This is exactly what a practitioner would want for identifying at-risk nodes before deployment.

**S7. Continuous-to-discrete transfer validation (Table V, $\tau$ column).**  
Reporting Kendall $\tau$ between continuous $S_c$ scores and brute-force discrete edge-removal ground truth is an excellent experimental design choice. Positive $\tau$ across 6/7 architectures (+0.22 to +0.54) demonstrates practical utility despite the perturbation-model mismatch.

**S8. Extensive ablations and diagnostic reporting.**  
Subgraph size ablation (Section V-E), hyperparameter sensitivity ($d$, $c$), convergence diagnostics ($\eta$), defense-informed edge protection (Section V-G), degree-weighted $S_c$ (negative result), and Proposition 3 bound tightness ratios (1.4x--5.9x) collectively demonstrate scientific thoroughness.

---

## 3. Weaknesses

**W1. The perturbation model (continuous edge weights on existing edges) is restrictive and under-justified.**  
The threat model (Eq. 4) perturbs $\hat{A}$ continuously while forbidding new edges. Real-world graph attacks overwhelmingly involve discrete edge insertions/deletions -- Nettack, Mettack, and virtually all practical attacks operate in the discrete domain. The paper acknowledges this (Conclusion limitation 2) and provides the $\tau$ transfer evidence, but does not formally analyze the gap between continuous first-order sensitivity and discrete perturbation impact. The $\tau = -0.04$ for GCN-2 (Table V) shows the transfer can fail.

*Why it matters*: If AEGIS's primary claim is "pre-deployment diagnostic," practitioners face discrete attacks, not continuous edge-weight modulation. The continuous model is mathematically convenient but may not capture the actual threat.

*Suggested fix*: Add a formal analysis or bound on the continuous-to-discrete gap, even if approximate. At minimum, report $\tau$ values for all datasets (not just Cora) for the discrete transfer evaluation, and characterize when the transfer fails (e.g., shallow models, low-degree graphs).

**W2. First-order sensitivity radii lack second-order error bounds, making them informational rather than certifiable.**  
Proposition 2 (Eq. 9) gives $r_v = m_v / (\|W_{y_v} - W_{c^*}\|_2 \cdot \|S_v\|_2)$, but the $O(\varepsilon^2)$ remainder in Eq. 5 is never bounded. The paper calls these "local sensitivity guarantees, not global certificates" (Section IV, after Proposition 2), which is honest, but without a second-order bound the radius could be arbitrarily wrong at any finite $\varepsilon$. The 0% breach rate at $\varepsilon = 0.01$ (Table III) is empirical, not a formal guarantee.

*Why it matters*: The paper positions AEGIS alongside randomized smoothing (Section V-B), which provides probabilistic certificates with explicit confidence levels. AEGIS radii have no error bar on the approximation quality.

*Suggested fix*: Derive a second-order correction bound using the Hessian of the equilibrium map. Even a loose bound of the form $|\Delta z^* - S \cdot \text{vec}(\delta A)| \leq C \cdot \varepsilon^2$ with a computable constant $C$ would transform the radii into rigorous local certificates. Alternatively, compute the Hessian-vector product and report the second-order correction magnitude empirically across all datasets to quantify the neglected term.

**W3. Theorem 1's critical budget $\varepsilon_{\text{crit}}$ is conservative and potentially vacuous for practical perturbation sizes.**  
The critical budget $\varepsilon_{\text{crit}} = (1-\kappa)/\|W\|_2$ uses worst-case norm bounds. From Table I: $\varepsilon_{\text{crit}} = 0.41$ (Citeseer) to 0.86 (Amazon). These are Frobenius-norm budgets on $\delta\hat{A}$, which is very large -- a budget of 0.41 on a 50-node subgraph means average per-edge perturbation of $0.41/\sqrt{|E|} \approx 0.04$. The practical regime of interest ($\varepsilon = 0.01$--$0.10$) is always deep in the subcritical zone, so the phase transition never actually manifests in experiments. The "critical" and "supercritical" regimes are never empirically validated.

*Why it matters*: The three-regime characterization is the main theoretical novelty, but it is never observed in practice. The critical regime is an artifact of worst-case norm bounds.

*Suggested fix*: (i) Empirically demonstrate the phase transition by artificially increasing $\kappa$ toward 1 (e.g., relaxing spectral normalization) and showing divergence near $\varepsilon_{\text{crit}}$. The paper mentions "$83\times$ amplification" at $\kappa = 0.99$ (Section V-D) but does not show the full transition curve. (ii) Report per-direction critical budgets using the SVD structure of $S_c$ rather than worst-case norms.

**W4. The random perturbation baseline is too weak to demonstrate attack advantage convincingly.**  
The attack advantage metric (AtkAdv = AEGIS/random damage, Table I: 3.2x--4.1x) uses *random perturbation* as the denominator. In a high-dimensional perturbation space ($|E|$ dimensions), any structured direction will massively outperform random by concentration of measure. This is not surprising and does not demonstrate that the SVD direction is meaningfully better than other structured attacks (e.g., degree-based heuristics, gradient-based edge scoring without IFT, or spectral methods).

*Why it matters*: The claim "SVD-optimal attack inflicts 2--8x more damage than random" (abstract) sounds impressive but is mathematically expected in high dimensions. The actual surprise would be *how close* AEGIS gets to the true (nonlinear) optimum.

*Suggested fix*: Add comparisons against: (i) top-eigenvalue perturbation of $A$ (spectral heuristic), (ii) degree-proportional perturbation, (iii) gradient-based edge scoring without the IFT resolvent (just $J_A$ alone). Report the ratio of AEGIS damage to PGD damage (already in Table III: 1.2x--2.2x), which is the more meaningful metric.

**W5. Per-edge vulnerability column norms may be dominated by degree effects, limiting diagnostic value.**  
The vulnerability score $v_{ij} = \|[S_c]_{:,k}\|_2$ is a column norm of the sensitivity matrix. In message-passing GNNs, high-degree nodes participate in more message-passing paths, so their incident edges will naturally have larger sensitivity simply due to aggregation fan-in, not due to structural vulnerability per se. The paper tests degree-weighted $S_c$ (Section V-H) and finds no improvement, but this only shows that *additional* degree weighting is redundant -- it does not rule out that degree already dominates the raw scores.

*Why it matters*: If vulnerability rankings are primarily degree rankings, the diagnostic adds little beyond degree centrality, which is trivially computable.

*Suggested fix*: Report Kendall $\tau$ between $v_{ij}$ and endpoint degree across all datasets. If strongly correlated, consider degree-normalized vulnerability scores $v_{ij} / f(d_i, d_j)$ and show that the residual still predicts discrete removal impact.

**W6. Proposition 3 (explicit GNN extension) adds limited theoretical content.**  
Proposition 3 states that $S_K = \partial \text{vec}(Z_K) / \partial \text{vec}(A)$ can be decomposed via the chain rule as a sum of products of per-layer Jacobians (Eq. 10). This is a direct application of the multivariate chain rule and does not constitute a novel mathematical result. The operator-norm bound (Eq. 11) is a standard submultiplicativity/triangle inequality application. The practical contribution (computing $S_c$ for explicit GNNs) is valuable, but presenting it as a "Proposition" overstates its theoretical novelty.

*Why it matters*: Readers expecting theoretical depth from a numbered proposition will find a chain-rule identity. This sets incorrect expectations.

*Suggested fix*: Relabel as "Remark" or "Observation" and emphasize the *computational* contribution (matrix-free JVP evaluation for explicit architectures) rather than the mathematical content.

**W7. Subgraph extraction introduces uncontrolled approximation error.**  
The default experiment uses 50-node BFS ego-subgraphs, but the sensitivity of node $v$ depends on the *entire* graph through multi-hop message passing. The subgraph ablation (Section V-E) shows Kendall $\tau = 0.80$--$0.86$ between subgraph and full-graph rankings on small power grids, but this is not validated on citation networks where the full graph has thousands of nodes and long-range dependencies may matter more.

*Why it matters*: The "full-graph analysis" capability (matrix-free pipeline) is presented as a contribution, but most experiments use subgraphs, creating a gap between what is validated and what is claimed.

*Suggested fix*: Run the full-graph matrix-free pipeline on at least Cora (which is demonstrated as feasible in 78s, Table IV) and compare per-edge vulnerability rankings against the 50-node subgraph results. Report the subgraph-to-full-graph ranking correlation on citation networks, not only power grids.

---

## 4. Detailed Technical Analysis

### 4.1 IFT Derivation Correctness (Eq. 3 and Proof of Theorem 1)

The IFT derivation in Section III is standard and correct. The equilibrium equation $G(z^*, A) = z^* - F(z^*, A) = 0$ is differentiated to give $\partial z^* / \partial p = (I - J_z)^{-1} \partial F / \partial p$. The key steps are:

1. **Jacobian structure**: $J_z = \text{diag}(\sigma') \cdot (\hat{A} \otimes W)$ is correctly identified for the IGNN operator $F(Z,A) = \sigma(\hat{A}ZW^\top + X_{\text{proj}})$. The Kronecker product arises from vectorizing the bilinear form $\hat{A}ZW^\top$.

2. **Norm bound**: $\kappa = \|J_z\|_2 \leq \|\hat{A}\|_2 \cdot \|W\|_2 \cdot \sup|\sigma'|$ is correct. For ReLU, $\sup|\sigma'| = 1$. For the symmetric normalized adjacency $\hat{A} = D^{-1/2}(A+I)D^{-1/2}$, we have $\|\hat{A}\|_2 \leq \lambda_{\max}(\hat{A})$, which equals 1 for connected graphs after self-loop normalization. So $\kappa \leq \|W\|_2$, and spectral normalization enforces $\|W\|_2 < 1$.

3. **Neumann convergence**: Correct. $\kappa < 1$ ensures the Neumann series $\sum_{k=0}^{\infty} J_z^k$ converges in operator norm.

4. **Resolvent bound**: $\|(I-J_z)^{-1}\|_2 \leq 1/(1-\kappa)$ follows from $\|(I-J_z)^{-1}\|_2 \leq \sum_{k=0}^{\infty} \|J_z\|_2^k = 1/(1-\kappa)$.

**One technical subtlety**: The Jacobian $J_z$ is evaluated at the fixed point $z^*$, which depends on $A$. The IFT requires $I - J_z$ to be nonsingular, which is guaranteed by $\kappa < 1$, but the IFT also requires smoothness of $F$ -- ReLU is not differentiable at zero. The paper implicitly assumes the fixed point avoids the ReLU kink (i.e., no pre-activation is exactly zero). This is a measure-zero event but should be stated as an assumption or addressed with the observation that ReLU is semismooth and the IFT extends to semismooth functions under mild regularity conditions.

**Verdict**: The IFT derivation is mathematically sound modulo the ReLU non-smoothness caveat, which is standard in the field and unlikely to cause practical issues.

### 4.2 Theorem 1 (Phase Transition)

**Assumptions**:
- (A1) 1-Lipschitz activation: Standard and satisfied by ReLU, sigmoid, tanh.
- (A2) Spectral-norm constraint on $W$: Enforced by construction via spectral normalization during training.
- (A3) $\|J_z\|_2 < 1$ at the fixed point: This is the operational assumption. It is verified empirically ($\kappa = 0.14$--$0.59$ across datasets, Table I).

**Tightness of the bound**: The critical budget $\varepsilon_{\text{crit}} = (1-\kappa)/\|W\|_2$ is a *sufficient* condition. The paper explicitly acknowledges conservativeness (Remark after Theorem 1 proof). The question is *how* conservative:

- For Citeseer ($\kappa = 0.59$, $\varepsilon_{\text{crit}} = 0.41$): The subcritical guarantee holds for $\varepsilon \leq 0.41$, but experiments only test up to $\varepsilon = 0.20$. The gap between the tested range and the critical budget is a factor of 2x.
- The bound uses $\|J_z'\|_2 \leq (\|A\|_2 + \|\delta A\|_F) \cdot \|W\|_2$, which replaces $\|\delta A\|_2$ (operator norm) with $\|\delta A\|_F$ (Frobenius). For sparse perturbations, $\|\delta A\|_F \gg \|\delta A\|_2$, making the bound loose.

**Are the three regimes meaningful?** The subcritical regime (part a) is well-validated. The critical regime (part b, divergence as $\varepsilon \to \varepsilon_{\text{crit}}$) is mathematically correct but never observed experimentally. The supercritical regime (part c) is a negative result -- guarantees cease -- and adds minimal information. The theorem would be equally useful with only part (a).

**Verdict**: The theorem is correct and the assumptions are reasonable, but the phase-transition framing oversells what is essentially a first-order perturbation bound with a contractivity-preservation condition. The critical and supercritical regimes, while mathematically valid, are not empirically accessible.

### 4.3 Proposition 2 (SVD-Based Optimal Perturbation)

**Correctness**: The claim that $\delta A^* = \varepsilon \cdot \text{reshape}(v_1, N \times N)$ maximizes $\|\Delta z^*\|$ to first order is a direct consequence of the variational characterization of the largest singular value: $\sigma_1(S) = \max_{\|v\|=1} \|Sv\|$. This is textbook linear algebra.

**The constrained version** ($S_c$ instead of $S$) is where the contribution lies. Projecting to symmetric, edge-only perturbations via $S_c$ turns the vacuous unconstrained result into a tight, practical tool. This is correctly formulated (Eq. 8).

**Novelty**: The SVD-optimality result itself is not novel (it is the Eckart-Young theorem applied to $S$). The novelty is the $S_c$ construction and the empirical demonstration that constrained first-order optimality achieves tightness $\approx 1.00$. This is a good engineering contribution wrapped in standard mathematics.

**One concern**: The paper claims AEGIS's SVD attack inflicts 1.2x--2.2x more damage than PGD (Table III, ratios 0.46--0.84 inverted). This is surprising: PGD with access to the same IFT gradients should converge to the first-order optimal direction if the landscape is smooth. The gap suggests either (i) PGD has convergence issues (50 steps may be insufficient), or (ii) the landscape has local optima. The paper attributes it to PGD getting "trapped in local optima," but this deserves more analysis -- if the first-order landscape has local optima, the SVD direction may not be globally optimal at finite $\varepsilon$ either.

### 4.4 Proposition 3 (Explicit GNN Extension)

As noted in W6, this is a chain-rule identity. The mathematical content is:

$$S_K = \sum_{l=1}^{K} \left(\prod_{k=l+1}^{K} J_z^{(k)}\right) J_A^{(l)}$$

This is the standard formula for sensitivity propagation through a feedforward composition. The bound $\sigma_1(S_K) \leq \sum_l \prod_k \|J_z^{(k)}\|_2 \cdot \|J_A^{(l)}\|_2$ follows from submultiplicativity of operator norms and the triangle inequality.

**What it adds**: Practically, it justifies applying the $S_c$ machinery to GCN, GIN, GAT, SAGE, APPNP. The empirical validation (Table V) showing tightness 0.99--1.02 across 6 explicit architectures is the valuable contribution, not the proposition itself.

**Bound looseness**: Reported as 1.4x (SAGE-2) to 5.9x (APPNP), with deeper models being looser. This is expected: the product of norms overestimates the norm of the product due to direction-dependent cancellations.

### 4.5 Per-Node Sensitivity Radii

The radius formula $r_v = m_v / (\|W_{y_v} - W_{c^*}\|_2 \cdot \|S_v\|_2)$ (Eq. 9) is derived correctly from the first-order approximation:

1. $\Delta z_v^* \approx S_v \cdot \text{vec}(\delta A)$ (first-order)
2. $|\Delta f_{y_v}(\zstar_v)| \leq \|W_{y_v} - W_{c^*}\|_2 \cdot \|S_v\|_2 \cdot \|\delta A\|_F$ (Cauchy-Schwarz)
3. Misclassification requires $|\Delta f_{y_v}| \geq m_v$
4. Inversion gives $r_v$

**Meaningfulness as first-order quantities**: These radii are first-order Taylor approximations. Their accuracy depends on:
- The magnitude of the $O(\varepsilon^2)$ remainder relative to $m_v$
- The curvature of the equilibrium manifold $z^*(A)$

The paper's empirical validation (0% breach rate at $\varepsilon = 0.01$, $< 1\%$ at $\varepsilon = 0.10$) is reassuring but does not substitute for a formal second-order bound. At $\varepsilon = 0.10$, the tightness is already 1.15 (Cora), meaning the first-order approximation underestimates the actual shift by 15%. For nodes near the decision boundary ($m_v$ small), this 15% error could cause breach.

**Comparison with certified methods**: The paper correctly notes (Section V-B) that randomized smoothing gives *uniform* radii (all nodes get the same $r$) while AEGIS gives *differentiated* radii. This is a genuine advantage for vulnerability ranking, but the comparison is somewhat apples-to-oranges: smoothing radii are probabilistic certificates (correct with probability $1 - \alpha$), while AEGIS radii are first-order approximations with unknown error.

### 4.6 Statistical Validity

**Seeds**: 10 seeds, explicitly listed. This is good practice. Mean $\pm$ std is reported throughout.

**Missing elements**:
- No confidence intervals (95% CI would be more informative than $\pm$ std for 10 samples).
- No effect sizes (Cohen's $d$ or similar) for comparisons.
- No statistical tests for the Mettack comparison ("149/150 wins" is reported but no $p$-value or binomial test).
- The claim "$83\times$ amplification" at $\kappa = 0.99$ (Section V-D) is a single data point, not a statistical comparison.

**Sample size adequacy**: $n = 10$ seeds is adequate for mean estimation but marginal for detecting small effects. The standard errors (std/$\sqrt{10}$) are small enough that the main findings (tightness $\approx 1.00$, AtkAdv $> 1$) are statistically significant by inspection, but formal significance testing would strengthen the claims.

### 4.7 Adaptive Attacker Design (Section V-C)

**Fairness**: The adaptive attacker uses PGD with IFT gradients -- the same information AEGIS uses for $S_c$ computation. This is the correct design: it tests whether the first-order analysis captures the attacker's capability.

**Potential issues**:
1. **PGD hyperparameters**: 50 steps with step size $\varepsilon/10$ may be insufficient for convergence. The paper does not report PGD convergence (loss vs. iteration).
2. **Constraint enforcement**: PGD enforces $\|\delta\hat{A}\|_F \leq \varepsilon$ via projection, but does it also enforce symmetry and edge-only constraints? If not, the comparison is between constrained AEGIS and unconstrained PGD, which is unfair.
3. **Loss function**: PGD optimizes classification loss (cross-entropy), while AEGIS maximizes equilibrium shift ($\|z^*\|$). These objectives are different; AEGIS's advantage may partly reflect this objective mismatch rather than the SVD direction's superiority.

**The AEGIS > PGD result is surprising and under-analyzed**: If both use first-order gradients and the landscape is quadratic near the optimum, PGD should converge to the SVD direction. The persistent gap (ratio 0.46--0.84) suggests either PGD convergence failure or objective function mismatch. This should be investigated and reported.

---

## 5. Questions for Authors

**Q1**: The critical budget $\varepsilon_{\text{crit}}$ is never experimentally reached. Can you show the phase transition by training IGNN with relaxed spectral normalization ($\kappa \to 1$) and demonstrating the predicted divergence of $\|(I - J_z')^{-1}\|_2$ as $\varepsilon \to \varepsilon_{\text{crit}}$? Even a single-dataset experiment would validate the main theorem's practical relevance.

**Q2**: The PGD adaptive attacker consistently underperforms the SVD direction (Table III, ratios 0.46--0.84). Does PGD optimize the same objective as AEGIS (equilibrium shift $\|\Delta z^*\|_F$) or classification loss? If the latter, the comparison is between two different optimization problems, not between two optimization methods for the same problem.

**Q3**: How correlated are per-edge vulnerability scores $v_{ij}$ with endpoint degree? If the Kendall $\tau$ between $v_{ij}$ and $\max(d_i, d_j)$ exceeds 0.5, the vulnerability spectrum may be substantially a degree proxy. Could you report this correlation across all 9 datasets?

**Q4**: The ReLU non-differentiability at zero is never discussed. What fraction of pre-activations at the fixed point are exactly zero (or within numerical tolerance)? If substantial, the IFT does not formally apply. Do you observe sensitivity discontinuities when pre-activations cross zero under perturbation?

**Q5**: For the power grid case study, the training data uses uniform load scaling (70--130% of nominal). How sensitive are the $S_c$ vulnerability rankings to the training distribution? If the load profile changes substantially (e.g., renewable integration scenarios), do the same edges remain critical?

---

## 6. Minor Issues

**M1. Notation inconsistency**: $z^*$ is used both as a scalar (vectorized) and matrix $Z^*$ depending on context. The IFT (Eq. 3) uses lowercase $z^*$, while the operator definition (Eq. 2) uses uppercase $Z$. The vec/reshape conventions should be made explicit once and referenced consistently.

**M2. Proposition numbering**: The paper has Theorem 1, Proposition 2 (attack), Proposition 3 (radius), and Proposition 4 (explicit). In the main text, Proposition 2 is cited as "Proposition 2" but the label is `\prop:attack`. In the experiments, references to "Prop. 4 bound tightness" appear. The numbering should be verified for consistency (especially since Theorem 1 and Propositions share a counter in the LaTeX setup).

**M3. Missing $\hat{A}$ vs. $A$ distinction in Theorem 1**: The theorem statement uses $A$ (unnormalized) in the operator $F_\theta(Z, A) = \sigma(AZW^\top + X_{\text{proj}})$, but the IGNN definition (Eq. 2) uses $\hat{A}$ (normalized). The contractivity bound $\kappa \leq \|\hat{A}\|_2 \cdot \|W\|_2$ requires $\hat{A}$, not $A$. The critical budget $\varepsilon_{\text{crit}}$ derivation in the proof uses $\|A\|_2 + \|\delta A\|_F$, mixing normalized and unnormalized adjacency. This should be cleaned up.

**M4. Tightness definition**: Tightness = actual/predicted shift is $> 1$ throughout (Table I: $1.00$--$1.01$; Table II: up to $1.39$). This means the first-order prediction consistently *underestimates* the actual shift, which is expected (convexity of the equilibrium map). But the abstract claims "tightness $1.00 \pm 0.01$," which could be read as overestimation as well. Clarify that tightness $\geq 1$ is expected.

**M5. Table I column "Cert%"**: Described as "coverage (fraction of nodes with positive first-order radius)." But any correctly classified node has positive margin ($m_v > 0$), hence positive radius. So Cert% should equal clean accuracy on the subgraph, not the full graph. The caption says "full-graph / subgraph accuracy" which is ambiguous. Clarify.

**M6. GAT modification**: The edge-weighted GAT variant (multiplying attention by $\hat{A}_{ij}$) changes the model's representational behavior, not just its sensitivity computation. The tightness and $\tau$ results for GAT$^\dagger$ are for this modified model, not standard GAT. This should be more prominently flagged.

**M7. The "matrix-free formulation ... enables full-graph analysis" claim (abstract, introduction)**: Full Cora analysis takes 78s. For Pubmed ($N = 19{,}717$) or WikiCS ($N = 11{,}701$), the $O(K \cdot Nd)$ cost per matvec would be substantially higher, and no timing is reported for these graphs. The claim should be qualified.

---

## 7. Scores

| Dimension | Score (0--100) | Justification |
|-----------|---------------|---------------|
| **Mathematical Rigor** | 72 | IFT derivation is correct; Theorem 1 is sound but conservative; Proposition 3 is trivial; ReLU non-smoothness unaddressed; $\hat{A}$ vs $A$ confusion in theorem statement; no second-order error bound |
| **Statistical Validity** | 75 | 10 seeds with std, tightness across $\varepsilon$ range, adaptive attacker -- all good. Missing CIs, effect sizes, significance tests. PGD objective mismatch unanalyzed |
| **Reproducibility** | 80 | Seeds listed, architecture details given, wall-clock times reported, code release promised. Missing: learning rate schedules for explicit GNNs, PGD convergence diagnostics |
| **Novelty of Method** | 68 | $S_c$ construction is the novel contribution. IFT for sensitivity is standard (Lorraine et al., Gould et al.). SVD-optimality is textbook. Phase transition framing is not empirically validated. Matrix-free pipeline is engineering, not methodological novelty |
| **Overall** | **Weak Accept** | A solid engineering contribution ($S_c$, matrix-free pipeline, cross-architecture validation) wrapped in a theory paper. The theoretical novelty (Theorem 1) is overstated relative to its practical impact (the phase transition is never observed). The perturbation model is restrictive. But the constrained sensitivity matrix, the adaptive attacker evaluation, and the power grid case study are well-executed and the 10-seed protocol is commendable. The paper would benefit from (i) a second-order error bound, (ii) stronger baselines than random perturbation, and (iii) empirical demonstration of the phase transition. With revisions addressing W1--W4, this would be a clear accept. |

---

*Reviewed as Reviewer 1 (Methodology) for IEEE ICDM.*
