# Reviewer 2 — Domain Review (GNN theory)

**Role.** GNN-theory researcher; deep familiarity with implicit / equilibrium GNNs, graph signal processing, Lipschitz GNNs, spectral methods.
**Mode.** Independent.
**Page-budget calibration.** 10-page IEEEtran cap binding.

---

## 1. Summary as I read it

The paper introduces a sensitivity object $S = \partial \mathrm{vec}(z^*) / \partial \mathrm{vec}(\hat{A})$ for a contractive implicit GNN, computed via the implicit function theorem as $(I - J_z)^{-1} J_A$ at the equilibrium. It then constructs a symmetric edge-supported projection $S_c = S P_c$ and identifies three outputs (SVD direction, column norms, per-node Jacobian-norm ratio). Theorem 1 characterises three regimes (subcritical, critical, supercritical) for the contraction certificate, and Observation 1 bounds the pseudospectral index $\eta$ by the eigenvector conditioning of $W$ (graph-independent under symmetric $\hat{A}$). Proposition 5 (Observation in the text) extends the construction to $K$-layer explicit GNNs via the unrolled Jacobian.

## 2. Strengths

**S1. The IFT-resolvent + symmetrized edge projection is a clean object.** Stating it as $S_c$, computing it matrix-free via Neumann + randomized SVD, and using it for three downstream tasks is genuinely tidy. The unification language ("from one object") is justified at the formal level: rankings = column norms, direction = leading right singular vector, radius = block-row norms.

**S2. Observation 1 (graph-independent η).** This is a non-trivial result. The fact that $\eta \leq \kappa(V_W)$ — eigenvector conditioning of the weight matrix alone, no dependence on $\hat{A}$ — is a useful structural statement about why $\eta$ stays near 1 in practice. The proof sketch via $J_z = \mathrm{diag}(\phi') \cdot (\hat{A} \otimes W)$ + Kronecker eigendecomposition is correct.

**S3. Honest distinction between $\kappa$ and $\rho$.** The paper does not use spectral radius (which would tighten bounds for normal $J_z$) but uses the operator norm (required by Neumann convergence). This is the right choice and is justified explicitly. The "1.02–1.28" empirical $\eta$ shows the gap is small in practice.

**S4. The conservative-IFT extension for ReLU.** Citing Bolte-Pauwels 2021 for the piecewise-affine case is the correct way to handle nonsmoothness. The measure-zero exceptional set argument is standard but underappreciated in GNN-robustness papers, so its inclusion is welcome.

**S5. Position vs Lipschitz-GNN stability bounds.** The Related Work distinguishes between scalar Lipschitz bounds (Gama 2020) and directional Jacobian information (this work). This is a real and underused distinction.

## 3. Weaknesses (numbered)

### W1. **The "constrained sensitivity matrix" $S_c$ is presented as the novel construction, but the symmetrization is a standard matrix-calculus reduction.** [Moderate]

The construction $[S_c]_{:,k} = S_{:, iN+j} + S_{:, jN+i}$ with edge basis $b_k = (e_i e_j^\top + e_j e_i^\top)/\sqrt 2$ is the natural restriction of $\partial / \partial A$ to the symmetric-edge subspace. This is what every careful derivative-w.r.t.-symmetric-matrix calculation does (cf. Magnus & Neudecker, Matrix Differential Calculus, ch. 3; the duplication-matrix formalism). The paper does not cite this literature.

The novelty is then *operational* (using $P_c$ to enforce the symmetry constraint inside the matrix-free pipeline, so $S_c v$ never materialises $S$) rather than *mathematical*. This is a real contribution — but the abstract's "we introduce the constrained sensitivity matrix" reads as a mathematical innovation, not an engineering one.

**Concretely:** add one sentence in §Theory acknowledging $P_c$ as the standard duplication-matrix-style reduction, and credit Magnus–Neudecker or similar. The intellectual contribution is the matrix-free $S_c v$ implementation; that should be made the explicit claim.

**Fix fits the budget:** ≤ 1 sentence + 1 citation.

### W2. **Theorem 1 part (b) "Critical divergence" is worst-case, not typical-case; the abstract's "three-regime characterisation" elides this.** [Major]

The paper states the critical-regime divergence $\|(I-J_z')^{-1}\|_2 = \Omega(1/(\varepsilon_\text{crit} - \varepsilon))$ holds *"along worst-case directions (aligned with the top singular vector of $\hat A$); generic directions diverge more slowly since $\|J_z'\|_2$ need not approach 1."*

This is technically correct, but the abstract advertises "a three-regime characterisation" without specifying that the critical-regime statement binds only in a worst-case slice. A reader expecting three regimes of *behaviour* (analogous to subcritical / critical / supercritical phase transitions in statistical mechanics) will be disappointed: there is no claim of a generic-direction critical regime, no empirical demonstration of the three regimes as ε crosses $\varepsilon_\text{crit}$.

**Concretely:**
- Either (a) qualify the abstract to "a three-regime *worst-case* characterisation" or "three-regime characterisation along the top sensitivity direction"; or
- (b) Add a small empirical demonstration (e.g., 1 plot, 6 lines: $\|\Delta z^*\|_F$ vs ε swept across [0, 2 × $\varepsilon_\text{crit}$] for 3 datasets, showing the transition.) The repository already has `exp_phase_transition.py` — this should be one figure, fits the budget if a smaller existing figure is shrunk.

The (b) option would substantially strengthen the theorem's claim; (a) is the minimum.

### W3. **Observation 1 hypothesis: "all activations positive" is restrictive.** [Moderate]

Observation 1(a) requires $\phi'_i = 1$ for all $i$, i.e., all activations in the linear region of ReLU. This holds in *some* network states but not in trained networks at typical inputs (ReLU networks rely on the masking for representation). Part (b) extends to general activation patterns via $\kappa(V_{J_z})$, but the bound now depends on the eigenvector conditioning of the *masked* Jacobian, which can depend on the graph structure indirectly (via which nodes have which activation patterns).

The "graph-independent" framing in the headline is exact under (a) but degraded under (b). The paper should make clear that the headline "graph-independent nonnormality bound" applies in the all-positive regime, not in general operation.

**Concretely:** add half a sentence in the Observation discussion stating that the strict graph-independence is the (a) bound, and (b) introduces an indirect graph dependence via the activation mask.

### W4. **The umbrella claim "any GNN with continuous edge-weight-modulated message passing" hides per-architecture variability.** [Major]

The explicit-GNN extension (Observation 2 / Proposition 4) defines $S_K$ as a sum of products of layer Jacobians. The bound $\sigma_1(S_K) \leq \sum_l (\prod_{k>l} \|J_z^{(k)}\|_2) \|J_A^{(l)}\|_2$ can be vacuous: with $K=4$ and per-layer Jacobian norm $> 1$, the product term dominates and the bound is uninformative.

The Table on explicit-GNN tightness shows IGNN tight=1.01, GCN-2 to GIN-2 ranging 0.99–1.02 — empirically tight despite the a priori bound being potentially loose. The text says "the empirical tightness 0.99–1.02 validates the approximation regardless of the a priori bound" — fair, but this means the *useful* claim is empirical, not formal: the bound does not certify why the framework works on explicit GNNs.

Additionally, the cross-architecture $\tau$ in Table tau_cross varies wildly: IGNN $-0.15$ on Amazon Photo, GCN-2 $-0.03$ to $+0.04$ on Cora/Citeseer, GCN-4 $+0.45$ to $+0.83$. The "any GNN" umbrella is true at the construction level but false at the predictive level — the framework's per-edge ranking transfers well on deeper-than-2 layers, badly on shallow ones.

**Concretely:**
- Replace "any GNN with continuous edge-weight-modulated message passing" with "the construction applies to any such GNN; predictive transfer of the per-edge ranking is architecture-dependent (Table tau_cross), with deeper-than-2-layer models showing the strongest transfer."
- This is more accurate and gives the practitioner a usable guideline.

**Fix fits the budget:** Abstract + introduction rewording.

### W5. **Missing recent literature on graph adversarial robustness.** [Major]

The Related Work cites Zügner 2018/2019, Geisler 2021, Wu 2019, Bojchevski 2020, Schuchardt 2023, Li 2025 — but the field has moved. The following are notable absences for a paper claiming a unification:

- **Gosch et al. 2024** ("Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions", NeurIPS 2024) — challenges prior defense benchmarks; relevant to the §Defense ablation.
- **Mujkanovic et al. 2022** ("Are Defenses for Graph Neural Networks Robust?", NeurIPS 2022) — methodology for honest defense evaluation; the defense-informed-masking experiment should engage with this protocol.
- **Bojchevski & Günnemann 2019** ("Certifiable Robustness to Graph Perturbations", NeurIPS) — earlier than the cited Bojchevski 2020 smoothing paper; relevant to the certification thread.
- **Schuchardt et al. 2021** ("Collective Robustness Certificates") — collective vs per-node certification; relevant to the radius claim.
- **PRBCD** (Geisler 2021, NeurIPS) — cited only as the smaller GR-BCD variant. PRBCD is the actual scalable structural attack baseline.
- **El-Hamri et al. 2021** (cited) but **Revay et al. 2020** (Lipschitz networks) is only in the background — a single line in Related Work tying the input-Lipschitz vs structural-Lipschitz distinction would help.

In 10 pages this is a real constraint. My recommendation: PRBCD must appear (this is R1's W3 too); the other four can be cited briefly in Related Work without dedicated discussion.

### W6. **The "structural Lipschitz" framing is implicit but not named.** [Minor]

The paper computes $\sigma_1(S_c)$, the structural Lipschitz constant under symmetric edge perturbations. This is a meaningful quantity — it is the GNN-robustness analogue of the Lipschitz constant for input-feature perturbations. The paper does not name it as such. Doing so would clarify the contribution: AEGIS is computing a structural Lipschitz constant *and* its leading eigenvector *and* its per-edge norms, in a single pass.

**Concretely:** add the phrase "structural Lipschitz constant $\sigma_1(S_c)$" somewhere in §Theory; this gives the contribution a clean name. Optional.

### W7. **Theorem 1 (c) supercritical regime is descriptive, not constructive.** [Minor]

Part (c) says "the contraction certificate is void, Banach no longer guarantees uniqueness, and the part-(a) first-order guarantees lapse." This is true and important to flag, but the theorem makes no positive statement about what happens. The empirical demonstration (W2) would resolve this if it included the supercritical regime; otherwise (c) reads as a disclaimer.

## 4. Specific corrections / clarifications

- §Theory: "the joint eigenvector matrix has condition number $\kappa(V_W)$" — verify this carefully. $J_z = \mathrm{diag}(\phi') \cdot (\hat A \otimes W)$. Under all-positive activations, $\mathrm{diag}(\phi')=I$, so $J_z = \hat A \otimes W$. The eigenvector matrix of $\hat A \otimes W$ is $U_{\hat A} \otimes V_W$ with condition number $\kappa(U_{\hat A}) \cdot \kappa(V_W) = 1 \cdot \kappa(V_W) = \kappa(V_W)$ since $\hat A$ is symmetric (orthonormal eigenvectors). The text is correct; the algebra is fine. **No change needed**, but consider adding the Kronecker eigendecomposition identity in one line for the reader.
- §Background eq:ift: $\norm{(I-J_z)^{-1}}_2 \leq 1/(1-\kappa)$ via Neumann — standard; would benefit from a parenthetical "(Neumann series convergence requires $\kappa < 1$, i.e., (A3))".
- §Theory eq:radius: the radius is $m_v / (\|W_{y_v} - W_{c^*}\|_2 \cdot \|S_v\|_2)$ — please confirm the $\|S_v\|_2$ is the **operator** norm of the block-rows (not the Frobenius); the conservative bound uses operator norm.
- Theorem 1 (a) statement / proof: "Taylor remainder exact, $R_k = 0$" for ReLU on each linear region — verify this carefully across the activation boundaries. The conservative IFT handles boundary crossings, but the within-region Taylor remainder is exact only because the operator is piecewise *affine*, not piecewise smooth. The phrasing is fine, but please make sure the proof is unambiguous on this point.

## 5. Recommendation

**Major Revision.** The theoretical core is sound and the unification is real, but W2 (worst-case framing of three regimes), W4 (per-architecture transfer variability hidden under the umbrella claim), and W5 (missing recent literature, especially PRBCD and Mujkanovic 2022) need substantive response. W1 (Magnus-Neudecker positioning) is a minor framing fix. The paper would benefit from a small empirical phase-transition figure (W2 option (b)).

## 6. Scores

| Dimension | Score (0–100) |
|---|---|
| Theoretical correctness | 75 |
| Literature coverage | 60 |
| Framework generality (as advertised) | 62 |
| Framework generality (as actually demonstrated) | 70 |
| Position vs prior work | 68 |
| **Domain overall** | **66** |
