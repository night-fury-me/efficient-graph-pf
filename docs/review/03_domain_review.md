# Domain Review Report — AEGIS

**Reviewer**: Prof. Aleksandar Bojchevski, CISPA Helmholtz Center for Information Security
**Expertise**: Adversarial robustness of GNNs, certified defenses (randomized smoothing on graphs, convex relaxations), spectral methods, graph poisoning attacks (Nettack/Mettack lineage), scalable robustness certificates
**Confidence**: 5/5 (core domain expertise; author of several cited works)

## Summary (200-300 words)

This paper introduces AEGIS, a framework for pre-deployment structural vulnerability analysis of GNNs. The central construct is the constrained sensitivity matrix $S_c \in \mathbb{R}^{Nd \times |E|}$, derived by applying the implicit function theorem (IFT) to the equilibrium equation of contractive implicit GNNs (IGNN-class) and then projecting the full $N^2$-dimensional perturbation space onto the $|E|$-dimensional space of symmetric, edge-only perturbations. From $S_c$, three outputs are extracted: (i) the SVD-optimal first-order attack direction, (ii) per-edge vulnerability rankings, and (iii) per-node first-order sensitivity radii.

The paper positions itself in a gap between adversarial attack methods (which search for damaging perturbations) and certified defenses (which provide robustness guarantees), claiming that neither provides a "structural vulnerability map." Theorem 1 characterizes three perturbation regimes (subcritical, critical, supercritical) for contractive implicit GNNs, with a critical budget $\varepsilon_{\text{crit}} = (1-\kappa)/\|W\|_2$. The framework is extended to explicit GNNs (GCN, GAT, GIN, SAGE, APPNP) via unrolled Jacobian computation, though without formal convergence guarantees.

The experimental evaluation is thorough: 9 datasets, 7 architectures, 10 seeds, with a power grid case study demonstrating N-1 contingency recovery. The matrix-free formulation (Neumann series + randomized SVD) scales to $N = 7{,}650$ on a single GPU.

The paper makes a genuine contribution to the vocabulary of GNN robustness analysis. However, I have concerns about the novelty of the theoretical machinery, the accuracy of the claimed literature gap, several missing references, and the practical significance of first-order sensitivity radii relative to existing certification methods.

## Strengths

1. **Well-defined constrained projection $S_c$ (Sec. III-IV).** The reduction from $N^2$ to $|E|$ dimensions via the symmetry and edge-only constraint is the paper's most valuable technical contribution. The authors correctly identify that unconstrained sensitivity bounds are vacuous for realistic graph perturbations, and the constrained projection transforms first-order analysis into a tight, practical tool. The tightness results ($1.00 \pm 0.01$ at $\varepsilon = 0.01$, Table I) are convincing and well-validated.

2. **Comprehensive experimental methodology (Sec. V).** The evaluation is among the most thorough I have seen in the GNN robustness literature. The four-quadrant attack taxonomy (gradient-based vs. gradient-free, same-objective vs. different-objective) in Sec. V-C is methodologically sound. The comparison against brute-force greedy-optimal discrete edge removal (Table V) is the right ground truth. The 330-run cross-architecture evaluation (Table VII, Fig. 3) with 10 seeds each provides strong statistical evidence. The structured baselines (degree, spectral, betweenness in Table III) and the honest reporting of the narrow AEGIS-vs-degree gap for continuous perturbation are commendable.

3. **Honest self-assessment of limitations.** The paper is unusually forthright about where its contributions end: the theoretical guarantees apply only to IGNN-class models (Sec. III, Observation 1); the critical budget $\varepsilon_{\text{crit}}$ is conservative by $\sqrt{|E|}$ (Remark after Theorem 1); Amazon Photo shows negative $\tau$ for IGNN (Sec. V-F); and the power flow $\tau = 0.37$--$0.67$ is "insufficient for direct operational use" (Sec. VI). This transparency strengthens the paper's credibility.

4. **Matrix-free scalability (Sec. IV-B, Table VI).** The Neumann-series resolvent combined with autograd JVPs and randomized SVD is a practical engineering contribution. Scaling from $N \leq 200$ (dense path) to $N = 7{,}650$ (Amazon Photo) with sub-linear memory growth is useful. The dense-vs-matrix-free comparison (Table VI) clearly demonstrates the scalability gain.

5. **Power grid case study with domain validation (Sec. VI).** Connecting GNN sensitivity analysis to N-1 contingency analysis is creative and practically motivated. The comparison against LODF (an industry standard) at case57/118 with Wilcoxon signed-rank tests is rigorous. The binary-vs-admittance ablation (Sec. VI-C) demonstrates domain understanding.

## Weaknesses

1. **Theorem 1 is a restatement of standard contraction-mapping results with minor graph-specific dressing (Sec. III).** The three-regime characterization (subcritical/critical/supercritical) follows directly from the Banach fixed-point theorem and the Neumann series bound $\|(I - J_z)^{-1}\|_2 \leq 1/(1-\kappa)$. Part (a) is the IFT applied to the equilibrium equation — standard in implicit differentiation (Gould et al., 2021; Lorraine et al., 2020). Part (b) is the trivial observation that $\|(I - J_z')^{-1}\|_2 \to \infty$ as $\|J_z'\|_2 \to 1$. Part (c) notes that the Banach theorem no longer applies. The critical budget $\varepsilon_{\text{crit}} = (1-\kappa)/\|W\|_2$ is a direct consequence of the sub-multiplicativity bound on the perturbed Jacobian. The observation that $\eta$ depends only on $W$ (Observation 1) is interesting but follows from the Kronecker-product eigenstructure of $J_z = \hat{A} \otimes W$ when $\hat{A}$ is symmetric. None of these steps represent a conceptual advance beyond what is known in the DEQ/implicit network literature.

   **Suggested fix:** Reframe Theorem 1 as a "specialization of known results to the graph perturbation setting" rather than a novel theorem. The novelty claim should center on $S_c$ (the constrained projection) and its empirical validation, not on the three-regime characterization.

2. **The claimed gap between attacks and defenses is overstated (Sec. I, Sec. VII).** The introduction states that certified defenses "are uniform: every node receives the same certificate, revealing nothing about which edges or nodes are structurally weak." This mischaracterizes the state of the art. Localized randomized smoothing (Schuchardt et al., 2023, which IS cited) provides per-node certificates. More importantly, per-edge gradient information from attacks like Mettack already provides a form of per-edge vulnerability ranking — the meta-gradient $\partial \mathcal{L} / \partial A_{ij}$ is exactly an edge-level vulnerability score, albeit optimizing classification loss rather than equilibrium shift. The paper acknowledges this in Sec. V-C (classification-gradient edge ranking) but does not reconcile it with the introduction's claim of a complete gap.

   **Suggested fix:** Revise the introduction to acknowledge that per-edge information IS available from gradient-based attacks and per-node information from localized smoothing. The genuine gap is that no existing method provides (i) the SVD-optimal attack direction, (ii) per-edge vulnerability rankings, AND (iii) per-node sensitivity radii from a single unified computation under realistic structural constraints. This is a more defensible and still compelling claim.

3. **Missing comparison with Lipschitz-based robustness analysis for GNNs.** The paper does not compare against or discuss the line of work on GNN Lipschitz constants and their use for robustness analysis. Specifically:
   - Gama et al. (2020) [cited but underutilized] derive stability bounds for graph filters and GNNs under graph perturbation — this is the closest existing work to AEGIS's sensitivity analysis and deserves a detailed comparison.
   - Kenlay et al. (ICML 2021, "Stability and Generalisation of Graph Neural Networks via Spectral Analysis") provide spectral perturbation bounds for GNNs that yield per-node stability guarantees.
   - Xu et al. (NeurIPS 2023, "On the Stability of Expressive Positional Encodings for Graphs") analyze stability under structural perturbation.
   - The monotone operator / Lipschitz-bounded equilibrium network literature (Winston & Kolter, 2020; Revay et al., 2020; Pabbaraju et al., 2021 — all cited) provides Lipschitz certificates that implicitly bound per-input sensitivity; the paper should discuss why these do not yield per-edge structural vulnerability maps.

   **Suggested fix:** Add a dedicated paragraph in Sec. VII comparing AEGIS's $S_c$-based analysis against spectral stability bounds (Gama et al., Kenlay et al.) and Lipschitz certificates (monotone operators). Clarify what $S_c$ provides that these bounds do not (specifically: directional sensitivity via SVD, not just worst-case Lipschitz).

4. **Continuous-to-discrete transfer is the practical bottleneck, but the theoretical bridge (Proposition 3) has limited practical utility.** The sufficient condition for ranking preservation (Eq. 11) requires knowledge of $L_J$ (the Lipschitz constant of $J_z$ w.r.t. $A$), which the paper estimates as $\|W\|_2^2$ — but this bound applies only within a fixed ReLU linear region. The remark on ReLU non-differentiability (after Proposition 3) acknowledges that $J_z$ jumps discontinuously across activation boundaries, and the "measure zero" argument is technically correct but practically irrelevant for ranking transfer, where the question is whether specific edge pairs satisfy the sufficient condition. The empirical results (Table VII) show the real picture: transfer is architecture-dependent, dataset-dependent, and can be negative (GCN-2 on Citeseer: $\tau = -0.28$, IGNN on Amazon Photo: $\tau = -0.15$).

   **Suggested fix:** Be more transparent that Proposition 3's sufficient condition is a theoretical justification for why transfer *can* work, not a practical prediction of when it *will* work. The empirical cross-architecture evaluation (Table VII) is the real contribution here; Proposition 3 should be presented as a motivating result rather than a predictive tool.

5. **The first-order sensitivity radii $r_v$ are fundamentally weaker than existing robustness certificates and the comparison in Sec. V-B is misleading.** The paper compares AEGIS radii against randomized smoothing (Sec. V-B) and claims complementarity. However, the comparison obscures a fundamental asymmetry: smoothing certificates are *valid* guarantees (with probability $1-\alpha$, no perturbation within the certified radius can change the prediction), while AEGIS radii are first-order approximations that can be violated at any perturbation magnitude (they are tight only in the limit $\varepsilon \to 0$). Calling both "radii" invites confusion. The breach rate data (Table VIII) shows 0.6% breach at $\varepsilon = 0.01$ on Cora, rising to 7.6% at $\varepsilon = 0.20$ — these are precisely the cases where the first-order approximation fails, and any truly certified guarantee would show 0% breach by definition.

   **Suggested fix:** Rename $r_v$ to "first-order sensitivity threshold" or "linear-regime tolerance" to avoid confusion with certified radii. Add a clear statement that $r_v$ provides no formal guarantee — it is a diagnostic quantity, not a certificate. The comparison with smoothing should emphasize that AEGIS provides *structural differentiation* (which edges/nodes are weak), not *certification* (which nodes are guaranteed robust).

6. **Several important recent references are missing (2023-2025).**
   - Gosch et al. (NeurIPS 2024, "Provably Robust Conformal Prediction with Improved Efficiency") — extends certified robustness to conformal prediction on graphs.
   - Geisler et al. (ICLR 2024, "Attacking Graph Neural Networks with Bit Flips") — a fundamentally different attack surface that should be acknowledged in the threat model discussion.
   - Mu et al. (ICML 2023, "Certifiably Robust Graph Contrastive Learning") — certified robustness for graph self-supervised learning.
   - Topping et al. (ICML 2022, "Understanding over-squashing and bottlenecks on graphs via curvature") — relevant to the depth-sensitivity connection the paper makes in the GCN-2 vs GCN-4 discussion.
   - Di Giovanni et al. (ICML 2023, "Over-squashing and curvature") — extends the curvature-based analysis, directly relevant to understanding why depth matters for sensitivity transfer.

   **Suggested fix:** Add citations and brief discussion of the curvature/over-squashing line of work, which provides an independent theoretical lens on edge sensitivity (edges in negatively-curved regions are bottlenecks, and these may correlate with high $S_c$ vulnerability scores). This connection would strengthen the paper's theoretical positioning.

## Literature Assessment

**Coverage of seminal works:** Good. Nettack, Mettack, randomized smoothing (Bojchevski et al., 2020), localized smoothing (Schuchardt et al., 2023), convex relaxations (Bojchevski & Gunnemann, 2019; Zugner & Gunnemann, 2019), DEQ/IGNN foundations are all cited. The survey coverage (Wu et al., 2019; Jin et al., 2021) is adequate.

**Coverage of certified defense literature:** Adequate but incomplete. The paper cites the key randomized smoothing works and convex relaxations, but misses the conformal prediction extensions and the graph contrastive learning certificates that represent the 2023-2024 frontier.

**Coverage of GNN stability/Lipschitz analysis:** Weak. Gama et al. (2020) is cited but not engaged with substantively. The spectral stability analysis literature (Kenlay et al., 2021; Xu et al., 2023) is absent. This is a significant gap because these works address the same fundamental question — how GNN outputs change under graph perturbation — from a spectral/Lipschitz perspective that directly relates to AEGIS's sensitivity analysis.

**Coverage of implicit network robustness:** Good. El Ghaoui et al. (2021), Revay et al. (2020), Winston & Kolter (2020), and Pabbaraju et al. (2021) are cited. The distinction between input sensitivity and structural sensitivity is correctly drawn.

**Coverage of GNN explainability vs. vulnerability:** Good. The distinction between GNNExplainer/PGExplainer (prediction fidelity) and AEGIS (adversarial vulnerability) is well-articulated and experimentally validated (Sec. V-F, negative $\tau$ for gradient attribution).

**Positioning accuracy:** Mostly accurate but overstated at the introduction. The gap is narrower than claimed — localized smoothing provides per-node differentiation, and meta-gradients provide per-edge information. The genuine contribution is the unified $S_c$ framework providing all three outputs under constrained perturbations, not the existence of per-edge vulnerability information *per se*.

## Novelty Assessment

The novelty of AEGIS can be decomposed into four components:

1. **IFT + Neumann series for equilibrium sensitivity (LOW novelty).** This is standard implicit differentiation applied to the DEQ fixed point. The resolvent $(I - J_z)^{-1}$ and its Neumann expansion have been used extensively in the DEQ literature for training (Bai et al., 2019; Gould et al., 2021; Fung et al., 2022). The three-regime characterization (Theorem 1) is a repackaging of the Banach fixed-point theorem's contractivity conditions.

2. **Constrained projection $S_c$ (MODERATE-HIGH novelty).** Projecting the full $N^2$-dimensional sensitivity onto the $|E|$-dimensional edge-only, symmetric subspace is the paper's key technical idea. While the construction itself is straightforward (sum corresponding columns for $(i,j)$ and $(j,i)$), its effect is dramatic: it transforms vacuous unconstrained bounds into tight predictions (tightness $\approx 1.00$). This has not been done in the GNN robustness literature before, and the empirical validation is convincing. The SVD of $S_c$ providing the optimal constrained attack direction is a clean result.

3. **Matrix-free computation pipeline (MODERATE novelty).** Combining Neumann-series resolvent, autograd JVPs, and randomized SVD is sound engineering. The individual components are well-known (Halko et al., 2011 for randomized SVD; standard autograd for JVPs), but their integration for graph sensitivity analysis at scale ($N = 7{,}650$) is new.

4. **Cross-architecture extension via unrolled Jacobian (LOW novelty).** Observation 1 (Sec. III-E) is the multivariate chain rule applied to a $K$-layer GNN. The authors correctly state this. The contribution is empirical (validating that rankings transfer), not theoretical.

**Overall novelty verdict:** The paper's novelty is concentrated in item 2 ($S_c$) and its comprehensive empirical validation. The theoretical contributions (Theorem 1, Observation 1, Proposition 3) are standard applications of existing mathematical tools. This is an acceptable novelty profile for a data mining venue if the empirical contribution is strong enough — and it is.

## Scores (0-100 scale)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Novelty | 58 | $S_c$ is genuinely new; Theorem 1 is standard IFT/contraction repackaged; cross-architecture extension is the chain rule. Concentrated novelty in the constrained projection and its validation. |
| Literature Coverage | 62 | Seminal works well-covered; missing spectral stability analysis (Kenlay et al.), curvature/over-squashing connection (Topping et al., Di Giovanni et al.), and 2024 certified robustness extensions. Gama et al. cited but underutilized. |
| Domain Accuracy | 78 | GNN robustness claims are mostly correct; the certificate comparison (Sec. V-B) is misleading on the semantics of $r_v$ vs. smoothing radii; the introduction overstates the gap. |
| Theoretical Contribution | 52 | Theorem 1 is standard contraction mapping + IFT; Observation 1 (nonnormality bound) is a nice observation but follows from Kronecker eigenstructure; Proposition 3 (continuous-to-discrete bridge) has limited practical predictive power. The constrained projection $S_c$ is the real theoretical contribution. |
| Practical Relevance | 75 | The vulnerability ranking tool has clear deployment value; the power grid case study is compelling; the defense-informed edge protection (Sec. V-E) demonstrates actionable use. Limited by the continuous-perturbation-only formal guarantee and the IGNN accuracy penalty. |
| Overall Domain | 65 | A solid applied contribution with one genuinely novel technical idea ($S_c$), strong experiments, but overselling of theoretical novelty and incomplete literature engagement with the spectral stability analysis community. |

## Questions for Authors

1. **On the relationship to spectral stability analysis:** Gama et al. (2020) derive stability bounds for graph filters under graph perturbation of the form $\|Z' - Z\| \leq C \cdot \|A' - A\|$, where $C$ depends on the filter's Lipschitz constant and the graph's spectral properties. How does your $\sigma_1(S_c)$ relate to such spectral stability constants? Is $\sigma_1(S_c)$ a tighter, directional version of the worst-case Lipschitz bound? A formal comparison would significantly strengthen the positioning.

2. **On the practical significance of $\varepsilon_{\text{crit}}$:** The phase transition experiments (Sec. V-D) show that the actual spectral radius $\rho(J_z)$ saturates at $\approx 0.42$ even at $\kappa_{\max} = 0.99$, and the resolvent norm grows only $1.5\times$ (not $70\times$). This means $\varepsilon_{\text{crit}}$ is conservative by orders of magnitude. Given this, what is the practical value of $\varepsilon_{\text{crit}}$ as a deployment criterion? Would a practitioner ever use $\varepsilon_{\text{crit}}$ to decide whether to deploy a model, or is the first-order tightness at a chosen $\varepsilon$ the operationally relevant quantity?

3. **On the GCN-2 failure mode:** GCN-2 shows negative $\tau$ on Cora and Citeseer (Table VII). You attribute this to shallow depth. However, GCN-2 is the most widely deployed GNN architecture. If the $S_c$ framework cannot produce meaningful vulnerability rankings for 2-layer GCNs, this is a significant practical limitation. Have you investigated whether the failure is due to the small receptive field (aggregation over only 2-hop neighborhoods) or the near-uniform sensitivity landscape of shallow models? Would using $S_c$ from APPNP or GCN-4 to rank edges for a GCN-2 model (cross-architecture transfer) be a viable workaround?

4. **On curvature and vulnerability:** Recent work on over-squashing (Topping et al., 2022; Di Giovanni et al., 2023) shows that edges with negative Ricci curvature are information bottlenecks. Do high-$v_{ij}$ edges from AEGIS correlate with negatively-curved edges? If so, this would provide an independent geometric interpretation of vulnerability and connect your work to a growing theoretical literature.

5. **On the threat model's practical realism:** The continuous edge-weight perturbation model (Sec. II-B) is analytically convenient but does not match real-world attack scenarios, which involve discrete edge insertions/deletions. While Proposition 3 bridges continuous-to-discrete rankings, the formal guarantee (Theorem 1, including $\varepsilon_{\text{crit}}$ and the three regimes) holds only for continuous perturbations. How should a practitioner interpret the formal guarantees when the actual threat involves discrete topology changes?

## Recommendation

**Minor Revision.**

The paper makes a genuine contribution to the GNN robustness analysis toolkit through the constrained sensitivity matrix $S_c$ and its comprehensive empirical validation across 7 architectures and 9 datasets. The power grid case study is a compelling real-world application. The experimental methodology is among the strongest I have seen in this area.

However, the paper needs revision on three fronts: (1) the theoretical novelty is oversold — Theorem 1 should be reframed as a specialization of known results, with $S_c$ clearly identified as the primary contribution; (2) the literature gap claimed in the introduction is overstated and must be corrected to acknowledge existing per-edge and per-node information from attacks and localized smoothing; (3) the comparison with smoothing certificates must be clarified to avoid conflating first-order approximations with formal guarantees. Additionally, the missing references to spectral stability analysis (Kenlay et al., 2021) and curvature-based bottleneck analysis (Topping et al., 2022; Di Giovanni et al., 2023) should be addressed, as they provide the closest theoretical parallels to AEGIS's per-edge sensitivity analysis.

With these revisions — primarily rewriting/repositioning rather than new experiments — the paper would be a strong contribution to a top-tier data mining venue.
