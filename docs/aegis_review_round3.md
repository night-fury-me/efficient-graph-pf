# AEGIS Paper — Simulated Review Round 3

**Paper:** "AEGIS: Mining Graph Structure for Adversarial Vulnerability Analysis of Graph Neural Networks"
**Target Venue:** ICDM 2026 (IEEE International Conference on Data Mining)
**Review Date:** 2025-05-25
**Mode:** Full (5 reviewers)

---

## Phase 0: Field Analysis & Reviewer Configuration

**Primary Discipline:** Machine Learning / Graph Neural Networks
**Secondary Discipline:** Adversarial Robustness, Sensitivity Analysis
**Research Paradigm:** Theoretical + Empirical (hybrid)
**Methodology Type:** Mathematical framework with experimental validation
**Target Journal Tier:** A* conference (ICDM)
**Paper Maturity:** Near-final (post-revision)

### Reviewer Panel Configuration

| # | Role | Identity | Focus |
|---|------|----------|-------|
| R1 | Editor-in-Chief | Senior ICDM Area Chair, graph mining and adversarial ML | Novelty, significance, venue fit |
| R2 | Methodology | Faculty specializing in matrix perturbation theory, implicit networks | Mathematical rigor, proof correctness, reproducibility |
| R3 | Domain Expert | Researcher in GNN robustness, certified defenses | Literature positioning, incremental contribution |
| R4 | Perspective | Cross-disciplinary: power systems + ML | Practical impact, case study validity, broader applicability |
| R5 | Devil's Advocate | Adversarial reviewer targeting logical gaps | Core argument challenges, confirmation bias |

---

## Phase 1: Independent Reviews

---

### Reviewer 1 (Editor-in-Chief) — Novelty, Significance, Venue Fit

**Overall Score: 6.5/10 (Weak Accept)**

#### Summary
The paper introduces AEGIS, a structural vulnerability analysis framework for GNNs based on a constrained sensitivity matrix $S_c$. The framework applies to both implicit (via IFT) and explicit (via unrolled Jacobian) GNNs. Experiments span 7 architectures, 9 datasets, and a power grid case study.

#### Strengths
1. **Clear central object.** The constrained sensitivity matrix $S_c$ is a well-defined, elegant mathematical construction that unifies attack, ranking, and certification.
2. **Breadth of validation.** 7 GNN architectures, 9 datasets across 4 domains, 10 seeds each — thorough empirical evaluation for a 10-page paper.
3. **Cross-domain case study.** The power flow contingency application demonstrates real-world utility beyond standard benchmarks.
4. **Well-structured paper.** Clear pipeline description (Fig. 1), logical flow from theory to experiments, honest limitations section.

#### Weaknesses
1. **Venue fit concern.** ICDM emphasizes data mining applications and scalability. The $N \leq 200$ subgraph limitation and dense Jacobian computation ($O(D \cdot N^2)$) raise questions about applicability to large-scale graphs relevant to ICDM's audience (social networks, web graphs with millions of nodes).
2. **First-order tightness at $\varepsilon = 0.01$ is expected.** The authors acknowledge this but the main quantitative result (tightness 1.00) is a mathematical tautology for sufficiently small perturbations. The interesting regime (larger $\varepsilon$) shows degradation to 1.36.
3. **No comparison with node-level attacks.** Nettack and PGDAttack are standard baselines in ICDM adversarial papers. The Mettack comparison is brief and acknowledged as unfair (surrogate mismatch).

#### Questions for Authors
- Q1: What is the practical use case where a practitioner would use AEGIS over existing tools like Nettack or randomized smoothing?
- Q2: Can the subgraph approach be validated on a graph with >100K nodes?

#### Recommendation: Weak Accept
The paper is technically sound with a clean central contribution ($S_c$). The breadth of architectures is impressive for a framework paper. However, the scalability gap and the tautological nature of the small-$\varepsilon$ tightness result somewhat weaken the empirical contribution for ICDM's applied audience.

---

### Reviewer 2 (Methodology) — Mathematical Rigor & Reproducibility

**Overall Score: 7.0/10 (Accept)**

#### Summary
The theory section presents Theorem 1 (vulnerability characterization via three regimes), Proposition 2 (SVD-optimal attack), Proposition 3 (per-node radius), and Proposition 4 (generalization to explicit GNNs). The mathematical framework is built on the implicit function theorem and operator-norm contractivity.

#### Strengths
1. **Correct use of operator norm.** Using $\kappa = \|J_z\|_2$ instead of spectral radius $\rho(J_z)$ is the right choice for Neumann series convergence. The remark on conservativeness is appropriate.
2. **Clean proof structure.** Theorem 1 follows logically from A1-A3 using standard perturbation theory. The three-regime characterization (subcritical/critical/supercritical) is well-motivated.
3. **Honest conservativeness discussion.** The authors clearly state that $\varepsilon_{\text{crit}}$ is sufficient, not necessary, and that the bound uses worst-case $\|\cdot\|_F$ relaxation.
4. **Proposition 4 elegantly extends.** The unrolled Jacobian bound $\sigma_1(S_K) \leq \sum_{l=1}^K (\prod_{k=l+1}^K \|J_z^{(k)}\|_2) \|J_A^{(l)}\|_2$ correctly generalizes via telescoping.

#### Weaknesses
1. **Gap between theory and experiments for explicit GNNs.** Proposition 4 provides only an upper bound on $\sigma_1(S_K)$, but the experiments use finite differences to compute $S_K$ directly. The theoretical bound is never evaluated or compared to the actual $\sigma_1$. How tight is the Proposition 4 bound?
2. **Assumption A3 is restrictive.** $\|J_z\|_2 \leq \kappa < 1$ holds by construction for IGNN (spectral normalization), but the paper claims generality. For explicit GNNs, there is no such guarantee — the theoretical results (Theorem 1) do not apply.
3. **Per-node radius denominator.** The formula $r_v = m_v / (\|W_{y_v} - W_{c^*}\|_2 \cdot \|S_v\|_2)$ assumes a linear classification head. Many GNN architectures (especially GAT) use nonlinear heads or multi-hop attention. Is this assumption validated for all 7 architectures?
4. **Missing second-order analysis.** The paper acknowledges the first-order limitation but provides no analysis of the remainder term $O(\|\delta A\|_F^2)$. At $\varepsilon = 0.20$, tightness degrades to 1.36 — is this because the second-order term dominates, or because the perturbation leaves the linear regime?

#### Minor Issues
- The pseudospectral index $\eta = 1.02$-$1.28$ is reported but its computation is not described.
- The notation switches between $\hat{A}$ (normalized) and $A$ (raw) without always being explicit.

#### Reproducibility Assessment
- Seeds listed (10 specific values): good.
- Hyperparameters specified: good.
- Code promised but not yet available: cannot verify implementation correctness.
- GPU memory estimate (6.5 GB at $N=200$): helpful.

#### Recommendation: Accept
The mathematical framework is sound and correctly executed. The operator-norm choice, proof strategy, and extensions are technically solid. The gap between theoretical bounds and experimental practice for explicit GNNs is a minor concern that could be addressed with a table comparing the Prop. 4 bound to actual $\sigma_1(S_K)$.

---

### Reviewer 3 (Domain Expert) — Literature & Contribution

**Overall Score: 6.0/10 (Borderline)**

#### Summary
The paper positions itself in the intersection of structural attacks, certified robustness, and sensitivity analysis. With 56 citations across 7 thematic threads, the literature coverage is comprehensive.

#### Strengths
1. **Strong literature coverage.** 56 citations spanning attacks (Nettack, Mettack, topology attack), certified methods (Bojchevski, Zugner IBP, smoothing variants), implicit networks (DEQ, IGNN, EIGNN), and sensitivity analysis. The related work is well-organized.
2. **Clear positioning.** The paper correctly identifies its niche: structural vulnerability *analysis* (not defense), per-edge differentiation (not uniform), and architecture-agnostic (not model-specific).
3. **Comparison with smoothing.** The complementarity argument (smoothing = uniform probabilistic, AEGIS = differentiated deterministic) is well-articulated.

#### Weaknesses
1. **Incremental over IFT sensitivity.** The core computation $S = (I - J_z)^{-1} J_A$ is a direct application of the implicit function theorem to IGNN. The projection to $S_c$ (summing columns for symmetry) is straightforward. What is the *intellectual novelty* beyond "apply IFT, then project to edges"?
2. **Explicit GNN extension is shallow.** For GCN/GIN/SAGE/APPNP, AEGIS reduces to computing a finite-difference Jacobian — a standard numerical technique. The "unrolled sensitivity" $S_K$ in Proposition 4 is just the chain-rule Jacobian. The paper overstates this as a "generalization" when it is simply a different (simpler) computation.
3. **Missing comparison with GNNExplainer/GradCAM.** Per-edge vulnerability ranking is functionally similar to edge importance scores from GNN explanation methods (GNNExplainer [Ying et al., 2019], PGExplainer [Luo et al., 2020], GradCAM [Pope et al., 2019]). The paper does not discuss or compare against these.
4. **No defense integration.** The paper claims AEGIS could inform defense design but provides no experiment. GNNGuard, Pro-GNN, or even simply removing top-$k$ vulnerable edges would be a natural experiment to include.
5. **GAT limitation undermines generality claim.** Requiring "edge-weighted GAT" (a non-standard modification) means AEGIS does not apply to vanilla GAT, which is one of the most popular architectures.

#### Missing References
- Ying et al., 2019 — GNNExplainer
- Luo et al., 2020 — PGExplainer
- Pope et al., 2019 — Explainability Methods for GNNs
- Feng et al., 2022 — KerGNNs (kernel perspective on GNN robustness)
- Mujkanovic et al., 2022 — Are Defenses for GNNs Robust?

#### Recommendation: Borderline
The paper is technically competent and well-written, but the core contribution (IFT + edge projection) may be viewed as incremental by adversarial robustness specialists. The explicit GNN extension strengthens the paper but is computationally trivial (FD Jacobian). Missing comparison with explainability methods is a gap.

---

### Reviewer 4 (Perspective) — Cross-Disciplinary Impact & Practical Applicability

**Overall Score: 7.0/10 (Accept)**

#### Summary
The power flow case study connects AEGIS to a well-established engineering problem (N-1 contingency analysis). The cross-disciplinary application is the most distinctive aspect of this paper relative to standard adversarial robustness literature.

#### Strengths
1. **Genuine cross-domain validation.** Using IEEE standard test cases (14, 30, 57, 118 buses) is rigorous — these are established benchmarks in power systems research.
2. **Operationally relevant metrics.** Precision@10 (0.66-0.81) measures what matters: can AEGIS identify the top critical lines? The answer is yes, with meaningful improvement over the effective-resistance baseline.
3. **Honest assessment of limitations.** Kendall $\tau = 0.37$-$0.67$ is acknowledged as insufficient for standalone use — AEGIS as a screening layer, not replacement, is the right framing.
4. **Binary vs. weighted adjacency insight.** The finding that binary adjacency outperforms admittance-weighted is counterintuitive and practically useful.

#### Weaknesses
1. **Case study is disconnected from main narrative.** The paper's core contribution is the $S_c$ framework for adversarial vulnerability. The power flow application is interesting but not adversarial — it is more of a "sensitivity analysis as anomaly detection" story. The connection between "adversarial perturbation" and "contingency analysis" is conceptual, not formal.
2. **Small grid sizes.** IEEE case118 (118 buses, 179 edges) is a toy grid by modern standards. Real transmission grids have 5,000-50,000 buses. The subgraph limitation ($N \leq 200$) is not a problem here, but scalability to real grids is undemonstrated.
3. **Missing operational baselines.** Power systems have well-established screening tools: LODF (Line Outage Distribution Factor), PTDF (Power Transfer Distribution Factor), and generation shift factors. Only effective resistance is compared. LODF is the standard screening proxy in industry.
4. **No dynamic contingency.** N-1 is static; operators increasingly care about N-1-1 (cascading) and dynamic stability. AEGIS is inherently static (first-order, steady-state).

#### Questions
- Q3: Has the binary-outperforms-weighted finding been verified on real (non-synthetic) grid data?
- Q4: How does AEGIS-based screening compare to LODF in terms of computational cost? (LODF is $O(|E|)$ from pre-computed PTDF.)

#### Recommendation: Accept
The cross-domain story elevates this paper above a standard adversarial robustness contribution. The honesty about limitations and the counterintuitive binary-adjacency finding add genuine scientific value. Scalability to real grids remains a concern for follow-up work.

---

### Reviewer 5 (Devil's Advocate) — Core Argument Challenges

**Overall Score: 5.5/10 (Weak Reject)**

#### Strongest Counter-Argument (250 words)

The paper's central claim is that AEGIS provides "architecture-agnostic structural vulnerability analysis for any differentiable GNN." This claim rests on two legs: (1) the implicit case via IFT, and (2) the explicit case via $S_K$. Leg (1) is mathematically sound but applies only to a niche architecture family (IGNN) that is rarely used in practice. Leg (2) is a finite-difference Jacobian computation — a generic numerical technique that requires no framework, no theory, and no "AEGIS." Any practitioner can compute $\partial Z_K / \partial A$ with a few lines of PyTorch autograd. The $S_c$ projection (sum columns for symmetry) adds minimal intellectual content.

The paper therefore faces a dilemma: the *interesting* theoretical results (Theorem 1, $\varepsilon_{\text{crit}}$, three regimes) apply only to IGNN, while the *general* claim (any GNN) reduces to a trivial computation (autograd + column sum). The "7 architectures" headline obscures this: 6 of 7 use the trivial path. The paper would be more honest positioned as "deep equilibrium sensitivity analysis with a computational recipe for other architectures" — but that was the previous version's framing, which reviewers rejected as too narrow.

This is not a fatal flaw, but it is a fundamental tension that the paper's current positioning does not resolve.

#### Issue List

| # | Severity | Dimension | Location | Issue |
|---|----------|-----------|----------|-------|
| 1 | CRITICAL | Logic | Abstract, Intro, Conclusion | **Generality claim is misleading.** "Any differentiable GNN" technically true, but the non-trivial theory (Theorem 1, $\varepsilon_{\text{crit}}$, convergence guarantees) applies only to IGNN. For other GNNs, AEGIS offers no theoretical guarantees beyond $\sigma_1(S_K)$ being an upper bound on first-order shift. |
| 2 | MAJOR | Methodology | §5.7 (Explicit Extension) | **Finite-difference Jacobian is not a contribution.** Computing $S_K$ via FD is standard numerical analysis. It is unclear what AEGIS adds over `torch.autograd.functional.jacobian`. |
| 3 | MAJOR | Logic | §4.4 (Stage 4) | **Per-node radii are not certificates.** The paper alternates between calling them "first-order radii" (correct) and implying they provide robustness guarantees. Without a second-order remainder bound, they are sensitivity measures, not certificates. The 0% breach rate at $\varepsilon = 0.01$ does not validate them as certificates — it validates that small perturbations cause small shifts. |
| 4 | MAJOR | Cherry-picking | §5.1 | **Tightness at $\varepsilon = 0.01$ is cherry-picked.** This is the smallest budget tested. At realistic attack budgets ($\varepsilon \geq 0.10$), tightness degrades to 1.15-1.39. The abstract and introduction lead with 1.00, burying the degradation in a table. |
| 5 | MINOR | Overgeneralization | §5.8 (Power Flow) | **N-1 contingency analogy is strained.** Continuous edge-weight perturbation $\neq$ discrete edge removal. The mapping between $v_{ij}$ and true contingency severity is approximate and the $\tau$ correlation confirms this (0.37-0.67). |
| 6 | MINOR | Logic | §3.3 (Threat Model) | **Continuous perturbation model is unrealistic.** Real graph attacks add/remove discrete edges. Continuous edge-weight changes assume a relaxed threat model that may not reflect realistic adversaries. |

#### Ignored Alternative Explanations
- The high AtkAdv ratio (2-8x over random) may simply reflect that *any* gradient-based attack outperforms random — this is true for image adversarial examples too. It does not demonstrate that $S_c$ specifically is necessary.
- WikiCS showing near-perfect tightness even at $\varepsilon = 0.20$ (1.05) may indicate that WikiCS has a particularly well-conditioned adjacency, not that AEGIS is universally accurate.

#### Missing Stakeholder Perspectives
- Defense practitioners: no experiment shows AEGIS improving any defense.
- GNN explanation community: strong overlap with edge attribution methods, no comparison.

#### "So What?" Test
If a GNN practitioner has AEGIS, what can they *do* that they couldn't before? They get vulnerability rankings — but GNNExplainer already provides edge importance. They get radii — but these are first-order only and conservative. They get an optimal attack — but it requires continuous perturbation in the normalized adjacency, which is hard to deploy as a real attack. The killer application remains unclear.

#### Recommendation: Weak Reject
The fundamental tension between the generality claim and the theory's applicability is unresolved. The paper is technically sound but overpromises. If repositioned more modestly — as a sensitivity analysis framework with strong IGNN-specific theory — the contribution would be more defensible, but that framing was already tried and found too narrow.

---

## Phase 2: Editorial Synthesis & Decision

### Consensus Points (All/Most Reviewers Agree)
1. **Paper is technically sound.** No mathematical errors detected in the proofs (R2 confirms). The operator-norm choice is correct.
2. **Comprehensive experiments.** 7 architectures, 9 datasets, 10 seeds is thorough (R1, R2, R4 agree).
3. **Good writing quality.** Well-structured, honest limitations, clear pipeline description (R1, R4).
4. **Scalability is a limitation.** $N \leq 200$ subgraphs limit applicability (R1, R4, R5).
5. **Explicit GNN extension is computationally trivial.** FD Jacobian is standard (R3, R5 agree).

### Disagreement Points
| Issue | For | Against |
|-------|-----|---------|
| Novelty sufficient for ICDM | R2, R4 (the $S_c$ projection + cross-domain application is novel) | R3, R5 (IFT application + trivial FD = incremental) |
| Power flow case study value | R4 (genuine cross-domain, distinguishing feature) | R5 (strained analogy, disconnected from adversarial narrative) |
| Generality claim | R1 (technically supported, well-qualified) | R5 (misleading: nontrivial theory is IGNN-only) |

### Devil's Advocate CRITICAL Issue Resolution

**Issue #1 (Generality claim is misleading):** This is the paper's central tension. The EIC notes that the abstract already qualifies the claim ("additionally yields a critical budget" for implicit models), but agrees with R5 that the framing overstates the explicit-GNN contribution. **Resolution:** The generality claim should more clearly acknowledge that for explicit GNNs, AEGIS provides a *computational recipe* (not theoretical guarantees), while for implicit GNNs it provides both.

Per Iron Rule #4: Devil's Advocate found one CRITICAL issue → **Decision cannot be Accept.**

---

### Editorial Decision: **Minor Revision**

#### Decision Rationale
The paper presents a technically sound and well-validated framework with a clean central contribution ($S_c$). The experiments are comprehensive and the cross-domain case study is distinctive. However, the tension between the generality claim and the theory's actual scope (flagged as CRITICAL by Devil's Advocate) must be addressed before acceptance.

#### Priority Revision Roadmap

##### P0 — Must Fix (blocking acceptance)

| # | Issue | Source | Action Required |
|---|-------|--------|-----------------|
| 1 | Generality framing overstates explicit-GNN contribution | R5 (CRITICAL), R3 | Revise abstract/intro/conclusion to clearly distinguish: (a) $S_c$ as a *computational tool* for any GNN, vs. (b) theoretical guarantees (Theorem 1, $\varepsilon_{\text{crit}}$) that apply only to contractive implicit models. Remove or qualify "any differentiable GNN" where it implies theoretical backing for explicit GNNs. |
| 2 | Clarify "radii" are not certificates | R5 (#3) | Ensure *every* mention of $r_v$ explicitly states "first-order" or "local sensitivity radius." Never imply global guarantee without second-order bound. Check abstract for unqualified "robustness radii." |

##### P1 — Strongly Recommended

| # | Issue | Source | Action Required |
|---|-------|--------|-----------------|
| 3 | Compare with explanation methods | R3, R5 | Add 2-3 sentences in Related Work discussing GNNExplainer/PGExplainer and how AEGIS differs (perturbation-space optimality vs. attribution). Ideally, add one experiment comparing edge rankings. |
| 4 | Tightness presentation is front-loaded with $\varepsilon = 0.01$ | R5 (#4) | In abstract and intro, also mention the $\varepsilon = 0.10$ tightness (1.15) alongside the 1.00 result. Currently the degradation is buried. |
| 5 | Evaluate Proposition 4 bound tightness | R2 (#1) | Add a row to Table VII showing the Prop. 4 upper bound vs. actual $\sigma_1(S_K)$ for each architecture. Shows whether the bound is useful or vacuous. |
| 6 | LODF baseline for power flow | R4 (#3) | Add LODF as a screening baseline. It is the industry standard and takes one line to compute from the B matrix. |

##### P2 — Nice to Have

| # | Issue | Source | Action Required |
|---|-------|--------|-----------------|
| 7 | Defense integration experiment | R3 (#4) | Remove top-$k$ AEGIS-ranked edges before attack; measure if vulnerability decreases. Simple experiment, high impact on narrative. |
| 8 | Scalability to larger graphs | R1, R4 | Test on OGB-arxiv or a larger power grid (Polish/European test cases). Even a single result at $N > 200$ via sparse approximation would address concern. |
| 9 | GAT without modification | R3 (#5) | Discuss in more detail why standard GAT fails and whether attention-weighted formulations (GATv2) naturally admit continuous sensitivity. |
| 10 | Practical use case articulation | R5 ("So What?") | Add a paragraph in the introduction or conclusion articulating the concrete workflow: who uses AEGIS, when, and what decision does it inform? |

---

### Score Summary

| Reviewer | Score | Confidence | Recommendation |
|----------|-------|------------|----------------|
| R1 (EIC) | 6.5 | High | Weak Accept |
| R2 (Methodology) | 7.0 | High | Accept |
| R3 (Domain) | 6.0 | Medium | Borderline |
| R4 (Perspective) | 7.0 | High | Accept |
| R5 (Devil's Advocate) | 5.5 | High | Weak Reject |

**Aggregate: 6.4/10 — Minor Revision (conditional acceptance)**

The paper will be accepted upon satisfactory resolution of P0 items. P1 items would elevate the paper from borderline to solid accept territory.

---

*Review generated: 2025-05-25 | Simulated panel for ICDM 2026 | Round 3 (post-scope-broadening)*
