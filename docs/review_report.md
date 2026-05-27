# AEGIS: Full Peer Review Report

## Editorial Decision Package — Phase 2 Synthesis

**Paper:** AEGIS: Mining Graph Structure for Adversarial Vulnerability Analysis of GNNs
**Date:** 2026-05-26
**Reviewers:** 5 (EIC + 3 Peer Reviewers + Devil's Advocate)

---

## Panel Summary

| Reviewer | Role | Score | Recommendation |
|----------|------|-------|----------------|
| Prof. Elena Marchetti (EIC) | Venue fit, significance, clarity | 7.5/10 | Minor Revision |
| Dr. Ravi Chandrasekaran (R1) | Methodology, proofs, statistics | 7/10 | Minor Revision |
| Prof. Stephan Gunnemann (R2) | Domain, literature, baselines | 7/10 | Minor Revision |
| Dr. Amara Nakiganda (R3) | Power systems, cross-disciplinary | 6/10 | Major Revision |
| Dr. Marcus Chen (DA) | Core argument challenges | 3 CRITICAL / 6 MAJOR | Cannot Accept |

---

## Consensus Issues (4+ reviewers agree)

### 1. case300 Model Quality Invalidates Power Flow Claims [CRITICAL — 5/5 agree]

**Source:** EIC W3, R1 W5, R2 W3, R3 W1, DA Issue 6

Table 7 shows case300 has ΔS = 10.9 p.u. (100× worse than case14–118) and θ RMSE = 0.394 rad (~22°), yet the paper reports τ = +0.72 and P@10 = 0.87. All five reviewers independently flagged this as the single most problematic result. The high τ likely reflects topological structure captured by the 200-node BFS subgraph, not learned power flow physics. The footnote ("1,000 training samples") is easy to miss.

### 2. Pubmed Tightness Discrepancy with Abstract Claims [CRITICAL — 3/5 flag]

**Source:** EIC Minor, R2 Minor, DA Issue 1

Table 1 reports Pubmed tightness = 1.01 ± 0.01; Table 2 reports 0.79 ± 0.01 at the same ε = 0.01. Amazon Photo shows a similar gap (1.00 vs. 0.92). The abstract claims "1.00 ± 0.01 at ε = 0.01" without qualification. The discrepancy is never explained. If the two tables use different experimental setups, this must be stated explicitly.

### 3. Continuous Threat Model Limits Practical Scope [MAJOR — 4/5 flag]

**Source:** EIC W1, R2 Threat Model (6/10), R3 Practical Applicability (4/10), DA Issue 5

The continuous, edge-only, no-insertion perturbation model is analytically elegant but misaligned with standard discrete flip attacks (Nettack/Mettack). The paper's own continuous-to-discrete transfer shows 4/33 architecture-dataset combinations with negative τ. Edge insertions — often the most damaging attack vector — are excluded by construction.

### 4. IGNN Accuracy Gap Undermines Deployment Argument [MAJOR — 4/5 flag]

**Source:** EIC W2, R2 W6, DA Issue 6, R3 (implicit)

IGNN achieves 77.5% on Cora vs. 82.2% APPNP. The full theoretical apparatus (Theorem 1, ε_crit, three regimes) applies only to IGNN. Practitioners will choose higher-accuracy models, receiving only the computational tool without formal guarantees — which is the paper's main theoretical contribution.

---

## Majority Issues (2–3 reviewers agree)

### 5. Degree Baseline Near-Equivalence Underexplored [MAJOR — 2/5 flag]

**Source:** R2 W2, DA Issue 4

Degree-proportional attack achieves 94–98% of AEGIS's damage on WikiCS (3.93 vs. 4.02). Degree requires zero model access and runs in milliseconds. The marginal gain of AEGIS (6–8%) may not be statistically significant. No formal significance tests are reported.

### 6. ReLU Non-Differentiability Undermines Remainder Bounds [MAJOR — 2/5 flag]

**Source:** R1 W1, R1 Theorem 1 analysis, DA (implicit via logic-gap)

Proposition 3's remainder bound requires Lipschitz continuity of J_z w.r.t. A (via L_J), but ReLU makes J_z piecewise constant with discontinuous jumps at activation boundaries. The estimate L_J ≈ ‖W‖₂² is heuristic. The paper needs either a smooth-activation assumption, a measure-theoretic argument, or a piecewise bound.

### 7. Amazon Photo IGNN Negative τ [CRITICAL per DA — 2/5 flag]

**Source:** R2 Claims vs. Evidence, DA Issue 2

IGNN on Amazon Photo has τ = −0.15, meaning AEGIS rankings are anti-correlated with discrete ground truth. This is the largest standard benchmark (7,650 nodes) and the regime (κ = 0.09) with the strongest formal guarantees (largest ε_crit). The logic chain breaks: strongest theory → worst practical output.

### 8. Adaptive Attack Evaluation Partly Circular [MAJOR per DA — 2/5 flag]

**Source:** DA Issue 3, R2 (implicit)

Both AEGIS and the equilibrium-shift PGD optimize the same objective using the same IFT gradients. AEGIS wins because SVD is the exact linearized solution while PGD is iterative. The classification-loss PGD partially addresses this, but a genuinely independent baseline (black-box search, Nettack on IGNN) is missing.

### 9. "Any GNN" Overclaim [MAJOR — 2/5 flag]

**Source:** R2 Novelty, DA Issue 5

Standard GAT, GraphSAGE with sampling, and GNNs with discrete aggregations fail the continuous differentiability requirement. The paper creates a non-standard GAT† variant. The claim should be "any GNN with continuous edge-weight-modulated message passing."

### 10. Unconstrained vs. Constrained Radius Ambiguity [MINOR — 1/5 flag]

**Source:** R1 W3

Proposition 2 states r_v using unconstrained S_v, but the constrained threat model should use S_{c,v}, yielding tighter (larger) radii.

---

## Reviewer-Specific Issues (single-reviewer, not duplicated above)

### From R1 (Methodology):
- Norm mixing in Theorem 1 contractivity preservation (Frobenius vs. operator norm)
- Proposition 3(b) condition w_{k1} ≥ w_{k2} conflicts with observed positive degree-vulnerability correlation
- No formal analysis of randomized SVD error propagation to v₁ stability
- Kronecker product vectorization convention unstated
- Pseudospectral index η defined non-standardly relative to Trefethen & Embree
- Breach rate reporting: mean ± std misleading for skewed distributions (Pubmed: 10.3 ± 11.0)

### From R2 (Domain):
- Missing references: Schuchardt et al. ICML 2023, Gosch et al. NeurIPS 2023, Geng et al. 2023, Gama et al. 2020, Pei et al. 2020
- No heterophilic dataset evaluation (Texas, Cornell, Actor)
- "Cert%" terminology misleading — should be "Coverage" to avoid confusion with formal certificates
- No comparison against topology-based certified robustness methods (Bojchevski & Gunnemann 2019, Wang et al. 2021)
- L_J estimation (≈ ‖W‖₂²) not validated empirically

### From R3 (Perspective):
- Training data generation unrealistic (uniform 70–130% load scaling only)
- No thermal/voltage violation analysis in power flow case study
- ε_crit framed as "stability margin" is misleading — conflates model robustness with physical stability
- Kirchhoff compliance comparison not iso-parameter (IGNN d=64 vs. PIGNN d=10)
- Missing comparison with AC-based contingency screening (fast-decoupled, PTDF)
- Small grid sizes (max 300 buses vs. real 5,000–70,000)
- Rank stability concerning on small grids (case14: τ = 0.40 ± 0.29)

### From DA (Devil's Advocate):
- Mettack comparison included despite being acknowledged as uninformative
- Phase transition never empirically observable — vacuous worst-case bound
- Effective resistance never compared across all datasets
- No model ensembling (degree + S_c) explored
- Dual-use risk inadequately addressed

---

## Disagreements and Arbitration

### Disagreement 1: Minor vs. Major Revision

EIC, R1, R2 recommend Minor Revision. R3 recommends Major Revision. DA finds 3 CRITICAL issues.

**Arbitration:** The DA's CRITICAL issues are procedurally binding (Iron Rule #4). However, assessing their substance:
- **Pubmed tightness discrepancy** (DA-CRITICAL-1): A presentation/disclosure issue, fixable by reconciling tables and qualifying the abstract. Not a fundamental flaw.
- **Amazon Photo negative τ** (DA-CRITICAL-2): A genuine limitation that the paper partially acknowledges. Fixable by prominent disclosure and theoretical explanation, but requires new analysis.
- **Circular adaptive attack** (DA-CRITICAL-3): Partially addressed by the classification-loss PGD. Fully fixable by adding one independent baseline.

R3's Major Revision call is driven by power systems engineering validity, which requires case300 model improvement and several missing comparisons — substantive work but feasible.

**Decision: The combination of DA CRITICALs and R3's engineering concerns pushes this to Major Revision.** The core framework is sound (all reviewers acknowledge this), but the presentation overclaims relative to evidence, and several experimental gaps need filling.

### Disagreement 2: Power Grid Case Study Value

R3 rates the case study 5–6/10 with specific engineering concerns. EIC and R2 find it compelling. DA calls τ = 0.37–0.67 insufficient.

**Arbitration:** R3's domain expertise is authoritative here. The case study is a valid proof-of-concept but overclaimed for operational utility. The fix is presentation: reframe as "structural screening tool" not "contingency assessment tool," and address the engineering gaps R3 identifies.

### Disagreement 3: Degree Baseline Threat

R2 and DA flag degree centrality as a near-equivalent baseline. EIC and R1 do not flag this.

**Arbitration:** The concern is legitimate for the vulnerability ranking component specifically. However, AEGIS provides three outputs (rankings, SVD attacks, radii) — degree centrality only approximates one. The paper should add statistical significance tests and a "value-added over degree" analysis to quantify when S_c provides meaningful improvement.

---

# EDITORIAL DECISION

## Decision: MAJOR REVISION

## Decision Rationale

AEGIS makes a genuine contribution to GNN vulnerability analysis: the constrained sensitivity matrix S_c is a novel and well-engineered construction that transforms vacuous unconstrained bounds into tight predictions under realistic perturbation constraints. The matrix-free pipeline is a solid engineering achievement, the experimental coverage across 7 architectures and 9 datasets is thorough, and the paper's intellectual honesty about limitations is exemplary.

However, three categories of concern prevent acceptance in the current form:

1. **Presentation integrity**: The Pubmed tightness discrepancy (Tables 1 vs. 2), the unqualified abstract claims, and the "any GNN" overclaim require reconciliation. These are fixable but essential for maintaining reader trust.

2. **Experimental gaps**: The near-equivalence of degree centrality (no significance tests), the absence of independent attack baselines, and the missing heterophilic benchmarks leave important questions unanswered.

3. **Power systems validity**: The case300 model quality (ΔS = 10.9 p.u.) invalidates claims in Section 7. R3's engineering concerns about training data realism, thermal limit awareness, and the ε_crit-as-stability-margin framing require substantive revision.

The theoretical framework and core empirical results are strong enough that a thorough revision should produce a publishable paper. The authors' existing habit of honest self-assessment provides a solid foundation for addressing these concerns.

---

# REVISION ROADMAP

## Priority 1: CRITICAL (Must Fix — blocks acceptance)

### C1. Reconcile Pubmed/Amazon tightness discrepancy
- **What**: Tables 1 and 2 report different tightness for Pubmed (1.01 vs. 0.79) and Amazon Photo (1.00 vs. 0.92) at ε = 0.01
- **How**: Explain experimental setup differences between tables. Qualify abstract claim to exclude outliers or report the full range
- **Effort**: Low (1–2 hours)
- **Flagged by**: EIC, R2, DA

### C2. Address Amazon Photo negative τ
- **What**: IGNN achieves τ = −0.15 on the largest standard benchmark, in the regime with strongest formal guarantees
- **How**: (a) Add theoretical explanation for why over-contraction causes near-uniform S_c columns, (b) Test whether explicit GNN rankings on Amazon Photo serve as a practical fallback, (c) State explicitly that AEGIS vulnerability rankings are unreliable for highly contractive models (κ < 0.15)
- **Effort**: Medium (1–2 days)
- **Flagged by**: R2, DA

### C3. Fix or remove case300 results
- **What**: ΔS = 10.9 p.u., θ RMSE = 0.394 rad — model quality is unacceptable
- **How**: Option A (preferred): Retrain with more data (≥2,000 samples), larger hidden dim, or physics-informed loss until ΔS < 0.5 p.u. Option B: Remove case300 from main results, place in supplementary as "topological-only analysis" with explicit caveat
- **Effort**: Medium (2–3 days for Option A)
- **Flagged by**: All 5 reviewers

### C4. Add independent attack baseline
- **What**: Adaptive attack comparison is partly circular (same IFT gradients, same objective)
- **How**: Add at least one of: (a) black-box evolutionary edge search, (b) Nettack adapted to target IGNN directly, (c) finite-difference per-edge ranking as a competing method. The classification-loss PGD partially addresses this but shares the differentiation pathway
- **Effort**: Medium (2–3 days)
- **Flagged by**: DA, R2

## Priority 2: MAJOR (Strongly recommended — strengthens paper significantly)

### M1. Degree baseline significance tests
- **What**: AEGIS vs. degree-proportional: 6–8% margin, no statistical tests
- **How**: Report paired Wilcoxon signed-rank test p-values across all datasets. If not significant on some datasets, acknowledge and characterize when S_c provides meaningful value over degree
- **Effort**: Low (half day)
- **Flagged by**: R2, DA

### M2. Qualify "any GNN" claim
- **What**: Standard GAT, max/min aggregations, hard attention fail differentiability requirement
- **How**: Replace "any GNN whose message passing is differentiable" with "GNNs with continuous edge-weight-modulated message passing" in abstract and intro. List incompatible architectures explicitly. Report standard GAT results (even if τ ≈ 0) alongside GAT†
- **Effort**: Low (1 hour)
- **Flagged by**: R2, DA

### M3. Address ReLU non-differentiability in Proposition 3
- **What**: L_J undefined at ReLU kinks; remainder bound formally requires smooth activation
- **How**: Option A: Add measure-theoretic argument (problematic paths have measure zero). Option B: State Prop. 3 for smooth activations, note empirical validation covers ReLU. Option C: Replace L_J with piecewise bound + argument bounding number of activation-pattern crossings
- **Effort**: Medium (1–2 days)
- **Flagged by**: R1

### M4. Use constrained S_{c,v} in Proposition 2
- **What**: Radius r_v uses unconstrained S_v but threat model is constrained
- **How**: State the constrained version (tighter radii). Compare unconstrained vs. constrained radii experimentally
- **Effort**: Low (half day)
- **Flagged by**: R1

### M5. Add heterophilic benchmarks
- **What**: All 5 graph benchmarks are homophilic
- **How**: Add at least one heterophilic dataset (Texas, Cornell, or Actor from Pei et al. 2020)
- **Effort**: Medium (1–2 days)
- **Flagged by**: R2

### M6. Reframe phase transition prominence
- **What**: Three-regime characterization is never empirically observable (ρ saturates at ~0.42)
- **How**: Present as "conservative theoretical safety boundary" in contributions, not as predictive behavior. Reduce prominence relative to S_c construction
- **Effort**: Low (1 hour)
- **Flagged by**: DA

### M7. Reframe power flow ε_crit
- **What**: ε_crit framed as "stability margin" conflates model robustness with physical stability
- **How**: Rename to "model sensitivity threshold." Add paragraph distinguishing from voltage/transient/frequency stability margins
- **Effort**: Low (1 hour)
- **Flagged by**: R3

### M8. Report Prop. 3(b) condition coverage
- **What**: Ranking preservation condition (w_{k1} ≥ w_{k2} when v_{k1} > v_{k2}) may rarely hold
- **How**: Report fraction of edge pairs satisfying the condition across datasets
- **Effort**: Low (half day)
- **Flagged by**: R1

## Priority 3: MINOR (Recommended — polish and completeness)

### m1. Add missing references (R2): Schuchardt et al. ICML 2023, Gosch et al. NeurIPS 2023, Geng et al. 2023, Gama et al. 2020

### m2. Replace "Cert%" with "Coverage" throughout (R2)

### m3. Report breach rates as median + IQR for skewed distributions (R1)

### m4. State Kronecker product vectorization convention (R1)

### m5. Clarify pseudospectral index η definition vs. Trefethen & Embree (R1)

### m6. Compare standard GAT results alongside GAT† (R2)

### m7. Add BFS center sensitivity analysis (random vs. highest-degree center) (R1)

### m8. Discuss singular value gap σ₁ − σ₂ and v₁ stability (R1)

### m9. Quantify ε_crit Frobenius-vs-operator norm gap empirically (R1, DA)

### m10. Strengthen ethical considerations: consider tiered release protocol (R3, DA)

### m11. Move Mettack comparison to supplementary or reframe as sanity check (DA)

### m12. Report wall-clock time for full pipeline (training + S_c) in power flow comparison (R2)

---

## Estimated Revision Effort

| Priority | Items | Est. Time |
|----------|-------|-----------|
| Critical (C1–C4) | 4 | 1–2 weeks |
| Major (M1–M8) | 8 | 1–2 weeks |
| Minor (m1–m12) | 12 | 2–3 days |
| **Total** | **24** | **3–4 weeks** |

---

## What We Liked (Consensus Strengths)

All 5 reviewers independently praised:

1. **Intellectual honesty** — The paper consistently identifies its own limitations, flags when comparisons are uninformative (Mettack mismatch), and distinguishes what holds for IGNN vs. explicit GNNs. This is rare and commendable.

2. **Experimental thoroughness** — 7 architectures × 9 datasets × 10 seeds, with structured baselines, adaptive attacks, ablations, and honest reporting of negative results (GCN-2 negative τ, Amazon Photo failure).

3. **Unified three-output design** — Attacks, rankings, and radii from a single S_c computation is genuinely useful compared to running separate tools.

4. **Matrix-free scalability** — The Neumann + randomized SVD pipeline enabling N = 7,650 analysis is a solid engineering contribution.

5. **S_c constrained projection** — The N² → |E| reduction is the central technical contribution and has no precedent in GNN robustness literature.

---

## Individual Review Reports

The full text of all 5 review reports follows below for reference.

---

# EIC Review Report — Prof. Elena Marchetti

### Summary

AEGIS introduces a constrained sensitivity matrix S_c that projects the full adjacency perturbation space onto realistic (symmetric, edge-only) perturbations, enabling extraction of SVD-optimal attacks, per-edge vulnerability rankings, and per-node sensitivity radii from a single computation. The framework applies as a practical tool to any edge-weight-differentiable GNN and provides additional formal guarantees (critical budget, three vulnerability regimes) for contractive implicit models. Validation spans 7 architectures, 9 datasets across 4 domains, and includes a power grid contingency case study demonstrating cross-domain transfer.

### Venue Fit & Significance — 8/10

This paper addresses a genuine gap between adversarial attack methods (which find damaging perturbations but lack optimality guarantees) and certified defenses (which provide uniform certificates but no structural vulnerability maps). The problem of identifying which edges and nodes are structurally vulnerable before deployment is practically important for safety-critical GNN applications. The cross-domain scope and breadth of architectures tested position this as a systems-level contribution. The power grid case study connecting to N-1 contingency analysis is a compelling demonstration of real-world relevance.

### Originality — 7/10

The authors are commendably transparent that the individual mathematical ingredients are classical (IFT resolvent, Neumann series, SVD, Banach contraction). The novelty lies in the constrained projection S_c, the three-output design, and the cross-architecture pipeline. This is a "novel combination" contribution rather than fundamentally new theory. The matrix-free formulation is a solid engineering contribution. The continuous-to-discrete transfer result (Proposition 3) is useful though conditions are restrictive.

### Clarity & Presentation — 9/10

Exceptionally well-written. The paper maintains consistent scope delineation between IGNN guarantees and explicit GNN tools. Tables are well-designed. The "certificate semantics" paragraph carefully prevents misinterpretation.

### Impact Potential — 7/10

The S_c framework fills a practical niche for pre-deployment vulnerability screening. However: (1) strongest formal guarantees limited to IGNN, (2) scalability ceiling at N ≈ 7,650, (3) continuous perturbation model doesn't address discrete attacks.

### Key Strengths
1. Unified three-output framework (Section 3, 5)
2. Extraordinary experimental thoroughness (330 runs in Table 5 alone)
3. Intellectual honesty throughout
4. Matrix-free scalability to N = 7,650 (Table 6)
5. Cross-domain validation including power grids (Section 6)

### Key Weaknesses
1. Continuous perturbation model limits practical threat coverage (Section 2.2)
2. IGNN accuracy gap undermines deployment argument (Table 8)
3. case300 model quality poor (Table 7, ΔS = 10.9 p.u.)
4. Scalability ceiling at N ≈ 7,650 (Table 6)

### Overall Score: 7.5/10 | Recommendation: Minor Revision | Confidence: High

---

# Methodology Review Report — Dr. Ravi Chandrasekaran (R1)

### Summary

AEGIS derives S_c from the IFT applied to GNN equilibria, reducing N² to |E| dimensions. The theoretical contribution is cleanest for contractive IGNNs; extension to explicit GNNs is honest. Mathematically competent and self-aware about limitations, though several proof steps deserve tighter treatment.

### Theoretical Rigor — 7/10

**Theorem 1**: Parts (a) and (b) are sound. Part (a) has a norm-mixing issue (‖A‖₂ vs ‖δA‖_F) that is valid but potentially loose. Part (b)'s Ω(1/(ε_crit − ε)) rate assumes linear relationship between ε and ‖J_z'‖₂, which ReLU complicates. Part (c) is vacuous but honest. Hidden assumption: J_z treated as fixed at Z* but ReLU causes discontinuous jumps.

**Proposition 1**: Completely standard and correct (SVD variational characterization).

**Proposition 2**: Sound but uses unconstrained S_v instead of constrained S_{c,v}.

**Proposition 3**: Most technically interesting but most problematic. L_J undefined for ReLU. The ranking condition w_{k1} ≥ w_{k2} conflicts with observed degree-vulnerability correlation.

**Observation 1**: Correctly labeled, standard chain rule. No issues.

### Computational Methods — 8/10

Neumann-series resolvent, adaptive depth, randomized SVD — all well-justified. Memory scaling to 6.5 GB for Amazon Photo exceeds the O(Nd) claim (autograd tape overhead).

### Statistical Validity — 7/10

10 seeds with explicit values, appropriate confidence intervals. Missing: formal significance tests, multiple comparison correction, median/IQR for skewed breach distributions.

### Key Strengths
1. S_c constrained projection (Eq. 5) — central novel contribution
2. Honest scope delineation (Theorem 1 vs. Observation 1)
3. Matrix-free pipeline engineering
4. Comprehensive ablation
5. Adaptive attack evaluation
6. Cross-architecture validation (330 runs)

### Key Weaknesses
1. ReLU non-differentiability undermines Prop. 3 remainder bounds
2. Norm mixing in Theorem 1 contractivity preservation
3. Proposition 2 uses unconstrained S_v
4. Proposition 3(b) condition rarely holds in practice
5. case300 model quality poor
6. No randomized SVD error propagation analysis
7. Breach rate reporting misleading for skewed distributions

### Overall Score: 7/10 | Recommendation: Minor Revision | Confidence: High

---

# Domain Review Report — Prof. Stephan Gunnemann (R2)

### Summary

AEGIS proposes S_c for structural vulnerability analysis of GNNs, validated across 7 architectures and 9 datasets. Thorough and honest, though the threat model and positioning warrant scrutiny.

### Literature Coverage — 8/10
Key works covered. Missing: Schuchardt et al. (ICML 2023), Gosch et al. (NeurIPS 2023), Geng et al. (2023), Gama et al. (2020), Pei et al. (2020).

### Novelty & Positioning — 7/10
S_c constrained projection is genuinely new. IFT resolvent analysis is standard in DEQ literature. Continuous-to-discrete transfer (Prop. 3) is the most novel theoretical element.

### Threat Model Assessment — 6/10
Continuous edge-weight perturbations are unrealistic for most graph domains. No edge insertions. Perturbation on normalized Â, not raw A. Appropriate for power grid domain but not standard adversarial robustness.

### Baseline & Evaluation Adequacy — 7/10
Strong structured baselines and adaptive attackers. Missing: Nettack, topology-based certified robustness, GNN Lipschitz analysis, PTDF. Degree baseline within 6–8% is concerning. No heterophilic datasets.

### Claims vs. Evidence — 7/10
Well-supported: tightness, SVD advantage, cross-architecture, scalability, defense. Overclaimed: "mathematically optimal" (first-order only), "formal guarantees" (empirically vacuous), case300 results, "without line-impedance data" (requires physics simulator for training).

### Key Strengths
1. Honest self-assessment throughout
2. Comprehensive experimental design
3. Matrix-free scalability
4. Continuous-to-discrete transfer theory
5. Power grid case study

### Key Weaknesses
1. Threat model mismatch with standard adversarial robustness
2. Degree baseline competitiveness underexplored
3. case300 model quality invalidates conclusions
4. No heterophilic dataset evaluation
5. First-order radii called "certificates" in places
6. IGNN accuracy gap limits practical relevance

### Overall Score: 7/10 | Recommendation: Minor Revision | Confidence: High

---

# Perspective Review Report — Dr. Amara Nakiganda (R3)

### Summary

AEGIS maps structural sensitivity to N-1 contingency screening on IEEE power grids. Technically ambitious with commendable cross-domain scope. However, the power flow case study has several engineering shortcomings.

### Power Flow Case Study Validity — 5/10
Unrealistic training data (uniform 70–130% only). Binary adjacency discards essential physics. case300 results suspect (ΔS = 10.9 p.u.). IEEE test cases too small (max 300 buses vs. real 5,000–70,000). LODF baseline appropriate but incomplete (missing PTDF, fast-decoupled).

### Engineering Interpretation — 6/10
S_c-to-contingency mapping structurally sound but N-1 severity depends on voltage violations and line overloads, not just equilibrium displacement. ε_crit-as-stability-margin conflates model robustness with physical stability. Kirchhoff compliance comparison not iso-parameter.

### Practical Applicability — 4/10
Timing too slow for real-time (2–23s vs. milliseconds for DC screening). No thermal/voltage limit awareness. No generator re-dispatch model. τ = 0.37–0.67 insufficient for operational use. Strongest value: works without line-impedance data.

### Cross-Disciplinary Potential — 8/10
Genuinely domain-agnostic. Natural extensions: communication networks, water distribution, transportation, supply chains, cyber-physical systems.

### Broader Impact & Ethics — 7/10
Dual-use risk real. case300 P@10 = 0.87 from topology alone is a security concern. Consider tiered release.

### Key Strengths
1. Unified three-output framework
2. Rigorous continuous-to-discrete transfer analysis
3. Honest operational caveat
4. Binary adjacency finding
5. Scalability to 7,650 nodes
6. Cross-architecture generality

### Key Weaknesses
1. case300 model quality invalidates results
2. No thermal/voltage violation analysis
3. Training data unrealistic
4. Missing AC-based contingency screening comparison
5. ε_crit as "stability margin" misleading
6. Kirchhoff comparison not iso-parameter

### Overall Score: 6/10 | Recommendation: Major Revision | Confidence: High

---

# Devil's Advocate Report — Dr. Marcus Chen

### Strongest Counter-Argument

The continuous-to-discrete gap fundamentally undermines the practical claims. IGNN on Amazon Photo produces τ = −0.15 (anti-correlated rankings) in the regime with strongest formal guarantees (κ = 0.09, largest ε_crit). The model satisfying the theory's assumptions most strongly produces the worst practical output. Furthermore, degree-proportional attack achieves 94–98% of AEGIS's damage on WikiCS, and GCN-2 (the most widely deployed GNN) shows negative τ on 3/5 datasets. A practitioner cannot know which architecture will produce useful rankings without running the brute-force ground truth that AEGIS is supposed to replace.

### Issue List (3 CRITICAL, 6 MAJOR, 3 MINOR)

**CRITICAL:**
1. Pubmed/Amazon tightness discrepancy between Tables 1 and 2 (cherry-picking)
2. Amazon Photo IGNN negative τ in strongest-guarantee regime (logic-gap)
3. Circular adaptive attack evaluation (confirmation-bias)

**MAJOR:**
4. Degree baseline near-equivalence (alternative-explanation)
5. "Any GNN" overclaim (overgeneralization)
6. IGNN accuracy gap + case300 model quality (so-what)
7. GCN-2 negative τ on most deployed architecture (cherry-picking)
8. Phase transition never empirically observable (logic-gap)
9. Dual-use risk inadequately addressed (stakeholder-blind-spot)

**MINOR:**
10. Abstract P@10 range inconsistency with Table 7
11. ε_crit Frobenius-vs-operator norm gap unquantified
12. Mettack comparison included despite being acknowledged as uninformative

### Observations (Non-Defects)
- Tightness ~1.01 at small ε genuinely consistent across architectures
- GCN-2 negative τ honestly reported and well-explained
- Neumann-series convergence engineering is sound
- WikiCS tightness stability consistent with low κ
- Paper's own disclaimers are more self-aware than most

---

*Generated by the Academic Paper Reviewer v1.9.1 — multi-perspective peer review simulation*
*5 independent reviewers, 3 phases (field analysis → parallel review → editorial synthesis)*
*Date: 2026-05-26*
