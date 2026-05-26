# Editorial Decision Letter & Revision Roadmap

**Paper**: AEGIS: Mining Graph Structure for Adversarial Vulnerability Analysis of GNNs  
**Venue**: IEEE International Conference on Data Mining (ICDM)  
**Track**: Graph Mining  
**Date**: 2026-05-26

---

## 1. Decision: Major Revision

Per Iron Rule #4, the Devil's Advocate identified 4 CRITICAL issues (C1--C4). Acceptance is not possible until these are resolved. The 4 peer reviewers (EIC, R1, R2, R3) all recommended Weak Accept, indicating the paper has substantial merit and is above the acceptance threshold contingent on revisions. The DA recommended Major Revision (4.5/10). The editorial decision is therefore **Major Revision**: the paper's core contribution is sound and the experimental effort is commendable, but the gap between claims and evidence must be closed before acceptance.

---

## 2. Summary

AEGIS introduces a constrained sensitivity matrix $S_c$ for pre-deployment adversarial vulnerability analysis of GNNs, extracting per-edge vulnerability rankings, SVD-optimal attack directions, and per-node sensitivity radii from a single computation. All five reviewers agree that the $S_c$ construction -- projecting unconstrained $N^2$-dimensional sensitivity onto the $|E|$-dimensional space of symmetric, edge-only perturbations -- is the paper's strongest and most genuinely novel contribution [EIC-S2, R1-S2, R2-S2, R3-S5, DA-Sec.9]. The experimental protocol (7 architectures, 9 datasets, 10 seeds) substantially exceeds ICDM standards [EIC-S3, R1-S3, R2-S3, R3-S5]. The power grid case study is creative and well-framed [EIC-S5, R2-S5, R3-S1].

The paper falls short in four areas. First, attack baselines are weak: comparing against random perturbation is uninformative, and the adaptive attacker uses the same IFT gradients, making the comparison partly circular [EIC-W1, R1-W4, R2-W5, DA-M2]. Second, the continuous perturbation model limits practical relevance, with moderate-to-negative transfer to discrete attacks [EIC-W2, R1-W1, R2-W3, R3-W2, DA-C3]. Third, the "architecture-agnostic" framing overstates generality: formal guarantees apply only to IGNN (which has an accuracy deficit), while explicit GNNs receive only the computational tool [EIC-W3, R1-W6, R2-W2, DA-C1, DA-M1]. Fourth, scalability is limited to $N \approx 300$ for dense analysis and $N \approx 5{,}000$ for matrix-free, far below the safety-critical domains invoked [EIC-W4, R3-W1, DA-C2].

---

## 3. Consensus Strengths (3+ reviewers agree)

| Strength | Reviewers | Description |
|----------|-----------|-------------|
| **$S_c$ constrained projection** | EIC, R1, R2, R3, DA | The $N^2 \to |E|$ projection enforcing symmetry and edge-only perturbation is the paper's most important contribution. Transforms vacuous unconstrained bounds into tight practical predictions (tightness $1.00 \pm 0.01$ at $\varepsilon = 0.01$). |
| **Experimental breadth** | EIC, R1, R2, R3 | 7 architectures, 9 datasets, 4 domains, 10 seeds with std. dev. -- substantially above ICDM standards. |
| **Honest self-assessment** | EIC, R1, R2, R3, DA | Six explicit limitations in the Conclusion, transparent Mettack caveat, operational caveat on power grid $\tau$ values. Unusual and commendable intellectual honesty. |
| **Power grid case study** | EIC, R2, R3 | Creative cross-disciplinary bridge connecting adversarial ML to N-1 contingency analysis. P@10 = 0.66--0.81 on IEEE test cases. The "screening layer, not standalone tool" framing is appropriate. |
| **Matrix-free scalability** | EIC, R1, R2, R3 | Concrete engineering contribution: dense pipeline OOMs at $N > 200$; matrix-free handles $N = 2{,}708$ in 78s / 1.1 GB. Makes the framework practically deployable on medium-scale graphs. |
| **Adaptive attack evaluation** | EIC, R1, R2 | Using PGD with identical IFT gradients correctly tests first-order radius reliability. 0% breach at $\varepsilon = 0.01$, <1% at $\varepsilon = 0.10$. Methodologically sound design. |
| **Novel mining perspective** | EIC, R2 | Framing adversarial vulnerability analysis as a graph structure mining problem -- rather than attack or defense -- is a genuine conceptual contribution well-suited to ICDM. |

---

## 4. Consensus Weaknesses (3+ reviewers agree)

| Weakness | Reviewers | Description |
|----------|-----------|-------------|
| **Weak attack baselines** | EIC, R1, R2, DA | Primary comparison is against random perturbation (trivially weak in high dimensions). Mettack uses GCN surrogate (acknowledged mismatch). Adaptive attacker uses same IFT gradients (partly circular). No modern structural attack baseline (PR-BCD, topology attack, RL-S2V). |
| **Continuous perturbation model** | EIC, R1, R2, R3, DA | Real attacks are discrete (edge add/remove). Transfer to discrete is moderate ($\tau = +0.22$ to $+0.54$) and fails for GCN-2 ($\tau = -0.04$). Formal guarantees strictly apply only to continuous perturbations. The continuous-discrete gap is acknowledged but not formally analyzed. |
| **IGNN accuracy gap** | EIC, R2, DA | IGNN achieves 77.5% on Cora vs. ~82% for APPNP. Formal guarantees (Theorem 1) apply only to this underperforming architecture. Practitioners may not deploy IGNN, making the theoretical core practically irrelevant without broader applicability. |
| **Scalability ceiling** | EIC, R3, DA | Dense analysis limited to $N \approx 200$--$300$. Matrix-free reaches $N \approx 5{,}000$. Safety-critical domains (power grids, financial networks) have thousands to millions of nodes. The 50-node BFS subgraph workaround changes the analysis target and lacks theoretical justification for sufficiency. |
| **Theoretical novelty incremental** | R1, R2, DA | Individual components (IFT, Neumann series, SVD, resolvent bounds) are textbook. Proposition 3/4 (explicit GNN extension) is the chain rule. The novelty is in the combination and the $S_c$ projection, not the individual mathematical results. |

---

## 5. Disputed Issues (reviewers disagree, with arbitration)

### 5.1 Significance of Tightness at $\varepsilon = 0.01$

- **EIC, R1, R2**: Treat tightness $1.00 \pm 0.01$ as a strong empirical result validating the $S_c$ framework.
- **DA** (m1): Argues this is a mathematical tautology -- any differentiable function equals its Taylor expansion at small $\varepsilon$. The contribution is tractability, not tightness.

**Arbitration**: Both perspectives have merit. The tightness result validates the *implementation* (no bugs, correct projection) but is not surprising *mathematically*. The paper should retain the result but present it as validation of correctness rather than a surprising finding. The more informative data is tightness at $\varepsilon = 0.05$--$0.20$ (Table II), which should be promoted to equal prominence. **Action**: Reframe tightness at $\varepsilon = 0.01$ as a correctness check; emphasize degradation profile across $\varepsilon$ as the substantive result.

### 5.2 Adaptive Attacker Evaluation

- **R1** (S4), **R2** (S4): Commend the adaptive attacker design as methodologically sound.
- **DA** (M2): Argues the comparison is circular -- SVD is the exact solution to the linearized problem; PGD is an approximate solver for the same problem.

**Arbitration**: The adaptive attacker is well-designed *for what it tests* (first-order radius reliability), but it does not substitute for comparison against structurally different attacks. The paper should keep the adaptive attacker but add at least one attack that uses different information (not IFT gradients). **Action**: Retain adaptive attack; add a non-IFT-based structural attack baseline.

### 5.3 Proposition 3/4 (Explicit GNN Extension) -- Theoretical Status

- **R1** (W6): "Relabel as Remark or Observation."
- **R2** (Sec. 5.1): "Moderately novel -- cleanly stated and empirically validated."
- **DA** (C4): "The multivariate chain rule."

**Arbitration**: The mathematical content is indeed standard, but the empirical validation across 6 architectures is the substantive contribution. **Action**: Downgrade from "Proposition" to "Observation" or "Remark" and emphasize the computational/empirical contribution rather than mathematical novelty.

### 5.4 Power Grid Case Study -- Sufficiency for ICDM

- **EIC** (S5): "Compelling cross-domain transfer."
- **R2** (S5): "Goes beyond a toy demonstration."
- **R3**: Notes IEEE test cases are pedagogical benchmarks (case14--118 are toy-scale), $\Delta S = 0.106$ p.u. is poor model quality, and the study would not pass at IEEE PES.
- **DA** (M4): "Moderate correlation on toy-scale grids."

**Arbitration**: For ICDM (a data mining venue), the case study is acceptable as proof-of-concept and cross-disciplinary illustration. It would not satisfy a power systems venue. The paper should add at least one medium-scale grid (PEGASE 1354 or similar) if feasible, but the primary audience is data mining, not power engineering. **Action**: Qualify case study claims; attempt one larger grid; acknowledge the study is illustrative, not operational validation.

### 5.5 Smoothing Comparison -- Certificate Semantics

- **EIC**: Does not flag this comparison.
- **R1** (W2): Notes AEGIS radii lack second-order error bounds, making comparison with smoothing apples-to-oranges.
- **DA** (M3): "Conflates two certification paradigms of fundamentally different strength."

**Arbitration**: The comparison is valid directionally (AEGIS gives per-node differentiation, smoothing gives uniform radii) but the "1.9--7.7x larger radii" claim conflates certificate types. **Action**: Add a paragraph explicitly distinguishing deterministic-local (AEGIS, first-order) from probabilistic-global (smoothing) certificates. Present the comparison as complementary rather than competitive.

---

## 6. Devil's Advocate CRITICAL Issues -- Evaluation & Resolution

### C1. "Architecture-Agnostic" Claim Is Misleading

**DA finding**: The abstract claims AEGIS "applies to any differentiable GNN," but formal guarantees (Theorem 1: phase transition, critical budget, three regimes) apply only to contractive implicit GNNs. For explicit GNNs, the contribution reduces to Jacobian + SVD. GAT requires architectural modification.

**Corroboration**: EIC (W3) flags the IGNN accuracy gap. R2 (W2) identifies the same concern ("who would deploy IGNN when APPNP gives better accuracy?"). R1 (W6) notes Proposition 3 is the chain rule. **3/4 peer reviewers partially corroborate.**

**Verdict**: CRITICAL status justified. The framing must be revised.

**Resolution path**:
1. Restructure claims into two tiers: Tier 1 (explicit GNNs) = practical vulnerability tool with empirical validation; Tier 2 (implicit GNNs) = formal guarantees with $\varepsilon_{\text{crit}}$ and three regimes.
2. Replace "any differentiable GNN" with "any GNN whose message passing is differentiable with respect to continuous edge weights" and add the tier distinction.
3. Qualify GAT compatibility by noting the edge-weighted variant is a modification, not standard GAT.
4. Downgrade Proposition 3/4 to "Observation" or "Remark."

**Acceptance criterion**: The abstract, introduction, and conclusion clearly distinguish what is provided for explicit vs. implicit GNNs. No unqualified "architecture-agnostic" claims remain.

---

### C2. Scalability Makes the Framework Impractical for Stated Use Cases

**DA finding**: Dense analysis limited to $N \le 300$; 50-node BFS subgraph changes the analysis target; safety-critical domains require thousands to millions of nodes.

**Corroboration**: EIC (W4) flags the $N \approx 5{,}000$ ceiling not being stress-tested. R3 (W1) flags the absence of power grids above 200 buses. **2/4 peer reviewers corroborate the scalability concern, though with less severity than DA.**

**Verdict**: CRITICAL status justified for the *claim-evidence gap*, though the matrix-free pipeline (acknowledged by EIC-S4, R1-S5, R2-S6, R3-S8 as a genuine contribution) partially addresses this. The issue is overclaiming, not missing capability.

**Resolution path**:
1. Run matrix-free pipeline on Pubmed ($N = 19{,}717$) or Amazon Photo ($N = 7{,}650$) full-graph and report wall-clock time and memory. If infeasible, clearly state the scalability boundary.
2. Provide theoretical or empirical justification for subgraph analysis sufficiency: e.g., show vulnerability influence decays with graph distance for contractive GNNs, or compare subgraph vs. full-graph rankings on Cora (feasible at 78s).
3. Remove or qualify "safety-critical" framing for domains that exceed the demonstrated scalability.
4. If possible, add one power grid above 300 buses (PEGASE 1354 or similar).

**Acceptance criterion**: The paper's scalability claims match demonstrated capabilities. Full-graph analysis is reported on at least one graph with $N > 5{,}000$, or the scalability boundary is explicitly stated and safety-critical framing is appropriately scoped.

---

### C3. Continuous-Discrete Perturbation Gap Undermines Core Value Proposition

**DA finding**: Real attacks are discrete; AEGIS analyzes continuous perturbations; $\tau$ transfer is moderate (+0.22 to +0.54) and negative for GCN-2 ($-0.04$). The connection between continuous sensitivity and discrete impact is assumed, not proven.

**Corroboration**: **All 4 peer reviewers corroborate.** EIC (W2), R1 (W1), R2 (W3), R3 (W2) all flag this as a weakness. This is the strongest consensus weakness across all reviewers.

**Verdict**: CRITICAL status justified. This is a fundamental limitation that all reviewers identified.

**Resolution path**:
1. Report $\tau$ values for continuous-to-discrete transfer across **all datasets and all architectures** (not just Cora, Table V). The current single-dataset evaluation is insufficient.
2. Explain the GCN-2 negative $\tau$: analyze what graph/model properties predict good vs. poor transfer (depth, degree distribution, spectral gap).
3. Add a formal analysis or bound connecting continuous sensitivity scores to discrete edge-removal impact, even if approximate.
4. For the power grid case study specifically, add a discrete-removal experiment: compare $S_c$ scores against actual power flow shift from full edge removal (weight 1 to 0).
5. Add precision@k for discrete removal alongside $\tau$.

**Acceptance criterion**: Discrete transfer is evaluated on all datasets and architectures. The GCN-2 failure case is explained. The conditions under which continuous-to-discrete transfer works (and fails) are clearly characterized.

---

### C4. Theoretical Novelty Is Incremental

**DA finding**: Every component (IFT, Neumann series, SVD, resolvent bounds) is textbook. The assembly does not produce surprising results.

**Corroboration**: R1 (W6) agrees Proposition 3 is trivial. R2 (Sec. 5.1) rates theoretical novelty at 58/100 and agrees individual components are standard. EIC gives novelty 75/100, noting "incremental technical novelty." **3/4 peer reviewers partially corroborate** but frame it more positively: the combination and $S_c$ projection are valued.

**Verdict**: Partially justified. The $S_c$ construction is genuinely novel and all reviewers acknowledge it. However, the framing overstates the theoretical contribution of the other components. This is addressable through reframing rather than new results.

**Resolution path**:
1. Add a "Relationship to Existing Sensitivity Analysis" paragraph (as R2 suggests in W1) explicitly distinguishing what is novel ($S_c$ projection, three-output design, cross-architecture pipeline) from what is standard (IFT, Neumann, SVD).
2. Relabel Proposition 3/4 as "Observation" or "Remark."
3. Present Theorem 1 honestly: the individual bounds are standard; the contribution is their application to structural vulnerability analysis and the three-regime packaging.
4. Emphasize the empirical and engineering contributions (tightness validation, matrix-free pipeline, cross-architecture coverage) as co-equal with theory.

**Acceptance criterion**: The paper clearly delineates novel contributions ($S_c$, three-output framework, cross-architecture validation) from standard mathematical tools. No overclaiming of theoretical novelty for textbook results.

---

## 7. Score Summary Table

### Peer Reviewer Scores

| Dimension | EIC | R1 (Method.) | R2 (Domain) | R3 (Perspective) |
|-----------|:---:|:---:|:---:|:---:|
| Novelty | 75 | 68 (Method) | 58 (Theoretical) | -- |
| Technical Soundness | 82 | 72 (Math Rigor) | -- | -- |
| Significance | 70 | -- | 74 (Domain) | 48 (Practical Appl.) |
| Clarity | 78 | -- | -- | -- |
| Reproducibility | 80 | 80 | -- | -- |
| Statistical Validity | -- | 75 | -- | -- |
| Literature Coverage | -- | -- | 72 | -- |
| Positioning Accuracy | -- | -- | 70 | -- |
| Cross-Disc. Value | -- | -- | -- | 72 |
| Case Study Quality | -- | -- | -- | 55 |
| Broader Impact | -- | -- | -- | 65 |
| **Recommendation** | **Weak Accept** | **Weak Accept** | **Weak Accept** | **Weak Accept** |

### Devil's Advocate Scores

| Dimension | Score | Key Issue |
|-----------|:-----:|-----------|
| Novelty | 4/10 | Standard tools assembled; Proposition 3 is chain rule |
| Methodology | 6/10 | Sound execution; weak baselines, circular adaptive attack |
| Significance | 4/10 | $N \le 300$ scalability + continuous-only = niche |
| Clarity | 7/10 | Well-written; "architecture-agnostic" misleading |
| Reproducibility | 5/10 | Good protocol; no code release |
| **Overall** | **4.5/10** | **Major Revision** |

### Aggregated Assessment

- **Peer reviewer consensus**: Weak Accept (4/4 reviewers)
- **Devil's Advocate**: Major Revision (4 CRITICAL issues)
- **Editorial decision**: **Major Revision** (DA CRITICAL issues corroborated by peer reviewers; addressable through revision)

---

# Revision Roadmap

## Priority 1: Must Fix (Blocks Acceptance)

### P1.1 -- Restructure "Architecture-Agnostic" Claims [C1]

**Raised by**: DA (C1), EIC (W3), R1 (W6), R2 (W2)

**Action items**:
1. Revise the abstract to distinguish two tiers: "For explicit GNNs (GCN, GIN, GAT, SAGE, APPNP), AEGIS provides a practical vulnerability ranking tool; for contractive implicit GNNs, it additionally provides formal robustness guarantees including a critical perturbation budget and three-regime vulnerability characterization."
2. In Section I (Contributions), split Contribution 1 into 1a (computational: $S_c$ for any differentiable GNN) and 1b (theoretical: formal guarantees for contractive implicit GNNs).
3. Qualify every instance of "architecture-agnostic" or "any differentiable GNN" with the tier distinction.
4. Downgrade Proposition 3/4 (explicit GNN extension) from "Proposition" to "Observation" or "Remark" and emphasize the computational/empirical contribution.
5. Note GAT compatibility requires an edge-weighted variant that modifies model semantics.

**Acceptance criterion**: No unqualified "architecture-agnostic" claims. Reader can immediately understand which guarantees apply to which architecture class.

---

### P1.2 -- Add Meaningful Attack Baselines [DA-M2, EIC-W1, R1-W4, R2-W5]

**Raised by**: DA (M2), EIC (W1), R1 (W4), R2 (W5)

**Action items**:
1. Add at least one gradient-based structural attack applied directly to the target model (not through a surrogate). Candidates: PR-BCD (Geisler et al., 2021) or topology attack (Xu et al., 2019). Report damage relative to AEGIS SVD direction.
2. Add a simple structured baseline: degree-proportional perturbation and/or top-eigenvalue perturbation of $A$ (spectral heuristic). Report attack advantage relative to these baselines.
3. If GNNExplainer/PGExplainer are feasible, add them to Table V as edge-importance baselines [R2-W4]. Even if they optimize for different objectives, the comparison clarifies positioning.
4. Downweight the Mettack comparison: reduce to one sentence noting the surrogate mismatch, and make the adaptive attack + new baselines the primary comparisons [R2-W5].

**Acceptance criterion**: At least one non-trivial structural attack baseline beyond random. The paper demonstrates AEGIS's advantage (or honestly reports its limitations) against a method that uses gradient information but different optimization than SVD.

---

### P1.3 -- Address Continuous-to-Discrete Gap [DA-C3, all peer reviewers]

**Raised by**: DA (C3), EIC (W2), R1 (W1), R2 (W3), R3 (W2) -- **strongest consensus weakness**

**Action items**:
1. Expand Table V to report $\tau$ for continuous-to-discrete transfer across **all 9 datasets** (or at minimum all 5 benchmark datasets) and all 7 architectures. The current single-dataset (Cora) evaluation is insufficient.
2. Explain the GCN-2 negative $\tau$ ($-0.04$): analyze what model/graph properties predict good vs. poor transfer. Correlate $\tau$ with model depth, graph density, and spectral gap [R2-W3].
3. Report precision@k for discrete edge removal alongside $\tau$ [EIC-W2].
4. Add a brief formal analysis connecting continuous $S_c$ scores to discrete removal impact, even if approximate. At minimum, prove that edges with the highest continuous sensitivity scores cause above-median damage under discrete removal [DA-C3].
5. For the power grid case study, add a discrete-removal experiment specifically comparing $S_c$ scores against actual power flow shift from full edge removal [R3-W2].

**Acceptance criterion**: Discrete transfer evaluated on all datasets. GCN-2 failure explained. Conditions for transfer success/failure characterized. Reader can assess when the continuous framework is vs. is not a reliable proxy for discrete threats.

---

### P1.4 -- Scope Scalability Claims to Match Evidence [DA-C2, EIC-W4, R3-W1]

**Raised by**: DA (C2), EIC (W4), R3 (W1)

**Action items**:
1. Run the matrix-free pipeline on at least one graph with $N > 5{,}000$ (e.g., Pubmed $N = 19{,}717$ or Amazon Photo $N = 7{,}650$). Report wall-clock time and memory. If infeasible, state the boundary explicitly.
2. Compare subgraph (50-node BFS) vs. full-graph (matrix-free) per-edge rankings on Cora (feasible at 78s per Table IV). Report ranking correlation to justify or qualify the subgraph approximation [R1-W7].
3. Qualify or remove "safety-critical" framing for domains that exceed demonstrated scalability. Replace with honest scope: "small-to-medium graphs ($N \le 5{,}000$) and subgraph-level analysis for larger graphs."
4. If possible, add one power grid case above 300 buses (e.g., PEGASE 1354 or IEEE 300-bus) [R3-W1].
5. Provide theoretical justification for subgraph sufficiency (e.g., sensitivity decay with graph distance for contractive GNNs) or acknowledge it as an unvalidated assumption [DA-C2].

**Acceptance criterion**: Scalability claims match demonstrated capabilities. Either full-graph analysis is shown at $N > 5{,}000$, or the boundary is explicitly stated and claims are appropriately scoped.

---

## Priority 2: Should Fix (Significantly Strengthens Paper)

### P2.1 -- Delineate Novel vs. Standard Contributions [DA-C4, R1-W6, R2-W1]

**Raised by**: DA (C4), R1 (W6), R2 (W1, Sec. 5.1)

**Action items**:
1. Add a "Relationship to Existing Sensitivity Analysis" paragraph in Section III or IV. Explicitly compare $S_c$ to existing Jacobian-based sensitivity analyses (Chen et al. 2018 Neural ODE, Bai et al. 2021 Jacobian regularization for DEQs). State what is novel ($S_c$ projection, three-output design, cross-architecture pipeline) vs. standard (IFT, Neumann, SVD).
2. Relabel Proposition 3/4 as "Observation" or "Remark."
3. Frame Theorem 1 honestly: the individual bounds are standard; the contribution is their *combination and application* to structural vulnerability analysis.

**Acceptance criterion**: A reader can clearly distinguish the paper's genuine novelty ($S_c$, practical pipeline, cross-domain validation) from well-known mathematical tools.

---

### P2.2 -- Second-Order Error Analysis [R1-W2, DA-M3]

**Raised by**: R1 (W2), DA (M3)

**Action items**:
1. Derive a second-order correction bound (even if loose): $|\Delta z^* - S \cdot \text{vec}(\delta A)| \le C \cdot \varepsilon^2$ with computable $C$. This transforms first-order radii into rigorous local certificates.
2. Alternatively, compute Hessian-vector products empirically and report the second-order correction magnitude across all datasets.
3. Report empirical breach rates at all $\varepsilon$ values ($0.01, 0.05, 0.10, 0.20$), not just $0.01$ and $0.10$ [DA-M3].
4. Add a paragraph distinguishing deterministic-local (AEGIS) from probabilistic-global (smoothing) certificate semantics [R1-W2, DA-M3].

**Acceptance criterion**: Either formal second-order bounds or comprehensive empirical breach rates across all $\varepsilon$ values and datasets. Certificate comparison with smoothing includes semantic distinction.

---

### P2.3 -- Address IGNN Accuracy Gap [EIC-W3, R2-W2, DA-M1]

**Raised by**: EIC (W3), R2 (W2), DA (M1)

**Action items**:
1. Option A: Test EIGNN (Liu et al. 2021) to show whether implicit-model accuracy can be improved while retaining contractivity [R2-W2].
2. Option B: Present an accuracy-vs-guarantee frontier table: for each architecture, show what formal guarantees are available and at what accuracy cost [EIC-W3].
3. Frame Theorem 1 as providing *understanding* of vulnerability phase transitions (qualitative insight applicable to all architectures) rather than *practical deployment guidance* restricted to IGNN.

**Acceptance criterion**: The paper either demonstrates a higher-accuracy implicit model, or explicitly frames the theoretical contribution as providing qualitative insight with the practical tool ($S_c$ rankings) being architecture-independent.

---

### P2.4 -- Empirically Validate Phase Transition [R1-W3, DA-Sec.7]

**Raised by**: R1 (W3), EIC (Q1 implicit), DA (Sec. 7 Alt. Expl. 3)

**Action items**:
1. Demonstrate the phase transition empirically by training IGNN with relaxed spectral normalization ($\kappa \to 1$) and showing divergence of $\|(I - J_z')^{-1}\|_2$ as $\varepsilon \to \varepsilon_{\text{crit}}$ [R1-Q1].
2. Plot the full transition curve (not just the $83\times$ amplification at $\kappa = 0.99$ data point).
3. Report per-direction critical budgets using SVD structure of $S_c$ rather than worst-case norms [R1-W3].

**Acceptance criterion**: At least a single-dataset experiment showing the phase transition or, alternatively, an honest acknowledgment that the critical/supercritical regimes are not empirically accessible and the theorem's practical value is the subcritical bound.

---

### P2.5 -- Strengthen Power Grid Case Study [R3-W1/W3/W4/W5/W6/W7]

**Raised by**: R3 (multiple weaknesses)

**Action items**:
1. Add at least one medium-scale grid (PEGASE 1354 or IEEE 300-bus) if feasible [R3-W1].
2. Plot $\tau$ vs. $\Delta S$ across cases and seeds to quantify model-quality-to-ranking-quality relationship [R3-W7].
3. Report LODF wall-clock time and data generation cost alongside AEGIS cost for a fair comparison [R3-W4].
4. Investigate the binary-vs-admittance-weighted result more deeply: does the IGNN learn impedance-like information from training data? Report impedance variance across test cases [R3-W6].
5. Discuss N-2/N-k connection: report top-k edges from SVD attack direction $v_1$ and evaluate whether they correspond to known N-2 critical pairs [R3-W5].
6. Report rank stability of top-10 vulnerable edges across 10 seeds [R3-W7].

**Acceptance criterion**: At least 2 of the 6 items above are addressed. The case study includes fair comparison context (LODF cost) and addresses the largest outstanding concern (grid scale or $\tau$ variance).

---

### P2.6 -- Report $\kappa$-Based Quantities as Primary [DA-M5]

**Raised by**: DA (M5)

**Action items**:
1. Report $\kappa$-based $\varepsilon_{\text{crit}}$ as the primary safety quantity in all tables.
2. Report $\rho$ as a secondary diagnostic for comparison.
3. Either prove a structural bound on $\eta$ for spectrally-normalized ReLU operators, or state that $\eta$ must be computed post-hoc for each new graph.

**Acceptance criterion**: Tables report formal ($\kappa$-based) quantities as primary. The 28% gap between $\kappa$ and $\rho$ is not buried.

---

### P2.7 -- Degree Correlation Analysis [R1-W5, DA-Sec.7 Alt.4]

**Raised by**: R1 (W5), DA (Sec. 7, Alternative Explanation 4)

**Action items**:
1. Report Kendall $\tau$ between per-edge vulnerability scores $v_{ij}$ and endpoint degree ($\max(d_i, d_j)$) across all 9 datasets [R1-Q3].
2. If strongly correlated, consider degree-normalized scores $v_{ij} / f(d_i, d_j)$ and show the residual still predicts discrete removal impact.
3. For power grids, test whether a simple centrality-based ranking achieves comparable $\tau$ without any GNN or IFT computation [DA-Alt.4].

**Acceptance criterion**: The degree-vulnerability correlation is reported. If high, the paper demonstrates that $S_c$ provides information *beyond* degree centrality.

---

## Priority 3: Nice to Fix (Minor Issues)

| Issue | Source | Action |
|-------|--------|--------|
| Notation: $\kappa$ vs. $\rho$ vs. $\eta$ inconsistency | EIC, R1 (M1) | Define all spectral quantities once; use consistently. |
| Notation: $\sigma$ for both activation and singular values | R2 | Use $\phi$ for activation function. |
| Notation: $z^*$ vs. $Z^*$ (vec/reshape) | R1 (M1) | State vec/reshape convention once; reference throughout. |
| Notation: $\hat{A}$ vs. $A$ in Theorem 1 statement | R1 (M3) | Use $\hat{A}$ consistently in theorem; clean up proof. |
| Table density -- move Table II or IV to supplement | EIC | Improve visual breathing room for remaining tables. |
| Related work placement (after experiments) | EIC | Consider moving to Section III (before technical sections). |
| Missing std. dev. in Table VI power flow columns | EIC | Add $\pm$ std. dev. for RMSE values. |
| Acronym overload in abstract | EIC | Reduce acronyms; define in body instead. |
| Proposition numbering verification | R1 (M2) | Verify LaTeX counter consistency. |
| Tightness definition: clarify $\ge 1$ is expected | R1 (M4) | Note underestimation is expected from convexity. |
| Cert% column clarification | R1 (M5) | Clarify subgraph vs. full-graph accuracy. |
| GAT$^\dagger$ formal definition | R2 | Add 2-line equation for edge-weighted GAT variant. |
| "149/150 wins" decomposition | R2 | Clarify which dataset/seed/$k$ lost. |
| Binary vs. admittance result discussion | R2 | Expand discussion of this surprising finding. |
| Citeseer accuracy discrepancy (Appendix B vs. Table I) | DA (m3) | Reconcile the ~20 percentage point gap. |
| Tightness degradation: add Pubmed and Amazon Photo to Table II | DA (m2) | Report all 5 benchmark datasets. |
| GCN-2 negative $\tau$ discussion | DA (m5) | Discuss in text, not just table. |
| Defense ablation: validate against discrete attacker | DA (m6) | Test edge masking defense against discrete edge removal. |
| ReLU non-differentiability at zero | R1 (Sec. 4.1) | Add assumption or smoothness remark. |
| PGD convergence diagnostics | R1 (Sec. 4.7) | Report loss vs. iteration; clarify objective (shift vs. CE). |
| Missing references: PGExplainer, GRAND/GRAND++, Entezari et al., Mujkanovic et al. | R2 (Sec. 4.2) | Add citations and brief positioning. |
| Pseudospectral index $\eta$: integrate into theory or move to experiments | R2 | Currently defined but not used in bounds. |
| Norm conflation ($\|\cdot\|_2$ vs. $\|\cdot\|_F$) in Theorem 1 proof | DA (M6) | State $\varepsilon_{\text{crit}}$ is Frobenius-norm sufficient condition; quantify gap. |

---

## Summary for Authors

Your paper has genuine merit: the $S_c$ construction is novel and practically impactful, the experimental protocol is thorough, and the cross-domain case study is creative. All four peer reviewers recommended Weak Accept, which is a strong signal for ICDM. The Major Revision decision reflects not a lack of quality but a gap between what the paper claims and what it demonstrates. The four Priority 1 items -- scoping the architecture-agnostic claims, adding meaningful attack baselines, addressing the continuous-discrete gap across all datasets, and matching scalability claims to evidence -- are all addressable through additional experiments and careful reframing. None requires fundamental changes to the methodology or theoretical framework. We look forward to a revised submission.
