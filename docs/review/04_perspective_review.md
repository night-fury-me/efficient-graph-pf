# Perspective Review Report — AEGIS

**Reviewer**: Dr. Line Roald, University of Wisconsin-Madison
**Expertise**: Power systems optimization, GNNs for grid operations, N-1/N-2 contingency analysis, physics-informed ML, safe ML deployment in critical infrastructure
**Confidence**: 4/5 (high confidence on power systems aspects; strong familiarity with GNN adversarial robustness literature)

## Summary (280 words)

AEGIS introduces the constrained sensitivity matrix $S_c$ as a unified object for adversarial vulnerability analysis of graph neural networks. The framework projects the full $N^2$-dimensional adjacency perturbation space onto the $|E|$-dimensional space of realistic (symmetric, edge-only) perturbations, then extracts three outputs from a single computation: SVD-optimal attack directions, per-edge vulnerability rankings, and per-node first-order sensitivity radii. For contractive implicit GNNs (IGNN-class), formal guarantees including a critical perturbation budget and three-regime vulnerability characterization are additionally provided; for explicit GNNs with edge-weight-differentiable message passing, the practical analysis tools transfer without formal regime guarantees. A matrix-free formulation using Neumann-series resolvent iteration and randomized SVD enables full-graph analysis up to N=7,650 nodes on a single GPU.

From a cross-disciplinary perspective, the most compelling contribution is the power grid case study (Section VII), which demonstrates that the $S_c$ vulnerability spectrum recovers N-1 contingency rankings on five IEEE test cases (14-300 buses) with $\tau = 0.37$-$0.72$ and P@10 = 0.66-0.87, without requiring line-impedance data. This bridges two communities that rarely interact at this technical depth: the GNN adversarial robustness community and the power systems contingency analysis community. The analogy between structural edge perturbation and transmission line outage is physically intuitive and mathematically grounded through Proposition 3's continuous-to-discrete transfer result.

However, significant gaps remain between the current demonstration and practical deployment in power systems operations. The binary adjacency representation discards essential electrical information, the training data covers only uniform load scaling, and the comparison with industry-standard tools (LODF) needs deeper engagement with the power systems literature. Despite these gaps, the paper opens a genuinely novel research direction at the intersection of adversarial ML and power grid security assessment.

## Strengths

1. **Principled cross-domain transfer (Section VII, Table VI).** The paper does not merely apply an ML tool to power data; it identifies a structural isomorphism between adversarial edge perturbation and N-1 contingency analysis. The observation that binary adjacency outperforms admittance-weighted adjacency (P@10 = 0.81 vs. 0.27 on case118) is a non-obvious and well-explained finding: N-1 is an all-or-nothing event, so binary sensitivity correctly models the discrete nature of line trips. This insight demonstrates genuine understanding of the power systems domain.

2. **Honest operational caveat (Section VII).** The paper explicitly states that "$\tau = 0.37$-$0.67$ is insufficient for direct operational use" and positions AEGIS as a "screening layer, not a standalone contingency tool." This intellectual honesty is rare in cross-disciplinary ML papers and builds trust with the power systems audience. The distinction between model sensitivity and physical measurement (i.e., $\tau$ reflects "the fidelity of the GNN's learned physics, not a direct physical measurement") is correctly drawn.

3. **Scale progression through IEEE test cases (Table VI).** Testing on case14 through case300 (with the 200-node subgraph approach for case300) demonstrates meaningful scalability awareness. The improving $\tau$ with grid size ($+0.42$ at case14 to $+0.72$ at case300) is an encouraging trend, though the subgraph extraction for case300 introduces a confound that should be discussed more carefully.

4. **LODF baseline comparison (Section VII).** Including the industry-standard Linear Outage Distribution Factor comparison is essential for credibility with the power systems audience. The result that AEGIS matches or exceeds LODF on case57/118 ($\tau = 0.62$-$0.67$ vs. $0.44$-$0.58$) with statistical validation (Wilcoxon signed-rank $p < 0.01$) is a meaningful benchmark, even if the comparison merits deeper analysis (see Weaknesses).

5. **N-2 extension via SVD (Section VII).** The observation that the leading singular vector $v_1$ naturally identifies multi-edge vulnerabilities, with 40-64% edge-level overlap against brute-force N-2 ground truth, is a genuinely useful result. N-2 contingency screening is computationally expensive ($O(|E|^2)$ full power flow solves), and any data-driven pre-screening has practical value.

6. **Rank stability analysis (Section VII).** Reporting pairwise Kendall $\tau$ of vulnerability rankings across 10 seeds ($+0.78 \pm 0.07$ on case57, $+0.67 \pm 0.12$ on case118) with the observation that 60% of top-10 edges appear in every seed's top-10 on case118 addresses a concern that power systems engineers would immediately raise: "Can I trust this ranking to be reproducible?"

7. **Comprehensive experimental methodology (Section VI).** The 10-seed protocol with explicit seed values, the four-quadrant attack taxonomy (gradient-based/free, same/different objective), and the honest discussion of where AEGIS underperforms (Amazon Photo IGNN $\tau = -0.15$; Cora greedy-optimal at 54%) set a high standard for empirical rigor.

## Weaknesses

1. **Binary adjacency discards essential electrical information (Section VII).** While the authors correctly argue that binary adjacency suits N-1's all-or-nothing character, this framing is incomplete. In practice, the severity of an N-1 contingency depends critically on line impedance, thermal limits, and the resulting power flow redistribution. Two lines with identical binary connectivity but different impedances (e.g., a 138 kV tie-line vs. a 500 kV backbone) produce vastly different post-contingency states. The paper's admittance-weighted comparison (P@10 = 0.27) does not mean impedance information is unimportant---it means the current model fails to incorporate it effectively.

   **Suggested fix:** Discuss alternative encoding strategies: edge-feature GNNs where impedance, thermal rating, and voltage level are edge attributes rather than adjacency weights; or a physics-informed loss that penalizes violation of Kirchhoff's laws. Acknowledge explicitly that binary adjacency limits the framework to topological screening and cannot capture impedance-dependent severity.

2. **Training data covers only uniform load scaling (Section VII, "Limitation" paragraph).** The 2,000 samples at 70-130% of nominal load represent a single operating mode. Real power grids exhibit seasonal load patterns, generator commitment schedules, renewable intermittency, and scheduled maintenance that create fundamentally different operating points. A vulnerability ranking derived from a narrow operating envelope may not generalize to stressed conditions where contingencies are most dangerous.

   **Suggested fix:** Add at least one experiment with non-uniform load variation (e.g., scale individual bus loads independently at 50-150%, or include generator outage scenarios). If computationally prohibitive, explicitly quantify the limitation: "Our training data spans X% of the feasible operating space estimated by [method]."

3. **LODF comparison lacks depth (Section VII).** The comparison reports $\tau$ values but does not analyze where AEGIS and LODF disagree and why. LODF is a DC approximation that ignores reactive power, voltage magnitude, and transformer tap ratios---precisely the information that the ContractiveGCN-PF model attempts to learn from AC power flow data. If AEGIS outperforms LODF, it should be because the GNN captures AC effects that DC linearization misses. The paper does not verify this hypothesis.

   **Suggested fix:** Identify the specific lines where AEGIS ranks higher than LODF (and vice versa) and correlate with physical characteristics: are the AEGIS-unique critical lines those with high reactive power sensitivity, voltage-constrained corridors, or transformer-connected? This analysis would demonstrate that AEGIS adds value precisely where DC approximation fails. Also compare against PTDF (Power Transfer Distribution Factors), which is more widely used than LODF for pre-contingency screening.

4. **No voltage or thermal limit violations in contingency assessment (Section VII).** The N-1 ground truth is described as "brute-force contingency ranking" based on "voltage and flow deviations," but the paper does not specify the severity metric. In practice, N-1 severity is measured by post-contingency thermal overloads (MW flow exceeding line rating) and voltage violations (p.u. voltage outside 0.95-1.05). The per-unit RMSE reported in Table VI ($|V|$ RMSE = 0.007-0.033, $\theta$ RMSE = 0.020-0.076) does not directly map to operational severity.

   **Suggested fix:** Define the N-1 severity metric explicitly (e.g., maximum post-contingency line loading as fraction of thermal limit, or maximum voltage deviation in p.u.). Report whether AEGIS's top-ranked lines correspond to those causing the most severe thermal or voltage violations, not just the largest $\Delta|V|$ or $\Delta\theta$.

5. **Missing transient stability and voltage stability considerations (Section VII).** N-1 contingency analysis in practice involves three timescales: (a) steady-state power flow redistribution (what the paper addresses), (b) voltage stability (seconds to minutes), and (c) transient/angular stability (milliseconds to seconds). The paper's static analysis captures only (a). For certain contingencies---particularly loss of large generators or key interconnecting lines---the dynamic response dominates the severity ranking, and a purely steady-state ranking may be misleading.

   **Suggested fix:** Add a paragraph acknowledging this limitation and discussing how dynamic GNN models (e.g., temporal GNNs or recurrent architectures) might extend AEGIS to capture time-domain vulnerability. At minimum, state that the current framework addresses only steady-state severity and that dynamic contingencies require separate analysis.

6. **Case300 uses subgraph extraction, confounding the scalability claim (Table VI).** The $\tau = +0.72$ and P@10 = 0.87 for case300 are obtained on a 200-node BFS subgraph with only 1,000 training samples (vs. 2,000 for smaller cases). The subgraph ablation in Section VI.D shows that subgraph-to-full-graph ranking correlation degrades significantly on larger, sparser graphs (Cora: $\tau = 0.16$). While power grids are denser than citation networks, the case300 result should be interpreted cautiously: the 200-node subgraph covers 67% of the 300-bus grid, but this coverage ratio will drop sharply for realistic grids (2,000+ buses).

   **Suggested fix:** Add full-graph matrix-free analysis for case300 (the matrix-free pipeline handles N=2,708 for Cora in 78s; case300 with N=300 should be tractable). If the dense path OOMs at N>200 but the matrix-free path should handle N=300 easily, explain why the subgraph approach was chosen over matrix-free for case300. Also discuss the coverage ratio issue for larger grids (e.g., Polish 2383-bus system).

7. **ContractiveGCN-PF model quality degrades at scale (Table VI).** The $\theta$ RMSE of 0.394 p.u. for case300 is an order of magnitude worse than case14-118 (0.020-0.076). In a 300-bus system, a 0.394 radian ($\approx 22.6$ degree) average angle error is physically unrealistic and suggests the model has not adequately learned the power flow physics. If the GNN's learned physics is poor, the vulnerability ranking measures model sensitivity rather than physical criticality, undermining the N-1 analogy.

   **Suggested fix:** Investigate the case300 model quality issue. Try deeper networks, larger hidden dimension, or more training samples. If the quality cannot be improved with the current architecture, report case300 results with a prominent caveat and focus the N-1 claims on case14-118 where model quality is adequate.

8. **No discussion of bus-type heterogeneity (Section VII).** Power grids have three bus types (slack/swing, PV/generator, PQ/load) with fundamentally different physical behavior. The two binary bus-type indicators (is-slack, is-PV) are a reasonable encoding, but the paper does not discuss whether vulnerability rankings correlate with bus-type topology. In practice, edges connecting to generator buses are often more critical than load-to-load connections because generator outages cascade differently.

   **Suggested fix:** Report vulnerability rankings stratified by endpoint bus types. Do the top-ranked edges disproportionately connect to PV or slack buses? This analysis would strengthen the physical interpretability of the results.

## Power Systems Case Study Assessment

### Setup Validity

The experimental setup in Section VII is reasonable for a proof-of-concept demonstration. The choice of IEEE test cases (14-300 buses) is standard in the power systems ML literature. Using PandaPower's Newton-Raphson solver for training data generation is appropriate, and the 5 bus features (is-slack, is-PV, P, Q, |V|) capture the essential node-level physics.

However, several aspects would concern a power systems practitioner:

**Graph construction.** The binary adjacency from the admittance matrix treats all transmission lines identically regardless of voltage level, impedance, or thermal rating. In a real grid, a 500 kV backbone line and a 69 kV distribution feeder have fundamentally different roles in system security, but they receive equal treatment in the binary representation. The finding that admittance-weighted adjacency performs worse is interesting but does not resolve this concern---it suggests that a more sophisticated edge encoding is needed, not that impedance information is irrelevant.

**Training data.** Uniform load scaling at 70-130% of nominal produces a narrow operating envelope. Real contingency screening must be valid across seasonal peaks, light-load conditions, and N-1-1 cascading scenarios. The absence of generator commitment variation is particularly limiting because the dispatch pattern fundamentally changes power flow directions on the network.

**Model architecture.** The IGNN architecture with spectral normalization sacrifices accuracy ($\sim$6% penalty per Section VI) for formal guarantees. In power systems, model accuracy directly affects the reliability of vulnerability rankings. The $\theta$ RMSE of 0.394 for case300 raises concerns about whether the model has learned meaningful physics at this scale.

### N-1 Analogy Validity

The mapping from $S_c$ vulnerability to N-1 contingency is conceptually sound: both quantify the impact of removing/perturbing a single edge (transmission line) on system state (equilibrium representation / power flow solution). Proposition 3's continuous-to-discrete transfer result provides formal justification, and the empirical $\tau$ values (0.37-0.72) confirm partial ranking agreement.

The analogy has fundamental limitations that should be more prominently discussed:

- **Continuous vs. discrete**: N-1 is inherently discrete (a line is either in or out). The first-order continuous approximation works well for small perturbations but may miss discontinuous effects (e.g., islanding, loss of path connectivity).
- **Single-point sensitivity vs. post-contingency state**: AEGIS measures the first-order sensitivity of the GNN's learned equilibrium. N-1 severity depends on the full post-contingency power flow, which involves nonlinear redistribution, generator re-dispatch, and potentially corrective actions.
- **Topological vs. electrical**: AEGIS captures topological vulnerability (which edges matter for the learned representation). N-1 severity is electrical (which line outages cause the worst thermal or voltage violations). These are correlated but distinct.

### Missing Elements for Power Systems Credibility

- **PTDF comparison**: Power Transfer Distribution Factors are the most common industry tool for pre-contingency screening. LODF is a related but less commonly used metric. Including PTDF would strengthen the baseline comparison.
- **Thermal limit analysis**: The most operationally relevant N-1 metric is the post-contingency line loading as a fraction of the thermal limit. This is absent.
- **Reactive power and voltage stability**: The case study focuses on $\Delta|V|$ and $\Delta\theta$ but does not analyze Q-V curves or voltage stability margins, which drive many critical contingencies.
- **Larger test systems**: The 300-bus system is the largest tested; realistic N-1 screening operates on 2,000-30,000 bus systems. The matrix-free pipeline's scalability to N=7,650 suggests this is feasible but untested for power grids.

## Cross-Disciplinary Impact

### Bridge Quality

The paper does a commendable job of bridging ML and power systems, though each audience will find gaps:

**For ML readers:** The power systems background (Section VII.A) provides adequate context on N-1 contingency. The analogy table (edge perturbation $\leftrightarrow$ line trip, vulnerability spectrum $\leftrightarrow$ contingency severity, $\varepsilon_{\text{crit}}$ $\leftrightarrow$ sensitivity threshold) is well-structured. However, ML readers may not appreciate why binary adjacency is a significant limitation or why $\tau = 0.67$ is both impressive and insufficient for operational use.

**For power systems readers:** The theoretical sections (III-IV) are dense but well-motivated. The connection to LODF (both are linearized sensitivity tools) is the right framing. However, power systems readers will expect discussion of thermal limits, voltage stability, and dispatch-dependent vulnerability---topics absent from the current version.

### Other Application Domains

The AEGIS framework has clear potential beyond power grids:

- **Water distribution networks**: Pipe failure analysis maps directly to edge perturbation; pressure/flow sensitivity is analogous to voltage/power sensitivity.
- **Transportation networks**: Link failure in traffic networks; vulnerability of route recommendations in navigation GNNs.
- **Communication networks**: Router/link failure in network topology; QoS degradation under edge perturbation.
- **Chemical process networks**: Equipment failure propagation in process flow diagrams modeled as graphs.
- **Supply chain networks**: Supplier/logistics link disruption; resilience of GNN-based demand forecasting.

The paper briefly mentions fraud detection and drug interaction but does not develop these connections. A table mapping AEGIS concepts to 3-4 application domains would significantly strengthen the cross-disciplinary contribution.

## Practical Deployment Assessment

### Gap Between Paper and Deployment

For a power systems engineer considering AEGIS for operational contingency screening, the following gaps exist:

1. **Accuracy gap.** P@10 = 0.66-0.87 means 1-3 of the top-10 critical lines are missed. In power systems operations, missing even one critical contingency can lead to cascading failure. The screening tool must be supplemented by full N-1 analysis on the flagged subset, which the paper correctly acknowledges.

2. **Speed gap.** AEGIS requires 2-23 seconds (training + $S_c$ computation) vs. 0.1-2 seconds for brute-force N-1 and <0.13 seconds for LODF. For grids small enough for brute-force N-1, AEGIS offers no speed advantage. The value proposition depends on scaling to grids where brute-force N-1 is expensive (thousands of buses with AC power flow), but this scaling is not yet demonstrated.

3. **Interpretability gap.** Power systems operators need to understand *why* a line is critical (thermal overload path, voltage support loss, stability margin). AEGIS provides a vulnerability score without physical interpretation. Combining AEGIS rankings with PTDF/LODF decomposition could bridge this gap.

4. **Validation gap.** The IEEE test cases, while standard, are synthetic. Deployment requires validation on realistic utility-scale models (e.g., WECC, ERCOT) with actual operating data, seasonal variation, and generator commitment schedules.

5. **Regulatory gap.** NERC reliability standards require that N-1 analysis use validated power flow models with specific accuracy thresholds. A GNN-based screening tool would need to demonstrate compliance with TPL-001 standards before operational use.

### Realistic Deployment Scenario

The most realistic near-term deployment would be as a **pre-screening filter** in a two-stage pipeline: (1) AEGIS identifies the top-K most vulnerable edges quickly, then (2) full AC power flow N-1 analysis is run only on the top-K set, reducing computation by a factor of $|E|/K$. For a 10,000-bus grid with 15,000 lines, reducing from 15,000 to 500 full AC solves would provide meaningful computational savings if AEGIS can maintain P@500 > 0.95 at that scale. This use case is not discussed in the paper but would be the strongest practical argument.

## Scores (0-100 scale)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Cross-disciplinary Value | 78 | Genuinely novel bridge between adversarial ML and power systems contingency analysis; the structural isomorphism is well-identified but incompletely developed |
| Application Validity | 62 | IEEE test case results are encouraging but limited by binary adjacency, narrow training data, and degraded model quality at scale (case300 $\theta$ RMSE = 0.394) |
| Practical Impact | 55 | Currently slower than brute-force N-1 on tested grids; value proposition depends on scaling to large grids, which is untested in the power domain; P@10 insufficient for standalone use |
| Communication Quality | 74 | Good bridging of both audiences; honest operational caveats; theory sections dense but well-motivated; missing key power systems context (thermal limits, voltage stability, PTDF) |
| Broader Significance | 80 | Framework generalizes to any domain where graph edge failure drives system-level consequences; 7-architecture validation and matrix-free scalability are strong contributions beyond power systems |
| Overall Perspective | 72 | A promising cross-disciplinary contribution that opens a new research direction; the power grid case study is well-conceived but needs deeper engagement with power systems reality to be fully convincing |

## Questions for Authors

1. **Impedance encoding.** The binary adjacency result is surprising and well-argued for the all-or-nothing N-1 case. But for operational use, one needs to rank contingencies by *severity*, which depends on impedance-driven power redistribution. Have you experimented with edge-feature GNNs (where impedance, thermal rating, and voltage level are edge attributes rather than adjacency weights)? This would preserve the binary connectivity structure that works well while adding the electrical information needed for severity ranking.

2. **Full-graph matrix-free for case300.** The matrix-free pipeline handles Cora (N=2,708) in 78 seconds. Why was a 200-node subgraph extraction used for case300 (N=300) instead of full-graph matrix-free analysis? If the matrix-free pipeline were applied to case300 (and potentially to larger IEEE cases like the 2383-bus Polish system), would the P@10 improve or degrade relative to the subgraph result?

3. **Where do AEGIS and LODF disagree?** You report aggregate $\tau$ values but do not analyze the disagreement structure. On case118, which specific lines does AEGIS rank in the top-10 but LODF does not, and vice versa? Do the AEGIS-unique critical lines correspond to those with high reactive power sensitivity or voltage-constrained corridors---i.e., effects that the DC-based LODF cannot capture? This analysis would demonstrate the specific value-add of an AC-trained GNN over DC linearization.

4. **Contingency severity metric.** The paper describes "brute-force contingency ranking" based on "voltage and flow deviations" but does not specify the exact severity index. Is it $\max_i |\Delta V_i|$, $\sum_i (\Delta V_i)^2$, maximum post-contingency line loading, or a composite performance index? The choice of severity metric significantly affects the ground-truth ranking and thus the meaning of the reported $\tau$.

5. **Generator dispatch sensitivity.** N-1 contingency rankings are dispatch-dependent: a line that is critical at summer peak may be non-critical at spring light load because the power flow direction reverses. Your uniform load scaling preserves the relative dispatch pattern. Have you tested whether AEGIS rankings are stable across different dispatch scenarios (e.g., training on multiple dispatch patterns and comparing per-dispatch vulnerability rankings)?

## Recommendation

**Minor Revision.** Accept with revisions addressing the following priority items:

The paper presents a genuinely novel cross-disciplinary contribution. The structural isomorphism between adversarial edge sensitivity and N-1 contingency is well-identified, and the empirical validation across five IEEE test cases with honest operational caveats demonstrates intellectual maturity. The broader AEGIS framework (7 architectures, 9 datasets, matrix-free scalability) is a strong ML contribution in its own right.

The power grid case study, however, needs targeted strengthening to be fully convincing to both audiences:

**Required revisions:**
- Specify the N-1 severity metric explicitly (Weakness 4).
- Add full-graph matrix-free analysis for case300, or explain why the subgraph was used when matrix-free should handle N=300 (Weakness 6).
- Investigate and address the case300 model quality issue ($\theta$ RMSE = 0.394; Weakness 7).
- Add one paragraph acknowledging the transient/voltage stability limitation and the dispatch-dependence of rankings (Weaknesses 5, Question 5).

**Recommended revisions (would strengthen the paper significantly):**
- Analyze AEGIS-vs-LODF disagreement structure on case118 (Question 3).
- Discuss edge-feature GNN alternatives to binary adjacency (Question 1).
- Report vulnerability rankings stratified by endpoint bus types (Weakness 8).
- Discuss the pre-screening filter use case as the realistic deployment scenario (Practical Deployment Assessment).

The theoretical contributions (Theorem 1, Proposition 3, Observation 1) are sound and the experimental methodology is thorough. The paper's weaknesses are primarily in the depth of power systems engagement, which is addressable through targeted revisions without altering the core contribution. This work is a valuable step toward bridging adversarial ML and safety-critical infrastructure analysis.
