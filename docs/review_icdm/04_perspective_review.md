# Reviewer 3 (Perspective / Cross-Disciplinary) -- IEEE ICDM Review

**Paper:** AEGIS: Mining Graph Structure for Adversarial Vulnerability Analysis of GNNs

**Reviewer expertise:** Power systems engineering (N-1/N-k contingency analysis, LODF/PTDF screening, AC/DC power flow), network reliability, critical infrastructure protection, ML applications in power systems.

---

## 1. Summary (Cross-Disciplinary Perspective)

AEGIS proposes a sensitivity-matrix framework that, given any trained differentiable GNN and a graph, produces per-edge vulnerability rankings, an SVD-optimal attack direction, and per-node tolerance radii. The central object is the constrained sensitivity matrix S_c, which projects the full N^2-dimensional perturbation space onto the |E|-dimensional space of realistic (symmetric, edge-only) perturbations. For contractive implicit GNNs (IGNN-class), the framework additionally provides a critical perturbation budget and convergence guarantees via the implicit function theorem. The paper validates across 7 architectures and 9 datasets spanning citation networks, e-commerce, encyclopedias, and power grids. The cross-disciplinary contribution is a power grid case study (Section VII) where the vulnerability spectrum is shown to recover N-1 contingency rankings on IEEE test cases (case14 through case118) with P@10 = 0.66--0.81, without any domain-specific power systems inputs beyond the graph topology and standard bus features. The paper positions AEGIS as a screening layer analogous to LODF in power systems, bridging adversarial ML and classical infrastructure reliability analysis. The practical applicability claim -- that structural sensitivity analysis transfers to real engineering problems -- is the paper's most ambitious and most scrutiny-worthy assertion.

---

## 2. Strengths

- **S1. Genuine cross-disciplinary bridge.** The paper draws an explicit and technically sound analogy between adversarial edge perturbation in GNNs and N-1 contingency analysis in power systems (Section VII). The mapping -- edge perturbation to line trip, vulnerability spectrum to contingency severity, critical budget to stability margin -- is conceptually clean and would be recognizable to a power systems engineer. This is not a superficial application; it connects two literatures that rarely interact.

- **S2. Honest self-assessment of the case study.** The paper explicitly states that AEGIS is "a screening layer, not a standalone contingency tool" and that tau = 0.37--0.67 "is insufficient for direct operational use" (Section VII). This intellectual honesty is commendable and rare in cross-disciplinary ML papers, which often overclaim.

- **S3. LODF baseline comparison provides calibration.** Comparing against LODF -- the industry-standard DC approximation for contingency screening -- gives readers from the power systems community a meaningful reference point. AEGIS outperforming LODF on case57/case118 (tau = 0.62--0.67 vs. 0.44--0.58) is a noteworthy result, particularly because AEGIS does not require line reactances.

- **S4. Equilibrium-physics observation is insightful.** The finding that IGNN achieves low power-balance residuals (Delta S = 0.03--0.11 p.u.) without any physics loss term, and the comparison against explicit GCN and PIGNN (Section VII), is a genuine contribution to the understanding of implicit models in physics-informed ML. The hypothesis that the fixed-point condition absorbs structural constraints from the topology is well-supported.

- **S5. Breadth of validation is impressive.** 7 architectures, 9 datasets, 4 domains, 10 seeds each. The cross-architecture generalization (Table V) is particularly important because it demonstrates that the S_c framework is not tied to a single model class, increasing practical applicability.

- **S6. Practical speedup is meaningful.** For case118, AEGIS ranks in 2.3s vs. 179 full AC solves for brute-force N-1. While the N-1 brute-force is not prohibitively expensive for 118 buses, this speedup ratio becomes operationally significant as grid size scales.

- **S7. Defense-informed edge protection (Section VI-G).** Masking the top-5 AEGIS-ranked edges reduces attack damage by 42% vs. 11% for random masking. This directly demonstrates actionable utility: a defender can use the vulnerability map to prioritize hardening.

- **S8. Matrix-free scalability removes a genuine barrier.** The transition from O((Nd)^3) to O(K * Nd) per evaluation, enabling full Cora (N=2,708) in 78s with 1 GB, is a legitimate engineering contribution. Without this, the framework would be limited to toy-size graphs.

---

## 3. Weaknesses

- **W1. IEEE test cases are too small for credible power systems claims.**
  - *Evidence:* case14 (14 buses, 20 edges), case30 (30 buses, 41 edges), case57 (57 buses, 78 edges), case118 (118 buses, 179 edges). These are pedagogical benchmarks. Real transmission grids have 2,000--70,000+ buses. The Polish 2383-bus system, PEGASE 1354/2869/9241, and synthetic Texas 2000-bus cases are standard in modern power systems research.
  - *Impact:* A reviewer at IEEE PES or PSCC would immediately flag the absence of grids above 200 buses. It is unknown whether AEGIS's correlation holds on larger, more realistic topologies with meshed structures, multiple voltage levels, and heterogeneous generation.
  - *Suggested fix:* Add at least one medium-scale grid (PEGASE 1354 or Polish 2383). If full AC power flow training is too expensive, use the DC approximation for the larger case and report it as such.

- **W2. The continuous perturbation model has limited fidelity for power systems contingency.**
  - *Evidence:* The threat model (Section II-B) restricts perturbations to continuous edge-weight modifications. N-1 contingency analysis involves discrete line removal (0/1). The paper acknowledges this mismatch (Section VIII, Limitation 2) and shows positive tau for continuous-to-discrete transfer (Table V).
  - *Impact:* In power systems, what matters is whether a line is in or out -- not how much its admittance is scaled. The continuous model misses threshold effects (e.g., a line at 99% loading that trips on a 1% perturbation). Thermal limits, protection relay logic, and cascading failures are inherently discrete. The tau = 0.37--0.67 likely reflects this mismatch.
  - *Suggested fix:* Add a discrete-removal experiment in the power flow case study: for each edge, compare the S_c vulnerability score against the actual power flow shift from removing that edge entirely (not just the continuous first-order prediction). Report tau and P@10 for this discrete ground truth specifically for the power grids. This is already done for citation networks (Table V) but not reported separately for power grids.

- **W3. Power flow model quality is marginal for larger grids.**
  - *Evidence:* Table VI reports per-unit RMSE for |V| (0.007--0.033), theta (0.020--0.076), and Delta S (0.033--0.106). On case14, Delta S = 0.106 p.u. means the predicted voltages violate Kirchhoff's current law by about 10% of nominal per bus on average. On case57, theta RMSE = 0.059 rad (3.4 degrees).
  - *Impact:* For power systems work, theta errors above 2 degrees and Delta S above 0.05 p.u. are significant. The case14 Delta S = 0.106 p.u. would not pass muster at IEEE PES as a credible power flow solver. The vulnerability ranking may still be useful (P@10 is reasonable), but the underlying model is not physically trustworthy for quantitative power flow prediction.
  - *Suggested fix:* (a) Report model quality metrics more prominently -- ideally with comparison to the Newton-Raphson reference. (b) Consider adding a physics-informed loss term (power balance penalty) to improve model fidelity. (c) Acknowledge explicitly that the model is used for ranking, not for quantitative power flow prediction. (d) Compare against DC power flow (B-theta) as an additional baseline -- DC-PF is trivially accurate for voltage angles on meshed grids.

- **W4. LODF baseline comparison is not fully fair.**
  - *Evidence:* Section VII states LODF achieves tau = 0.44--0.58 vs. AEGIS's tau = 0.37--0.67. On case14, LODF (tau = 0.44) actually outperforms AEGIS (tau = 0.42). On case30, LODF (not separately reported, inferred from range) is competitive.
  - *Impact:* LODF is a linearized DC-based method that runs in microseconds, requires only the B-matrix (line reactances), and is already deployed in every ISO/RTO control room worldwide. AEGIS requires training a GNN from scratch on thousands of power flow samples, plus the IFT computation. The comparison should include wall-clock time and data requirements. LODF also provides actionable quantities (MW flow redistribution on each line), whereas AEGIS provides a unit-free sensitivity norm.
  - *Suggested fix:* (a) Report LODF wall-clock time alongside AEGIS's 2.3s. (b) Report data generation cost (2,000 AC power flow solves for training). (c) Discuss when AEGIS would be preferred over LODF -- the argument should be about capturing nonlinear AC effects (reactive power, voltage, losses) that DC-based LODF misses, but this needs to be demonstrated explicitly on cases where DC approximation fails (e.g., high-R/X ratio distribution grids).

- **W5. No treatment of N-2 or N-k contingencies.**
  - *Evidence:* The paper focuses exclusively on N-1 (single-line removal). N-2 and N-k contingency analysis is increasingly important for grid reliability (NERC TPL standards require N-2 for certain categories). The SVD-optimal attack direction from AEGIS is inherently multi-edge (it perturbs all edges simultaneously), so there is a natural connection to N-k.
  - *Impact:* N-2 screening is where brute-force becomes truly expensive (O(|E|^2) solves) and where a screening tool like AEGIS could provide the most value. Missing this application is a significant lost opportunity.
  - *Suggested fix:* (a) Discuss the connection between the SVD-optimal attack and N-k contingency. (b) Report the top-k edges from the SVD attack direction v_1 and evaluate whether they correspond to known N-2 critical pairs. Even a brief analysis on case118 would substantially strengthen the case study.

- **W6. Binary adjacency vs. admittance weighting result is underexplored.**
  - *Evidence:* Section VII notes that binary adjacency outperforms admittance-weighted (P@10 = 0.81 vs. 0.27) "because N-1 contingency is a discrete event better modeled by uniform sensitivity."
  - *Impact:* This is a surprising and counterintuitive result. In power systems, line parameters (impedance, thermal rating) are fundamental to contingency severity. A line with 10 ohms of reactance has a very different impact from one with 0.1 ohms. The explanation offered ("discrete event better modeled by uniform sensitivity") is not convincing -- a high-impedance line removal causes less redistribution than a low-impedance one, which is exactly what admittance weighting would capture.
  - *Suggested fix:* Investigate this result more deeply. Does the binary model succeed because the IGNN is learning impedance-like information from the power flow training data? Or is it an artifact of the small test cases where most lines have similar impedances? Report the impedance variance across test cases and correlate it with the binary vs. weighted gap.

- **W7. No uncertainty quantification for the vulnerability rankings.**
  - *Evidence:* Table VI reports tau with standard deviations across 10 seeds (e.g., +0.42 +/- 0.19 for case14). The variance is large -- case14's tau ranges roughly from 0.23 to 0.61 across seeds.
  - *Impact:* For operational use, a screening tool must provide consistent rankings. A tau that swings from 0.23 to 0.61 depending on the random seed is unreliable. Power systems operators need confidence bounds on the vulnerability ranking, not just a point estimate.
  - *Suggested fix:* (a) Report the rank stability across seeds: how often does the same edge appear in the top-10 across all 10 seeds? (b) Consider ensembling across seeds to produce a consensus ranking with confidence intervals. (c) Discuss what drives the variance -- is it model training stochasticity or sensitivity of the IFT computation?

---

## 4. Power Grid Case Study Assessment

### 4.1 N-1 Contingency Framing Accuracy

The framing is conceptually accurate. The mapping from edge perturbation to line outage, vulnerability spectrum to contingency severity, and critical budget to stability margin is a valid analogy. However, there are important nuances the paper glosses over:

- **N-1 contingency is fundamentally about post-contingency state feasibility** (voltage limits, thermal limits, transient stability), not just about the magnitude of the equilibrium shift. AEGIS measures ||Delta z*||_F, which captures total hidden-state displacement but does not directly map to voltage violations, line overloads, or frequency deviations.
- **The paper conflates contingency severity ranking with contingency screening.** In practice, screening answers "does this contingency cause a violation?" (binary), while ranking orders contingencies by severity. AEGIS does ranking, not screening. This distinction matters operationally.

### 4.2 IEEE Test Cases: Appropriateness

Case14/30/57/118 are appropriate for proof-of-concept validation. They are NOT appropriate for claiming practical applicability. These cases are used in textbooks and introductory courses. Any serious power systems publication would require at least one grid above 1,000 buses.

- case14: 5 generators, 14 buses -- essentially a toy. Almost all lines are critical because the grid is radial/near-radial.
- case30: similar limitations.
- case57: somewhat more realistic with meshed structure.
- case118: the most credible test case, but still far below real-grid complexity.

The absence of larger grids (PEGASE 1354, Polish 2383, ACTIVSg 2000/10000) is the single largest weakness of the case study from a power systems perspective.

### 4.3 LODF Baseline Fairness

The LODF comparison is directionally correct but incomplete:

- **Fairness:** LODF is a DC-based linearization that ignores reactive power and voltage. Comparing it against an AC-trained GNN is somewhat favorable to AEGIS (AEGIS captures AC effects that LODF cannot). This is fine, but should be acknowledged more explicitly.
- **Cost:** LODF computation from the B-matrix takes microseconds and requires no training data. AEGIS requires 2,000 AC power flow solves for training plus model training time plus IFT computation. The total cost comparison heavily favors LODF for small grids.
- **Interpretability:** LODF gives MW redistribution on each line (directly actionable for operators). AEGIS gives a unit-free sensitivity norm (requires interpretation).

### 4.4 Kendall tau = 0.37--0.67: Practical Meaning

From a power systems screening perspective:

- **tau = 0.37 (case30):** This is weak. It means the ranking is better than random but would miss many critical contingencies. Not operationally useful.
- **tau = 0.67 (case57):** This is moderate. Combined with P@10 = 0.66, it means about 2/3 of the top-10 critical lines are correctly identified. For a first-pass screening to reduce the N-1 list before detailed analysis, this could be useful.
- **P@10 = 0.81 (case118):** This is the strongest result and is operationally meaningful. Correctly identifying 8 of the top 10 critical lines in a 179-line grid with a 2.3-second computation is a credible screening contribution.

Overall assessment: **Useful as a screening pre-filter for reducing the number of full AC power flow solves required, but not reliable enough to replace N-1 analysis.** The paper's own characterization ("screening layer, not standalone tool") is appropriate.

### 4.5 Power Flow Model Physical Meaningfulness

The ContractiveGCN-PF model produces:
- |V| RMSE: 0.007--0.033 p.u. (acceptable for screening; 0.033 p.u. = 3.3% voltage error is marginal)
- theta RMSE: 0.020--0.076 rad (1.1--4.4 degrees; 4.4 degrees is significant for angle-based stability)
- Delta S: 0.033--0.106 p.u. (power balance residuals; 0.106 is poor)

**Would a power systems practitioner trust these results?** For contingency ranking (ordinal): cautiously yes, given P@10 results. For quantitative power flow prediction (cardinal): no. The Delta S = 0.106 on case14 means the model is not solving the power flow equations to any reasonable accuracy.

The comparison with PIGNN-Attn-LS is interesting -- IGNN matches a purpose-built physics-informed model without physics loss. But both are far from Newton-Raphson accuracy.

### 4.6 N-2 and N-k Contingencies

The paper does not address N-2 or N-k contingencies, which is a missed opportunity. The SVD decomposition naturally produces a multi-edge perturbation direction (v_1 has nonzero components on all edges). The top-2 or top-3 edges in v_1 could be interpreted as an N-2 or N-3 critical set. This would be a much more compelling demonstration of AEGIS's value because:

1. N-2 brute-force is O(|E|^2), making screening far more valuable.
2. N-k interactions (where two individually non-critical lines are jointly critical) are exactly the kind of nonlinear effect that first-order sensitivity analysis might capture.

---

## 5. Practical Applicability Assessment

### 5.1 Applications Beyond Power Grids

The framework has natural applicability to several domains:

- **Telecommunications networks:** Link vulnerability analysis for fiber/backbone networks. The continuous perturbation model maps well to signal attenuation (not binary failure).
- **Supply chain networks:** Identifying critical supplier-manufacturer edges whose disruption maximally propagates.
- **Financial transaction networks:** Fraud detection robustness -- which edges, if manipulated, would evade detection? (Already mentioned in the introduction.)
- **Water distribution networks:** Pipe criticality analysis, analogous to power grid contingency.
- **Social network manipulation:** Identifying edges whose removal most changes community structure or information flow.

### 5.2 Deployment Barriers

1. **Training requirement:** AEGIS requires a trained GNN on the target graph. For power systems, this means generating thousands of power flow solutions. For other domains, labeled training data may not be available.
2. **Computational cost:** 78 seconds for Cora (N=2,708). Production social networks have millions of nodes. The paper acknowledges the N~5,000 ceiling.
3. **Interpretability:** The vulnerability score v_ij is a norm of sensitivity columns -- it has no direct physical interpretation (MW, voltage, etc.). Power systems operators work with MW, MVAr, kV, and degrees. A bridge to physical units is needed.
4. **Integration with existing tools:** Power systems operators use PSS/E, PowerWorld, PSCAD. There is no integration pathway described.
5. **Model maintenance:** If the grid topology or generation dispatch changes, the GNN must be retrained and AEGIS recomputed. N-1 analysis with Newton-Raphson adapts to the current operating point automatically.

### 5.3 Continuous Perturbation Model vs. Real-World Threats

- **Power grids:** Threats are discrete (line trips, generator outages, cyberattacks on breakers). The continuous model is a linearized approximation, analogous to PTDF/LODF. Acceptable for screening; not for detailed analysis.
- **Social networks:** Adversarial edge manipulation (fake connections) is discrete. The continuous model is an approximation.
- **Fraud detection:** Transaction graph manipulation can be continuous (modifying transaction amounts) or discrete (adding/removing edges). The continuous model is more natural here.
- **Molecular graphs:** Bond perturbations are inherently discrete. Poor match.

### 5.4 Scalability to Production Graphs

The matrix-free pipeline reaches N=2,708 (Cora) in 78s. Production-scale graphs:
- Power grids: 2,000--70,000 buses. case118 is handled; larger grids need demonstration.
- Social networks: millions of nodes. Far beyond current capability.
- Financial networks: thousands to millions. Marginal at best.

The paper's future work mentions distributed JVP computation for N > 5,000, but no concrete path is provided.

---

## 6. Broader Impact and Ethics

### 6.1 Dual-Use Concerns

The paper explicitly identifies and addresses dual-use risk (Section VIII). The vulnerability map reveals optimal attack directions and the most critical edges. In power systems, this is information that could be used by adversaries to target specific transmission lines for maximum grid disruption.

**Assessment:** The dual-use concern is real but manageable:

1. **N-1 contingency results are already publicly available** for most grids through NERC reliability assessments and ISO planning studies. AEGIS does not reveal fundamentally new information for power grids.
2. **The GNN attack surface is less obvious.** For fraud detection or social network applications, the vulnerability map could enable targeted evasion. This is the more concerning use case.
3. **The defense application (Section VI-G) is concrete and compelling.** The same map that identifies vulnerable edges directly informs which edges to protect. This is the standard argument in adversarial ML, and the paper's defense ablation supports it.

### 6.2 Responsible Disclosure

The paper promises code release with responsible-use guidelines. This is appropriate. For power systems applications specifically, the training data (IEEE test cases) is public and the grids are not real infrastructure. For deployment on real grids, the vulnerability rankings should be treated as Critical Energy Infrastructure Information (CEII) under FERC regulations.

---

## 7. Questions for Authors

**Q1.** The binary adjacency outperforming admittance-weighted adjacency (P@10 = 0.81 vs. 0.27) is counterintuitive from a power systems perspective. Have you investigated whether the IGNN is implicitly learning impedance-like information from the power flow training data? Specifically, what happens if you train on a grid with highly heterogeneous line impedances (e.g., a mix of 69 kV and 345 kV lines in the same model)?

**Q2.** The tau variance across seeds is large (e.g., +0.42 +/- 0.19 on case14). What is the rank stability of the top-10 vulnerable edges across seeds? If the same edges consistently appear in the top-10 regardless of seed, the mean tau may be misleading -- the ordinal top-set may be more stable than the full ranking correlation suggests.

**Q3.** Have you considered connecting the SVD-optimal attack direction v_1 to N-2 contingency analysis? Specifically, if the top-2 components of v_1 identify edges (i,j) and (k,l), does their simultaneous removal correspond to a known critical N-2 pair in the IEEE test cases?

**Q4.** For practical deployment in power systems, how would AEGIS handle topology changes (switching operations, line maintenance outages) that alter the graph structure? Would the entire GNN need to be retrained, or could the IFT computation be updated incrementally?

**Q5.** The Delta S = 0.106 p.u. on case14 suggests significant Kirchhoff violations. Have you investigated whether the vulnerability ranking quality (tau, P@10) degrades when the underlying power flow model is less accurate? A controlled experiment degrading model quality and tracking ranking correlation would help establish the minimum model fidelity required for useful screening.

---

## 8. Scores

| Dimension | Score (0--100) | Justification |
|---|---|---|
| **Cross-Disciplinary Value** | 72 | Genuine and well-executed bridge between adversarial ML and power systems. The analogy is technically sound and the paper is honest about limitations. Loses points for not engaging deeply enough with the power systems literature (no PTDF discussion, no larger grids, no N-k). |
| **Practical Applicability** | 48 | The framework has clear conceptual applicability but faces significant deployment barriers: continuous perturbation model mismatch, scalability ceiling at N~5,000, lack of physical-unit interpretability, and no integration pathway with existing tools. The power grid case study demonstrates proof-of-concept, not practical readiness. |
| **Case Study Quality** | 55 | Honest and well-structured, with appropriate baselines (LODF) and caveats. Weakened by toy-scale test cases (max 118 buses), marginal model accuracy (Delta S = 0.106), large tau variance, and missing N-2 analysis. Would not satisfy IEEE PES reviewers but is acceptable for a case study in a data mining venue. |
| **Broader Impact** | 65 | Dual-use concerns are real but responsibly addressed. The defense application (edge protection ablation) is concrete. The framework's potential for network vulnerability analysis extends beyond power grids to telecom, supply chain, and financial networks. |
| **Overall** | **Weak Accept** | The paper makes a genuine and technically sound cross-disciplinary contribution. The power grid case study, while limited in scale, demonstrates that structural sensitivity analysis transfers to real engineering problems. The honesty about limitations is commendable. The primary weaknesses -- toy-scale power grids, continuous perturbation mismatch, and scalability ceiling -- are addressable in revision. For a data mining venue (ICDM), the cross-disciplinary demonstration is sufficient; for a power systems venue (IEEE PES), it would require substantially larger grids and deeper engagement with power systems methodology. |

---

*Reviewer 3 -- Perspective / Cross-Disciplinary*
*Expertise: Power systems, network reliability, critical infrastructure*
