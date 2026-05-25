# Perspective Review Report -- AEGIS

## Reviewer Profile
- **Role**: Peer Reviewer 3 (Cross-Disciplinary Perspective)
- **Expertise**: Power systems engineering, applied ML for critical infrastructure
- **Review Focus**: Cross-disciplinary connections, practical impact, broader implications

## Summary Assessment

AEGIS presents a mathematically elegant framework for structural vulnerability analysis of GNNs, grounded in the implicit function theorem and constrained sensitivity matrices. The core ML contribution -- the constrained sensitivity matrix $S_c$ that reduces perturbation space from $N^2$ to $|E|$ dimensions while enforcing symmetry -- is sound and well-validated across 7 architectures and 9 datasets. The first-order tightness results (1.00 +/- 0.01 at epsilon=0.01) are impressive and the SVD-optimal attack's 2-8x advantage over random perturbation is convincing.

The power systems case study (Section VII) is the most novel and cross-disciplinarily interesting contribution, but also the most vulnerable to scrutiny from a domain perspective. The N-1 contingency analogy is conceptually appealing -- edge perturbation maps to line trips, vulnerability spectrum maps to contingency severity -- but the mapping involves several engineering simplifications that the paper only partially acknowledges. The use of binary adjacency rather than admittance-weighted edges, the continuous perturbation model versus discrete line outages, and the uniform load scaling training data all limit the operational relevance of the results. The authors are appropriately cautious in calling tau=0.37-0.67 "insufficient for direct operational use," but this honesty also undercuts the practical motivation.

The paper occupies an interesting niche: it is neither a pure attack paper nor a pure defense paper, but a diagnostic tool. This positioning is a strength for the ML community but creates tension in the power systems framing, where practitioners need actionable tools with well-characterized failure modes. The dual-use implications of providing an attack toolkit for safety-critical infrastructure receive no discussion, which is a notable omission for a paper that explicitly targets "safety-critical domains."

## Scores (0-100)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Cross-Disciplinary Value | 72 | The power systems bridge is genuinely novel but the engineering model is too simplified to convince domain experts. The conceptual connection between IFT sensitivity and contingency analysis is valuable but underdeveloped. |
| Practical Impact | 58 | The N~300 subgraph limit and 2.3s single-IFT timing are useful for screening on small grids but do not scale to real transmission networks (1000+ buses). Binary adjacency discards impedance information that is critical for contingency severity ranking. |
| Case Study Validity | 62 | P@10 = 0.66-0.81 is promising for a topology-only method, but the comparison with LODF is not entirely fair (LODF uses reactances; AEGIS uses a trained model). The claim of "outperforming on larger grids" (tau=0.62-0.67 vs LODF 0.44-0.58 on case57/118) needs more careful analysis of what each method captures. |
| Broader Implications | 65 | The framework's generality across GNN architectures is well-demonstrated, but the paper misses opportunities to connect to other safety-critical network domains. No discussion of dual-use risks. |
| Ethical Considerations | 40 | Complete absence of dual-use discussion for a paper that provides attack optimization tools and explicitly targets safety-critical infrastructure. This is a significant gap. |
| Overall | 63 | Solid ML contribution with an ambitious but underdeveloped cross-disciplinary case study. The power systems framing needs substantial strengthening or honest rescoping. |

## Strengths

1. **Sound mathematical framework with practical tightness.** The constrained sensitivity matrix $S_c$ (Section IV, Eq. 7) is a genuine contribution that makes first-order sensitivity analysis practical for realistic graph perturbations. The $N^2 \to |E|$ reduction with enforced symmetry is elegant and the empirical tightness of 1.00 +/- 0.01 at epsilon=0.01 (Table I) validates the theory convincingly.

2. **Architecture-agnostic generalization.** Proposition 4 (Section IV-D) and the validation across 7 architectures (Table IV) demonstrate that $S_c$ is not an IGNN-specific trick. The unrolled sensitivity matrix $S_K$ for explicit GNNs provides a unified framework. The acknowledgment that explicit models lack the critical budget $\varepsilon_{crit}$ and convergence regime guarantees is appropriately honest.

3. **Appropriate calibration of claims in the power systems case study.** The operational caveat in Section VII-C ("AEGIS is a screening layer, not a standalone contingency tool") is commendably honest. The training data limitation ("covers uniform load scaling but not seasonal or generator-outage variation") is also clearly stated. This calibration builds trust.

4. **The LODF comparison provides a meaningful engineering baseline.** Comparing against the industry-standard DC screening tool (Section VII-C) rather than only ML baselines shows awareness of domain practice. The observation that AEGIS outperforms LODF on larger grids by "capturing nonlinear AC effects" is an interesting hypothesis worth further investigation.

5. **The implicit physics observation is genuinely interesting.** The finding that ContractiveGCN-PF achieves Delta-S = 0.03-0.11 p.u. without explicit power-balance penalty (Section VII-C) connects equilibrium GNN architecture to physical self-consistency in a way that could inform future physics-informed ML design.

6. **Defense-informed edge protection experiment.** Section VI-G demonstrates practical utility: masking the top-5 AEGIS-identified edges reduces SVD attack damage by 42 +/- 8% vs 11 +/- 6% for random masking. This is directly actionable for defense design.

## Weaknesses

1. **Binary adjacency discards critical engineering information.** The paper reports that "binary adjacency outperforms admittance-weighted (P@10 = 0.81 vs 0.27)" (Section VII-C), interpreting this as evidence that "N-1 contingency is a discrete event better modeled by uniform sensitivity." This interpretation is problematic. In power systems engineering, the severity of a line trip is determined precisely by the line's impedance and loading relative to the network: a tripped 500kV tie-line with 1000MW flow causes far more disruption than a tripped 69kV radial feeder at 10MW. The poor performance of admittance-weighted adjacency likely reflects a feature engineering failure (e.g., poor normalization of admittance values) rather than a fundamental property of contingency analysis. **Suggestion**: Investigate whether log-admittance or normalized susceptance weighting improves P@10 before concluding that binary is inherently superior. Report the admittance-weighted model's training loss and voltage RMSE to rule out a training failure.

2. **The N-1 contingency analogy conflates continuous perturbation with discrete outage.** AEGIS perturbs edge weights continuously (Section III-B: "edge weights are perturbed continuously in R, not discretely flipped"), but N-1 contingency is inherently discrete: a line is either in service or out. The paper acknowledges this ("continuous first-order vs. discrete removal") but does not quantify how much this mismatch degrades the ranking. LODF, despite being a linearization, at least models the correct discrete event (full line removal). **Suggestion**: Add an experiment comparing AEGIS rankings when using large epsilon (approaching full edge removal) versus the small-epsilon first-order rankings. If the two diverge significantly, the analogy weakens.

3. **Uniform load scaling training data severely limits generalization claims.** The ContractiveGCN-PF is trained on "2,000 load samples per case, uniformly sampled at 70-130% of nominal load" (Section VII-B). Real power grids operate with spatially heterogeneous load patterns, generator redispatch, and renewable intermittency. A model trained on uniform scaling will not capture contingency severity that depends on the specific dispatch pattern. **Suggestion**: At minimum, test on a few non-uniform load scenarios (e.g., heavy north-south transfer, peak summer, light load with high renewable penetration) to characterize sensitivity to operating point. If P@10 degrades substantially, the "recovers N-1 contingency rankings" claim should be qualified.

4. **No discussion of dual-use risks.** The paper explicitly targets "safety-critical domains" (abstract, introduction) and provides an optimized attack toolkit (SVD-optimal perturbation directions, per-edge vulnerability rankings). For power grids specifically, identifying the most critical lines to attack is precisely the information a malicious actor would seek. The paper provides no discussion of responsible disclosure, access controls, or the ethics of publishing attack tools for critical infrastructure. **Suggestion**: Add a dedicated "Ethical Considerations" subsection discussing: (a) the dual-use nature of vulnerability analysis, (b) how the screening-layer positioning mitigates risk (an attacker still needs physical access to trip lines), (c) whether code release should include safeguards, and (d) how the defensive application (edge protection, Section VI-G) balances the offensive capability.

5. **The LODF comparison is not apples-to-apples.** LODF requires line reactances but no training data; AEGIS requires 2,000 training samples from a full AC solver but no line parameters. The paper claims AEGIS "outperforms on larger grids" but does not discuss the practical cost of generating training data, which requires the same Newton-Raphson solver that brute-force N-1 uses. For a fair comparison, report the total computation (training data generation + model training + AEGIS analysis) versus brute-force N-1. **Suggestion**: Add a wall-clock comparison including data generation time. If generating 2,000 AC power flow samples takes longer than 179 brute-force N-1 solves, the computational advantage claim is misleading.

6. **The tau=0.37-0.67 range has high variance across grid sizes with no clear explanation.** Case14 achieves tau=0.42, case30 drops to tau=0.37, then case57 jumps to tau=0.67 and case118 achieves tau=0.62. This non-monotonic behavior is unexplained. Is case30 harder because of its meshed topology? Does case57's high tau reflect a more tree-like structure where topological sensitivity better predicts contingency severity? **Suggestion**: Analyze what structural properties (mesh density, tree-likeness, degree distribution) correlate with AEGIS ranking accuracy. This would help practitioners judge when AEGIS is trustworthy.

7. **Bus type features embed domain knowledge that undermines "without domain-specific inputs."** The model uses 5 bus features including "bus type indicators (slack, PV)" (Section VII-B). The claim that AEGIS works "without domain-specific modifications" (abstract) and "without domain-specific inputs" (abstract) is technically about the AEGIS analysis stage, but the underlying model encodes significant domain knowledge through these features. A power systems engineer would note that slack bus identification and PV/PQ bus classification are non-trivial domain knowledge. **Suggestion**: Clarify that "without domain-specific inputs" refers to the AEGIS analysis pipeline, not the GNN model itself. Consider an ablation removing bus-type features to test how much domain knowledge the model actually needs.

8. **The N~300 subgraph limit is impractical for real transmission networks.** Real transmission grids have 1,000-60,000+ buses. The paper notes the limit is N~300 due to dense Jacobian memory (Section V-E: 12.6 GB at N=300), and the conclusion lists "sparse solvers for larger subgraphs" as future work. For the power systems community, this is not a minor limitation -- it means AEGIS cannot analyze even a medium-sized utility's transmission network in its current form. **Suggestion**: Discuss whether hierarchical decomposition (analyzing zones/areas separately) could extend the approach, and whether the BFS ego-subgraph extraction introduces boundary artifacts for power flow problems where electrical distance matters more than topological distance.

## Cross-Disciplinary Opportunities

1. **Chemical process safety and HAZOP analysis.** Chemical plants are modeled as directed graphs of unit operations connected by material/energy streams. The analog of N-1 contingency is single-equipment failure analysis (part of HAZOP). AEGIS could identify which stream disruptions most destabilize a process simulation GNN. The continuous perturbation model maps well to gradual fouling, catalyst deactivation, or flow rate changes. This domain has established safety analysis standards (IEC 61882) that would provide rigorous validation frameworks.

2. **Transportation network vulnerability.** Traffic flow on road/rail networks exhibits equilibrium behavior (Wardrop equilibrium) analogous to the DEQ fixed point. Link removal (road closure, bridge failure) is the transportation N-1 analog. The Bureau of Public Roads function gives a known physics model for comparison, similar to LODF for power systems. AEGIS could identify critical links whose disruption causes disproportionate network-wide delay increases.

3. **Water distribution network resilience.** Water networks are governed by conservation laws (mass balance at nodes, energy balance in loops) that are structurally similar to Kirchhoff's laws. Pipe breaks are the N-1 analog. The hydraulic modeling community has well-established vulnerability analysis tools (e.g., EPANET-based methods) that would provide strong baselines.

4. **Supply chain network stress testing.** Global supply chains are graphs where node/edge disruptions propagate non-locally (COVID-19 demonstrated this dramatically). The continuous perturbation model maps to gradual capacity degradation. Supply chain GNNs are an active research area, and vulnerability analysis would have immediate practical value.

5. **Telecommunications network reliability.** The AEGIS framework maps directly to identifying single points of failure in communication networks, where the analog of contingency analysis is well-established (network survivability analysis). The equilibrium behavior of routing protocols provides a natural DEQ connection.

6. **Connection to classical network reliability theory.** The paper could strengthen its theoretical positioning by connecting $S_c$ to established concepts in network reliability: importance measures (Birnbaum importance, Fussell-Vesely importance) for components in fault trees. The per-edge vulnerability score $v_{ij}$ is conceptually related to Birnbaum structural importance. Making this connection explicit would position AEGIS within a well-understood theoretical landscape.

## Questions for Authors

1. **On the binary vs. admittance-weighted result**: Can you provide the training curves and voltage RMSE for the admittance-weighted model? The dramatic P@10 drop (0.81 to 0.27) suggests a possible training failure rather than a fundamental property. Did you normalize the admittance values? What was the condition number of the admittance-weighted adjacency matrix?

2. **On implicit physics**: You observe that the DEQ architecture recovers approximate power balance (Delta-S = 0.03-0.11 p.u.) without explicit enforcement. Is this genuinely surprising, or is it expected because the model is trained to predict voltages that are generated by a power flow solver that enforces Kirchhoff's laws? The training targets embed the physics; the question is whether the DEQ architecture recovers it more faithfully than an explicit GNN would. Did you compare Delta-S for a 2-layer GCN on the same task?

3. **On computational cost fairness**: How long does it take to generate the 2,000 PandaPower training samples for case118? If it takes more than 2.3 seconds per sample (i.e., >76 minutes total), the total AEGIS pipeline (data generation + training + analysis) is slower than brute-force N-1, which would significantly qualify the "2.3s single IFT" advantage.

4. **On the epsilon_crit interpretation for power grids**: What is the physical meaning of epsilon_crit = (1-kappa)/||W||_2 in the power systems context? Can you map it to a meaningful engineering quantity (e.g., maximum tolerable impedance change, loading margin)?

5. **On the non-monotonic tau across grid sizes**: What structural property of case57 makes it easier for AEGIS (tau=0.67) than case30 (tau=0.37)? Is it related to the ratio of radial vs. meshed topology, the number of parallel paths, or the distribution of line impedances?

6. **On the edge-only constraint**: Real power system attacks can involve node removal (generator trip, bus fault). Your threat model restricts to edge perturbation (Section III-B). How would extending to node perturbation change the analysis? Is there a natural $S_c$ construction for combined node-and-edge perturbation?

7. **On scalability to real grids**: Have you considered applying AEGIS to the ACTIVSg synthetic grids (2000, 10000, 25000 buses)? Even with subgraph extraction, does the BFS ego-subgraph capture enough electrical neighborhood for meaningful contingency ranking on these grids?

8. **On the training data distribution**: If you trained on non-uniform load patterns (e.g., Monte Carlo samples from historical load profiles rather than uniform 70-130% scaling), would you expect P@10 to improve because the model sees more realistic operating conditions, or degrade because the model must generalize over a larger state space?

9. **On responsible disclosure**: Given that AEGIS identifies optimal attack directions on power grid models, have you consulted with grid operators or followed any responsible disclosure protocol before planning code release? IEEE and NERC have specific guidelines for vulnerability disclosure in the power sector.

10. **On the defense application**: The edge protection experiment (Section VI-G) shows 42% damage reduction from masking top-5 edges. In a power systems context, "masking" an edge from the perturbation space would mean hardening a transmission line. Is there a practical mapping from "edge masking" to real protective actions (e.g., adding redundancy, installing FACTS devices, or rerouting power flow)?

## Recommendation

**Minor Revision**, leaning toward Major.

The core ML contribution (the $S_c$ framework) is technically sound and well-validated. The cross-architecture generalization and the tightness results are convincing. However, the paper's ambitious positioning as a tool for "safety-critical domains" is not fully supported by the power systems case study, which contains several engineering simplifications that would concern domain reviewers.

Specific conditions for acceptance:

1. **Required**: Add an ethical considerations subsection addressing dual-use risks of publishing attack optimization tools for critical infrastructure (Weakness 4). This is a minimum requirement for any paper targeting safety-critical applications.

2. **Required**: Clarify the "without domain-specific inputs" claim to distinguish between the AEGIS analysis pipeline and the underlying GNN model, which uses bus-type features (Weakness 7).

3. **Strongly recommended**: Investigate the binary vs. admittance-weighted result more carefully (Weakness 1). Report training diagnostics for the admittance-weighted variant.

4. **Strongly recommended**: Add wall-clock comparison including training data generation time for the power systems case (Weakness 5).

5. **Recommended**: Discuss the non-monotonic tau behavior across grid sizes (Weakness 6) and the limitations of uniform load scaling (Weakness 3).

6. **Recommended**: Compare Delta-S for DEQ vs. explicit GNN to contextualize the "implicit physics" claim (Question 2).

The paper makes a genuine contribution at the intersection of adversarial ML and structural sensitivity analysis, but the cross-disciplinary bridge to power systems needs more careful construction to be convincing to both communities.
