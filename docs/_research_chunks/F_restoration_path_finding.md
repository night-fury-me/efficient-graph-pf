# F. Grid Restoration Path Finding (Survey Research Chunk, A6 — 2023-2026)

Scope: Post-blackout restoration sequencing, black-start, service
restoration in transmission and distribution networks. Methods span
classical mixed-integer optimisation, graph-theoretic combinatorial
formulations, learning-based / hybrid solvers, decision-making under
uncertainty, and resilience-aware multi-agent coordination. The cut
captures the 2023-2026 window and emphasises path-finding (action
ordering, switch sequencing, energisation paths) rather than steady-state
re-dispatch.

---

## Landscape — Classical Optimization

The 2023-2026 service-restoration literature is dominated by mixed-integer
programmes that jointly select switch states, microgrid boundaries, and
generator/DER startup ordering, increasingly coupled with field-crew or
mobile-resource dispatch.

**MILP formulations of restoration sequencing.** Pang et al. (IEEE Trans.
Power Systems, 2023, [PWA-23]) cast dynamic active-distribution-network
restoration as a multi-period MILP that co-optimises repair-crew dispatch
with cold-load pickup, capturing inrush at re-energisation. Xie et al.
(EPSR, 2024, [XIE-24]) extend parallel transmission restoration to
mobile-energy-storage support, modelling MESS routing, dock-charge
constraints, and time-coupled generator cranking. Zhao et al. (J. Modern
Power Systems & Clean Energy, 2024, [ZHA-24a]) include
battery-swap/charge stations and repair-crew routing in a single MILP for
distribution service restoration. Peng et al. (Electronics, 2025,
[PEN-25]) propose a two-stage MILP+MISOCP earthquake restoration that
combines time-domain generator start-up with prioritised load recovery
under voltage/current constraints. Heidari-Akhijahani & Butler-Purry
(Energies, 2023, [HBP-23]) provide a survey of black-start MILP for
active distribution and microgrids, summarising radiality, energisation,
and frequency-pickup constraints.

**Mixed-integer convex relaxations.** Restoration OPF is consistently
solved through SOC or DistFlow relaxation. Zhou et al. (Frontiers in
Energy Research, 2024, [ZHO-24]) cast a fault-repair + sequential
reconfiguration model as a mixed-integer SOC programme, exploiting branch
flow with on/off Big-M. Lou et al. (IJEPES, 2024, [LOU-24]) develop a
three-phase unbalanced SOC restoration for hybrid AC/DC distribution
networks with MESS, handling unbalance through phase-specific cone
constraints. Several IEEE Access and PCMP 2024 papers solve microgrid
service restoration as MISOCP with linearised DistFlow, exploiting tight
formulations for warm-startable MILP solvers.

**Decomposition methods.** Chai et al. (CICED, 2024, [CHA-24]) implement
generalized Benders decomposition for distributed service restoration of
interconnected distribution networks, separating per-area MILP
subproblems from a master coordination cut. Nasiri et al. (IEEE Access,
2024, [NAS-24]) propose a bi-level decentralised distributionally robust
restoration of coupled electricity/natural-gas systems with pump-storage
and wind, solved via privacy-preserving primal-dual decomposition. Zhang
et al. (IEEE TPWRS, 2024, [ZHA-24b]) develop a multi-stage stochastic
restoration of integrated gas-electricity systems with dynamic islanding
that enforces nonanticipativity through scenario decomposition. Column
generation for crew-routing+restoration has appeared in adjacent
power-transportation co-optimisation literature (Wang et al., 2025,
[WAN-25a]).

**Stochastic / robust restoration.** Distributionally robust microgrid
formation is increasingly the default for extended events: Zhu et al.
(IJEPES, 2025, [ZHU-25]) propose Wasserstein-ball ambiguity sets for
load and DER output; Shi et al. (IJEPES, 2024, [SHI-24]) design risk-
averse microgrid formation against subsequent contingencies (cascading
failures during restoration). Yang et al. (Reliability Engineering &
System Safety, 2025, [YAN-25]) handle multi-timescale risk-averse
restoration of interdependent water-power networks. Shi et al. (J. Mod.
Power Syst. & Clean Energy, 2024, [SHI-24b]) include traffic-congestion
uncertainty in transportable-power-source dispatch for restoration.

---

## Landscape — Graph Algorithms

Distribution service restoration is intrinsically a constrained
combinatorial graph problem (radial trees, energisation paths, switch
sequences). Several 2023-2025 papers exploit this structure directly
rather than embedding it in generic MILP.

**Steiner-tree and minimum-spanning-tree formulations.** Saki et al.
(EPSR, 2024, [SAK-24]) develop a distributed minimum-spanning-tree
algorithm for critical-load restoration via microgrid formation in
resilient distribution systems. Each candidate root (DER or
black-start-capable generator) grows a feasible energised subtree
respecting capacity and radiality. Jafarpour & Amirioun (IET GTD, 2023,
[JAF-23]) use a directed multi-commodity flow formulation for
adaptable microgrid formation in coupled electricity/gas distribution,
which is essentially a Steiner-arborescence relaxation with
priority-weighted node weights.

**Network-flow / energisation sequencing.** Time-expanded network-flow
models continue to underpin parallel-restoration formulations. Xie et
al. (EPSR, 2024, [XIE-24]) explicitly model MESS routing on a
time-expanded graph with energisation cuts. Monteiro et al. (IEEE
TPWRS, 2024, [MON-24]) build a hierarchical T+D restoration where
transmission-level energisation arcs feed distribution-level microgrid
subgraphs.

**Resource-constrained shortest paths and routing.** Repair-crew and
mobile-storage dispatch are routinely cast as time-windowed vehicle
routing on an augmented road-grid graph: Shi et al. (J. Mod. Power
Syst. & Clean Energy, 2024, [SHI-24b]) handle traffic-congestion
uncertainty in restoration; Fan et al. (EPSR, 2024, [FAN-24]) couple
power-gas-transportation distribution networks for restoration; Wang
et al. (IEEE Access, 2025, [WAN-25b]) co-optimise distribution-microgrid-
transportation systems with tiered loads. Hu et al. (Electronics, 2025,
[HU-25]) introduce helicopter-scheduling routing (3-D, non-Euclidean
metric) for inaccessible fault sites.

**Quantum / hybrid combinatorial.** Fu et al. (Energy, 2023, [FU-23])
present a hybrid quantum-classical formulation for coordinated
post-disaster restoration of urban distribution systems, encoding the
combinatorial switch decisions on a QUBO and warm-starting a classical
MILP from sampled bitstrings.

---

## Landscape — Learning-Based & Hybrid

The most active sub-field, with explosive growth in 2023-2026 RL and
GNN literature. Three patterns dominate: (i) multi-agent RL with graph
encoders for switch actions, (ii) hybrid RL+MILP feasibility wrappers,
(iii) imitation / meta-learning from synthetic operator traces.

**GNN-based restoration policies.** Wadhwa et al. (Nature Communications,
2024, [WAD-24]) propose real-time outage management in active
distribution networks using RL over graphs, achieving real-time switch
recommendations on IEEE 8500-node feeders. Fan et al. (IEEE Trans.
Artificial Intelligence, 2023, [FAN-23]) introduce attention-based
multi-agent graph RL for service restoration, with topology-aware
attention scoring switch candidates. A 2024-2025 wave of works
(Khattar et al. PESGM 2025, [KHA-25]; Si et al. Applied Energy 2024,
[SIR-24]) embeds GraphSAGE / GAT encoders inside MARL critics for
hierarchical critical-load restoration under uncertain topology.

**RL for restoration (DQN, PPO, MARL).** Igder & Liang (IEEE Trans.
Industry Applications, 2023, [IGL-23]; ICPS 2023, [IGL-23b]) propose
DRL with dynamic microgrid formation; Vu et al. (IEEE Trans. Smart
Grid, 2023, [VU-23]) develop a multi-agent DRL for distributed load
restoration. Yao et al. (PESGM 2023, [YAO-23]) and Si et al. (SMC
2023, [SIR-23]) develop transferable MARL methods, addressing the
classical issue that RL policies trained on one feeder do not transfer
to another. An (Applied Energy, 2026, [AN-26]) introduces a dynamic
gradient masking embedded meta-DRL for hybrid mobile-power-source
dispatch. Chen et al. (Entropy-driven MARL, IJEPES, 2025, [CHE-25])
coordinates MESS dispatch and switch operation for resilient
distribution. Yao et al. (preprint, NREL/OSTI, 2023, [YAO-23b])
provide a DOE-funded open implementation.

**Hybrid RL + MILP.** Liang et al. (IET Renewable Power Generation,
2025, [LIA-25]) design a two-stage strategy coordinating DRL (for
generator-startup ordering — a high-dimensional combinatorial action
space) with MILP (for feasibility-constrained dispatch) in transmission
black-start. This pattern — RL proposes, MILP verifies and projects —
is becoming the de facto safe-RL recipe.

**Safe RL / cyber-resilient RL.** A 2025 IEEE Trans. Smart Grid paper
[SAFE-25] proposes safe deep RL for resilient self-proactive
distribution grids against wildfires, encoding chance constraints via
Lagrangian penalties; another work [CYBR-23] sets up a reinforcement-
learning environment for cyber-resilient power distribution systems.

**Pointer-network / Transformer combinatorial solvers.** Although core
restoration papers using pure pointer-network solvers are sparse, the
adjacent literature (Quantum-Inspired Hyperheuristic Framework for
Dynamic Multi-Objective Combinatorial Disaster Problems, World Electric
Vehicle Journal, 2025) and broader neural combinatorial optimisation
surveys (AI Review, 2025) signal a trend; transformer-based switching
sequence generators are an emerging direction (see "Breakthrough
Directions" below).

**Imitation learning.** Direct imitation from operator logs is still
rare in published peer-reviewed work, but DOE EERE and NREL have funded
synthetic-trajectory pipelines (cf. [YAO-23b]). Liu et al. (CSEE J.
Power & Energy Systems, 2025, [LIU-25]) use improved conditional GANs
to synthesise restoration scenarios — implicitly an imitation-data
augmentation step.

**Foundation models / LLM-based restoration.** No mature peer-reviewed
LLM-based restoration planner exists yet; the closest 2024-2026 entries
are general-purpose surveys (Harnessing the Power of LLMs in Practice,
ACM TKDD, 2024 [HAR-24]) and graph-RL surveys (Energy & AI, 2026
[GRL-26]). This is a clear gap (see below).

---

## Landscape — Uncertainty & Multi-agent

**POMDP / sequential observation models.** Li et al. (arXiv 2601.02958,
2026, [LI-26]) cast post-earthquake restoration of electricity-gas
distribution as a POMDP-like problem in which repair vehicles
simultaneously collect damage information and execute repair — a
classical exploration-exploitation trade-off for restoration. Safe DRL
under wildfire uncertainty [SAFE-25] is a partially observable MDP in
practice. Khattar et al. [KHA-25] explicitly model uncertain topology
changes during restoration through a hierarchical MARL POMDP.

**Robust restoration.** Robust microgrid-formation literature (Shi et
al. 2024, [SHI-24]; Zhu et al. 2025, [ZHU-25]) handles adversarial /
worst-case damage realisations. The 2023 review on power system
restoration with large renewable penetration [REV-23] enumerates robust
formulations as future work.

**Multi-agent restoration with crew dispatch.** The 2024-2025 wave is
unified by joint crew-routing + reconfiguration MILP/MARL: Zhou et al.
[ZHO-24] (frontier-energy, RC-NR coordination), Wang et al. [WAN-25b]
(LSTM-AdaBoost-driven priority + repair-crew dispatch in coupled
distribution-transportation), Hu et al. [HU-25] (helicopter scheduling),
Zhao et al. [ZHA-24a] (battery-swap stations + crews), Shi et al. [SHI-
24b] (traffic congestion).

**Restoration with DERs / inverters.** Grid-forming-inverter-driven
black start: Seo et al. (ISGT, 2024, [SEO-24]) provide power-hardware-
in-the-loop validation; Seo et al. (IEEE Access, 2025, [SEO-25]) survey
GFM-inverter microgrid black-start challenges; Konar & Srivastava (IEEE
Access, 2023, [KON-23]) propose an MPC-based black-start for
DER-rich distribution; Huang et al. (Applied Energy, 2025, [HUA-25])
develop a frequency-secured load-pickup strategy for IBR-rich black
start under dynamic microgrid formation; Yusoff et al. (Frequency
Nadir-Constrained Power System Restoration Planning with Energy
Storage, 2026, [NAD-26]) embed analytical frequency-nadir constraints
into restoration MILP.

**Resilience metrics & equity.** Equity-driven distribution planning
for resilience (Tatari et al., EPSR, 2024, [TAT-24]) addresses spatial
fairness of load restoration; Mohamed et al. (Smart Grids and
Sustainable Energy, 2024, [MOH-24]) studies social disparities in US
power-outage mitigation. Most existing restoration objectives still
use weighted-load not equity-corrected metrics.

---

## Research Gaps

1. **AC-feasibility-guaranteed learning-based restoration.** No published
   learning-based restoration planner provides certified feasibility
   over the AC power-flow feasible set during sequential energisation.
   Current methods either (a) fix a DC / linearised DistFlow
   approximation and accept post-hoc voltage violations, or (b) wrap a
   MILP/SOCP projector around RL outputs (cf. [LIA-25]) with no
   end-to-end gradient. A differentiable AC-feasibility certificate
   embedded in the RL policy is missing.

2. **Inrush-aware sequencing.** Almost all MILP/RL restoration papers
   use static load models; cold-load pickup with frequency-nadir and
   transformer-inrush physics is handled at most in linearised form
   ([HUA-25], [NAD-26], [PWA-23]). A formulation that captures
   transient frequency, voltage, and cold-load dynamics within a
   tractable optimisation horizon is open.

3. **Provably tight Steiner-arborescence relaxations for radial
   restoration with capacity + voltage + branch losses.** [SAK-24]
   delivers a heuristic distributed MST; there is no theoretically
   tight integrality-gap-bounded Steiner relaxation that simultaneously
   handles branch losses, voltage cones, and dynamic microgrid borders.

4. **Online restoration with streaming damage observations.** Most POMDP-
   ish papers ([LI-26], [SAFE-25]) treat observations through
   discretised scenario trees. A scalable deep-POMDP solver for
   distribution restoration with continuous damage-state belief and
   feasibility-aware actions does not yet exist.

5. **Robust restoration against adversarial damage with adaptive crew
   reaction.** Existing DRO microgrid formation ([ZHU-25]) handles
   load/DER ambiguity but not adversarial damage adaptive to crew
   movement. A min-max-min restoration with crew-aware adversary is
   missing.

6. **Transferable / foundation-model restoration policies.** Transfer
   across feeder topologies is brittle ([SIR-23] addresses it partially;
   most RL policies overfit one feeder). No pre-trained restoration
   foundation model exists that generalises across PG&E / DOE / IEEE
   feeders with one-shot fine-tuning.

7. **Equity-aware restoration with formal guarantees.** Equity-weighted
   objectives appear ([TAT-24], [MOH-24]) but no work provides
   axiomatic fairness guarantees (envy-freeness, max-min, proportional
   fairness across census tracts) over restoration sequence selection.

8. **Co-optimisation of restoration with cyber recovery.** Communication
   recovery and cyber-resilience are studied separately ([UAV-23],
   [CYBR-23], [BEL-24]); a joint feasibility-coupled co-optimisation
   of cyber and physical restoration paths is open.

9. **Differentiable MILP layers for end-to-end restoration training.**
   No restoration paper exploits modern differentiable MIO
   (cvxpylayers, MIPLearn, dual-decomposition implicit differentiation)
   to back-propagate through the restoration decision graph; current
   hybrids ([LIA-25]) decouple the two stages.

10. **Stochastic resource-constrained vehicle routing for restoration
    under traffic and weather noise.** Crew-routing papers ([SHI-24b])
    handle congestion in a chance-constrained way but not jointly with
    weather-degraded link reliability and dynamic re-routing during
    execution.

11. **Restoration in inverter-dominated systems with grid-forming
    coordination.** Black-start of GFM-inverter clusters
    ([SEO-24], [SEO-25], [HUA-25]) is treated as device-level control;
    a network-level energisation sequence MILP that explicitly models
    GFM droop, virtual inertia, and protection interaction during
    pickup remains an active gap.

12. **Multi-energy joint restoration (electricity-gas-water-heat) with
    physically consistent couplings.** Multi-stage stochastic gas-
    electricity restoration exists ([ZHA-24b], [YAN-25], [JAF-23],
    [NAS-24]) but most still rely on linearised gas flow / Weymouth
    relaxations; tight conic relaxations co-deployed with restoration
    sequencing are open.

13. **Foundation-model / LLM-assisted restoration planning.** No
    published peer-reviewed work uses an LLM as either a
    natural-language operator interface that calls verified MILP/MARL
    sub-tools, or as a few-shot pattern matcher across historical
    operator playbooks.

14. **Equity-/risk-coupled multi-agent restoration with mechanism-design
    primitives.** Cooperative MARL ([VU-23], [FAN-23]) optimises team
    return; nothing combines individual-rational/credit-assignment
    mechanism design with restoration objectives, which would matter
    for inter-utility / public-private crew coordination.

15. **Benchmarks and standardised testbeds.** There is no widely
    adopted, open, reproducible restoration benchmark with damage
    scenario generators, crew/MESS pools, IBR cranking models, and
    metric definitions (analogous to PGLib for OPF). This blocks
    cross-paper comparison of all the above methods.

---

## Breakthrough Directions

Each direction below ties to specific mathematical primitives that
practitioners can immediately operationalise.

**B1. Differentiable Steiner-arborescence layer for radial restoration.**
Embed a Sinkhorn-/perturbed-optimiser-relaxed Steiner-arborescence solver
(e.g. Berthet et al., NeurIPS 2020) inside a GNN policy. Forward pass
returns a feasible radial energisation tree; backward pass gives implicit
gradients w.r.t. node priorities. Closes Gap 3 + Gap 9.

**B2. AC-feasibility projection layer.** Plug a small Newton-corrected
AC-PF solver (with implicit differentiation, cf. AC-OPF unrolled
solvers, [FPL-OPF-26]) downstream of the RL switch action, so that
policy gradients incorporate post-action voltage feasibility. Closes
Gap 1.

**B3. Frequency-nadir-constrained restoration QP layer.** Treat the
analytic frequency-nadir constraint of [NAD-26] / [HUA-25] as a
differentiable QP layer; the policy then learns load-pickup magnitudes
that respect the analytic nadir bound without post-hoc rejection.
Closes Gap 2.

**B4. Deep POMDP with topology-attention belief state.** Maintain a
belief over the post-event topology using a Graph-Set-Transformer
(MoCo-style contrastive update); use a particle-based critic
(DeepMind R2D2 / IMPALA) and Lagrangian safety on AC-PF residuals.
Closes Gap 4.

**B5. Robust min-max-min crew-aware adversary.** Formulate restoration
as a three-level robust optimisation (operator + adversary damage +
operator adaptive recourse) and solve via column-and-constraint
generation with affine-decision-rule adversary. Closes Gap 5.

**B6. Restoration foundation model with topology pre-training.** Pre-
train a graph transformer on synthetic feeder topologies + damage
scenarios (PGLib + OpenDSS) with masked-switch-and-status modelling
(BERT-style), then fine-tune for restoration on each utility's feeder.
Closes Gap 6 + Gap 13.

**B7. Fair restoration as a constrained CSP with Lorenz-dominance
constraint.** Encode equity as a Lorenz-dominance / Gini-bound
constraint inside MILP. Combine with chance-constrained ENS objective.
Closes Gap 7.

**B8. Cyber-physical joint restoration via product-graph routing.**
Construct the Cartesian product of the cyber communication graph and
the physical feeder graph; solve a single restoration shortest-path /
Steiner problem on the product graph. Closes Gap 8.

**B9. Implicit-MIP differentiation for end-to-end RL+MILP training.**
Use IntOpt / MIPLearn dual-decomposition gradients (Mandi et al.
NeurIPS 2022, Mulamba et al.) so the policy and the MILP cuts are
jointly learnable. Closes Gap 9.

**B10. Crew + restoration as time-expanded resource-constrained
shortest-path with Bayesian belief update.** Solve via column
generation on a time-expanded graph where edge cost is
expected-energy-not-served and resource budget is crew hours, with
Bayesian belief updates on damage state. Closes Gap 10.

**B11. GFM-cluster network-level black-start MILP with droop, virtual
inertia, and protection coordination as logical constraints.** Build
on [SEO-24] device models; add network-level energisation constraints
with disjunctive protection-misoperation cuts. Closes Gap 11.

**B12. Open restoration benchmark suite.** Curate a reproducible
benchmark akin to PGLib-OPF: feeder + damage generator + crew/MESS
pool + IBR cranking + evaluation harness on ENS, GWh-not-supplied,
restoration time, and equity index. Closes Gap 15.

---

## Bibliography

Notation: [TAG-YEAR] author, title, venue, DOI/preprint.

- [PWA-23] K. Pang, C. Wang, N. Hatziargyriou, F. Wen,
  "Dynamic Restoration of Active Distribution Networks by Coordinated
  Repair Crew Dispatch and Cold Load Pickup," IEEE Trans. Power
  Systems, 2023. doi:10.1109/tpwrs.2023.3309862
- [XIE-24] Y. Xie, S. Cai, J. Wang, Y. Chen, "A MILP-based power
  system parallel restoration model with the support of mobile energy
  storage systems," Electric Power Systems Research, 2024.
  doi:10.1016/j.epsr.2024.110592
- [ZHA-24a] X. Zhao, Q. Xu, Y. Yang, "Service Restoration of
  Distribution System Considering Novel Battery Charging and Swapping
  Station, Repair Crews, and Network Reconfigurations," J. Modern
  Power Syst. & Clean Energy, 2024. doi:10.35833/mpce.2024.000010
- [PEN-25] L. Peng et al., "A Two-Stage Restoration Method for
  Distribution Networks Considering Generator Start-Up and Load
  Recovery Under an Earthquake Disaster," Electronics, 2025.
  doi:10.3390/electronics14153049
- [HBP-23] A. Heidari-Akhijahani, K.L. Butler-Purry, "A Review on
  Black-Start Service Restoration of Active Distribution Systems and
  Microgrids," Energies, 2023. doi:10.3390/en17010100
- [ZHO-24] F. Zhou et al., "Resilience-oriented repair crew and
  network reconfiguration coordinated operational scheduling for
  post-event restoration," Frontiers in Energy Research, 2024.
  doi:10.3389/fenrg.2024.1369452
- [LOU-24] C. Lou, L. Zhang, W. Tang, J. Yang, "A coordinated
  restoration method of three-phase AC unbalanced distribution
  network with DC connections and mobile energy storage systems,"
  IJEPES, 2024. doi:10.1016/j.ijepes.2024.109895
- [CHA-24] Y. Chai et al., "Distributed Service Restoration Strategy
  for Interconnected Distribution Networks with Generalized Benders
  Decomposition Algorithm," CICED, 2024.
  doi:10.1109/ciced63421.2024.10754405
- [NAS-24] N. Nasiri, S. Zeynali, S. Najafi Ravadanegh, S. Kubler,
  Y. Le Traon, "Decentralized Privacy-Preserving Distributionally
  Robust Restoration of Electricity/Natural-Gas Systems...," IEEE
  Access, 2024. doi:10.1109/access.2024.3354891
- [ZHA-24b] Y. Zhang, C. He, X. Liu, L. Nan, "Coordinated Restoration
  of Integrated Gas-Electricity Distribution System With Dynamic
  Islanding: A Multi-Stage Stochastic Model With Nonanticipativity,"
  IEEE Trans. Power Systems, 2024. doi:10.1109/tpwrs.2024.3497981
- [ZHU-25] R. Zhu, H. Liu, W. Yu, W. Gu, "Distributionally robust
  microgrid formation for service restoration in distribution systems
  against extended extreme events," IJEPES, 2025.
  doi:10.1016/j.ijepes.2025.110720
- [SHI-24] H. Shi, S. Cai, Y. Xie, Q. Wu, "A robust microgrid
  formation method for risk-resistant service restoration considering
  subsequent contingency," IJEPES, 2024.
  doi:10.1016/j.ijepes.2024.109994
- [SHI-24b] Z. Shi, Y. Xu, D. Xie, S. Xie, "Optimal Coordination of
  Transportable Power Sources and Repair Crews for Service Restoration
  of Distribution Networks Considering Uncertainty of Traffic
  Congestion," J. Mod. Power Syst. & Clean Energy, 2024.
  doi:10.35833/mpce.2023.000012
- [YAN-25] Y. Yang, Z. Li, E.Y.M. Lo, "Multi-timescale risk-averse
  restoration for interdependent water-power networks with joint
  reconfiguration and diverse uncertainties," Reliability Engineering
  & System Safety, 2025. doi:10.1016/j.ress.2025.111083
- [SAK-24] H. Saki, A. Zangeneh, J. Aghaei, "Distributed minimum
  spanning tree approach for critical load restoration using
  microgrid formation in resilient distribution systems," EPSR, 2024.
  doi:10.1016/j.epsr.2024.111186
- [JAF-23] S. Jafarpour, M.H. Amirioun, "A resilience-motivated
  restoration scheme for integrated electricity and natural gas
  distribution systems using adaptable microgrid formation," IET
  GTD, 2023. doi:10.1049/gtd2.13032
- [MON-24] M.R. Monteiro, A.C. Zambroni de Souza, M. Abdelaziz,
  "Hierarchical Load Restoration for Integrated Transmission and
  Distribution Systems With Multi-Microgrids," IEEE Trans. Power
  Systems, 2024. doi:10.1109/tpwrs.2024.3381120
- [FAN-24] J. Fan, P. He, C. Li, C. Zhao, "A post-disaster
  restoration model for power-gas-transportation distribution networks
  considering spatial interdependency and energy hubs," EPSR, 2024.
  doi:10.1016/j.epsr.2024.110505
- [WAN-25a] (Coupled power-transportation resilience tri-stage paper,
  2025) — see SciDirect / EPSR coverage.
- [WAN-25b] L. Wang et al., "Coordinated Post-Disaster Restoration
  for Coupled Distribution-Transportation System," IEEE Access, 2025.
  doi:10.1109/access.2025.3605974
- [HU-25] S. Hu, X. Jing, X. Hu, M. Zhang, C. Li, "Model for
  Post-Disaster Restoration of Power Systems Considering Helicopter
  Scheduling and Its Cost-Benefit Analysis," Electronics, 2025.
  doi:10.3390/electronics14193903
- [FU-23] W. Fu, H. Xie, H. Zhu, H. Wang, L. Jiang, "Coordinated
  post-disaster restoration for resilient urban distribution systems:
  A hybrid quantum-classical approach," Energy, 2023.
  doi:10.1016/j.energy.2023.129314
- [WAD-24] Wadhwa et al., "Real-time outage management in active
  distribution networks using reinforcement learning over graphs,"
  Nature Communications, 2024. doi:10.1038/s41467-024-49207-y
- [FAN-23] B. Fan, X. Liu, G. Xiao, Y. Kang, "Attention-Based
  Multiagent Graph Reinforcement Learning for Service Restoration,"
  IEEE Trans. AI, 2023. doi:10.1109/tai.2023.3314395
- [KHA-25] V. Khattar, Y. Yao, F. Ding, M. Jin, "Distribution Grid
  Critical Load Restoration under Uncertain Topology Changes via a
  Hierarchical Multi-Agent Reinforcement Learning Approach," IEEE
  PES GM, 2025. doi:10.1109/pesgm52009.2025.11225150
- [SIR-24] R. Si, S. Chen, J. Zhang, J. Xu, "A multi-agent
  reinforcement learning method for distribution system restoration
  considering dynamic network reconfiguration," Applied Energy, 2024.
  doi:10.1016/j.apenergy.2024.123625
- [IGL-23] M. Afshari Igder, X. Liang, "Service Restoration Using
  Deep Reinforcement Learning and Dynamic Microgrid Formation in
  Distribution Networks," IEEE Trans. Industry Applications, 2023.
  doi:10.1109/tia.2023.3287944
- [IGL-23b] M. Afshari Igder, X. Liang, "Dynamic Microgrid Formation-
  Based Service Restoration Using Deep Reinforcement Learning in
  Distribution Networks," ICPS 2023.
  doi:10.1109/icps57144.2023.10142118
- [VU-23] L. Vu, T. Vu, T.L. Vu, A.K. Srivastava, "Multi-Agent Deep
  Reinforcement Learning for Distributed Load Restoration," IEEE
  Trans. Smart Grid, 2023. doi:10.1109/tsg.2023.3310893
- [YAO-23] Y. Yao, X. Zhang, J. Wang, F. Ding, "Multi-Agent
  Reinforcement Learning for Distribution System Critical Load
  Restoration," IEEE PES GM, 2023.
  doi:10.1109/pesgm52003.2023.10252887
- [YAO-23b] Y. Yao, "Multi-Agent Reinforcement Learning for
  Distribution System Critical Load Restoration: Preprint," NREL/OSTI,
  2023. osti.gov/biblio/1992817
- [SIR-23] R. Si, Q. Ji, X. Wang, K. Ji, "A Transferable Multi-Agent
  Reinforcement Learning Method for Distribution Service Restoration,"
  IEEE SMC, 2023. doi:10.1109/smc53992.2023.10394147
- [AN-26] H. An, Y. Xu, G. Zhang, Y. Xing, "Coordinated dispatch of
  hybrid mobile power sources for distribution network restoration:
  A dynamic gradient masking embedded multi-agent meta-deep
  reinforcement learning method," Applied Energy, 2026.
  doi:10.1016/j.apenergy.2026.127377
- [CHE-25] (Entropy-driven MARL) "Entropy-driven multi agent deep
  reinforcement learning for resilient distribution networks:
  coordinating MESS ...," IJEPES, 2025.
- [LIA-25] L. Liang, H. Zhang, W. Xiao, X. Zhao, "A Two-Stage Strategy
  for Black-Start Restoration by Coordinating Deep Reinforcement
  Learning and Mixed-Integer Linear Programming," IET RPG, 2025.
  doi:10.1049/rpg2.70081
- [SAFE-25] "Safe Deep Reinforcement Learning for Resilient
  Self-Proactive Distribution Grids Against Wildfires," IEEE Trans.
  Smart Grid, 2025.
- [CYBR-23] "Reinforcement Learning Environment for Cyber-Resilient
  Power Distribution System," IEEE Access, 2023.
- [LI-26] M. Li, W. Wei, Y. Xu, C. Zhang, "Post-Earthquake
  Restoration of Electricity-Gas Distribution Systems with Damage
  Information Collection and Repair Vehicle Routing," arXiv:2601.02958
  / CSEE J. Power Energy Syst., 2026.
  doi:10.17775/cseejpes.2025.06100
- [REV-23] "Power system restoration with large renewable Penetration:
  State-of-the-Art and future trends," IJEPES, 2023.
- [UAV-23] X. Qi, J. Chen, H. Zhao, Y. Zhang, "Post-Disaster
  Distribution System Restoration Considering UAV-Based Communication
  Recovery Based on Multi-Agent Reinforcement Learning," IECON, 2023.
  doi:10.1109/iecon51785.2023.10311875
- [SEO-24] G.-S. Seo, J. Sawant, F. Ding, "Parallel Grid-Forming
  Inverter-Driven Black Start: Power-Hardware-in-the-Loop
  Validation," ISGT 2024. doi:10.1109/isgt59692.2024.10454153
- [SEO-25] G.-S. Seo, W. Wang, B. Mirafzal, "Microgrid Black Start
  Challenges: The Role of Grid-Forming Inverters," IEEE Access, 2025.
  doi:10.1109/access.2025.3634530
- [KON-23] S. Konar, A.K. Srivastava, "MPC-Based Black Start and
  Restoration for Resilient DER-Rich Electric Distribution System,"
  IEEE Access, 2023. doi:10.1109/access.2023.3292254
- [HUA-25] Y. Huang, S. Lei, J. Liu, C. Wang, "A frequency-secured
  load pickup strategy for black-start restoration in IBR-rich
  distribution systems under dynamic microgrid formation," Applied
  Energy, 2025. doi:10.1016/j.apenergy.2025.126752
- [NAD-26] "Frequency Nadir-Constrained Power System Restoration
  Planning with Energy Storage," 2026.
- [TAT-24] "Equity-driven distribution power system planning for
  resilience enhancement," EPSR, 2024.
- [MOH-24] "Power System Resilience: The Role of Electric Vehicles
  and Social Disparities in Mitigating the US Power Outage Burden,"
  Smart Grids and Sustainable Energy, 2024.
- [BEL-24] Y. Nait Belaid, Y. Fang, Z. Zeng, P. Coudray, A. Barros,
  "Communication-aware Restoration of Smart Distribution Grids Based
  on Optimal Allocation of Resilience Resources," J. Mod. Power Syst.
  & Clean Energy, 2024. doi:10.35833/mpce.2024.000015
- [LIU-25] W. Liu, Y. Wang, Q. Shi, Q. Yao, H. Wan, "Multi-Stage
  Restoration Strategy to Enhance Distribution System Resilience with
  Improved Conditional Generative Adversarial Nets," CSEE J. Power &
  Energy Systems, 2025. doi:10.17775/cseejpes.2021.09080
- [HAR-24] "Harnessing the Power of LLMs in Practice: A Survey on
  ChatGPT and Beyond," ACM Trans. KDD, 2024. doi:10.1145/3649506
- [GRL-26] "Graph reinforcement learning for power grids: A
  comprehensive survey," Energy and AI, 2026.
- [FPL-OPF-26] Zhang et al., "Unsupervised Learning for AC Optimal
  Power Flow with Fast Physics-Aware Layer," 2026.
  doi:10.1145/3744255.3811718
- [MRCSR-24] W. Zhang et al., "Multi-Resource Collaborative Service
  Restoration of a Distribution Network with Decentralized
  Hierarchical Droop Control," Protection & Control of Modern Power
  Systems, 2024. doi:10.23919/pcmp.2023.000530
- [OAC-23] L. Zhang, S. Yu, B. Zhang, G. Li, "Outage management of
  hybrid AC/DC distribution systems: Co-optimize service restoration
  with repair crew and mobile energy storage system dispatch,"
  Applied Energy, 2023. doi:10.1016/j.apenergy.2022.120422
- [TES-24] S. Zhao, K. Li, M. Yin, J. Yu, "Transportable energy
  storage assisted post-disaster restoration of distribution networks
  with renewable generations," Energy, 2024.
  doi:10.1016/j.energy.2024.131105
- [MES-23] Y. Xu, M. Zhao, H. Wu, S. Xiang, Y. Yuan, "Coordination of
  network reconfiguration and mobile energy storage system fleets to
  facilitate active distribution network restoration under forecast
  uncertainty," Frontiers in Energy Research, 2023.
  doi:10.3389/fenrg.2022.1024282
- [SSR-25] H. Shu, H. Zhao, X. Zhao, "Service restoration strategy
  for active distribution networks considering source-load
  uncertainty," IJEPES, 2025. doi:10.1016/j.ijepes.2025.111250
- [COR-26] Z. Shi, Z. Li, S. Chen, Y. Xu, "Coordinated Repair and
  Restoration of a Multienergy Distribution System Under Diverse
  Uncertainties via Joint Network Reconfiguration," IEEE Trans.
  Industrial Informatics, 2026. doi:10.1109/tii.2025.3598441
- [VPP-26] L. Chen, X. Lv, W. Yang, Y. Zhou, "Virtual Power Plant
  Enhanced Load Restoration in Distribution Networks based on
  Multi-Agent Reinforcement Learning," SSRN, 2026.
  doi:10.2139/ssrn.6576039
- [AUT-23] "Autonomous Restoration of Networked Microgrids Using
  Communication-Free Smart Sensing and Protection Units," IEEE Trans.
  Sustainable Energy, 2023. doi:10.1109/tste.2023.3245881

(Total ~50 distinct references in the 2023-2026 window.)
