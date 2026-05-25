# C. Reinforcement Learning for Power Flow & Restoration (Survey Research Chunk, A3 — 2023–2026)

Scope: Reinforcement-learning (RL) methods for AC/DC and security-constrained
optimal power flow, distribution / transmission service restoration and
black-start sequencing. The cut captures 2023–2026 publications spanning
constraint-aware policy optimization (CMDP, primal-dual / Lagrangian PPO,
safety-shielded actor-critic, projection layers), graph-RL with GNN policies,
multi-agent RL (MARL) for distributed grid control and restoration,
hierarchical RL for sectionalizing + switching, offline RL on historical
SCADA, model-based / world-model RL with learned grid dynamics, imitation
learning from operator logs, and differentiable power-flow surrogates for
back-prop through time. Topics adjacent to OPF that are *not* steady-state
re-dispatch nor restoration (pure inverter inner loops, market bidding,
cyber-attack injection, EV-as-load) are mentioned only when they introduce
load-bearing methodological primitives.

---

## Landscape — RL for PF / OPF

The 2023–2026 RL-OPF literature is consolidating around **constrained
policy optimization** with explicit primal-dual / safety projection
machinery, **graph-structured policies**, and **hybrid knowledge-data**
formulations that combine RL with a classical OPF or QP block.

- **Safe / constrained-MDP RL for OPF.** Wu et al. (IEEE TPWRS, 2024,
  [WU-24]) cast real-time OPF as a CMDP solved by primal–dual PPO
  (PD-PPO) with prior-knowledge guidance, dynamically tuning the multiplier
  on each operational constraint. Yi et al. (IEEE TPWRS, 2023, [YI-23])
  develop a hybrid knowledge-data-driven SAC variant for sequential
  security-constrained OPF that injects warm-start cuts from a physical
  model. Feng et al. (CSEE-JPES, 2026, [FEN-26]) propose a near-optimal safe
  DRL solver for AC-OPF using a Lagrangian critic. Shi et al. (IEEE TSG,
  2023, [SHI-23a]) develop an augmented-Lagrangian safe off-policy DRL for
  carbon-oriented EV-aggregator OPF. Zhang et al. (MPCE, 2026, [ZHA-26])
  introduce a graph-based safe RL for dynamic OPF with hybrid action
  spaces under time-varying topology.
- **Projection / safety-shielded actor-critic.** Sayed et al. (IEEE
  TPWRS, 2023, [SAY-23]) propose a DNN-assisted projection layer that
  projects each sampled action onto an inner approximation of the AC
  feasible set, attaining hard constraint satisfaction during training.
  Malik (arXiv, 2026, [MAL-26]) layers a runtime safety shield on top of
  hierarchical PPO for grid topology control. Stability-constrained RL is
  developed by Feng et al. (IEEE TCNS, 2023, [FEN-23]) with explicit
  Lyapunov certificates for decentralized voltage control. Wan & Xu (IEEE
  TIE, 2025, [WAN-25]) impose Lyapunov-based stability shaping on RL
  control of grid-forming converters.
- **Graph-RL for OPF and dispatch.** Deihim et al. (EPSR, 2024, [DEI-24])
  use a GNN warm-start as the initial AC-OPF estimate before RL fine-tuning.
  Cao et al. (IEEE TSG, 2023, [CAO-23]) propose physics-informed graphical
  representation + DRL for robust VVC. Pei et al. (IET GTD, 2023, [PEI-23])
  apply graph-DRL for undervoltage load shedding. Sun et al. (IEEE TSG,
  2023, [SUN-23]) integrate a human-in-the-loop DRL for unbalanced VVC.
  Several SCUC + graph-RL hybrids (Tang et al., 2024, [TAN-24a]; Yasirroni
  et al., 2025, [YAS-25]) factorise the action over generators and use a
  Benders / DDPG outer loop.
- **Imitation / hybrid expert-guided RL.** Liu et al. (IJEPES, 2024,
  [LIU-24]) train PPO with grid-expert imitation for real-time dispatch.
  Lin et al. (MPCE, 2024, [LIN-24]) build an improved PPO for sequential
  SC-OPF that initialises from MILP solutions. Hussain et al. (IEEE TSG,
  2025, [HUS-25]) hybridise imitation + RL for soft open-point operation.
  Li et al. (IEEE TPWRS, 2024, [LI-24]) use imitative expert experience in
  an interpretable DRL for smart EV charging that respects grid limits.
- **Topology control and L2RPN benchmarks.** Chauhan et al. (AAAI 2023,
  [CHA-23]) introduce **PowRL** for robust L2RPN management with goal-based
  topological actions and contingency replay. Lan et al. (Energy & AI,
  2023, [LAN-23]) systematically compare rule-based agents vs DRL for
  topology actions on L2RPN-WCCI tracks. Dogoulis & Cordy (arXiv 2026,
  [DOG-26]) propose physics-informed RL with Gibbs priors for combinatorial
  topology control. Zhang et al. (arXiv 2026, [ZHA-26b]) embed an LLM
  search prior into safe RL for topology reconfiguration.

---

## Landscape — RL for Restoration

Restoration RL is split between **distribution service restoration via
microgrid formation**, **transmission-scale recovery / black-start**, and
**resilience scheduling of mobile / crew resources**. The 2023–2026 trend
is strongly toward MARL with hierarchical structure (sectionalising +
switching) and GNN policies that ingest the contingency-modified topology.

- **Hierarchical / two-layer RL.** Hosseini et al. (IEEE TSTE, 2023,
  [HOS-23]) combine a high-level DRL controller with a low-level QP that
  enforces feasibility for distribution restoration. Khattar et al.
  (IEEE PESGM, 2025, [KHA-25]) propose hierarchical MARL for community
  critical-load restoration under uncertain post-event topology. Malik
  (2026, [MAL-26]) sits at the intersection of hierarchical and shielded
  RL for grid operation. Jo et al. (Results in Engineering, 2024, [JO-24])
  treat self-healing radial reconfiguration as a stack of high-level
  area-decision + low-level switch-pick DRL.
- **MARL for distributed restoration.** Vu et al. (IEEE TSG, 2023,
  [VU-23]) develop MADRL for distributed load restoration with networked
  microgrids and per-MG critic ensembles. Wang et al. (IEEE TPWRS, 2023,
  [WAN-23]) propose MARL for mobile-power-source + repair-crew
  coordination. Qiu et al. (Applied Energy, 2023, [QIU-23]) tackle
  repair-crew dispatch as hierarchical MARL with cross-energy coupling.
  Si et al. (IEEE SMC, 2023, [SI-23]) provide a transferable MARL for
  distribution service restoration with shared encoder + per-bus head.
  Zou et al. (IEEE TIA, 2024, [ZOU-24]) handle MARL scheduling of mobile
  energy resources during typhoons. Cai et al. (IJEPES, 2025, [CAI-25])
  introduce entropy-driven MADQL with action-masking for resilient DNs.
- **GNN policies over the post-fault topology.** Jacob et al. (Nature
  Comms, 2024, [JAC-24]) train RL over graphs for real-time outage
  management in active distribution networks, generalising to unseen
  topologies. Badakhshan et al. (SEGN, 2025, [BAD-25]) extend the same
  graph-RL recipe to *controlled islanding* for self-healing transmission
  grids. Zhang et al. (IEEE TNNLS, 2024, [ZHA-24a]) apply multi-agent
  graph-attention DRL for post-contingency grid emergency voltage
  control. Wang et al. (IEEE TIA, 2025, [WNX-25]) propose robust safe RL
  with DER fleets for typhoon-resilient grid recovery.
- **Imitation from operator logs and expert replay.** Igder & Liang (IEEE
  TIA, 2023, [IGD-23]) seed DRL for service restoration with dynamic
  microgrid-formation demonstrations. Liu et al. (IJEPES, 2024, [LIU-24])
  use grid-operator imitation as warm-start for dispatch RL during fault
  rides. Li et al. (Advances in Wind Engineering, 2025, [LI-25]) propose
  knowledge-enhanced DRL for post-hurricane interdependent infrastructure
  recovery, pre-training from expert restoration sequences.
- **Multi-energy and integrated-system restoration.** Wang et al. (IEEE
  TSTE, 2023, [WAY-23]) coordinate multi-energy microgrids for integrated
  resilience with multi-task learning. Cui et al. (IEEE TSG, 2023,
  [CUI-23]) treat networked-microgrid energy management with real-time
  pricing as RL.
- **Adversarial / cyber-physical restoration.** Selim et al. (IEEE TSG,
  2023, [SEL-23]) develop adaptive DRL for cyber-attack defence with
  high-PV penetration that includes restorative reconfiguration.
- **Surveys.** Gautam (Electricity 2023, [GAU-23]) reviews DRL for
  resilient power and energy systems; Heidari-Akhijahani & Butler-Purry
  ([HBP-23], from chunk F) cover black-start MILP and connect to RL
  hybrids; Liu et al. (CSEE-JPES, 2026, [LIX-26]) survey DRL for
  distribution resilience under extreme weather.

---

## Cross-cutting: safety, generalization, sample efficiency

- **Safety mechanisms in practice.** The dominant 2023–2026 strategies
  are: (i) **Lagrangian / primal-dual updates** on a separate cost critic
  (PD-PPO, augmented-Lagrangian, e.g. [WU-24], [SHI-23a], [FEN-26]); (ii)
  **projection layers** onto inner approximations of the AC feasible set
  ([SAY-23]) and **action masking** ([CAI-25], [LAN-23]); (iii) **runtime
  safety shielding** with an MPC / OPF fallback ([MAL-26], [YOO-24]); and
  (iv) **Lyapunov-shaped reward / critic** with stability certificates
  ([FEN-23], [WAN-25]). None of these provide anytime-feasibility during
  exploration on the full AC model — Lagrangian methods admit transient
  violations and projection assumes an inner approximation that can be
  empty under contingencies.
- **Generalization.** Transfer to unseen topologies is most commonly
  pursued via GNN policies ([JAC-24], [BAD-25], [DEI-24]) and physics-
  informed graphical representations ([CAO-23]). Curriculum and synthetic-
  grid pretraining are explicit in [SI-23] (shared encoder transferring
  across feeders) and [LIX-26] (extreme-weather augmented synthetic
  scenarios), but most papers train and test on a single feeder.
- **Sample efficiency.** Hybrid learning + OPF/QP layers ([HOS-23],
  [HUS-25], [YI-23]) and warm-starts from imitation ([LIU-24], [LIN-24])
  consistently outperform tabula-rasa RL by orders of magnitude in
  simulator calls. Model-based / world-model RL (Dreamer-style) over
  power-flow dynamics is *almost absent* in the surveyed papers.
- **Action-space factorisation.** Most large-scale agents factorise the
  joint action over generators or substation busbars
  ([CHA-23], [TAN-24a], [ZHA-24a]), often with attention pooling or
  shared parameters per asset class. Decision-Transformer-style sequence
  models are not yet established in restoration.

---

## Research gaps

1. **No anytime-feasibility guarantee on full AC during exploration.**
   Lagrangian / PD-PPO methods ([WU-24], [SHI-23a], [FEN-26]) tolerate
   transient violations; projection approaches ([SAY-23]) project onto an
   inner approximation that may not contain *any* feasible point under
   N-1 / N-k contingencies.
2. **Lyapunov-shaped RL is restricted to inverter / voltage-control
   sub-problems.** [FEN-23], [WAN-25] handle linearised swing dynamics; no
   paper certifies stability for joint topology + dispatch + restoration
   actions on the unreduced grid.
3. **No standard offline RL on real SCADA traces.** The “offline RL” label
   in 2023–2026 grid papers usually means batch DDPG on simulator-generated
   trajectories. CQL / IQL on multi-year operator-action logs is missing.
4. **Restoration MDPs do not exploit optimal-stopping structure.**
   Sequential energisation under uncertain fault locations is naturally a
   POMDP with a stopping decision (when to re-close a tie); no surveyed
   paper formalises it as such, despite the close fit.
5. **Model-based RL with learned grid dynamics is essentially absent.**
   Dreamer / TD-MPC / MuZero-style world-model RL is found in adjacent
   energy domains but not in OPF or restoration; we found only a handful
   of papers with explicit learned dynamics, all confined to single-bus
   converters.
6. **No principled curriculum over contingency severity.** Training
   typically samples N-1 / load profiles uniformly; an automatic
   curriculum (e.g. unsupervised environment design, prioritised
   contingency replay) is unstudied in our cut.
7. **Synthetic-grid pretraining + transfer to real feeders is unevaluated
   at scale.** [SI-23], [JAC-24] hint at zero-shot transfer across small
   feeders but no foundation-model-style pretraining on 1k+ synthetic
   networks is published in 2023–2026 within this scope.
8. **Imitation learning rarely uses real operator dispatch logs.**
   [LIU-24], [LIN-24] imitate MILP solutions; demonstrations from
   *operator* SCADA are largely unavailable due to data confidentiality —
   no standard dataset exists.
9. **MARL communication structure is mostly fully-observed CTDE.**
   Communication-limited or bandwidth-budgeted MARL (graph attention with
   sparsity penalty, event-triggered exchange) is rare; most papers assume
   a central training oracle.
10. **Differentiable AC power-flow back-prop through time is unused for
    policy optimisation.** Differentiable simulators exist (e.g. JAX
    Newton-Raphson), but no surveyed RL paper backpropagates policy
    gradients through them; gradients are estimated by REINFORCE/PG even
    when a differentiable surrogate is available.
11. **Hierarchical sectionalising + switching split is implicit rather than
    learned.** Authors handcraft the high/low split (zone, then switch).
    Options-discovery / sub-goal learning for restoration is open.
12. **Generalisation to *unseen* topologies after switch failure is not
    benchmarked.** Even graph-RL papers ([JAC-24], [BAD-25]) test on
    feeders with the same node set as training; topology distribution
    shift caused by physical line/transformer damage is missing.

---

## Breakthrough directions

1. **Restoration as a POMDP with a learned belief over latent fault
   locations**, solved as an optimal-stopping problem. The action set
   factorises into "energise next branch" vs "wait for diagnosis", and the
   belief is updated by Bayesian filtering of post-energisation
   measurements. This converts the problem into well-studied territory
   (Whittle indices, POMDP value iteration with information rewards).
2. **Differentiable AC-OPF as the simulator backbone**, with policy
   gradients computed by back-propagating through implicit-function
   gradients of Newton-Raphson. This subsumes [DEI-24]'s warm-start use of
   GNNs and produces low-variance gradients comparable to model-based RL
   without learning the dynamics.
3. **Lyapunov-shielded primal-dual PPO on the full AC manifold**, combining
   [FEN-23]'s certificates with [WU-24]'s PD-PPO. The key open primitive
   is a *control-Lyapunov function over a low-dimensional voltage-stability
   margin* that is differentiable through GNN feature maps.
4. **Graph foundation-model pretraining on 1k+ synthetic feeders**,
   followed by RL fine-tuning per real feeder. Treats the policy as a
   graph transformer over the augmented bus-branch graph and uses
   masked-bus reconstruction + N-k contingency completion as
   self-supervised pretext tasks.
5. **Option-discovery for hierarchical sectionalising-then-switching**,
   using DIAYN / VIC-style mutual-information bonuses on the learned zone
   embedding so the high-level option corresponds to a *contiguous,
   energisable subtree* — sidesteps the handcrafted high-level controller
   of [HOS-23], [JO-24].
6. **CTDE-MARL with bandwidth-budgeted graph-attention messages**: model
   communication as a Bernoulli mask over edges with a Lagrangian penalty
   on the expected number of messages, providing a principled
   sparsity-vs-coordination trade-off absent from current MARL grid work.
7. **Offline RL on operator SCADA via IQL / CQL with conservative
   safety regularisation**, using a learned grid-state critic to reject
   actions outside the support of historical operator behaviour. This
   replaces the simulator-only "offline" setup of current papers.
8. **Decision-Transformer + return-conditioned restoration**, where the
   trajectory of switch closures is generated autoregressively conditioned
   on a target restored-load profile, with a safety post-filter that
   replaces infeasible tokens.
9. **Model-based RL with a learned hybrid dynamics**: discrete switching +
   continuous AC power flow handled by a Dreamer-style world model whose
   discrete head emits switch transitions and continuous head emits AC
   states, enabling 100× cheaper planning rollouts than the embedded MILP
   of [HOS-23].
10. **Curriculum via unsupervised environment design over contingency
    distributions**, automatically generating contingencies that are
    just-barely-solvable for the current policy (PAIRED / ACCEL applied
    to N-k masks), giving guaranteed coverage of the contingency space
    that uniform sampling cannot.

---

## Bibliography

- [WU-24] Wu, P. *et al.* (2024). Real-Time OPF via Safe DRL Based on
  Primal-Dual and Prior Knowledge Guidance. *IEEE TPWRS*.
  doi:10.1109/TPWRS.2024.3395248.
- [YI-23] Yi, Z., Wang, X., Yang, C., *et al.* (2023). Real-Time
  Sequential Security-Constrained OPF: A Hybrid Knowledge-Data-Driven RL
  Approach. *IEEE TPWRS*. doi:10.1109/TPWRS.2023.3262843.
- [FEN-26] Feng, B., Zhao, J., Huang, G. (2026). Safe Deep Reinforcement
  Learning for Real-time AC Optimal Power Flow: A Near-optimal Solution.
  *CSEE JPES*. doi:10.17775/CSEEJPES.2023.02070.
- [SHI-23a] Shi, X., Xu, Y., Chen, G. (2023). An Augmented Lagrangian-
  Based Safe RL Algorithm for Carbon-Oriented Optimal Scheduling of EV
  Aggregators. *IEEE TSG*. doi:10.1109/TSG.2023.3289211.
- [ZHA-26] Zhang, X., Ge, S., Zhou, Y. (2026). Graph-based Safe RL for
  Dynamic OPF with Hybrid Action Space Considering Time-Varying Topology.
  *J. Modern Power Systems & Clean Energy*. doi:10.35833/MPCE.2024.001198.
- [SAY-23] Sayed, A. *et al.* (2023). DNN-Assisted Projection-Based Deep
  RL for Safe Control of Distribution Grids. *IEEE TPWRS*.
  doi:10.1109/TPWRS.2023.3336614.
- [MAL-26] Malik, G. (2026). Hierarchical Reinforcement Learning with
  Runtime Safety Shielding for Power Grid Operation. arXiv.
- [FEN-23] Feng, J., Shi, Y., Qu, G. *et al.* (2023). Stability-
  Constrained RL for Decentralized Real-Time Voltage Control. *IEEE TCNS*.
  doi:10.1109/TCNS.2023.3338240.
- [WAN-25] Wan, Y., Xu, Q. (2025). Stability-Guided RL Control for Power
  Converters: A Lyapunov Approach. *IEEE TIE*.
  doi:10.1109/TIE.2024.3522491.
- [DEI-24] Deihim, A., Apostolopoulou, D., Alonso, E. (2024). Initial
  estimate of AC OPF with graph neural networks. *EPSR*.
  doi:10.1016/j.epsr.2024.110782.
- [CAO-23] Cao, D., Zhao, J., Hu, J. *et al.* (2023). Physics-Informed
  Graphical Representation-Enabled DRL for Robust Distribution System
  Voltage Control. *IEEE TSG*. doi:10.1109/TSG.2023.3267069.
- [PEI-23] Pei, Y., Yang, J., Wang, J. (2023). Undervoltage load
  shedding: A graph deep RL emergency control strategy. *IET GTD*.
  doi:10.1049/gtd2.12795.
- [SUN-23] Sun, X., Xu, Z., Qiu, J. (2023). Optimal Volt/Var Control for
  Unbalanced Distribution Networks With Human-in-the-Loop DRL. *IEEE TSG*.
  doi:10.1109/TSG.2023.3337843.
- [TAN-24a] Tang, X., *et al.* (2024). Security Constrained Unit
  Commitment Optimization Based on Graph RL.
  doi:10.1109/ACPEE60788.2024.10532756.
- [YAS-25] Yasirroni, M., Putranto, L. M., Sarjiya (2025). A RL Approach
  for Frequency-Based Security Constrained Unit Commitment.
  doi:10.1109/CPEEE64598.2025.10987385.
- [LIU-24] Liu, Y. *et al.* (2024). Real-time power system dispatch
  scheme using grid expert strategy-based imitation learning. *IJEPES*.
  doi:10.1016/j.ijepes.2024.110148.
- [LIN-24] Lin, J. *et al.* (2024). Improved PPO Algorithm for
  Sequential Security-constrained OPF Based on Expert Knowledge.
  *MPCE*. doi:10.35833/MPCE.2023.000232.
- [HUS-25] Hussain, S., Farrokhabadi, M., Zareipour, H. (2025). A Hybrid
  Imitation–RL Framework for Soft Open Points in Unbalanced Distribution
  Networks. *IEEE TSG*. doi:10.1109/TSG.2025.3600714.
- [LI-24] Li, S., Zhao, P., Gu, C. (2024). Interpretable DRL With
  Imitative Expert Experience for Smart Charging of EVs. *IEEE TPWRS*.
  doi:10.1109/TPWRS.2024.3425843.
- [CHA-23] Chauhan, A., Baranwal, M., Basumatary, A. (2023). PowRL: A RL
  Framework for Robust Management of Power Networks. *AAAI*.
  doi:10.1609/aaai.v37i12.26724.
- [LAN-23] Lan, T. *et al.* (2023). Managing power grids through topology
  actions: A comparative study between rule-based and RL agents. *Energy
  and AI*. doi:10.1016/j.egyai.2023.100276.
- [DOG-26] Dogoulis, P., Cordy, M. (2026). Physics-Informed RL with Gibbs
  Priors for Topology Control in Power Grids. arXiv.
- [ZHA-26b] Zhang, Z., Shen, C., Wan, X. (2026). LLM-Guided Safe RL for
  Energy System Topology Reconfiguration. arXiv.
- [HOS-23] Hosseini, M. M., Rodriguez-Garcia, L., Parvania, M. (2023).
  Hierarchical Combination of DRL and Quadratic Programming for
  Distribution System Restoration. *IEEE TSTE*.
  doi:10.1109/TSTE.2023.3245090.
- [KHA-25] Khattar, V., Yao, Y., Ding, F. (2025). Distribution Grid
  Critical Load Restoration under Uncertain Topology Changes via a
  Hierarchical Multi-Agent RL. *IEEE PESGM*.
  doi:10.1109/PESGM52009.2025.11225150.
- [JO-24] Jo, S., Oh, J.-Y., Yoon, Y. T. (2024). Self-healing radial
  distribution network reconfiguration based on DRL. *Results in
  Engineering*. doi:10.1016/j.rineng.2024.102026.
- [VU-23] Vu, L., Vu, T., Vu, T. L. (2023). Multi-Agent DRL for
  Distributed Load Restoration. *IEEE TSG*. doi:10.1109/TSG.2023.3310893.
- [WAN-23] Wang, Y., Qiu, D., Teng, F. *et al.* (2023). Towards Microgrid
  Resilience Enhancement via Mobile Power Sources and Repair Crews: A
  MARL Approach. *IEEE TPWRS*. doi:10.1109/TPWRS.2023.3240479.
- [QIU-23] Qiu, D., Wang, Y., Zhang, T. (2023). Hierarchical MARL for
  Repair Crews Dispatch towards Multi-Energy Microgrid Resilience.
  *Applied Energy*. doi:10.1016/j.apenergy.2023.120826.
- [SI-23] Si, R., Ji, Q., Wang, X. (2023). A Transferable MARL Method for
  Distribution Service Restoration. *IEEE SMC*.
  doi:10.1109/SMC53992.2023.10394147.
- [ZOU-24] Zou, Y., Wang, Z., Huang, J. (2024). MARL for Mobile Energy
  Resources Scheduling Amidst Typhoons. *IEEE TIA*.
  doi:10.1109/TIA.2024.3463608.
- [CAI-25] Cai, C., Gan, F., Cui, Y. (2025). Entropy-driven MADRL for
  resilient distribution networks: coordinating MESS and DG. *IJEPES*.
  doi:10.1016/j.ijepes.2025.110968.
- [JAC-24] Jacob, R. A., Paul, S., Chowdhury, S. (2024). Real-time outage
  management in active distribution networks using RL over graphs.
  *Nature Comms*. doi:10.1038/s41467-024-49207-y.
- [BAD-25] Badakhshan, S., Jacob, R. A., Li, B. (2025). Self-healing power
  systems using RL over graphs for controlled grid islanding. *Sustainable
  Energy Grids and Networks*. doi:10.1016/j.segan.2025.101937.
- [ZHA-24a] Zhang, Y., Yue, M., Wang, J. (2024). Multi-Agent Graph-
  Attention DRL for Post-Contingency Grid Emergency Voltage Control.
  *IEEE TNNLS*. doi:10.1109/TNNLS.2023.3341334.
- [WNX-25] Wang, X., Ke, J., Wu, H. (2025). A Robust Safe RL Approach for
  Power Grid Resilience Enhancement against Typhoons via DER Fleets.
  *IEEE TIA*. doi:10.1109/TIA.2025.3619007.
- [IGD-23] Igder, M. A., Liang, X. (2023). Service Restoration Using DRL
  and Dynamic Microgrid Formation in Distribution Networks. *IEEE TIA*.
  doi:10.1109/TIA.2023.3287944.
- [LI-25] Li, S., Wu, T. (2025). Knowledge-enhanced DRL for post-
  hurricane recovery of interdependent infrastructure systems. *Advances
  in Wind Engineering*. doi:10.1016/j.awe.2025.100039.
- [WAY-23] Wang, Y., Qiu, D., Sun, X. (2023). Coordinating Multi-Energy
  Microgrids for Integrated Energy System Resilience: A Multi-Task
  Learning Approach. *IEEE TSTE*. doi:10.1109/TSTE.2023.3317133.
- [CUI-23] Cui, G., Jia, Q.-S., Guan, X. (2023). Energy Management of
  Networked Microgrids With Real-Time Pricing by RL. *IEEE TSG*.
  doi:10.1109/TSG.2023.3281935.
- [SEL-23] Selim, A., Zhao, J., Ding, F. (2023). Adaptive DRL Algorithm
  for Distribution System Cyber Attack Defense With High Penetration of
  Smart Inverters. *IEEE TSG*. doi:10.1109/TSG.2023.3345314.
- [GAU-23] Gautam, M. (2023). Deep RL for Resilient Power and Energy
  Systems: Progress, Prospects, and Future Avenues. *Electricity*.
  doi:10.3390/electricity4040020.
- [LIX-26] Liu, X., Liu, J., Zhao, Y. (2026). DRL-based Resilience
  Enhancement Framework for Distribution Networks Under Extreme Weather.
  *CSEE JPES*. doi:10.17775/CSEEJPES.2022.07450.
- [GUO-23] Guo, G., Zhang, M., Magnússon, S. (2023). Data-Driven
  Decentralized Control of Inverter-Based RES Using Safe Guaranteed Multi-
  Agent RL. *IEEE TSTE*. doi:10.1109/TSTE.2023.3341632.
- [GUZ-23] Guo, G., Zhang, M., Gong, Y. (2023). Safe multi-agent DRL for
  real-time decentralized control of inverter-based renewable energy.
  *Applied Energy*. doi:10.1016/j.apenergy.2023.121648.
- [ZHN-23] Zhang, B., Cao, D., Hu, W. (2023). Physics-Informed Multi-
  Agent DRL enabled distributed voltage control for active distribution
  networks. *IJEPES*. doi:10.1016/j.ijepes.2023.109641.
- [LIS-23] Li, S., Cao, D., Hu, W. (2023). Multi-energy Management of
  Interconnected Multi-microgrid System Using Multi-agent DRL. *MPCE*.
  doi:10.35833/MPCE.2022.000473.
- [YOO-24] Yoon, Y., Yoon, M., Zhang, X. (2024). Safe DRL-Based Real-
  Time Operation Strategy in Unbalanced Distribution System. *IEEE TIA*.
  doi:10.1109/TIA.2024.3446735.
- [JEO-23] Jeon, S., Nguyen, T. T., Choi, D.-H. (2023). Safety-Integrated
  Online DRL for Mobile Energy Storage System Scheduling and Volt/VAR
  Control. *IEEE Access*. doi:10.1109/ACCESS.2023.3264687.
- [HED-24] Hedayatnia, A., Ghafourian, J., Sepehrzad, R. (2024). Two-
  stage data-driven optimal energy management and dynamic real-time
  operation in networked microgrid. *IJEPES*.
  doi:10.1016/j.ijepes.2024.110142.
- [GUO-24] Guo, C., Jiang, C., Liu, C. (2025). Dynamic Reconfiguration of
  Active Distribution Networks Based on Graph Attention Network RL.
  *Energies*. doi:10.3390/en18082080.
- [DEH-25] Dehkordi, N. M., Nekoukar, V. (2025). Adaptive distributed
  stochastic DRL control for voltage and frequency restoration in
  islanded microgrids. *Scientific Reports*.
  doi:10.1038/s41598-025-13010-6.
- [WTU-23] Wu, T., Scaglione, A., Surani, A. P. (2023). Network-
  Constrained RL for Optimal EV Charging Control. *IEEE SmartGridComm*.
  doi:10.1109/SmartGridComm57358.2023.10333926.
- [SHE-23] Shereen, E., Kazari, K., Dán, G. (2023). A RL Approach to
  Undetectable Attacks Against AGC. *IEEE TSG*.
  doi:10.1109/TSG.2023.3288676.
- [MNG-25] Meng, T., Li, X., Zhu, Z. (2025). Robust Voltage Control for
  Active Distribution Networks via Safe DRL Against State Perturbations.
  *Protection and Control of Modern Power Systems*.
  doi:10.23919/PCMP.2024.000342.
- [GOZ-23] Gholizadeh, N., Kazemi, N., Musílek, P. (2023). A Comparative
  Study of RL Algorithms for Distribution Network Reconfiguration with
  Deep Q-Networks. *IEEE Access*. doi:10.1109/ACCESS.2023.3243549.
</content>
</invoke>