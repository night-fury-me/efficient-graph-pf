# E. Algebraic / Geometric / Topological & Game-Theoretic Methods for Power Flow and Grid Restoration

Scope: 2023–2026 advances at the intersection of (a) abstract algebra, geometry, topology and (b) game theory with power-flow (PF) prediction and grid restoration.

---

## Landscape — Algebra / Geometry / Topology

### A.1  Cellular sheaves and sheaf neural networks (SNNs)

Cellular-sheaf theory equips a graph with *stalks* (vector spaces over nodes and edges) and *restriction maps* (linear maps between them); the resulting sheaf Laplacian generalises the graph Laplacian, and the diffusion equation it induces can represent heterophilic, signed, asymmetric, and dimension-varying relations that classical GNNs cannot.

- **Hansen & Gebhart (2020)** introduce sheaf neural networks (NeurIPS TDA workshop) and show empirical gains on signed/heterophilic graphs over GCNs.
- **Bodnar, Di Giovanni, Chamberlain, Liò, Bronstein (NeurIPS 2022)** prove that *non-trivial* cellular sheaves give parametric control over the diffusion's asymptotic class separation, breaking the over-smoothing and heterophily ceilings of GCNs; introduce *Neural Sheaf Diffusion* (NSD).
- **Barbero, Bodnar, et al. (ICML 2022 TAG workshop)** "Sheaf Neural Networks with Connection Laplacians" use a *manifold* prior (Riemannian alignment of neighbour tangent spaces) to compute SO(d)-valued restriction maps, reducing parameter overhead.
- **Caralt, Bernárdez, Duta, Liò, Cot (2024)** introduce opinion-dynamics-inspired sheaf-learning rules and synthetic ellipsoid benchmarks; identify limitations of standard benchmarks for SNNs.
- **Braithwaite, Borgi, Onorato, Liò et al. (2024 / "Heterogeneous Sheaf Neural Networks", HetSheaf)** generalise SNNs to *heterogeneous* graphs with type-conditioned restriction maps and a `SheafPool` operator invariant to local basis changes — the first stalk-space graph-level representation, with up to 10x parameter reduction vs. baseline heterogeneous GNNs.
- "Polynomial Neural Sheaf Diffusion" (2025) introduces spectral filtering on cellular sheaves; "Sheaf GNNs via PAC-Bayes Spectral Optimization" (2025) gives generalisation bounds.

**Power-grid relevance.** No paper indexed (2020–2026) explicitly instantiates an SNN on a transmission/distribution grid, but every structural ingredient required for one is now standard: signed restriction maps mirror line susceptance/conductance signs; per-bus stalks of dimension `4` naturally carry the `(V, θ, P, Q)` quartet; multi-phase grids would use stalks of dimension `12` with dihedral D_3 structure.

### A.2  Equivariant / symmetry-preserving GNNs for PF and OPF

- **Talebi & Zhou (NeurIPS 2025, 2502.05702)** systematically benchmark GCN, GAT, SAGEConv, GraphConv for AC-PF on IEEE test grids; all are *permutation* equivariant by construction but ignore phase-rotation and three-phase symmetries.
- **Arowolo & Cremer (2025, 2510.06860)** "HH-MPNN" — heterogeneous message passing + transformer + physics-informed positional encodings; zero-shot generalisation across 14–2000-bus grids, <1% optimality gap on default topology, <3% on N-1 contingencies; speed-ups up to 5000x vs. interior-point solvers.
- **Pham & Li (2024, 2411.06268)** virtual node-splitting in a hierarchical GNN to lift homogeneous graph assumptions and capture per-generator attributes.
- **Cremer group, Varbella et al. (PowerGraph, 2024)** standard benchmark dataset enabling fair architectural comparison.
- **Lie-group / Lie-algebra equivariance.** Shumaylov, Zaika, et al. (ICLR 2025, 2410.02698) "LieLAC" — Lie algebra canonicalisation for arbitrary (possibly non-compact) Lie groups, integrated as a pre-processing canoniser for pre-trained models. Demonstrated on Lie-point-symmetry-equivariant PDE solvers — directly applicable to PF whose governing equations admit a U(1) global-phase symmetry.
- **Dihedral / D_N equivariance.** EquiSym, axis-level dihedral detection, and equivariant U-shaped neural operators for phase-field models (E-UNO, 2509.01293) demonstrate D_N-equivariant architectures; not yet ported to symmetric three-phase systems.

### A.3  Topological data analysis (TDA) & higher-order networks for grid resilience

- **Hernández-García, Serrano, Sánchez Gómez (2025, 2505.10467)** introduce *thick* and *cohesive* Betti numbers, biparameter persistence modules where one filtration parameter tracks the attack progression and the other tracks structural refinement; explicitly framed for resilience of higher-order networks.
- **Lecha, Cavallo, Dominici, Isufi, Battiloro (NeurIPS-aligned 2024, 2409.08389)** *directed* simplicial neural networks (Dir-SNNs) — first higher-order message passing on directed simplicial complexes; provably more expressive than directed GNNs on isomorphism distinction.
- "Learning Higher-Order Interactions in Brain Networks via Topological Signal Processing" (2504.07695) and "Quantum Simplicial Neural Networks" (2501.05558) extend the simplicial-complex toolkit.
- Topology-aware RL over graphs for resilient distribution networks (2603.06964) reports 9–18% higher cumulative reward, 6% increase in power delivery, 6–8% fewer voltage violations.
- **Effective resistance / Kirchhoff index.** Long-standing line (Wang et al. 2015, Cetinay 2018, Dörfler 2019) uses spectral radius / effective resistance / algebraic connectivity as electrical-aware vulnerability metrics. **2025: "Effective Resistance in Simplicial Complexes as Bilinear Forms"** (2511.10749) lifts effective resistance to simplicial complexes — opens door to higher-order PF vulnerability metrics.
- **Algebraic graph theory of cascading failures.** Topological / spectral investigations of phase transitions in cascading failures (Asztalos 2014, Yang 2017) remain unmatched on the *persistent-homology* side: PH-based early-warning indicators of cascading failure are an open thread.

### A.4  Algebraic / spectral graph theory of the Y-bus

- **Grid-GSP (Ramakrishna & Scaglione, 2021; extensions through 2024)** establishes graph-signal processing as the spectral domain for voltage phasor measurements with the Y-bus as a low-pass graph filter; supports anomaly detection, network inference, compression.
- **Signed-Laplacian stability work.** Spectral-index results on signed Laplacians (1503.01069), positive-semi-definiteness via effective conductance (1906.07632), spectral integral variation of signed graphs (2401.02639) provide the algebraic substrate for DC- and linearised-AC- PF feasibility and synchronisation analysis.
- **Kuramoto / oscillator synchronisation.** "Stability and Synchronization of Kuramoto Oscillators" (2024, 2411.17925), "Cluster Synchronization via Graph Laplacian Eigenvectors" (2025, 2503.18978), "Threshold Graphs are Globally Synchronizing" (2025, 2511.12646) — graph-theoretic synchronisation criteria directly relevant to inertia / virtual-synchronous-machine analysis.

### A.5  Tropical / max-plus algebra for restoration scheduling

- "Algebraic solution of project scheduling problems with temporal constraints" (Krivulin 2024, 2401.09216) and the long line of max-plus scheduling literature give an algebraic calculus for "synchronise + earliest-start" semantics — directly applicable to crew dispatch and re-energisation sequence problems in black-start restoration.
- "Tropical Algebraic approach to Consensus over Networks" (1109.0418) and recent "Tropical linearization and stability analysis of discrete dynamical systems" (2602.15443) provide the dynamics underpinnings. No paper found connects max-plus scheduling to power-system restoration explicitly — a clear gap.

### A.6  Geometric deep learning on the grid-state manifold

- **"Physics-Constrained Neural Dynamics: A Unified Manifold Framework for Large-Scale Power Flow Computation"** (2512.01207) — explicit manifold framework linking gradient flow on the PF feasible manifold to neural network mappings; label-free learning via physics-constrained loss.
- Bronstein-style "geometric deep learning" framework (2104.13478) provides the umbrella under which sheaf, equivariant, and gauge-equivariant grid models can be unified.

---

## Landscape — Game Theory

### G.1  Mean-field games (MFG) and mean-field control (MFC)

- **Bo, Liu, Wang (2025, 2509.04963)** "EV Charging in Smart Grids: Mean Field Equilibrium and Approximate Non-Cooperative and Cooperative Strategies" — finite-horizon MFG; existence/uniqueness for the consistency equation; approximate optima for both non-cooperative and cooperative regimes.
- **Scalable Method for MFC with Kernel Interactions via Random Fourier Features** (2601.01175) — overcomes kernel scaling for large populations.
- **Efficient and Scalable Deep RL for Mean Field Control Games** (2501.00052) addresses MFCGs — *mixed* cooperative-competitive populations, the exact structure of multiple aggregators inside a TSO-DSO frame.
- Earlier 2022–2024 line: heterogeneous MFG for stochastic energy scheduling, MFG-based decentralised optimal EV charging (S0016003226001389), MFG + RL with state-of-charge probability density.

### G.2  Stackelberg and bilevel games for TSO-DSO and VPPs

- **Wang, Zhang, Badesa (PowerTech 2025, 2501.07715)** Stackelberg/bilevel model for DSO-led VPP electricity trading; KKT reformulation; quantifies the price the wholesale market pays for DSO intermediation.
- **Jiang, Bolognani, Belgioioso (2025, 2508.05378)** "Voltage Support Procurement in Transmission Grids: Incentive Design via Online Bilevel Games" — Stackelberg game where TSO designs incentives for DSO reactive-power injections; *online feedback optimisation* gives real-time, model-free implementation; numerical study on 5-bus grid.
- **DSO-Led Bilevel TSO-DSO Coordination** (2603.23099) inverts the usual leader/follower hierarchy: DSO first, then TSO over active distribution networks.
- Multi-leader multi-follower games for prosumer P2P trading with carbon co-trading (S0306261923016926).
- **Multiplayer Stackelberg game for intelligent frequency control with line-loss uncertainty** (IEEE 10381578) — load aggregator leader, micro-turbines followers.
- **V2G for frequency regulation as Stackelberg** (IEEE 10506707) handles endogenous uncertainty.

### G.3  Cooperative games and Shapley-based cost allocation

- **Bauer, Dai, Hagenmeyer (PES GM 2024, 2405.06439)** demonstrate that DC-OPF-based Shapley redispatch cost allocation differs *significantly* from AC-OPF on large grids — Shapley values for industrial CM require AC-OPF.
- **Feng, Sun, Meng, Yang, Feng (2025, 2511.01229)** "SurroShap" — deep-learning surrogate for the characteristic function inside Shapley sampling, 10^4–10^5x speed-ups, scales to 1951-entity (Texas 2000-bus) systems with provable ε-close convergence.
- **Park, Kwag, Molzahn, Gupta (2025, 2510.22321)** DLMP-based bilevel optimisation + Shapley for fair cost allocation across multiple energy communities — couples mechanism design (DLMP) and cooperative allocation (Shapley).
- IEEE 10691766 "Reserve Cost Allocation Method Based on the Improved Shapley Value Theory" (2024) addresses joint stochastic load/RES contributions.
- Coalition formation: hedonic / fractional-hedonic coalition games for P2P trading (IEEE 9494877), strategy-based coalitions for multi-microgrid systems with DR (S0142061521008747), coalition + graph theory for movable energy resource pre-positioning for distribution resilience (S2352467723001030).

### G.4  Adversarial / zero-sum games for attack-defense

- Markov-game RL solution against cyber–physical attacks in smart grid (S095741742401474X, 2024) — model-free MARL to find NE policies for both players with no full model knowledge.
- "Securing demand-response against false pricing attacks" (S2352484724004244, 2024) — operator with incomplete knowledge of private DR behaviour.
- "Moving-Target Defense via Game Theory" (Liu et al., 2006.07697 + 2024 follow-ups including 2504.03065) — zero-sum game over D-FACTS reactance perturbations; NE solved via exponential weights.
- "Learning-Enabled Adaptive Voltage Protection Against Load Alteration Attacks" (2411.15229, 2024).
- Stackelberg security investment for voltage stability (IEEE 9304301).

### G.5  Potential games for distributed OPF

- Classical potential-game OPF (IEEE 6345654 — Yang et al.) with Carnot best-response-with-inertia is the touchstone; not extended beyond DC-OPF.
- "Signal-Anticipation in Local Voltage Control" (1811.09365) characterises Volt/Var NE as a global optimum, proves asymptotic global stability.
- **Chen, Scherpen, Monshizadeh (2024, 2404.00968)** distributed *aggregative* game with power-flow constraints; fully distributed generalised NE algorithm via forward-backward splitting, only neighbour communication.
- "On Best-Response Dynamics in Potential Games" (1707.06465) gives modern convergence rate results.
- "Optimally Managing Convergence Tolerance for Distributed OPF" (2311.08305) studies how primal tolerances trade off against feasibility.

### G.6  Mechanism design for electricity markets with PF constraints

- "VCG for Electricity Markets" (Karaca & Kamgarpour, 1611.03044 + ScienceDirect 2017) remains the canonical incentive-compatible benchmark; recent work extends to electricity-gas (IEEE 9220680).
- "Locational marginal burden" (2405.12219, 2024) and "On Locational Marginal Emissions" (2603.19530, 2025) extend the LMP family with equity and emissions externalities.
- "Generalized class of locational pricing mechanisms" (S0140988316302687) frames LMP in a parameterised family; core-selecting mechanisms (2012.05047) generalise the LMP rationale.

### G.7  No-regret learning for grid coordination

- "Distributed No-Regret Learning in Multi-Agent Systems" (2002.09047) — relaxed Nash through per-player no-regret; foundational.
- "Centrally Coordinated MARL for Power Grid Topology Control" (2502.08681, 2025) and "Heterogeneous Multi-Agent PPO for Power Distribution System Restoration" (2511.14730, 2025) use MARL but stop short of explicit no-regret guarantees.
- "MARL for Energy Networks: Computational Challenges, Progress and Open Problems" (2404.15583, 2024) surveys.

### G.8  Coalition formation for restoration & microgrid islanding

- "Graph theory + coalitional game theory for movable-energy-resource pre-positioning" (S2352467723001030, 2023) is the closest cooperative-game framing of restoration.
- "Optimal coalition formation for DERs in smart grids" (S0142061522004975) with splitting/merging operations + Shapley payoff allocation.
- "Coalitional game theory for multi-microgrid energy systems considering service charge and power losses" (S2352467722000613).
- "Fractional Hedonic Coalition Formation for P2P Energy Trading" (IEEE 9494877).

---

## Research Gaps

1. **No PF surrogate exploits the U(1) global-phase-rotation gauge symmetry.** AC PF equations are invariant under `θ_i → θ_i + α` for all `i`; existing equivariant PF GNNs (HH-MPNN, PowerFlowNet, PINCO) only handle bus *permutation* equivariance. A gauge-equivariant architecture (cf. Favoni et al. 2012.12901 for lattice U(1)) would eliminate the slack-bus arbitrariness and improve generalisation across reference choices.

2. **No sheaf neural network has been instantiated on a power grid**, despite the sheaf-Laplacian being the natural algebraic generalisation of the Y-bus when nodal degrees of freedom differ across buses (PQ vs. PV vs. slack vs. converter-interfaced). HetSheaf (2024) provides the framework but no grid case study exists.

3. **Dihedral D_3 / symmetric-three-phase equivariance is unexploited.** Balanced three-phase systems carry an exact D_3 (or D_3 × Z_2 for sequence components) symmetry; no GNN-PF paper imposes this as an architectural prior even though it would shrink the per-phase parameter count by 6x in symmetric segments.

4. **Persistent homology is not yet used as an *early-warning indicator* for cascading failures.** Topological phase-transition work (Asztalos 2014, Yang 2017) and resilience work via Betti numbers (Hernández-García et al. 2025) exist but are not coupled with real-time PMU streams to produce a PH-time-series early-warning signal.

5. **Tropical / max-plus algebra has not been transferred to black-start restoration scheduling.** Synchronise-and-energise semantics map exactly to (max, +), yet no paper applies max-plus eigenvalues / cycle means to restoration crew dispatch or generator re-energisation order.

6. **Lie-point-symmetry inductive biases for the PF PDE/algebraic system are unexplored.** LieLAC (Shumaylov et al. 2025) demonstrates the canonicalisation pipeline for PINNs; the PF system's symmetry group has not been written down for ML purposes.

7. **Mean-field control in TSO-DSO interaction lacks a unified formulation.** MFG papers stop at the DER/EV aggregator level; bilevel TSO-DSO papers stop at small numbers of DSOs. No paper couples a *continuum* of DSOs (with PF constraints) inside an MFC formulation.

8. **Shapley-based contingency-reserve allocation under AC PF at real-time scale.** SurroShap (2025) targets carbon allocation; analogous surrogates for N-k reserve allocation with AC feasibility do not exist.

9. **No-regret learning has not been operationalised under AC PF constraints.** Multi-agent algorithms either ignore network constraints or use DC linearisation; cf. distributed-aggregative-game line (Chen et al. 2024) which is closer but still potential-game-based, not regret-based.

10. **Mechanism design for online / streaming markets with PF constraints.** VCG and core-selecting mechanisms exist for static markets; online VCG with rolling-horizon AC-OPF, especially with non-convexity, is open.

11. **Robust adversarial training of GNN-PF surrogates against simultaneous topology + measurement attacks.** Moving-target defence work (Liu 2006.07697, 2504.03065) handles model-based defenders; ML-based PF surrogates are not yet hardened against the same threat model.

12. **Algebraic topology of cascading failures via *cycle-space* homology.** The cycle space of a grid encodes loop flows; how Betti-1 generators reorganise under sequential trippings — and whether their persistence is a predictor of blackout magnitude — is an open question.

---

## Breakthrough Directions

1. **Sheaf-Laplacian PF surrogate ("Sheaf-PF-Net").** Build a GNN where every bus carries a stalk encoding `(V, θ, P, Q)` (or `(V_a, V_b, V_c, θ_a, θ_b, θ_c, …)` in three-phase), restriction maps along each edge encode the complex line admittance via a 4×4 (resp. 12×12) orthogonal-then-scaled map. The sheaf Laplacian *is* a non-trivial algebraic generalisation of the Y-bus that natively handles bus-type heterogeneity and signed/asymmetric coupling; learned restriction maps absorb tap, phase-shifter, and FACTS effects without retraining.

2. **U(1)-gauge-equivariant message passing.** Restrict the message-passing primitives to functions of `θ_i − θ_j` rather than `(θ_i, θ_j)` separately, so the model is exactly invariant to global phase rotation. Combine with LieLAC canonicalisation for non-compact symmetries (translation in `θ`).

3. **D_3 / sequence-component-equivariant three-phase GNNs.** Embed the symmetrical-components transform as a fixed equivariant projection layer; learn separately on positive-, negative-, and zero-sequence sub-graphs whose D_3 structure is enforced architecturally. Yields 6x parameter savings in balanced regions and explicit reasoning about unbalance.

4. **Persistent-homology early-warning monitor.** Stream PMU data → time-evolving weighted graph (weights = line-loading margins) → compute Betti-0 and Betti-1 persistence diagrams in a sliding window; train a classifier (or a TDA-kernel SVM) on Wasserstein-distance features to flag imminent cascade. Couples 2505.10467 thick/cohesive Betti numbers with online stream processing.

5. **Max-plus restoration scheduler.** Encode the restoration plan as a max-plus state-space `x(k+1) = A ⊗ x(k)`, where `A_ij` is the time-to-energise edge `(j → i)` (∞ if infeasible), with eigenvector/eigenvalue analysis giving the optimal cyclic re-energisation rhythm. The max-plus eigenvalue equals the *critical-path* restoration time; max-plus row-balancing yields crew load distribution.

6. **Mean-field-control bilevel TSO-DSO surrogate.** Outer layer: TSO sets price signals (Stackelberg leader). Inner layer: a continuum of DSOs solve an MFC problem with AC-PF constraints, represented by a *deep MFC* solver (cf. 2501.00052) trained offline; inner-loop gradient flows back via implicit-function theorem for end-to-end TSO incentive design.

7. **Shapley-AC-OPF surrogate for reserve allocation.** Combine SurroShap-style coalition sampling with an HH-MPNN AC-OPF surrogate as the characteristic-function evaluator; deliver real-time N-1 reserve cost shares with AC feasibility and provable ε-close Shapley convergence.

8. **No-regret learning for AC-OPF.** Cast distributed AC-OPF as a constrained online convex optimisation game; pair Optimistic-FTRL with a feasibility-restoration projection that solves a local AC power-flow; prove sub-linear regret w.r.t. the social cost under standard convexification assumptions.

9. **Online VCG with rolling-horizon AC-OPF.** Build an incentive-compatible online auction whose allocation rule is a *learned* AC-OPF surrogate; certified payments through dual prices computed by automatic differentiation through the surrogate. Bridges 1611.03044 (static VCG) and 2510.06860 (HH-MPNN).

10. **Cycle-space (Betti-1) restoration cost allocator.** Use Hodge / simplicial decomposition of post-fault loop flows on the cycle space; allocate redispatch cost via Shapley on the cycle generators — an algebraic-topology refinement of 2405.06439 that respects Kirchhoff's voltage law as a chain-complex constraint.

---

## Bibliography (selected, 2023–2026 unless cited as background)

### Sheaf / topological deep learning
- Hansen, J., Gebhart, T. (2020). *Sheaf Neural Networks*. NeurIPS TDA Workshop. arXiv:2012.06333.
- Bodnar, C., Di Giovanni, F., Chamberlain, B.P., Liò, P., Bronstein, M.M. (2022). *Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs*. NeurIPS 2022. arXiv:2202.04579.
- Barbero, F., Bodnar, C., Sáez de Ocáriz Borde, H., Bronstein, M., Veličković, P., Liò, P. (2022). *Sheaf Neural Networks with Connection Laplacians*. ICML TAG Workshop. arXiv:2206.08702.
- Caralt, F.H., Bernárdez, G., Duta, I., Liò, P., Cot, E.A. (2024). *Joint Diffusion Processes as an Inductive Bias in Sheaf Neural Networks*. arXiv:2407.20597.
- Braithwaite, L. et al. (2024/2026). *Heterogeneous Sheaf Neural Networks*. arXiv:2409.08036.
- Lecha, M., Cavallo, A., Dominici, F., Isufi, E., Battiloro, C. (2024). *Higher-Order Topological Directionality and Directed Simplicial Neural Networks*. arXiv:2409.08389.
- Hernández-García, P., Serrano, D.H., Sánchez Gómez, D. (2025). *From Persistence to Resilience: New Betti Numbers for Analyzing Robustness in Simplicial Complex Networks*. arXiv:2505.10467.
- Polynomial Neural Sheaf Diffusion (2025). arXiv:2512.00242.
- Sheaf GNNs via PAC-Bayes Spectral Optimization (2025). arXiv:2508.00357.
- Effective Resistance in Simplicial Complexes as Bilinear Forms (2025). arXiv:2511.10749.

### Equivariant / geometric deep learning
- Shumaylov, Z., Zaika, P., Rowbottom, J., Sherry, F., Weber, M., Schönlieb, C. (ICLR 2025). *Lie Algebra Canonicalization: Equivariant Neural Operators under arbitrary Lie Groups*. arXiv:2410.02698.
- Lie Group Decompositions for Equivariant Neural Networks (2023). arXiv:2310.11366.
- A General Framework for Equivariant Neural Networks on Reductive Lie Groups (2023). arXiv:2306.00091.
- Equivariant U-Shaped Neural Operators for Cahn–Hilliard Phase-Field (2025). arXiv:2509.01293.
- Lattice Gauge Equivariant CNNs (Favoni et al. 2020). arXiv:2012.12901.
- Gauge-Equivariant GNNs for Lattice Gauge Theories (2026). arXiv:2604.20797.

### GNNs for PF / OPF (with structural/topological focus)
- Talebi, S., Zhou, K. (NeurIPS 2025). *Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids*. arXiv:2502.05702.
- Arowolo, O., Cremer, J.L. (2025). *Towards Generalization of Graph Neural Networks for AC Optimal Power Flow* (HH-MPNN). arXiv:2510.06860.
- Pham, T., Li, X. (2024). *Constraints and Variables Reduction for OPF Using Hierarchical GNNs with Virtual Node-Splitting*. arXiv:2411.06268.
- Varbella, A., Gjorgiev, B., Sansavini, G. (2024). *PowerGraph: A power grid benchmark dataset for graph neural networks*. arXiv:2402.02827.
- PINCO: Physics-Informed GNN for Non-linear constrained AC-OPF (2024). arXiv:2410.04818.
- Generalizable GNNs for Robust Power Grid Topology Control (HetGNN, 2025). arXiv:2501.07186.
- Physics-Constrained Neural Dynamics: A Unified Manifold Framework for Large-Scale Power Flow Computation (2025). arXiv:2512.01207.

### Algebraic graph theory / spectral methods
- Grid-Graph Signal Processing (Ramakrishna, Scaglione 2021). arXiv:2103.06068.
- Spectral integral variation of signed graphs (2024). arXiv:2401.02639.
- Stability and Synchronization of Kuramoto Oscillators (2024). arXiv:2411.17925.
- Cluster Synchronization via Graph Laplacian Eigenvectors (2025). arXiv:2503.18978.
- Threshold Graphs are Globally Synchronizing (2025). arXiv:2511.12646.

### Tropical / max-plus
- Krivulin, N. (2024). *Algebraic solution of project scheduling problems with temporal constraints*. arXiv:2401.09216.
- Tropical linearization and stability analysis of discrete dynamical systems (2026). arXiv:2602.15443.

### Mean-field games / control
- Bo, L., Liu, F., Wang, S. (2025). *EV Charging in Smart Grids: Mean Field Equilibrium*. arXiv:2509.04963.
- Efficient and Scalable Deep RL for Mean Field Control Games (2025). arXiv:2501.00052.
- Scalable Method for Mean Field Control with Kernel Interactions via Random Fourier Features (2026). arXiv:2601.01175.

### Stackelberg / bilevel
- Wang, P., Zhang, X., Badesa, L. (2025, PowerTech). *Analyzing the Role of the DSO in Electricity Trading of VPPs via a Stackelberg Game Model*. arXiv:2501.07715.
- Jiang, Z., Bolognani, S., Belgioioso, G. (2025). *Voltage Support Procurement in Transmission Grids: Incentive Design via Online Bilevel Games*. arXiv:2508.05378.
- DSO-Led Bilevel Optimization for TSO-DSO Coordination (2026). arXiv:2603.23099.
- Multiplayer Stackelberg Intelligent Frequency Control with Line-Loss Uncertainty (IEEE 10381578, 2024).
- V2G Frequency Regulation Stackelberg with Endogenous Uncertainty (IEEE 10506707).

### Cooperative games / Shapley
- Bauer, R., Dai, X., Hagenmeyer, V. (PES GM 2024). *Industrial Application of Shapley-value-based Redispatch Cost Allocation Requires AC OPF*. arXiv:2405.06439.
- Feng, Y., Sun, T., Meng, Y., Yang, X., Feng, D. (2025). *Deep-Learning-Accelerated Shapley Value for Fair Allocation in Power Systems (SurroShap)*. arXiv:2511.01229.
- Park, H., Kwag, K., Molzahn, D.K., Gupta, R.K. (2025). *Fair Cost Allocation in Energy Communities: DLMP-based Bilevel Optimization with Shapley Value*. arXiv:2510.22321.
- Reserve Cost Allocation via Improved Shapley Value (IEEE 10691766, 2024).

### Adversarial / robust games
- Markov game RL against cyber-physical attacks (Elsevier S095741742401474X, 2024).
- Securing demand-response against false pricing attacks (Elsevier S2352484724004244, 2024).
- Moving Target Defense vs Adversarial FDIA in Power Grids (2025). arXiv:2504.03065.
- Learning-Enabled Adaptive Voltage Protection Against Load Alteration Attacks (2024). arXiv:2411.15229.

### Potential games / distributed OPF
- Chen, X., Scherpen, J.M.A., Monshizadeh, N. (2024). *Optimal Bidding in Network-Constrained Demand Response: A Distributed Aggregative Game*. arXiv:2404.00968.
- Optimally Managing Convergence Tolerance for Distributed OPF (2023). arXiv:2311.08305.

### Mechanism design
- Karaca, O., Kamgarpour, M. (2017/2016). *Exploring VCG for Electricity Markets*. arXiv:1611.03044.
- On Locational Marginal Emissions: A Two-Layered Dispatch Mechanism (2026). arXiv:2603.19530.
- Locational Marginal Burden: Equity of OPF Solutions (2024). arXiv:2405.12219.

### No-regret / MARL for grids
- Centrally Coordinated MARL for Power Grid Topology Control (2025). arXiv:2502.08681.
- Heterogeneous MAPPO for Power Distribution System Restoration (2025). arXiv:2511.14730.
- MARL for Energy Networks: Computational Challenges (2024). arXiv:2404.15583.

### Coalition formation / restoration
- Graph theory + coalitional game for pre-positioning of movable energy resources (Elsevier S2352467723001030, 2023).
- Optimal coalition formation for DERs (Elsevier S0142061522004975, 2022).
- Fractional Hedonic Coalition for P2P Energy Trading (IEEE 9494877, 2021).

---

*Indexed sources in this session's knowledge base (queryable via `ctx_search(source: "...")`):* `sheaf-neural-networks-bodnar`, `neural-sheaf-diffusion-bodnar`, `sheaf-connection-laplacians`, `joint-diffusion-sheaf`, `heterogeneous-sheaf-nn`, `directed-simplicial-nn`, `betti-resilience-simplicial`, `lie-algebra-canonicalization`, `gnn-generalization-acopf`, `gnn-ac-power-flow-pred`, `hierarchical-gnn-virtual-split`, `mfg-ev-smartgrid`, `stackelberg-vpp-dso`, `voltage-bilevel-game`, `aggregative-game-demand-response`, `shapley-redispatch-acopf`, `dl-shapley-fair-allocation`, `dlmp-shapley-bilevel`.
