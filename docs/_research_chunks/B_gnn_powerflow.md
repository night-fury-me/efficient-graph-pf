# Research Chunk B — Graph Neural Network Architectures for AC/DC Power Flow and Optimal Power Flow (2023–2026)

Scope: deep technical landscape, generalization frontier, gaps, and breakthrough directions for the *architectural* layer of GNN-based PF/OPF surrogates. Eight families: (B.1) message-passing variants, (B.2) self-attention / graph transformers, (B.3) equivariant / symmetry-aware, (B.4) heterogeneous / hypergraph, (B.5) multi-scale / hierarchical, (B.6) continuous-depth / implicit / DEQ, (B.7) topology-aware contingency methods, (B.8) operator-learning views. Host project [Kim+Karim+Conrad 2025, arxiv:2509.22458] (PIGNN-Attn-LS) and the domain-adaptation companion [Karim+Kim 2026, arxiv:2602.18227] sit primarily inside B.2 (edge-aware physics attention) and partially in B.4 (typed bus encoding). This chunk is complementary to Chunk A (PINN/foundation pillar) and avoids duplicating the PINN-loss analysis already covered there.

---

## B.1 Landscape — Message-Passing Variants (GCN / GAT / GIN / EdgeConv / NNConv / GraphSAGE) on Power Grids

The 2023–2026 message-passing literature on AC/DC PF has consolidated into three sub-clusters.

- **GCN / spectral baselines.** Physics-Guided GCN for OPF [Liu+Gao+Yang 2024, doi:10.1109/TPWRS.2023.3238377, also reported as Gao+Yu+Yang 2024 in IEEE TPWRS, 82 citations] uses a graph-convolution backbone with topology channels encoded as edge weights and a soft Kirchhoff residual; it remains the most-cited GNN-OPF method in 2024. The earlier Physics-Embedded GCN for PF Considering Uncertain Injections and Topology [Gao+Yu+Yang 2023, IEEE TNNLS, 31 cit] embeds Y-bus as a learnable filter inside spectral conv. PowerGNN [Suri+Mangal 2025, arxiv:2503.22721] couples a topology-aware GCN with branch-level structural priors for state prediction.
- **Edge-conditioned (NNConv / EdgeConv) and GraphSAGE.** PowerFlowNet [Lin+Orfanoudakis+Cardenas-Bojaca 2023, arxiv:2311.03415, IJEPES, 50 cit] is the canonical message-passing PF surrogate, with edge-conditioned NNConv-style updates feeding admittance Y_ij as edge features; it remains the most-cited recent GNN-PF baseline. The empirical benchmark of [Abelezele+Sung+Ramamurthy 2025, DESTION/CPS-IoT-Week] systematically tests GCN, GAT, GraphSAGE, GIN, EdgeConv, NNConv on AC-OPF: NNConv and EdgeConv dominate accuracy whereas GAT shows the best topology robustness — but no single operator wins on all metrics. Topology-aware Gated GNN [Jadhav+Sevak+Das+Su+Bui 2025, arxiv:2507.02078] couples GraphSAGE-style aggregation with gating to fuse line-status binary features.
- **GAT / attention message passing.** DPFAGA [Le+Le 2025, arxiv:2503.15563] joins GAT with clustering-by-adaptive-neighbours for dynamic PF + fault characterisation. Topology-Aware GAT state-estimator [Liu+Shi+Wang 2025, ICCCBDA] uses attention scores as learned line-importance weights. The early GNN-PF systematisation by [Tuo+Li+Zhao 2023, arxiv:2307.02049, NAPS, 11 cit] benchmarks MPNN/GAT/GCN on IEEE systems.
- **Power-flow specific message-design.** Heterogeneous Edge GCNN for fast PF [Wu+Liu+Xie 2024, J. Phys. Conf. Ser.] introduces per-edge type heterogeneous convolutions tracking branch vs transformer vs shunt edges. Owerko-Gama-Ribeiro [arxiv:2210.09277, ICASSP 2022, 38 cit] is the *unsupervised* graph-aggregation baseline that several 2024–2026 works extend.
- **Benchmark / sample-efficiency studies.** [Yaniv+Goebel 2025, IEEE Kiel PowerTech] benchmarks GCN/GAT/GraphSAGE/GIN/EdgeConv on distribution PF and finds GAT wins on small data, EdgeConv wins on large data, and GCN saturates earliest. [Conrad+Kim+Jäger+Maier+Bayer 2026, arxiv:2602.19667] (the host group) shows on a modified IEEE 5-bus dataset that dataset size dominates architecture choice for MLP vs two GNN variants — a sobering critical baseline. [Yaniv+Kumar+Beck 2023, EPSR, 30 cit] is the canonical "towards adoption of GNNs for distribution PF" reference.

Limitations now in plain view: (i) the canonical message-passing PF surrogates remain over-smoothed past 5–6 layers; (ii) most use undirected aggregation and lose phase-asymmetry information; (iii) edge features rarely encode the *complex-valued* admittance, only magnitudes; (iv) GraphSAGE / GIN / NNConv benchmarks rarely report KCL residual at inference.

## B.2 Landscape — Self-Attention and Graph Transformers (Graphormer, GT, GPS, Spectral GT, Polynormer) on Grids

This pillar is the youngest and where the host project sits.

- **Edge-aware physics-attention GTs.** PIGNN-Attn-LS [Kim+Karim+Conrad 2025, arxiv:2509.22458] is the first self-attention PF GNN where attention biases encode per-edge admittance Y_ij and a backtracking line-search re-imposes a descent criterion at inference; the LoRA-adapted variant PIGNN-Attn-LS-LoRA [Karim+Kim 2026, arxiv:2602.18227] reuses the architecture for MV→HV domain shift with 85% fewer trainable parameters. PIGTN [Elnour+Saleh+Atat 2026, arxiv:2603.00085] applies physics-informed graph transformers to *attack detection* and sensor placement on grids.
- **Heterogeneous graph transformers for PF.** [Li+Sun+Su 2026, ICEAAI] proposes a "physical mechanism + heterogeneous graph transformer" coupling for PF, fusing learned attention with Newton-step residuals. PowerModelsGAT-AI [Ezeakunne et al. 2026, arxiv:2603.16879] trains a physics-informed GAT (used as a transformer with edge weights) on 13 of 14 IEEE systems with continual EWC + replay.
- **Convolutional transformer hybrids.** [Tran+Mitra+Nguyen 2024, EPSR, 17 cit] combines convolutional encoders with a self-attention head for AC-OPF. The pure transformer Powerformer [Chen+Luo+Liu+Wei+Zhou+Qing+Zhang+Song+Song 2024, arxiv:2401.02771] introduces section-adaptive transformer attention for grid state representation across IEEE-118 / China 300-bus / European 9241-bus systems with section-specific attention plus GNN propagation.
- **Spectral graph transformer adjacencies.** SPGFormer [An+Dai+Wang 2025, IEEE TGRS] uses Laplacian positional encoding inside a graph transformer for hyperspectral images; the *power-grid analogue* (Y-bus Laplacian PE) has not yet been published in any PF paper. PolyFormer [Ma+He+Wei 2024, arxiv:2407.14459, KDD] introduces scalable node-wise polynomial filters inside a GT framework — directly applicable but unapplied to grids.
- **Spatio-temporal transformer / GNN fusions on grid forecasting.** [Madsen+Bank+Mirshekali 2025, Smart Power & Energy Security] uses a Transformer-GNN hybrid for Danish distribution-grid energy forecasting. DANF [Lv+Zhang+Li 2025, PSGAI] is a direction-aware non-graph transformer for renewable forecasting that exposes the limits of static-topology GTs. RTGT [unspec. authors 2025] is a relational temporal graph transformer for cybersecurity risk prediction on grids.
- **Wider GT / spectral GNN background.** Polynormer-class GNNs and the GPS/GraphGPS framework remain influential for general graph problems but have *not* yet been benchmarked on AC-PF. Multiresolution Graph Transformer [Ngô+Hy+Kondor 2023, arxiv:2302.08647, JCP] introduces wavelet positional encoding for hierarchical learning — directly transferable to multi-scale grids but unused. Graph Rewiring survey [Attali+Buscaldi+Pernelle 2024, arxiv:2411.17429, 12 cit; IJCAI 2026 survey] catalogues over-squashing fixes; over-squashing is a known pathology on grids (single-cut topologies → bottleneck nodes), but no PF GT paper applies spectral rewiring.

Bridging Chunk A: PIGNN-Attn-LS (host) is the only edge-aware physics-attention model published — but its positional encoding is bus-index based, not Y-bus spectral; this leaves a structural gap actionable in B.9.

## B.3 Landscape — Equivariant / Symmetry-Aware GNNs

Power grids carry several natural symmetries: (a) **bus relabeling** (permutation), (b) **phase rotation** (U(1) on complex voltages), (c) **sign equivariance** on phase angles, and (d) **per-phase swap** in three-phase systems. The literature only sparsely exploits these.

- **Permutation-equivariant deep sets for power systems.** [Zelaya-Arrazabal+Martinez-Lizana+Pulgar-Painemal+Zhao 2025, arxiv:2512.10232] (referenced from Chunk A) is the most direct: a permutation-equivariant Deep Sets architecture for dynamic security assessment of system frequency response; outperforms non-equivariant baselines on IEEE 39/118-bus. This is the *only* paper that proves permutation-equivariance for any AC-PF–adjacent task.
- **Typed / heterogeneous-equivariant encodings.** Physics-Informed Typed GNNs for OPF [Lopez-Garcia+Domínguez-Navarro 2024, IEEE TPWRS, 20 cit] treats generator / load / slack / branch as distinct node/edge types so the message function is equivariant under intra-type permutation but type-aware. Most other "heterogeneous" papers (B.4) inherit this property *implicitly* without proof.
- **Multigraph for three-phase symmetry.** PowerFlowMultiNet [Ghamizi+Cao+Ma+Rodriguez 2024, arxiv:2403.00892] explicitly encodes the per-phase swap symmetry by stacking one subgraph per phase; phases share message functions giving S_3 phase-permutation equivariance — though not formally proved.
- **No published E(3)-equivariant AC-PF GNN.** A search of arxiv/SS up to 2026-05 returns no paper that imposes E(3) or O(2) equivariance on bus state representations. Complex-valued GNNs that would naturally encode U(1) phase rotation are absent in the PF/OPF literature.
- **Sign-equivariant phase modelling.** The "phase angle is defined modulo 2π, with a sign-degenerate slack convention" is widely acknowledged (PowerModels.jl) but no GNN architecture imposes sign-equivariance on angle outputs; PIGNN-Attn-LS handles it via slack-bus normalisation, not architecture. [Qian et al. 2026, arxiv:2602.15883] does *reference-anchor normalisation* for flow-reconstruction PINNs, a closely related gauge-fix — directly applicable to phase angles but unported.
- **Equivariant ML for closely related tasks.** Cosmological Velocityformer [Tröster et al. 2026, arxiv:2605.21483] is a broken-symmetry-matched equivariant graph transformer, a useful template for the AC-PF analogue (broken U(1) under slack constraint).

Pillar gap: equivariance is essentially unsolved for AC-PF. PIGNN-Attn-LS is *not* permutation-equivariant by construction (positional encodings break it).

## B.4 Landscape — Heterogeneous and Hypergraph GNNs (Typed Nodes: Bus / Branch / Transformer / Generator)

This pillar is the most operationally mature.

- **Heterogeneous AC-OPF GNNs.** OPF-HGNN [Ghamizi+Ma+Cao 2024, IEEE PES GM, 13 cit] is the canonical generalisable heterogeneous GNN for AC-OPF, with bus / branch / transformer / generator typed nodes; topology generalisation across IEEE 14/30/57/118. Heterogeneous GNN with local + global message passing [Wen+Wen+Li 2026, Applied System Innovation] adds a global virtual-node summary on top of the typed graph for long-range dependencies. Physics-Informed Typed GNNs for OPF [Lopez-Garcia+Domínguez-Navarro 2024/2025, IEEE TPWRS, 20 cit] is the *physics-informed* twin: typed-node encoding + KCL residual + line-loading penalty. [Yang+Qiu+Liu 2024, IEEE TII, 25 cit] is the Topology-Transferable Physics-Guided GNN that handles AC/DC mixed topologies. [Yang+Qiu+Liu 2025, IEEE TII] adds Control-Mode Switching-Enabled Physics-Guided Multi-Agent Graph Learning for real-time AC/DC PF.
- **SafePowerGraph and SafePowerGraph-HIL.** SafePowerGraph-HIL [Ma+Ghamizi+Cao 2025, arxiv:2501.12427, Kiel PowerTech] adds real-time hardware-in-the-loop validation to heterogeneous GNNs, exposing a sim-to-real gap that benchmark-only studies miss. SafePowerGraph itself [Ghamizi+Cao et al. 2024] is the canonical safety/feasibility benchmark for typed GNN-OPF models.
- **DAG / directed heterogeneous.** A Directed Acyclic Graph Neural Network for AC-OPF [Guo+Sun+Park 2023, IEEE PES GM] exploits the rooted-tree structure of radial feeders to enforce directional flow.
- **Hypergraph PF.** No paper has yet defined a *true* hypergraph for power flow (zone-of-influence hyperedges spanning sets of buses), despite the obvious applicability to multi-bus equipment like FACTS, HVDC stations, or substations. The closest is the multigraph per-phase decomposition of PowerFlowMultiNet (B.3) and the line-graph DC-PF surrogate inside OptiGridML [Meng+Haider+Van Hentenryck 2025, arxiv:2508.01951] (referenced in Chunk A). Hypergraph attention has been applied to *load forecasting* but not PF.

Bridging Chunk A: SafePowerGraph and OptiGridML are also referenced as "scalable / hierarchical OPF GNNs" in Chunk A — they straddle B.4 and B.5.

## B.5 Landscape — Multi-Scale / Hierarchical GNNs (Zonal Pooling, Virtual-Node Summaries, Hierarchical Attention)

- **Hierarchical PF GNNs.** N-1 ROPF Augmented Hierarchical GNN (AHGNN) [Pham+Li 2024, arxiv:2402.06226, also in Chunk A] hierarchically predicts congested lines to reduce the N-1 OPF instance solved analytically. OptiGridML [Meng+Haider+Van Hentenryck 2025, arxiv:2508.01951] uses a line-graph DC-PF surrogate plus a HeteroGNN that respects breaker/substation hierarchy. Heterogeneous GNN with local + global message passing [Wen+Wen+Li 2026] (also B.4) uses a virtual-node summary as the explicit zonal pooling mechanism.
- **Two-stage zonal partitioning.** [Liu+Zhang+Liu 2026, IEEE PES IM] uses agglomerative hierarchical clustering to partition large hybrid grids into a primary sub-area and secondary sub-areas before running a deep-learning transient voltage stability assessment. Lopez-Garcia / Domínguez-Navarro Typed-GNN OPF (B.4) uses a 2-level coarse-to-fine training schedule.
- **Multigrid / U-Net templates from adjacent fields.** A Multigrid Graph U-Net for Multiphase Flow [Jiang+Chen+Yang 2024, arxiv:2412.12757] and DPGUNet [Ni+Yuan+Zheng 2024, IEEE TAES] are the most relevant *template architectures*; neither is applied to AC-PF. Multiresolution GT with wavelet PE [Ngô+Hy+Kondor 2023, arxiv:2302.08647] (also B.2) is the most natural hierarchical-attention template.
- **Multi-fidelity hierarchical GNNs.** Multi-Fidelity Graph Neural Network for OPF under uncertainty [Khayambashi+Hasnat+Alemazkoor 2024, JMLMC, 9 cit] cascades a low-fidelity DC-OPF GNN as a coarse layer and a high-fidelity AC-OPF GNN as a fine layer — closest to a true multi-scale OPF surrogate.
- **Spatio-temporal hierarchies.** Fusion of Transformer + Spatio-Temporal GCN [unspec. 2024] uses a hierarchical encoder for fault prediction, and [Lin+Zhang+Zhao 2025, IEEE TPWRS] builds a two-tier local-global convolutional graph network for regional PV forecasting.

Pillar gap: there is **no published zonal-pooling AC-PF GNN with learnable Kron-reduction**. The numerical-analysis literature has Kron reduction for centuries; only [Meng 2025] approximates it via line-graph DC; no AC analogue exists.

## B.6 Landscape — Continuous-Depth / Implicit / Deep-Equilibrium GNNs (Fixed-Point Nodal Balance)

The fixed-point view of AC-PF (`f_θ(z*) = z*` where the fixed point coincides with the PF solution) maps almost perfectly onto implicit GNNs and DEQs, yet adoption is scant.

- **Implicit GNN foundation.** IGNN [Gu+Chang+Zhu 2020, arxiv:2009.06211, NeurIPS, 177 cit] is the original implicit-GNN paper; IGNN: A Monotone Operator Viewpoint [Baker+Wang+Hauck 2023, ICML, 11 cit] makes the fixed-point well-posedness rigorous; IGNN-Solver [Lin+Ling+Feng 2024, arxiv:2410.08524] accelerates inference; Convergent Graph Solvers (CGS) [Park+Choo+Park 2021, arxiv:2106.01680, ICLR, 17 cit] proves convergence and is the closest "stationary-state predictor" template. **None has yet been instantiated on AC-PF as its fixed point.**
- **DEQ / neural-ODE for power-system dynamics (not PF).** Bossart+Lara+Roberts+Henriquez-Auba+Callaway+Hodge [2024, arxiv:2405.06827] (also Chunk A) proposes a DEQ layer + Neural ODE surrogate for power-system *dynamic* simulation, with the equilibrium initialised exactly at the PF solution — but the PF map itself is not a DEQ.
- **Fast-physics-aware unrolled solver.** FPL-OPF [Zhang+Yan+Sheng+Yu+Ye+Wang+Shi 2026, arxiv:2604.23548] (also Chunk A) embeds a fast PF iterative solver as the last differentiable block of an unsupervised AC-OPF NN; gradients are taken through only the last few iterations, and the truncated gradient is proved to be a high-fidelity surrogate of the true implicit gradient. This is the closest implicit-PF-layer published.
- **Energy / gradient-flow OPF.** [Liu 2025, arxiv:2512.01219] (Chunk A) reformulates OPF as energy minimisation on the constraint manifold and learns gradient-flow dynamics directly — effectively a continuous-time DEQ.
- **Graph Neural ODE for thermal hydraulics (template).** Graph Neural ODE Digital Twins [Almukhametov+Lim+Hu+Liu 2026, arxiv:2604.07292] uses a physics-informed message-passing GNN coupled with a controlled Neural ODE for nuclear reactor thermal-hydraulics — the closest mature template for "GNN + Neural ODE on a grid-like graph". Direct port to AC-PF is unpublished.
- **Implicit Z-Bus and hybrid solvers.** Hybrid GNN-IZR [Shamseldein 2025, arxiv:2510.04264] and Hybrid GNN-LSE [Shamseldein 2025, arxiv:2510.22020] couple a physics-informed GNN to the Implicit Z-Bus Recursive solver (an analytic fixed-point iteration), turning the GNN's failure cases into 0% failures on stressed IEEE-33 scenarios. This is the most mature "fixed-point fail-safe" hybrid.
- **NEO-Grid.** [Chehade+Zhu 2025, arxiv:2509.21668] introduces NEO-Grid, a neural approximation framework for optimisation and control in distribution grids — uses an iterative neural block with convergence to local Volt/Var optima.
- **Speed-up of implicit graph diffusion.** [Shi+Gao 2024, IJCNN] speeds up implicit graph neural diffusion via a simplified residual strategy — relevant to scaling DEQ-PF.

Bridging Chunk A: FPL-OPF and the DEQ-for-dynamics paper appear in both A.2 and B.6. The "PE-DEQ whose fixed point is the PF solution" gap is articulated *first* in Chunk A and is re-stated here in concrete architectural terms.

## B.7 Landscape — Topology-Aware Methods for Line Outages, N-1 / N-k, and Cross-Grid Generalization

- **Cross-topology heterogeneous GNNs.** OPF-HGNN [Ghamizi+Ma+Cao 2024] and Lopez-Garcia / Domínguez-Navarro Typed-GNN OPF [2024/2025] explicitly test on IEEE 14 / 30 / 57 / 118 cross-topology splits and show non-trivial transfer.
- **Generalisation benchmarks and analyses.** [Okoyomon+Goebel 2025, EECN] introduces a framework for assessing the generalisability of GNN-based AC-PF models — across topology, scale, and operating conditions. [Arowolo+Cremer 2025, arxiv:2510.06860] (Towards Generalisation of GNNs for AC-OPF) is the most direct attack on cross-grid generalisation; it explicitly motivates that "existing models struggle with scalability and topology flexibility". UGCN [Wu+Scaglione+Miguel+Arnold 2025, arxiv:2509.08672] (Chunk A) zero-shot transfers across reconfigurations of arbitrary dimensionality.
- **N-1 / N-k contingency-robust models.** CANOS [Piloto+Liguori+Madjiheurem et al. 2024, arxiv:2403.17660] is the canonical N-1-robust neural AC-OPF baseline. PowerModelsGAT-AI [Ezeakunne et al. 2026, arxiv:2603.16879] trains under N-2 contingencies on 13 IEEE systems. Multi-Model Disagreement Active Learning for N-1 SCOPF [Yu+Zheng+Chen 2025, CEECT] uses disagreement to *generate* N-1 stability constraints. Non-solution PF Diagnosis [Jiang+Ye+Ao 2025, AIP Advances] handles AC/DC hybrid topology changes including breaker switching and abrupt load variations.
- **Transferable busbar splitting.** Transferable Graph Learning for Transmission Congestion Management via Busbar Splitting [Rajaei+Palensky+Cremer 2025, arxiv:2510.20591] is one of the most explicit transfer-across-topology papers; it solves a mixed-integer NTO problem in near-real-time by exploiting a learned GNN policy.
- **Adversarial / verification.** [Parker 2026, arxiv:2602.17975] (Chunk A) constructs adversarial inputs for CANOS-PF that elicit 3.4 p.u. reactive-power errors under tiny voltage perturbations, exposing a *failure mode* of N-1 generalisation claims. This is the field's de facto impossibility-result: empirical N-1 robustness on average ≠ worst-case robustness.
- **Critical baselines.** The host group's sample-efficiency study [Conrad+Kim 2026, arxiv:2602.19667] finds dataset size dominates architecture for MLP vs GNN on PF; BOOST-RPF [Okoyomon+Goebel 2026, arxiv:2603.21977] (Chunk A) shows MLP/GNN suffer under topological shifts on ENGAGE while boosted-trees-with-physics-residual maintain precision across unseen feeders.
- **Out-of-distribution PF surveys.** [Liu+He+Chen 2026, arxiv:2601.02706] (Chunk A) shows scaling-law divergence between prediction accuracy and constraint feasibility — implying compute-optimal frontiers differ for in-distribution vs OOD PF.
- **Transient stability with GNNs.** Real-Time Multi-Stability Risk Assessment GNN [Chen+Bu+Wang 2025, IEEE TPWRS, 24 cit] applies GNN message-passing for stability risk visualisation, which feeds back into contingency screening.

## B.8 Landscape — Operator Learning Views (Fourier Neural Operator, DeepONet) for Power Flow

This is the *least-explored* but possibly most impactful pillar.

- **No published Fourier Neural Operator for AC/DC-PF.** Search hits return: Channel-Assisted FNO for wind-turbine wakes [Lee+Lee 2025, Physics of Fluids], Spectral-Refiner FNO for turbulent flows [Cao+Brarda+Li 2024, arxiv:2405.17211, ICLR], Dual-Branch Coupled FNO for multiphase flow [Hashim+Elyas+Williams 2025, Water]. None target power flow. The closest grid application is the FNO for fluid simulation analogy.
- **No published DeepONet for AC/DC-PF.** Search returns DeepONet applied to dynamics surveys and orientation tasks but nothing with the PF operator $(P, Q, Y) \mapsto (V, \theta)$ as the target. The closest "operator learning" cluster on grids is the Implicit Z-Bus hybrid in B.6, which is *not* a learned operator.
- **Empirical-operator equivalents in the GNN literature.** PowerFlowNet (B.1) and CANOS (B.4) effectively learn the discrete PF operator $f: (S_{\text{net}}, G) \mapsto (V, \theta)$ but as a graph regressor, not a function-space operator. No paper has framed PF as a parametric PDE-like operator amenable to FNO-style spectral conv.
- **LUMINA grid foundation model.** [Jin+Song+Memon 2026, arxiv:2605.02133] introduces LUMINA, a *grid foundation model* for benchmarking AC-OPF surrogate learning across network topologies — closest published artefact to a foundation operator for OPF. (Also referenced in Chunk A as a foundation model.) Its architecture is graph-transformer-based, not FNO/DeepONet.
- **Differentiable optimisation as quasi-operator.** Differentiable Optimisation for Deep Learning-Enhanced DC Approximation of AC-OPF [Rosemberg+Klamkin 2025, arxiv:2504.01970] makes the dispatch operator differentiable through the optimisation solver — a step toward operator learning but on the optimiser, not on PF.
- **Wider operator-learning context.** Spectral-Refiner [Cao+Brarda+Li 2024] and Multigrid Graph U-Net [Jiang+Chen+Yang 2024] are direct templates; the conceptual "FNO on Kron-reduced graph" or "DeepONet whose branch network is a GNN and trunk is a function-evaluation network" is unpublished for PF.

Pillar gap: this is essentially open territory. The Kron-reduced graph offers a small, dense, spectrally-meaningful operator domain ideal for FNO-style spectral conv.

---

## B.9 Generalization Frontier — what is *actually known* across topologies, grids, and OOD

This subsection isolates **empirical results** and **counter-examples** about cross-topology, cross-grid, OOD generalisation.

### Positive results

1. **Heterogeneous typed GNNs transfer across IEEE 14↔30↔57↔118** within a 5–15% accuracy degradation, when typed-edge encodings are used [Lopez-Garcia+Domínguez-Navarro 2024; Ghamizi+Ma+Cao 2024 OPF-HGNN; Yang+Qiu+Liu 2024 Topology-Transferable PG-GNN].
2. **Zero-shot transfer to N-1 outages** is feasible: CANOS [Piloto et al. 2024] retains ≈1% dispatch-cost error on 10,000-bus systems under random single-line outages.
3. **N-2 contingency generalisation** is possible with continual learning: PowerModelsGAT-AI [Ezeakunne et al. 2026] retains <2% base-task degradation after sequential pretraining on 13 IEEE systems.
4. **Universal cross-topology GCN** zero-shot transfer to arbitrary reconfigurations is achieved by UGCN [Wu+Scaglione+Miguel+Arnold 2025, arxiv:2509.08672] without retraining.
5. **MV→HV regime shift** is feasible with parameter-efficient adaptation (LoRA): PIGNN-Attn-LS-LoRA [Karim+Kim 2026, arxiv:2602.18227] adapts with 85% fewer trainable params.
6. **Multi-fidelity hierarchies** transfer well across uncertainty regimes [Khayambashi+Hasnat+Alemazkoor 2024 — multi-fidelity GNN].
7. **Boosted trees with physics-residual** match or beat MLP/GNNs across unseen feeders on ENGAGE [Okoyomon+Goebel 2026 BOOST-RPF, arxiv:2603.21977].

### Counter-examples and impossibility results

1. **Adversarial inputs break N-1 robustness.** Parker [2026, arxiv:2602.17975] generates adversarial inputs on CANOS-PF causing 3.4 p.u. reactive-power errors and 0.08 p.u. voltage-magnitude errors with as little as 0.04 p.u. voltage perturbation on a single bus. *Average-case N-1 robustness ≠ worst-case generalisation.*
2. **Topological shifts collapse MLP/GNN baselines.** [Okoyomon+Goebel 2026 BOOST-RPF] explicitly notes "global MLPs and GNNs often suffer from performance degradation under topological shifts" on Dorfnetz / ENGAGE.
3. **Dataset size dominates architecture.** [Conrad+Kim 2026, arxiv:2602.19667] (host group) shows on IEEE 5-bus that GNN < MLP differences are dwarfed by data-volume effects — *architectural priors are weaker than data scale in the low-data regime*.
4. **Accuracy ≠ feasibility scaling.** [Liu+He+Chen 2026, arxiv:2601.02706] (Chunk A) shows ML-OPF compute-optimal frontiers differ for prediction RMSE vs constraint feasibility, implying any *scaling-law-derived* generalisation claim is multi-dimensional.
5. **Sim-to-real gap.** SafePowerGraph-HIL [Ma+Ghamizi+Cao 2025, arxiv:2501.12427] reports a measurable sim-to-real gap on hardware-in-the-loop validation, undermining pure simulation-benchmark generalisation claims.
6. **Cross-grid pretraining is not yet a transfer regime.** No published model fine-tunes from one IEEE system to a non-IEEE realistic grid; UGCN and PIGNN-LoRA come closest but neither tests on a TSO-scale real grid.

### Open empirical questions

- Cross-voltage-regime generalisation (DC↔AC; LV↔MV↔HV↔EHV).
- Cross-frequency (50 Hz ↔ 60 Hz, multi-frequency AC) generalisation.
- Generalisation under HVDC link insertion/removal.
- Generalisation under FACTS device installation.
- Transfer from balanced to unbalanced three-phase.

---

## B.10 Research Gaps (10–12 specific, falsifiable)

1. **No graph transformer in PF literature uses spectral positional encodings derived from the Y-bus Laplacian.** All existing PF GTs (PIGNN-Attn-LS, PIGTN, Powerformer, heterogeneous GT) use bus-index or learned PE that ignores line susceptance asymmetries. SPGFormer [An+Dai+Wang 2025] does this for hyperspectral images; no PF analogue exists. Falsifiable target: replacing PE with `Φ = eigvecs(ℜ(Y) + jℑ(Y))` should improve cross-topology generalisation by ≥10 % on OPF-HGNN benchmarks.
2. **No equivariant AC-PF GNN exists.** Permutation equivariance over bus indices is unproved for any AC-PF GNN; phase-rotation U(1) equivariance is also unimplemented. Even PowerFlowMultiNet's S_3 phase-permutation equivariance lacks formal proof.
3. **No DEQ / implicit-GNN encodes nodal balance as its fixed point.** IGNN, CGS, IGNN-Solver exist; FPL-OPF unrolls a PF solver inside an AC-OPF NN; but no `z* = F_θ(z*; G, S_net)` model with the AC-PF solution as its unique Banach-contractive fixed point has been published.
4. **No true hypergraph PF model.** Multi-bus equipment (FACTS, HVDC links, substations) are modelled as bilateral edges; no paper uses zone-of-influence hyperedges. This loses sparsity and prevents zonal pooling.
5. **No learnable Kron reduction.** Kron reduction is the natural multi-scale primitive for PF; no GNN paper has introduced a differentiable Kron-reduction pooling layer with electrical-equivalent retention.
6. **No FNO / DeepONet for AC-PF.** Operator learning has not been ported to PF despite the natural parametric-PDE-like structure; spectral conv on the Kron-reduced graph is an obvious candidate.
7. **No GNN uses complex-valued edge features encoding admittance Y_ij as a single complex scalar with phase information.** Existing PF GNNs split Y into (G, B) real channels, losing phase asymmetry. Complex-valued attention is unimplemented.
8. **No sign-equivariant phase angle head.** Slack-bus normalisation is hand-coded in PIGNN-Attn-LS and all PF GNNs; no architecture imposes sign-equivariance on θ outputs by construction.
9. **No PF GNN uses spectral graph rewiring to mitigate over-squashing.** Single-cut topologies (radial feeders, tie-lines) are classical over-squashing pathologies; no PF GNN paper applies Cayley / spectral / commute-time rewiring [Attali+Buscaldi+Pernelle 2024 survey].
10. **No published worst-case-certified GNN-PF model.** Parker 2026 exposes adversarial failures; no PF GNN has any α-CROWN / IBP / Lipschitz certificate.
11. **Cross-grid pretraining at SynthCity scale is missing.** Existing largest training cohort is PowerGraph (~9,241 buses) or 13 IEEE systems (PowerModelsGAT-AI). No GNN-PF model is trained on ≥10^5 procedurally-generated grids.
12. **No GNN-PF method evaluates HVDC-link insertion/removal generalisation.** AC/DC hybrid PF GNNs [Yang+Qiu+Liu 2024/2025] only test on *fixed* AC/DC topologies, not on link insertion/removal.

(Gaps overlap intentionally with Chunk A.5 #5 — *PE-DEQ for PF* — and #11 — *equivariant DEQ*. The architectural framing here is complementary.)

---

## B.11 Breakthrough Directions (6–10 high-risk, each tied to specific mathematical structure)

1. **Y-Bus Spectral Graph Transformer (YBus-SGT).** Replace learned PE in PIGNN-Attn-LS with `Φ = ` eigenvectors of the complex Y-bus Laplacian; complex attention biases become invariant to bus relabeling and encode network impedance topology. Mathematical primitive: spectral decomposition of `Y = G + jB` over a connected sub-graph. Predicts ≥10 % cross-topology improvement.
2. **Permutation-Equivariant Deep-Equilibrium Layer for PF (PE-DEQ-PF).** Define `z* = MPNN_θ(z*; G, S_net)` with weight-sharing across bus indices and Anderson-accelerated forward solve. The unique fixed point is the AC-PF solution. Train with implicit differentiation through z* and Banach contractivity certificates. (Cross-listed with Chunk A.5 #1.) Falsifiable: zero KCL residual at inference AND zero-shot bus-permutation invariance.
3. **Operator-Learning View via Spectral Neural Operators on the Kron-Reduced Graph.** Frame contingency screening as `FNO_θ : (Y_ij outage mask, S_net) → (V, θ)` with spectral convolution restricted to the Kron-reduced subgraph spectrum. Mathematical primitive: graph FNO with eigen-basis defined per contingency. Provides amortised N-k screening at PDE-operator inference cost.
4. **Hypergraph PF GNN with Zone-of-Influence Hyperedges.** Define hyperedges for substations / FACTS / HVDC stations so the message function aggregates over multi-bus equipment in a single step. Combine with hypergraph attention. Mathematical primitive: hypergraph Laplacian `L_H = D − HW H^T D^{-1}`.
5. **Learnable Differentiable Kron Reduction as a Pooling Layer.** Define a pooling block that, given partition into boundary vs interior buses, computes the Kron-reduced Y-bus `Y_KK − Y_KI Y_II^{-1} Y_IK` differentiably (with Schur-complement back-prop). Provides zonal coarsening with electrical-equivalence retention.
6. **Complex-Valued Edge-Attention GNN.** Treat admittance Y_ij as a single complex-valued attention weight; use complex multiplication for message updates `m_ij = Y_ij · v_j`; enforce U(1) phase-rotation equivariance. Mathematical primitive: complex-valued GNN message passing with Hermitian aggregation. Direct extension of PIGNN-Attn-LS, replacing real (G,B) channels.
7. **DeepONet for AC-PF: Branch = GNN encoder of (S_net, G); Trunk = function-evaluation network on bus indices.** Mathematical primitive: tensor product of branch / trunk in operator-learning theory; the trunk evaluated at any subset of buses generalises to grid-resizing without retraining.
8. **Graph Rewiring via Y-Bus Commute-Time for Over-Squashing on Radial Feeders.** Apply spectral rewiring with commute-time edges proportional to (Z_bus)_ij; mitigates over-squashing on long tie-lines and radial feeders. Mathematical primitive: effective resistance on weighted graphs.
9. **Equivariant Graph Neural ODE for Continuous-Depth PF.** Combine [Almukhametov+Lim+Hu+Liu 2026 GNN-ODE template] with permutation-equivariant message passing and a Lyapunov-decreasing energy `H_θ(z)` vanishing at the PF solution. Provides certified contractive convergence and any-time inference.
10. **In-Context Power-Flow Transformer (ICL-PF) with Y-Bus-Conditioned Attention.** Tokenise grid as (P_i, Q_i, V_i, θ_i, Y_ij) triples; train a transformer to perform amortised Newton-step prediction conditioned on k in-context PF examples from similar grids. Mathematical primitive: amortised inference of the Jacobian inverse via in-context learning. (Cross-listed with Chunk A.5 #8.)

---

## B.12 Bibliography (handles verified during fetch)

### Message-passing variants (B.1)

1. Lin Y., Orfanoudakis I., Cardenas-Bojaca J. (2023). *PowerFlowNet: Power flow approximation using message passing Graph Neural Networks.* International Journal of Electrical Power & Energy Systems. arxiv:2311.03415.
2. Gao Y., Yu R., Yang B. (2024). *A Physics-Guided Graph Convolution Neural Network for Optimal Power Flow.* IEEE Transactions on Power Systems. doi:10.1109/TPWRS.2023.3238377.
3. Gao Y., Yu R., Yang B. (2023). *Physics Embedded Graph Convolution Neural Network for Power Flow Calculation Considering Uncertain Injections and Topology.* IEEE Transactions on Neural Networks and Learning Systems.
4. Suri D., Mangal S. (2025). *PowerGNN: A Topology-Aware Graph Neural Network for Electricity Grids.* arxiv:2503.22721.
5. Abelezele L., Sung J., Ramamurthy S. (2025). *Empirical Assessment of Graph Neural Network Convolution Operators for AC-OPF Learning.* DESTION @ CPSIoTWeek.
6. Jadhav S., Sevak B., Das S., Su W., Bui V.-H. (2025). *Enhancing Power Flow Estimation with Topology-Aware Gated Graph Neural Networks.* arxiv:2507.02078.
7. Le V., Le D. (2025). *DPFAGA — Dynamic Power Flow Analysis and Fault Characteristics: A Graph Attention Neural Network.* arxiv:2503.15563.
8. Liu Y., Shi H., Wang J. (2025). *A Topology-Aware Power Grid State Estimation Approach via Graph Attention Networks-Based Representation Learning.* ICCCBDA 2025.
9. Tuo M., Li X., Zhao Y. (2023). *Graph Neural Network-Based Power Flow Model.* NAPS 2023. arxiv:2307.02049.
10. Owerko D., Gama F., Ribeiro A. (2022). *Unsupervised Optimal Power Flow Using Graph Neural Networks.* ICASSP 2022. arxiv:2210.09277.
11. Yaniv A., Kumar P., Beck Y. (2023). *Towards adoption of GNNs for power flow applications in distribution systems.* Electric Power Systems Research.
12. Yaniv A., Goebel C. (2025). *Benchmarking Graph Neural Networks for Power Flow Prediction in Distribution Systems.* IEEE Kiel PowerTech 2025.
13. Wu B., Liu Q., Xie Y. (2024). *Fast power flow calculation method for power system based on heterogeneous edge graph convolutional neural network.* J. Phys.: Conf. Ser.
14. Conrad T., Kim C., Jäger J., Maier A., Bayer S. (2026). *Impact of Training Dataset Size for ML Load Flow Surrogates.* arxiv:2602.19667.
15. Talebi S., Zhou K. (2025). *Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids.* arxiv:2502.05702.
16. Tran D., Mitra A., Nguyen H. (2024). *Learning model combining of convolutional deep neural network with a self-attention mechanism for AC optimal power flow.* Electric Power Systems Research.

### Self-attention / graph transformers (B.2)

17. Kim C., Conrad T., Karim R., Oelhaf J., Riebesel D., Arias-Vergara T., Maier A., Jäger J., Bayer S. (2025). *Physics-informed GNN for medium-high voltage AC power flow with edge-aware attention and line search correction operator (PIGNN-Attn-LS).* ICASSP 2026. arxiv:2509.22458.
18. Karim R., Kim C., Conrad T., Gourmelon N., Oelhaf J., Riebesel D., Arias-Vergara T., Maier A., Jäger J., Bayer S. (2026). *Parameter-Efficient Domain Adaptation of Physics-Informed Self-Attention based GNNs for AC Power Flow Prediction.* arxiv:2602.18227.
19. Elnour M., Saleh O., Atat R. (2026). *Joint Sensor Deployment and Physics-Informed Graph Transformer for Smart Grid Attack Detection (PIGTN).* arxiv:2603.00085.
20. Li S., Sun W., Su F. (2026). *A Novel Power Flow Calculation Method Based on the Integration of Physical Mechanisms and Heterogeneous Graph Transformer.* ICEAAI 2026.
21. Chen Y., Luo Y., Liu Z., Wei P., Zhou J., Qing L., Zhang H., Song K., Song S. (2024). *Powerformer: A section-adaptive transformer for power flow representation.* arxiv:2401.02771.
22. Ezeakunne U. et al. (2026). *PowerModelsGAT-AI: Continual Physics-Informed Graph Attention Networks for Multi-Grid AC Power Flow.* arxiv:2603.16879.
23. Madsen B., Bank H., Mirshekali H. (2025). *Short-Term Spatial-Temporal Energy Forecasting in a Danish Distribution Grid Using a Hybrid Transformer-GNN Model.* Smart Power & Energy Security.
24. Ngô N.K., Hy T.S., Kondor R. (2023). *Multiresolution Graph Transformers and Wavelet Positional Encoding for Learning Hierarchical Structures.* JCP. arxiv:2302.08647.
25. Ma C., He J., Wei Y. (2024). *PolyFormer: Scalable Node-wise Filters via Polynomial Graph Transformer.* KDD 2024. arxiv:2407.14459.
26. Attali H., Buscaldi D., Pernelle N. (2024). *Graph Rewiring in GNNs to Mitigate Over-Squashing and Over-Smoothing: A Survey.* IJCAI 2026. arxiv:2411.17429.
27. An W., Dai Q., Wang H. (2025). *SPGFormer: Structure Perception Graph Transformer With Laplacian Position Encoding for Hyperspectral Image Classification.* IEEE TGRS.

### Equivariant / symmetry-aware (B.3)

28. Zelaya-Arrazabal F., Martinez-Lizana S., Pulgar-Painemal H., Zhao J. (2025). *Permutation-Equivariant Learning for Dynamic Security Assessment of Power System Frequency Response.* arxiv:2512.10232.
29. Lopez-Garcia M.A., Domínguez-Navarro J.A. (2024/2025). *Optimal Power Flow With Physics-Informed Typed Graph Neural Networks.* IEEE TPWRS.
30. Ghamizi S., Cao J., Ma A., Rodriguez P. (2024). *PowerFlowMultiNet: Multigraph Neural Networks for Unbalanced Three-Phase Distribution Systems.* arxiv:2403.00892.
31. Tröster A. et al. (2026). *Velocityformer: Broken-Symmetry-Matched Equivariant Graph Transformers for Cosmological Velocity Reconstruction.* arxiv:2605.21483.
32. Qian Y., Liu J., Xia Z., Chen S., Xu C., Cai S. (2026). *Distributed physics-informed neural networks via domain decomposition for fast flow reconstruction.* arxiv:2602.15883.

### Heterogeneous / hypergraph (B.4)

33. Ghamizi S., Ma A., Cao J. (2024). *OPF-HGNN: Generalisable Heterogeneous Graph Neural Networks for AC Optimal Power Flow.* IEEE PES GM 2024.
34. Wen H., Wen Y., Li G. (2026). *Heterogeneous Graph Neural Network with Local and Global Message Passing for AC-Optimal Power Flow Solutions.* Applied System Innovation.
35. Yang J., Qiu G., Liu T. (2024). *Topology-Transferable Physics-Guided Graph Neural Network for Real-Time Optimal Power Flow.* IEEE TII.
36. Yang J., Qiu G., Liu T. (2025). *Control Mode Switching-Enabled Physics-Guided Multiagent Graph Learning for Real-Time AC/DC Power Flow.* IEEE TII.
37. Ma A., Ghamizi S., Cao J. (2025). *SafePowerGraph-HIL: Real-Time HIL Validation of Heterogeneous GNNs for Bridging Sim-to-Real Gap in Power Grids.* IEEE Kiel PowerTech 2025. arxiv:2501.12427.
38. Guo S., Sun X., Park M. (2023). *A Directed Acyclic Graph Neural Network for AC Optimal Power Flow.* IEEE PES GM 2023.

### Multi-scale / hierarchical (B.5)

39. Pham T., Li X. (2024). *N-1 Reduced Optimal Power Flow Using Augmented Hierarchical Graph Neural Network (AHGNN).* arxiv:2402.06226.
40. Meng D., Haider R., Van Hentenryck P. (2025). *OptiGridML: Flow-Aware GNN for Transmission Network Reconfiguration via Substation Breaker Optimisation.* arxiv:2508.01951.
41. Khayambashi K., Hasnat M.A., Alemazkoor N. (2024). *Hybrid Chance-Constrained OPF under Load and Renewable Generation Uncertainty using Enhanced Multi-Fidelity Graph Neural Networks.* JMLMC.
42. Jiang Q., Chen J., Yang K. (2024). *A Multigrid Graph U-Net Framework for Simulating Multiphase Flow in Heterogeneous Porous Media.* arxiv:2412.12757.
43. Lin Y., Zhang Y., Zhao Q. (2025). *Short-Term Probabilistic Forecasting for Regional PV Power Based on Convolutional Graph Neural Network and Parameter Transferring.* IEEE TPWRS.

### Continuous-depth / implicit / DEQ (B.6)

44. Gu F., Chang H., Zhu W. (2020). *Implicit Graph Neural Networks (IGNN).* NeurIPS 2020. arxiv:2009.06211.
45. Baker M., Wang Z., Hauck S. (2023). *Implicit Graph Neural Networks: A Monotone Operator Viewpoint.* ICML 2023.
46. Lin Q., Ling X., Feng B. (2024). *IGNN-Solver: A Graph Neural Solver for Implicit Graph Neural Networks.* arxiv:2410.08524.
47. Park J., Choo J., Park N. (2021). *Convergent Graph Solvers.* ICLR 2021. arxiv:2106.01680.
48. Bossart M., Lara J.D., Roberts C., Henriquez-Auba R., Callaway D., Hodge B.-M. (2024). *Acceleration of Power System Dynamic Simulations using a Deep Equilibrium Layer and Neural ODE Surrogate.* arxiv:2405.06827.
49. Zhang J., Yan H., Sheng Z., Yu H., Ye S., Wang H., Shi Y. (2026). *FPL-OPF: Unsupervised Learning for AC Optimal Power Flow with Fast Physics-Aware Layer.* ACM e-Energy 2026. arxiv:2604.23548.
50. Liu X. (2025). *Neural Network Optimal Power Flow via Energy Gradient Flow and Unified Dynamics.* arxiv:2512.01219.
51. Almukhametov A., Lim D., Hu R., Liu Y. (2026). *Graph Neural ODE Digital Twins for Control-Oriented Reactor Thermal-Hydraulic Forecasting Under Partial Observability.* arxiv:2604.07292.
52. Shamseldein M. (2025). *A Hybrid GNN-IZR Framework for Fast and Empirically Robust AC Power Flow Analysis in Radial Distribution Systems.* arxiv:2510.04264.
53. Shamseldein M. (2025). *A Hybrid GNN-LSE Method for Fast, Robust, and Physically-Consistent AC Power Flow.* Electric Power Systems Research. arxiv:2510.22020.
54. Chehade S., Zhu H. (2025). *NEO-Grid: A Neural Approximation Framework for Optimisation and Control in Distribution Grids.* HICSS 2025. arxiv:2509.21668.
55. Shi Q., Gao C. (2024). *Speed-up Implicit Graph Neural Diffusion Model: A Simplified and Robust Strategy.* IJCNN 2024.

### Topology-aware contingency / generalisation (B.7)

56. Piloto L., Liguori S., Madjiheurem S., Zgubic M., Lovett S., Tomlinson H., Elster S., Apps C., Witherspoon S. (2024). *CANOS: A Fast and Scalable Neural AC-OPF Solver Robust to N-1 Perturbations.* arxiv:2403.17660.
57. Arowolo D., Cremer J. (2025). *Towards Generalisation of Graph Neural Networks for AC Optimal Power Flow.* arxiv:2510.06860.
58. Okoyomon E., Goebel C. (2025). *A Framework for Assessing the Generalisability of GNN-Based AC Power Flow Models.* Energy-Efficient Computing and Networking.
59. Rajaei A., Palensky P., Cremer J. (2025). *Transferable Graph Learning for Transmission Congestion Management via Busbar Splitting.* arxiv:2510.20591.
60. Jiang Y., Ye Z., Ao W. (2025). *Non-solution power flow diagnosis method for AC/DC hybrid power grid based on topology-aware graph neural network.* AIP Advances.
61. Yu H., Zheng Q., Chen J. (2025). *Multi-Model Disagreement Active Learning for N-1 Stability Constraint Construction in OPF.* CEECT 2025.
62. Parker R. (2026). *Generating adversarial inputs for a graph neural network model of AC power flow.* arxiv:2602.17975.
63. Wu Z., Scaglione A., Miguel A., Arnold M. (2025). *UGCN: A Universal Graph Convolutional Network for Power Systems.* arxiv:2509.08672.
64. Falconer T., Mones L. (2021). *Leveraging Power Grid Topology in Machine Learning Assisted Optimal Power Flow.* IEEE TPWRS. arxiv:2110.00306.
65. Chen J., Bu S., Wang Y. (2025). *Real-Time Multi-Stability Risk Assessment and Visualisation of Power Systems: A Graph-Neural-Network Approach.* IEEE TPWRS.
66. Okoyomon E., Goebel C. (2026). *BOOST-RPF: Boosted Sequential Trees for Radial Power Flow.* arxiv:2603.21977.

### Operator learning (B.8)

67. Jin Y., Song K., Memon S. (2026). *LUMINA: A Grid Foundation Model for Benchmarking AC Optimal Power Flow Surrogate Learning.* arxiv:2605.02133.
68. Rosemberg A., Klamkin M. (2025). *Differentiable Optimisation for Deep Learning-Enhanced DC Approximation of AC Optimal Power Flow.* arxiv:2504.01970.
69. Cao Z., Brarda S., Li S. (2024). *Spectral-Refiner: Accurate Fine-Tuning of Spatiotemporal Fourier Neural Operator for Turbulent Flows.* ICLR 2024. arxiv:2405.17211.
70. Lee J., Lee K. (2025). *Channel-assisted Fourier Neural Operator for high-fidelity far-wake prediction.* Physics of Fluids.
71. Hashim H., Elyas A., Williams M. (2025). *A Dual-Branch Coupled Fourier Neural Operator for High-Resolution Multi-Phase Flow Modelling.* Water.

### Surveys / critical baselines

72. Khaloie H., Dolányi M., Toubeau J.F. (2024). *Review of Machine Learning Techniques for Optimal Power Flow.* SSRN.
73. Jiang Q., Wang Q., Wu Z. (2024). *Advancements and Future Directions in the Application of Machine Learning to AC Optimal Power Flow: A Critical Review.* Energies.
74. Liu X., He X., Chen Y. (2026). *Scaling Laws of Machine Learning for Optimal Power Flow.* arxiv:2601.02706.
75. Habib R., Isufi E., Breda M. (2023). *Deep Statistical Solver for Distribution System State Estimation.* IEEE TPWRS. arxiv:2301.01835.
76. Donon B. (2022). *Deep statistical solvers & power systems applications (PhD thesis).*

### Adjacent operator / DEQ context

77. Tian H., Lian Y. (2025). *Unsupervised Deep Equilibrium Model Learning for Large-Scale Channel Estimation with Performance Guarantees.* arxiv:2508.10546.
78. Plakias C., Boutalis Y. (2025). *A Deep Equilibrium Model for Remaining Useful Life Estimation of Aircraft Engines.* Electronics.

---

End of Chunk B.
