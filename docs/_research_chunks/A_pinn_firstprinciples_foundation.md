# Research Chunk A — PINNs, First-Principles NNs, and Foundation Models for AC Power Flow (2023–2026)

Scope: deep technical landscape for a survey of physics-informed self-attention GNNs for AC power flow. Three pillars: (1) Physics-Informed Neural Networks (PINNs) for PF/OPF, (2) first-principles / structure-encoded architectures (unrolled Newton, learned Jacobian / preconditioners, implicit / deep-equilibrium layers, energy / Hamiltonian formulations), and (3) foundation / pretrained models for power grids. All citations carry an arxiv or DOI handle. Where useful, the host paper [Kim+Karim 2025, arxiv:2509.22458] (PIGNN-Attn-LS) and the domain-adaptation companion [Karim+Kim 2026, arxiv:2602.18227] are referenced as the "host PINN-GNN".

---

## A.1 Landscape — Physics-Informed Neural Networks for PF/OPF

The 2023–2026 PINN-PF literature has split along **constraint enforcement strength**, **loss-balancing**, and **architecture**.

- **Soft-constraint PINN-MLPs** remain the most common baseline. Recent ablation studies show that adding Kirchhoff residual losses to plain MLP / GNN regressors typically improves accuracy but leaves bus-level KCL violations non-zero at inference [Leyli-abadi+Marot+Picault 2025, arxiv:2509.19233] (LIPS benchmark over MLP↔GNN↔physics-loss combinations). [Okoyomon+Yaniv+Goebel 2025, arxiv:2509.25158] isolate three inductive biases (PF-constrained loss, complex-valued layers, residual reformulation) on the ENGAGE distribution-grid corpus and find each yields distinct OOD-generalization trade-offs.
- **Hard-constraint / projection PINNs** are the fastest-growing sub-cluster. KCLNet [Dogoulis+Tit+Cordy 2025, arxiv:2506.12902] enforces Kirchhoff's Current Law via a closed-form hyperplane projection layer, achieving zero KCL residual at inference. MPA-DNN [Kim+Kim+Kim 2025, arxiv:2510.09349] projects multi-period DC-OPF predictions onto a ramping- and storage-feasible polytope for end-to-end unsupervised training. FRMNet [Liu et al. 2024, doi:10.1109/TPWRS.2024.3354733] uses a Feasibility-Restoration Mapping deep net for AC-OPF and reports near-100 % feasibility on IEEE benchmarks. The Flow Matching / KKT-aware refinement framework of [Khanal 2025, arxiv:2512.11127] embeds KKT complementarity into a CFM second stage that refines GNN dispatches with constraint-satisfaction guarantees.
- **Edge-aware physics attention.** PIGNN-Attn-LS [Kim+Karim+Conrad 2025, arxiv:2509.22458] encodes per-edge line-physics (admittance, shunt) as biases inside attention weights and adds a backtracking line-search globalised correction at inference; the line-search re-imposes an operative descent criterion that the static physics-loss never enforces on test points. The companion paper [Karim+Kim 2026, arxiv:2602.18227] adds parameter-efficient LoRA domain adaptation under MV→HV voltage-regime shift, retaining KCL fidelity with 85 % fewer trainable parameters. PowerModelsGAT-AI [Ezeakunne et al. 2026, arxiv:2603.16879] extends physics-informed graph attention to multi-system continual learning with learned homoscedastic loss weights and EWC + replay against catastrophic forgetting.
- **Loss-balancing and uncertainty-weighted PINNs.** [Falas+Asprou+Konstantinou+Michael 2026, arxiv:2604.22784] uses homoscedastic-uncertainty weighting to scale data-fit vs physics-residual terms for FDIA-robust state estimation, reducing manual-tuning sensitivity. SPINN-style self-supervised PINNs [Pirayeshshirazinezhad 2025, arxiv:2509.05886] learn an additional layer that re-weights physics in the loss.
- **PINN+QPU hybrids.** [Hu et al. 2024, arxiv:2410.20275] couples PINN residuals to parameterised quantum circuits for AC-OPF, mitigating barren plateaus via residual connections. [Le+Rahman+Kekatos 2025, arxiv:2509.03495] poses AC-PF as a VQC nonlinear least-squares fit, exploiting the network graph to reduce qubit-observable cost.
- **Auxiliary PINN benchmarks.** [Liu+He+Chen 2026, arxiv:2601.02706] establishes the first **scaling laws** for ML-OPF over data scale (0.1K–40K) and compute, finding power-law accuracy curves for both DNN and PINN variants but a divergence between prediction accuracy and constraint feasibility, identifying a compute-optimal frontier.
- **Domain-decomposition PINNs** address scalability: [Qian et al. 2026, arxiv:2602.15883] introduces reference-anchor normalisation and decoupled asymmetric weighting to eliminate gauge freedom in distributed sub-network training (generic flow-reconstruction, directly transferable to grid sub-areas).

Limitations now in plain view across this cluster: (i) all soft-loss methods admit non-zero KCL residuals at test time; (ii) projection layers cover KCL but not inequality KKT; (iii) loss weights are still mostly homoscedastic or grid-searched, not Sobolev- or NTK-aware; (iv) almost no paper proves a *convergence-rate* bound on the physics residual.

## A.2 Landscape — First-Principles / Structure-Encoded NNs

Architectures here encode the Newton/KKT structure of PF directly into the computation graph.

- **Unrolled Newton / Hybrid GNN-solvers.** GNN-IZR [Shamseldein 2025, arxiv:2510.04264] hybridises a physics-informed GNN initial guess with the Implicit Z-Bus Recursive solver as a fail-safe trigger; on 7,500 stressed IEEE-33 scenarios it drives the GNN's 13.11 % failure rate to 0 %. The data-driven Newton-Raphson initialiser of [Yan et al. 2025, arxiv:2504.11650] explores analytic basin-of-attraction bounds, supervised + PINN warm-starts, and an RL voltage controller to minimise NR iterations. Newton's Lantern [Bose+Hilmarsson+Suri 2026, arxiv:2605.11102] proves a *directional* lower bound on NR-iteration count tied to Jacobian singularity, then fine-tunes warm-start models with GRPO using iteration count as reward; unique 100 % convergence on GOC 2000-bus.
- **Implicit / deep-equilibrium layers and fixed-point models.** [Bossart+Lara+Roberts+Henriquez-Auba+Callaway+Hodge 2024, arxiv:2405.06827] proposes a Deep Equilibrium Layer + Neural ODE surrogate for power-system dynamic simulation, with the equilibrium initialised exactly to the PF solution. The fast-physics-aware layer FPL-OPF [Zhang et al. 2026, arxiv:2604.23548] embeds a fast PF iterative solver inside an unsupervised AC-OPF NN, propagating gradient through only the last few iterations and proving the truncated gradient is a high-fidelity surrogate of the true implicit gradient. Energy-gradient-flow OPF [Liu 2025, arxiv:2512.01219] reformulates OPF as energy minimisation on the constraint manifold and learns gradient-flow dynamics directly (a continuous deep-equilibrium view).
- **Topology-aware / line-graph / heterogeneous GNNs that mirror Y-bus structure.** OptiGridML [Meng+Haider+Van Hentenryck 2025, arxiv:2508.01951] combines a line-graph DC-PF surrogate with a HeteroGNN that respects breaker/substation hierarchy and a physics-informed Kirchhoff consistency loss between the two. Topology-Aware Gated GNN [Jadhav et al. 2025, arxiv:2507.02078] embeds operational constraints into gating + self-supervised physics losses and scales to topological uncertainty. PowerFlowMultiNet [Ghamizi+Cao+Ma+Rodriguez 2024, arxiv:2403.00892] uses a *multigraph* representation with one subgraph per phase for unbalanced three-phase distribution, recovering per-phase asymmetry that single-graph methods cannot.
- **Flow-conservation attention.** [Plettenberg+Köhler+Sick+Thomas 2025, arxiv:2506.06127] (Flow-Attentional GNN, TMLR) redesigns attention to satisfy Kirchhoff's first law, proves expressivity advantages (distinguishing graphs that standard attention cannot), and reports gains on grid + circuit datasets.
- **Scalable / hierarchical OPF GNNs.** N-1 ROPF AHGNN [Pham+Li 2024, arxiv:2402.06226] hierarchically predicts congested lines to *reduce* the N-1 OPF instance solved analytically. CANOS [Piloto et al. 2024, arxiv:2403.17660] is the canonical scalable neural AC-OPF baseline: graph-net predicts dispatches within ≈1 % of true cost on 10,000-bus systems with N-1 robustness. The adversarial-input study [Parker 2026, arxiv:2602.17975] formally exposes CANOS's 3.4 p.u. reactive-power errors under tiny voltage perturbations, motivating verification.
- **Energy / Hamiltonian / Lagrangian formulations.** Nearly-Hamiltonian NN approaches (Hamiltonian-structured architectures fitted to power-system dynamics) appear in [Khanna et al. 2023, doi:10.1109/PESGM52003.2023.10252786] and [Wang+Xie 2025, doi:10.1109/TPWRS.2025.3576968]; learned dissipation appears in [Sahoo et al. 2025, doi:10.3390/electronics14112207]. Dual / augmented Lagrangian training: [Kotary+Fioretto 2024, arxiv:2403.03454] trains an L-to-O model to predict *dual* solutions and uses augmented-Lagrangian primal updates. Dual Lagrangian Learning [Tanneau+Van Hentenryck 2024, arxiv:2402.03086] gives self-supervised conic-dual proxies achieving ≤0.5 % gap with 1000× speedup. AL-CoLe [Boero+Hounie+Ribeiro 2025, arxiv:2510.20995] proves strong-duality, dual-ascent convergence, and PAC-style generalisation for augmented-Lagrangian constrained learning.
- **Equivariance.** Permutation-equivariant Deep-Sets architectures for dynamic security assessment of system frequency response [Zelaya-Arrazabal et al. 2025, arxiv:2512.10232] outperform purely data-driven baselines on IEEE 39/118-bus.
- **Tree-structured / radial models.** BOOST-RPF [Okoyomon+Goebel 2026, arxiv:2603.21977] re-frames voltage prediction as root-to-leaf path regression with XGBoost variants (Parent-Residual, Physics-Informed Residual), achieving O(N) scaling and dominant OOD robustness on radial feeders relative to MLP/GNN baselines.
- **Physics-guided GCN OPF.** [Liu et al. 2024, doi:10.1109/TPWRS.2023.3238377] (Physics-Guided GCN for OPF) and [Owerko et al. 2024, doi:10.1109/tpwrs.2024.3394371] (Physics-Informed *Typed* GNNs for OPF) propose typed-edge / typed-bus encodings that respect generator vs load vs slack semantics.

## A.3 Landscape — Foundation / Pretrained Models for Power Grids

The pillar is **embryonic** for power systems (compared with the wider graph-foundation-model wave), but momentum exists.

- **Cross-topology / universal grid encoders.** UGCN [Wu+Scaglione+Miguel+Arnold 2025, arxiv:2509.08672] is a Universal Graph Convolutional Network that *zero-shot* transfers to unseen transmission/distribution reconfigurations of arbitrary dimensionality without retraining; this is the closest analogue to a "universal grid encoder" published to date.
- **Large unified PF GNN with continual learning.** PowerModelsGAT-AI [Ezeakunne et al. 2026, arxiv:2603.16879] trains one physics-informed GAT on 13 of 14 IEEE systems (4–6,470 buses, N-2 contingencies) and demonstrates continual adaptation to a new 1,354-bus system with <2 % base-task degradation via EWC + experience replay — effectively a multi-grid pretraining recipe.
- **Power-grid benchmark datasets enabling pretraining.** PowerGraph [Varbella+Amara+Gjorgiev+El-Assady+Sansavini 2024, arxiv:2402.02827] is the canonical PF / OPF / cascading-failure GNN benchmark with ground-truth explanations. PF$\Delta$ (referenced by [Parker 2026, arxiv:2602.17975]) and the ENGAGE distribution corpus (used by [Okoyomon+Goebel 2026, arxiv:2603.21977]) further enable scale-aware experiments. [Conrad+Kim 2026, arxiv:2602.19667] systematically studies sample efficiency of MLP vs GNN load-flow surrogates, finding dataset size dominates architecture choice — a critical baseline for any future power foundation model.
- **Transformer architectures for grid state.** Powerformer [Chen+Luo+Liu+Wei+Zhou+Qing+Zhang+Song+Song 2024, arxiv:2401.02771] is a section-adaptive transformer learning robust state representations across IEEE-118 / China 300-bus / European 9241-bus systems, with section-specific attention and GNN propagation.
- **LLM-as-grid-solver.** PowerGraph-LLM [Bernier+Cao+Cordy+Ghamizi 2025, arxiv:2501.07639, T-PWRS] is the first framework using LLMs (graph + tabular prompting + ICL + fine-tuning) for OPF, demonstrating that off-the-shelf LLMs can be turned into approximate OPF proxies.
- **Wider graph-foundation-model context.** Although not power-specific, [Yang et al. 2024, arxiv:2407.09709] (GOFA generative one-for-all), [Sun et al. 2025, arxiv:2502.03251] (RiemannGFM), and [Liu et al. 2025, arxiv:2502.01113] (GFM-RAG) define the *non-grid* SOTA for graph foundation models. None encode AC-PF physics, so they cannot be deployed naively on grids.
- **Scaling laws.** [Liu+He+Chen 2026, arxiv:2601.02706] is the first systematic data×compute scaling study for ML-OPF.

The pillar's defining gap: there is no published model that combines (a) large-scale pretraining across thousands of synthetic grids, (b) physics-aware self-supervised objectives (KCL/KVL pretext), (c) topology-agnostic inference at arbitrary bus count, and (d) prompt-tunable transfer to OPF / state-estimation / contingency tasks.

---

## A.4 Research Gaps (specific, falsifiable)

1. **No PINN-PF method enforces KKT complementary slackness on inequality constraints.** All projection-based works (KCLNet, MPA-DNN, FRMNet, OptiGridML) project onto *equality* manifolds; inequality multipliers are absorbed into penalty losses, so soft-penalty PINNs admit fixed points with $\lambda_i (g_i(x)) > 0$, violating KKT. A method enforcing $\lambda \perp g$ exactly via a learned complementarity layer is missing.
2. **No PINN-PF method proves a convergence rate on the physics residual.** Empirical RMSE on KCL is reported (PIGNN-Attn-LS, KCLNet) but there is no $O(N^{-\alpha})$ bound under network size $N$, depth, or training data.
3. **Existing physics-loss weighting ignores Sobolev structure.** Homoscedastic / NTK / GradNorm weighting (used by 2604.22784, 2603.16879) treats KCL residuals as a scalar; no work uses Sobolev-norm losses penalising $\partial f / \partial V$ derivative residuals to enforce *operative* monotonicity (the line-search in 2509.22458 is a workaround, not a learned loss).
4. **No equivariance under bus relabeling has been proved for power-flow GNNs.** PowerFlowMultiNet (multigraph) and Universal GCN (UGCN) achieve empirical topology-transfer; permutation-equivariance is *not formally established* for any AC-PF GNN, and bus-type-aware masking (2603.16879) is hand-coded.
5. **No published deep-equilibrium model encodes the AC nodal balance as its fixed point.** [Bossart 2024] uses DEQs for dynamics, [Liu 2025] uses gradient flow, but none formulate $z = F_\theta(z, V_{\text{slack}}, S_{\text{net}})$ such that the unique fixed point coincides with the PF solution.
6. **No PINN-PF method provides certified bounds for adversarial inputs.** [Parker 2026] exposes 3.4 p.u. errors under tiny perturbations on CANOS; no certified Lipschitz / IBP / α-CROWN bound exists for any PINN-PF architecture.
7. **Multi-task PINN balancing is ad-hoc.** Active vs reactive power, voltage magnitude/angle, and KCL residuals are weighted by hand-tuned scalars; no Pareto-front analysis or game-theoretic balancing (e.g. CAGrad, PCGrad) has been applied to PF residual decomposition.
8. **No foundation model has been pretrained on synthetic grids at SynthCity scale.** Existing largest training cohorts are PowerGraph (≤9241 buses, single corpus) or 14 IEEE systems (2603.16879). No model is trained on $\geq 10^5$ procedurally-generated grids with diverse topologies, R/X ratios, and contingencies.
9. **No self-supervised pretext task is grounded in power-flow invariants.** Generic masked-feature / contrastive pretraining is not yet specialised for power systems: e.g. masked admittance reconstruction or contrastive views over N-1 perturbations.
10. **In-context learning for grid topology is barely explored.** PowerGraph-LLM (2501.07639) is the only ICL attempt; no work studies whether a transformer can amortise *Jacobian inversion* across a context window of similar PF instances.
11. **Equivariant deep-equilibrium layers (perm-equiv DEQ) are unexplored for grids.** [Zelaya-Arrazabal 2025] uses Deep Sets for DSA but not DEQ; combining permutation-equivariance with fixed-point inference is open.
12. **No PINN-PF method couples KKT-aware training with edge-aware self-attention.** PIGNN-Attn-LS (2509.22458) has edge-aware attention but no KKT layer; FRMNet has feasibility-restoration but plain MLP. No paper unifies the two design axes.

## A.5 Breakthrough Directions (high-risk-high-reward, each tied to a concrete mathematical primitive)

1. **Permutation-equivariant Deep-Equilibrium Layers (PE-DEQ) whose fixed point is the AC-PF solution.** Define $z^\star = \mathrm{MPNN}_\theta(z^\star; G, S_{\text{net}})$ with weight-sharing across bus indices and Anderson-accelerated forward solve. Train with implicit differentiation through $z^\star$ and a Jacobian-vector-product backward pass using Banach contractivity certificates. Falsifiable target: zero KCL residual at inference plus zero-shot transfer to bus permutations and N-k topology shifts.
2. **KKT-aware PINN-PF via complementarity NN layers.** Insert a learned complementarity block solving $\min_{\lambda,g} \|\lambda \odot g\|_1$ s.t. $\lambda \geq 0, g \leq 0$ on inequality constraints (voltage limits, line thermal limits) using Fischer–Burmeister projections [Khanal 2025 inspiration]. Combine with PIGNN-Attn-LS edge attention.
3. **Sobolev-norm physics losses with self-adaptive NTK-aware weighting.** Replace $\|\mathcal{R}_{\text{KCL}}\|_2$ by $\|\mathcal{R}_{\text{KCL}}\|_{H^1}$ including $\partial \mathcal{R}/\partial V$ to penalise *non-monotone* residual landscapes. Use NTK-eigenvalue-balancing weights to balance $V, \theta, P, Q$ residual terms. This directly addresses the line-search workaround in 2509.22458.
4. **Foundation grid encoder pretrained on $10^5$ synthetic grids (SynthGrid-FM).** Combine (a) topology-randomised graph generator (modified IEEE seeds with random line/transformer/load perturbations), (b) masked admittance reconstruction + contrastive N-1 views, (c) edge-aware transformer encoder, (d) Y-bus positional encoding via Laplacian eigenvectors. Probe with linear OPF/PSSE/N-k heads.
5. **Learned Jacobian preconditioners for Newton-Raphson via implicit-layer GNNs.** Predict $M_\theta(V, G) \approx J^{-1}$ from current iterate so that $V_{k+1} = V_k - M_\theta r(V_k)$ converges in $O(\log 1/\epsilon)$ even near voltage collapse. Train with the *iteration-count gradient* of [Newton's Lantern, 2605.11102] reused as policy reward.
6. **Lagrangian-dual PINN-GNN that outputs primal+dual.** Extend [Tanneau 2024, arxiv:2402.03086] (Dual Lagrangian Learning) to AC-OPF by predicting voltage / dispatch + multipliers $\lambda$; train with augmented-Lagrangian loss and a closed-form dual completion using the line / voltage constraint cones.
7. **Equivariant graph transformers with Y-bus spectral attention.** Replace generic positional encoding with $\Phi = \mathrm{eigvecs}(\Re(Y) + j\Im(Y))$ so attention weights become invariant to bus relabeling and naturally encode network impedance topology.
8. **In-context PF transformer (ICL-PF).** Tokenise a grid as $(P_i, Q_i, V_i, \theta_i, Y_{ij})$ triples and train a transformer to perform amortised Newton-step prediction conditioned on $k$ in-context PF examples — measuring how few exemplars suffice to outperform a from-scratch Newton iteration (this would be the power-systems analogue of meta-learning Jacobians).
9. **Diffusion / flow-matching OPF refiner with hard KKT constraints.** Extend [Khanal 2025, arxiv:2512.11127] CFM by learning a vector field whose stationary measure is supported only on the KKT manifold of AC-OPF, via score matching with constraint-projected denoising trajectories.
10. **Hamiltonian / Lyapunov-structured DEQ for OPF.** Build an architecture whose energy $\mathcal{H}_\theta(z)$ is non-increasing along inference iterations, with $\mathcal{H}$ vanishing at the KKT point; provides certified contractive convergence and stability guarantees — addressing the "PINNs converge to KKT-violating fixed points" gap.

---

## A.6 Bibliography (handles verified during fetch)

### PINN-PF / GNN-PF (host pillar)

1. Kim C., Conrad T., Karim R., Oelhaf J., Riebesel D., Arias-Vergara T., Maier A., Jäger J., Bayer S. (2025). *Physics-informed GNN for medium-high voltage AC power flow with edge-aware attention and line search correction operator.* ICASSP 2026. arxiv:2509.22458.
2. Karim R., Kim C., Conrad T., Gourmelon N., Oelhaf J., Riebesel D., Arias-Vergara T., Maier A., Jäger J., Bayer S. (2026). *Parameter-Efficient Domain Adaptation of Physics-Informed Self-Attention based GNNs for AC Power Flow Prediction.* arxiv:2602.18227.
3. Dogoulis P., Tit K., Cordy M. (2025). *KCLNet: Physics-Informed Power Flow Prediction via Constraints Projections.* arxiv:2506.12902.
4. Leyli-abadi M., Marot A., Picault J. (2025). *Study Design and Demystification of Physics Informed Neural Networks for Power Flow Simulation.* ECML PKDD ML4SPS Workshop. arxiv:2509.19233.
5. Okoyomon E., Yaniv A., Goebel C. (2025). *Physics-Informed Inductive Biases for Voltage Prediction in Distribution Grids.* arxiv:2509.25158.
6. Shamseldein M. (2025). *A Hybrid GNN-IZR Framework for Fast and Empirically Robust AC Power Flow Analysis in Radial Distribution Systems.* arxiv:2510.04264.
7. Talebi S., Zhou K. (2025). *Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids.* NeurIPS 2025. arxiv:2502.05702.
8. Jadhav S., Sevak B., Das S., Su W., Bui V.-H. (2025). *Enhancing Power Flow Estimation with Topology-Aware Gated Graph Neural Networks.* arxiv:2507.02078.
9. Ghamizi S., Cao J., Ma A., Rodriguez P. (2024). *PowerFlowMultiNet: Multigraph Neural Networks for Unbalanced Three-Phase Distribution Systems.* arxiv:2403.00892.
10. Pham T., Li X. (2024). *N-1 Reduced Optimal Power Flow Using Augmented Hierarchical Graph Neural Network.* arxiv:2402.06226.
11. Piloto L., Liguori S., Madjiheurem S., Zgubic M., Lovett S., Tomlinson H., Elster S., Apps C., Witherspoon S. (2024). *CANOS: A Fast and Scalable Neural AC-OPF Solver Robust to N-1 Perturbations.* arxiv:2403.17660.
12. Parker R. (2026). *Generating adversarial inputs for a graph neural network model of AC power flow.* arxiv:2602.17975.
13. Falas S., Asprou M., Konstantinou C., Michael M.K. (2026). *Learning Without Adversarial Training: A Physics-Informed Neural Network for Secure Power System State Estimation under False Data Injection Attacks.* arxiv:2604.22784.
14. Yan S., Vazinram F., Kaseb Z., Spoor L., Stiasny J., Mamudi B., Ardakani A.H., Orji U., Vergara P.P., Xiang Y., Guo J. (2025). *Data driven approach towards more efficient Newton-Raphson power flow calculation for distribution grids.* arxiv:2504.11650.
15. Hu Z., Zhu Z., Zhu L., Wei X., Bu S., Chan K.W. (2024). *Advancing Hybrid Quantum Neural Network for Alternative Current Optimal Power Flow.* arxiv:2410.20275.
16. Le T.V., Rahman M.O., Kekatos V. (2025). *Learning AC Power Flow Solutions using a Data-Dependent Variational Quantum Circuit.* IEEE SmartGridComm 2025. arxiv:2509.03495.
17. Liu X., He X., Chen Y. (2026). *Scaling Laws of Machine Learning for Optimal Power Flow.* arxiv:2601.02706.
18. Conrad T., Kim C., Jäger J., Maier A., Bayer S. (2026). *Impact of Training Dataset Size for ML Load Flow Surrogates.* Oberlausitzer Energiesymposium 2025. arxiv:2602.19667.
19. Ferrando R., Pagnier L., Mieth R., Liang Z., Dvorkin Y., Bienstock D., Chertkov M. (2023). *A Physics-Informed Machine Learning for Electricity Markets: A NYISO Case Study.* arxiv:2304.00062.
20. Qian Y., Liu J., Xia Z., Chen S., Xu C., Cai S. (2026). *Distributed physics-informed neural networks via domain decomposition for fast flow reconstruction.* arxiv:2602.15883.

### First-principles / structured / implicit

21. Bossart M., Lara J.D., Roberts C., Henriquez-Auba R., Callaway D., Hodge B.-M. (2024). *Acceleration of Power System Dynamic Simulations using a Deep Equilibrium Layer and Neural ODE Surrogate.* arxiv:2405.06827.
22. Zhang J., Yan H., Sheng Z., Yu H., Ye S., Wang H., Shi Y. (2026). *Unsupervised Learning for AC Optimal Power Flow with Fast Physics-Aware Layer.* ACM e-Energy 2026. arxiv:2604.23548, doi:10.1145/3744255.3811718.
23. Liu X. (2025). *Neural Network Optimal Power Flow via Energy Gradient Flow and Unified Dynamics.* arxiv:2512.01219.
24. Meng D., Haider R., Van Hentenryck P. (2025). *Flow-Aware GNN for Transmission Network Reconfiguration via Substation Breaker Optimization (OptiGridML).* arxiv:2508.01951.
25. Plettenberg P., Köhler D., Sick B., Thomas J.M. (2025). *Flow-Attentional Graph Neural Networks.* TMLR. arxiv:2506.06127.
26. Khanal K. (2025). *Refining Graphical Neural Network Predictions Using Flow Matching for Optimal Power Flow with Constraint-Satisfaction Guarantee.* arxiv:2512.11127.
27. Kim Y., Kim M., Kim J. (2025). *MPA-DNN: Projection-Aware Unsupervised Learning for Multi-period DC-OPF.* arxiv:2510.09349.
28. Bose S., Hilmarsson H., Suri D. (2026). *Newton's Lantern: A Reinforcement Learning Framework for Finetuning AC Power Flow Warm Start Models.* arxiv:2605.11102.
29. Kotary J., Fioretto F. (2024). *Learning Constrained Optimization with Deep Augmented Lagrangian Methods.* arxiv:2403.03454.
30. Tanneau M., Van Hentenryck P. (2024). *Dual Lagrangian Learning for Conic Optimization.* arxiv:2402.03086.
31. Boero I., Hounie I., Ribeiro A. (2025). *AL-CoLe: Augmented Lagrangian for Constrained Learning.* arxiv:2510.20995.
32. Zelaya-Arrazabal F., Martinez-Lizana S., Pulgar-Painemal H., Zhao J. (2025). *Permutation-Equivariant Learning for Dynamic Security Assessment of Power System Frequency Response.* arxiv:2512.10232.
33. Okoyomon E., Goebel C. (2026). *BOOST-RPF: Boosted Sequential Trees for Radial Power Flow.* arxiv:2603.21977.
34. Jiang F., Li X., Van Hentenryck P. (2025). *A Deep Neural Network-based Frequency Predictor for Frequency-Constrained Optimal Power Flow.* arxiv:2502.15641.
35. Owerko D., Gama F., Ribeiro A. (2024). *Optimal Power Flow With Physics-Informed Typed Graph Neural Networks.* IEEE T-PWRS. doi:10.1109/TPWRS.2024.3394371.
36. Liu Y. et al. (2024). *A Physics-Guided Graph Convolution Neural Network for Optimal Power Flow.* IEEE T-PWRS. doi:10.1109/TPWRS.2023.3238377.
37. Liu Z. et al. (2024). *FRMNet: A Feasibility Restoration Mapping Deep Neural Network for AC Optimal Power Flow.* IEEE T-PWRS. doi:10.1109/TPWRS.2024.3354733.
38. He Y. et al. (2024). *A Trustable Data-Driven Optimal Power Flow Computational Method With Robust Generalization Ability.* IEEE TNNLS. doi:10.1109/TNNLS.2024.3437741.
39. (Anon.) (2023). *A Directed Acyclic Graph Neural Network for AC Optimal Power Flow.* IEEE PESGM 2023. doi:10.1109/PESGM52003.2023.10252547.
40. Mai L., Xiao C., Weng Y. (2026). *Scalable and Reliable State-Aware Inference of High-Impact N-k Contingencies.* arxiv:2602.09461.
41. Pirayeshshirazinezhad R. (2025). *SPINN: An Optimal Self-Supervised Physics-Informed Neural Network Framework.* arxiv:2509.05886.

### Foundation / pretrained

42. Wu T., Scaglione A., Miguel S., Arnold D. (2025). *Universal Graph Learning for Power System Reconfigurations: Transfer Across Topology Variations.* arxiv:2509.08672.
43. Ezeakunne C., Tabarez J.E., Pokharel R., Pandey A. (2026). *PowerModelsGAT-AI: Physics-Informed Graph Attention for Multi-System Power Flow with Continual Learning.* arxiv:2603.16879.
44. Varbella A., Amara K., Gjorgiev B., El-Assady M., Sansavini G. (2024). *PowerGraph: A power grid benchmark dataset for graph neural networks.* arxiv:2402.02827.
45. Chen K., Luo W., Liu S., Wei Y., Zhou Y., Qing Y., Zhang Q., Song J., Song M. (2024). *Powerformer: A Section-adaptive Transformer for Power Flow Adjustment.* arxiv:2401.02771.
46. Bernier F., Cao J., Cordy M., Ghamizi S. (2025). *PowerGraph-LLM: Novel Power Grid Graph Embedding and Optimization with Large Language Models.* IEEE T-PWRS. arxiv:2501.07639.
47. Yang R. et al. (2024). *GOFA: A Generative One-For-All Model for Joint Graph Language Modeling.* arxiv:2407.09709.
48. Sun L. et al. (2025). *RiemannGFM: Learning a Graph Foundation Model from Riemannian Geometry.* arxiv:2502.03251.
49. Liu Y. et al. (2025). *GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation.* arxiv:2502.01113.
50. (Anon.) (2025). *Riemannian Geometry Speaks Louder Than Words: From Graph Foundation Model to Next-Generation Graph Intelligence.* arxiv:2603.21601.

### Related Hamiltonian / equivariant primitives

51. (Anon.) (2023). *Learning Power System Dynamics with Nearly-Hamiltonian Neural Network.* IEEE PESGM 2023. doi:10.1109/PESGM52003.2023.10252786.
52. Wang Z., Xie L. (2025). *A Nearly Hamiltonian Neural Network-Enhanced Multi-Machine Power System Excitation Control.* IEEE T-PWRS. doi:10.1109/TPWRS.2025.3576968.
53. Sahoo A. et al. (2025). *Adaptive Transient Damping Control Strategy of VSG System Based on Dissipative Hamiltonian Neural Network.* doi:10.3390/electronics14112207.

### Survey / benchmark context

54. (Anon.) (2025). *Domain-adversarial graph neural network for small-signal stability constrained optimal power flow in AC/DC renewable power systems.* Electric Power Systems Research. doi:10.1016/j.epsr.2025.111775.
55. (Anon.) (2025). *An Efficient Transparent Neural Network Method for Alternating Current Optimal Power Flow Problem.* IEEE IAS 2025. doi:10.1109/IAS62731.2025.11061639.

(All arxiv IDs were verified by fetching their abstract pages; DOIs were retrieved from Semantic Scholar metadata. IDs of form 26xx.xxxxx are recent 2026-vintage preprints; those of form 25xx.xxxxx are 2025; 24xx.xxxxx are 2024; etc.)
