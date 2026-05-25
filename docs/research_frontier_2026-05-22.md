# Research Frontier: Power Flow Prediction & Restoration Path Finding (2023–2026)

**Date:** 2026-05-22
**Host project:** Physics-Informed Self-Attention GNN for AC Power Flow (PIGNN-Attn-LS; arxiv:2509.22458) with LoRA+PHead domain adaptation (arxiv:2602.18227).
**Scope:** Deep cross-field survey across NNs, PINNs, first-principles NNs, foundation models, RL, information theory, statistics, abstract algebra/topology, and game theory. Two target problems: (a) **Power Flow Prediction** and (b) **Restoration Path Finding**.

This document is the synthesis layer. Detailed domain-specific landscapes, bibliographies, and per-domain gap lists live in the six research chunks under `docs/_research_chunks/`:

| Chunk | Domain | Path |
|---|---|---|
| A | PINNs · First-principles NNs · Foundation models | `_research_chunks/A_pinn_firstprinciples_foundation.md` |
| B | GNN architectures · Graph transformers · Operator learning | `_research_chunks/B_gnn_powerflow.md` |
| C | Reinforcement Learning (PF, OPF, restoration) | `_research_chunks/C_rl_powerflow_restoration.md` |
| D | Information theory · Statistics · UQ · Causality | `_research_chunks/D_infotheory_statistics.md` |
| E | Abstract algebra / Geometry / Topology · Game theory | `_research_chunks/E_algebra_gametheory.md` |
| F | Restoration path finding (MILP · graph · learning · uncertainty) | `_research_chunks/F_restoration_path_finding.md` |

Combined corpus: ~260+ peer-reviewed and preprint citations spanning 2023–2026, sourced from arxiv, OpenAlex, Semantic Scholar, IEEE Xplore, and major venue proceedings.

---

## 0. Executive Summary

Across the six surveyed domains, three structural observations recur.

**Observation 1 — The "soft-physics" plateau.** State-of-the-art PINN-PF and GNN-PF surrogates have converged on physics-loss regularisation (KCL/KVL residual penalties) and edge-aware self-attention. Empirical accuracy is excellent on in-distribution test grids, but six independent failure modes block deployment: (i) inequality constraints have no complementary-slackness enforcement (Chunk A gap #1), (ii) no worst-case Lipschitz / certified-robustness bound exists (B #10), (iii) Parker [2026, arxiv:2602.17975] demonstrates 3.4 p.u. errors from 0.04 p.u. adversarial voltage perturbations on CANOS, (iv) cross-topology generalisation is empirically fragile beyond minor relabelings (B.9 counter-examples), (v) the fixed-point structure of AC nodal balance has no published Deep-Equilibrium realisation (A #5, B #3), and (vi) UQ on outputs treats voltage components independently rather than as a joint feasible manifold (D #1). These six gaps are *not orthogonal* — they share a single underlying primitive: nobody has built a surrogate whose architecture *forces* the output to be a fixed point of an AC nodal-balance operator and whose calibration *certifies* that fixed point against the KKT cone.

**Observation 2 — The algebraic / topological / game-theoretic vocabulary is barely touched.** Sheaf neural networks, persistent homology of cascading failures, U(1) gauge equivariance under phase reference, D_3 three-phase symmetry, max-plus restoration scheduling, mean-field control TSO–DSO coupling, no-regret AC-OPF, online VCG, do-calculus restoration counterfactuals, and cycle-space (Betti-1) cost allocation are *all* either unstudied or studied without grid-aware instantiation (Chunk E). Each one corresponds to a published mathematical primitive that *exactly* describes a known grid property; the bottleneck is engineering instantiation, not theory development. This is the highest-leverage, lowest-effort breakthrough surface.

**Observation 3 — Restoration is being eaten by hybrid RL+MILP, but feasibility certification is the wall.** Restoration path-finding has decisively shifted to graph-policy MARL with MILP feasibility wrappers (Chunk F). No published work delivers (a) a differentiable Steiner-arborescence layer, (b) an AC-feasibility projection layer inside the RL loop, (c) a frequency-nadir constraint as a QP layer, (d) a deep POMDP solver with topology-attention belief state, or (e) a restoration foundation model. The same primitives that solve PF prediction's KKT problem (implicit-function-theorem gradients through Newton-Raphson, differentiable MIP) solve restoration's feasibility-during-exploration problem. These are *not two separate research programmes*.

**Bottom line.** The frontier is not "more accurate PF surrogates" or "better RL for restoration"; it is **physics-feasible-by-construction differentiable solvers that share a single algebraic backbone (sheaf-Laplacian / Y-bus-equivariant / DEQ fixed-point)** with **calibrated uncertainty over the AC feasibility manifold** and **online game-theoretic / no-regret coordination across TSO–DSO–DER agents**.

---

## 1. Power Flow Prediction — Cross-Domain Synthesis

### 1.1 Landscape map (where each domain stands, 2023–2026)

| Sub-field | Maturity | Frontier paper | Hardest open question |
|---|---|---|---|
| Soft-constraint PINN-PF | Mature | PIGNN-Attn-LS [arxiv:2509.22458] | Inequality complementary slackness |
| Hard-projection PINN-PF | Mid | KCLNet [2506.12902], FRMNet, MPA-DNN [2510.09349], FPL-OPF [2604.23548] | KKT manifold, not just equality manifold |
| Edge-attention GNN-PF | Mature | PIGNN-Attn-LS, Powerformer [2401.02771], Flow-Attentional GNN [2506.06127] | Y-bus spectral PE; U(1) equivariance |
| Equivariant GNN-PF | Nascent | HH-MPNN [2510.06860], PINCO | Only permutations; no U(1) / D_3 |
| Heterogeneous typed GNN | Mature | OPF-HGNN [Ghamizi+Ma+Cao 2024] | Hypergraph for multi-bus equipment |
| Hierarchical / multi-scale | Mid | Khayambashi+Hasnat+Alemazkoor 2024 | Learnable Kron reduction |
| Continuous-depth / DEQ | Empty for AC-PF | Liu 2025 gradient flow; Bossart 2024 (dynamics) | Fixed-point AC-PF DEQ unpublished |
| Topology-transfer / N-1 | Mid | CANOS, UGCN [2509.08672], PIGNN-LoRA [2602.18227] | Adversarial robustness [Parker 2026] |
| Operator learning (FNO/DeepONet) | Empty | — | No PF FNO/DeepONet published |
| Foundation models | Nascent | UGCN, PowerModelsGAT-AI [2603.16879], PowerGraph-LLM [2501.07639] | No SynthCity-scale pretraining |
| Conformal prediction | Nascent | Mollaali 2024; Alcántara 2026 N-k stratified CP; SPLICE 2026 | Manifold-CP on AC feasibility set |
| Bayesian NN UQ | Empty for Laplace | (37+ Laplace papers, none on PF) | Last-layer Laplace on GNN-PF |
| Wasserstein-DRO OPF | Mature | Brock 2025 KL-DRJCC; Zhou 2024 | Kernel-DRO / Sinkhorn-DRO |
| GP-PF | Mature | Pareek/Deka/Mitrovic 2023–25 | Sparse / manifold GP on grid graphs |
| Information bottleneck | Empty for PF | Speckhard 2025 [2505.11517] (topology only) | IB for SCADA→V GNN encoder |
| Causal / do-calculus | Empty for AC-PF | Ghosh 2024 [2410.19179] (latent cascade DAG) | Pearl do(·) on PF DAG |
| Sheaf neural networks | Empty for grids | Bodnar 2022, HetSheaf 2024 (general) | First sheaf-GNN PF surrogate |
| Persistent homology grid | Nascent | Hernández-García 2025 Betti-resilience | PH-time-series early warning |
| Mean-field / Stackelberg | Mid | Bo 2025 (MFG-EV); Jiang/Bolognani 2025 | Continuum-DSO MFC w/ AC-PF |
| RL OPF (safe) | Mid | Lagrangian/PD-PPO [WU-24, FEN-26]; projection [SAY-23] | Anytime feasibility under N-k |

### 1.2 Cross-cutting gaps (themes that appear in ≥3 chunks)

These 12 themes are the highest-leverage targets — each is a *named* gap in multiple chunks, meaning they are independently recognised as bottlenecks by researchers approaching the problem from different angles.

**G1. Inequality constraint satisfaction.** [A #1; B #10; C #1; D #2; F #1] No method enforces complementary slackness on AC-PF inequality constraints; all use soft penalties or post-hoc projection, both of which admit KKT-violating outputs. *Primitive:* Fischer–Burmeister projection layers / proximal complementarity / differentiable conic projection.

**G2. Deep-equilibrium / fixed-point structure.** [A #5, #11; B #3] The AC-PF map admits a unique Banach-contractive fixed point under standard conditions, yet no published model encodes nodal balance as a learned fixed-point operator. *Primitive:* `z* = F_θ(z*; G, S_net)` with Anderson acceleration and implicit-function gradients.

**G3. Equivariance beyond permutations.** [A #4; B #10 #2; E #1, #3] No PF surrogate exploits U(1) global-phase gauge symmetry (`θ_i → θ_i + α`), D_3 three-phase symmetry, or sequence-component symmetry. Even permutation equivariance is unproven for AC-PF GNNs. *Primitive:* gauge-equivariant message passing on `θ_i − θ_j`; equivariant projection through symmetrical components; Lie-point-symmetry canonicalisation (LieLAC).

**G4. Y-bus / Kron-reduction / sheaf-Laplacian structure.** [B #10 #1, #5; E #2] No graph transformer uses Y-bus-Laplacian spectral positional encodings; no learnable differentiable Kron reduction exists; the sheaf-Laplacian generalisation of the Y-bus (which natively handles bus-type heterogeneity) is unrealised on grids. *Primitive:* eigendecomposition of `Y = G + jB`; Schur-complement back-prop; cellular sheaves with admittance-valued restriction maps.

**G5. Foundation-model-scale pretraining.** [A #8; B #11; C #7; F #6 #13] No model is pretrained on ≥10⁵ synthetic grids; existing largest corpus is ~9k buses (PowerGraph) or 13 IEEE systems (PowerModelsGAT-AI). No restoration foundation model exists. *Primitive:* procedural synthetic grid generators, masked-admittance / contrastive-N-1 / masked-switch SSL pretexts.

**G6. Worst-case / certified robustness.** [A #6; B #10 #10; C #1] No PF model carries α-CROWN / IBP / Lipschitz certificates. Parker 2026 demonstrates adversarial failures. RL exploration violates constraints in transit. *Primitive:* Lyapunov-structured DEQ inference, control-Lyapunov functions, certified-projection layers.

**G7. Joint / manifold uncertainty quantification.** [D #1, #2, #11] Conformal prediction treats voltage components independently; no joint prediction region on the AC feasibility manifold; no conformal risk control for PF inequality constraints; no adaptive CP under topology drift. *Primitive:* manifold-CP with Riemannian conformity score; CRC scalar λ̂ for joint constraint coverage.

**G8. Operator-learning view.** [B #10 #6] FNO/DeepONet have not been ported to AC-PF despite the natural parametric-PDE-like structure; spectral conv on Kron-reduced graph is an obvious candidate. *Primitive:* spectral neural operator on power-grid eigen-basis.

**G9. Differentiable simulation + implicit gradients.** [C #10; F #9] Differentiable Newton-Raphson exists in JAX; no RL paper backpropagates policy gradients through it. Differentiable MIP layers (IntOpt, MIPLearn) are unused in restoration. *Primitive:* implicit-function-theorem gradients through optimisers.

**G10. Causal inference on PF DAGs.** [D #10; E (implicit)] Do-calculus has not been instantiated on AC-PF causal graphs; counterfactual restoration ordering ("if line L_47 first") is unrealised. *Primitive:* structural causal models with `do(s)` on switching actions; topological identifiability conditions.

**G11. Information-theoretic compression with physics feasibility.** [D #4, #5, #7] No paper minimises a rate-distortion-feasibility trilemma `(R, D, ε_PF)`; SCADA compression treats PF residual as external. *Primitive:* Lagrangian `I(X;Z) + λ_D D + λ_F E[max_g g_+]`.

**G12. Multi-task PINN balancing is ad-hoc.** [A #7] No Pareto-front analysis or game-theoretic balancing (CAGrad, PCGrad) of `(V, θ, P, Q, KCL_eq, KCL_ineq)` residuals. *Primitive:* multi-objective gradient surgery; NTK-eigenvalue-balanced loss weights; Sobolev-norm penalties on residual derivatives.

### 1.3 Breakthrough programme for PF prediction

The 60+ chunk-level breakthrough directions cluster into seven *moonshots* — each fuses primitives from at least three of the six domains and addresses ≥2 of the cross-cutting gaps. Pursuing any one moonshot fully would produce a publishable top-venue result; pursuing the joint programme would reshape the field.

**M1. PE+U(1)-equivariant Deep-Equilibrium AC-PF (PE-U-DEQ-PF).** [Fuses A.bd #1, B.bd #2, E.bd #2; closes G2 + G3 + G6.] Define `z* = MPNN_θ(z*; G, S_net)` where the message function is U(1)-equivariant (functions of `θ_i − θ_j`), weight-shared across bus indices (permutation-equivariant), with Anderson-accelerated forward solve and Banach contractivity certificates. Train with implicit differentiation through `z*`. Falsifiable: zero KCL residual at inference, zero-shot bus-permutation invariance, exact invariance under slack-bus rotation. *Risk:* contractivity may fail near voltage-collapse; mitigate with input-convex parametrisation.

**M2. Sheaf-Foundation Grid Model (Sheaf-FM).** [Fuses A.bd #4, B.bd #4 #5, E.bd #1; closes G4 + G5.] Backbone: a sheaf-neural-network whose stalk at each bus carries `(V, θ, P, Q)` (or 12-vector in 3-phase) and whose edge restriction maps encode learnable admittance-valued 4×4 (or 12×12) orthogonal-then-scaled transformations. Pretrain on ≥10⁵ synthetic grids with masked-admittance and contrastive N-1 pretexts. The sheaf-Laplacian *is* the rigorous algebraic generalisation of the Y-bus. Probe with linear OPF / PSSE / N-k heads. *Risk:* sheaf training on irregular topologies has not been demonstrated beyond small ZINC-scale graphs.

**M3. KKT-Aware PINN-GNN with Complementarity Layer (KKT-PIGNN).** [Fuses A.bd #2 #6 #9; closes G1.] Insert a learned Fischer–Burmeister complementarity block enforcing `λ ⊥ g` on inequality constraints (voltage / thermal limits), combined with PIGNN-Attn-LS edge attention and dual-output (primal + multipliers). Train with augmented-Lagrangian loss. Compose with M1 for fixed-point KKT solutions. *Risk:* complementarity is non-smooth; need smoothed Fischer–Burmeister with annealing.

**M4. Manifold Conformal + Conformal-Risk-Controlled PF Surrogate.** [Fuses D.bd #2 #6 #7; closes G7.] Output: joint prediction regions on the AC-PF feasibility manifold using Riemannian conformity score `s = d_g(V, V̂)`. Adds a single CRC scalar λ̂ giving finite-sample distribution-free guarantee `Pr[max_i g_i(V̂ + λ̂·s) ≤ 0] ≥ 1 − δ`. Stack onto any backbone (PIGNN, PE-U-DEQ-PF, Sheaf-FM). *Risk:* manifold conformity requires distance computation; may be expensive at inference.

**M5. Y-Bus Spectral Graph Transformer with Complex Attention (YBus-cGT).** [Fuses B.bd #1 #6 #8; closes G3 + G4.] Replace learned PE with eigenvectors of the complex Y-bus Laplacian; treat admittance `Y_ij` as a single complex-valued attention weight (Hermitian aggregation); apply commute-time graph rewiring to mitigate over-squashing on radial feeders. *Risk:* complex-valued backprop is unstable; mitigate with split real/imag with phase-coupled gradients.

**M6. Information-Bottleneck PF Surrogate with Rate-Distortion-Feasibility Loss.** [Fuses D.bd #1 #3; closes G11.] Single training objective `L = I(X;Z) + λ_D D(X, X̂) + λ_F E[max_g g_+(PF(X̂))]` compresses SCADA, reconstructs measurements, and penalises PF residual. Yields *minimum sufficient representation* of SCADA for PF — directly identifies which buses carry information. *Risk:* MI estimators (MINE, CLUB) are high-variance; bound carefully.

**M7. Persistent-Homology Early-Warning + Do-Calculus Counterfactual Reschedule.** [Fuses E.bd #4, D.bd #5; closes G10.] Stream PMU → time-evolving weighted graph (weights = line-loading margins) → Betti-0/Betti-1 persistence diagrams in a sliding window → TDA-kernel classifier flags imminent cascade. Counterfactual policy `π*(s_1,…,s_T) = argmax E[load | do(s_1)…do(s_T)]` reroutes flows pre-event. *Risk:* PH computation latency may not match grid timescales; need approximate persistence.

### 1.4 What probably *won't* work (negative directions)

- **Bigger MLPs/GNNs without structural priors.** [Conrad+Kim 2026, arxiv:2602.19667] (host group) shows architecture differences are dwarfed by data scale in low-data; but [BOOST-RPF, 2603.21977] shows scale alone breaks under topology shift. Scaling is necessary but not sufficient.
- **Soft KCL penalties at higher weights.** [Parker 2026] adversarial attacks succeed even when residuals are tiny; penalty methods *cannot* deliver worst-case guarantees in principle.
- **Conformal prediction that ignores feasibility geometry.** Independent-component CP gives nominal coverage but generates trajectories that violate AC constraints. Must use manifold-aware CP.
- **Foundation models without physics SSL.** Generic masked-feature pretraining on grid topologies has not improved over from-scratch in our cut; physics-grounded SSL pretexts are mandatory.

---

## 2. Restoration Path Finding — Cross-Domain Synthesis

### 2.1 Landscape map (where each sub-field stands, 2023–2026)

| Sub-field | Maturity | Frontier paper | Hardest open question |
|---|---|---|---|
| MILP / SCOPF restoration | Mature | Liang 2025 IET RPG (hybrid RL+MILP) | Inrush-aware, frequency-secured |
| SOC / SDP relaxations | Mid | Various 2024–25 conic restoration | Tight relaxations w/ dynamics |
| Stochastic / DRO | Mid | Zhu 2025 IJEPES (DRO microgrid); Shi 2024 | Crew-aware adversarial damage |
| Multi-stage gas-electric | Mid | Zhang 2024 IEEE TPWRS | Tight Weymouth + sequencing |
| Steiner / MST formulations | Empty for AC | Sakkour 2024 (distributed MST) | Tight gap-bounded AC Steiner |
| RL restoration (single-agent) | Mature | DQN/PPO variants 2023–25 | Offline RL on real SCADA |
| Graph-RL / MARL restoration | Mature | Jacquemart 2024 (Nature Comms); Fan 2023; Vu 2023 | Topology-shift generalisation |
| Hierarchical RL | Mid | Hosseini 2023; Jo 2024 | Learned option discovery |
| Deep POMDP restoration | Empty | Li 2026; SAFE-25 (scenario tree only) | Belief over latent faults |
| Differentiable MIP layers | Empty | (IntOpt, MIPLearn exist; no restoration use) | End-to-end RL+MILP gradients |
| Restoration foundation model | Empty | — | No published candidate |
| GFM-inverter black-start | Nascent | Seo 2024–25 (device only); Huang 2025 | Network-level GFM-coordinated MILP |
| Equity / fairness | Nascent | Tatari 2024 (weighted-load only) | Lorenz / envy-freeness / proportionality |
| Cyber-physical co-recovery | Nascent | UAV 2023; BEL 2024 | Joint feasibility coupling |
| Multi-energy joint restoration | Nascent | Jafarzadeh 2023; Yan 2025 | Physically consistent couplings |
| LLM-assisted restoration | Empty | — | No peer-reviewed work |
| Open benchmark | Empty | — | No PGLib-equivalent |

### 2.2 Cross-cutting gaps for restoration

**R1. AC-feasibility-certified learning-based restoration.** [F #1; C #1] No published learning-based planner provides certified AC feasibility during sequential energisation. Current methods use DC linearisation with post-hoc voltage violations or wrap MILP/SOCP around RL with no end-to-end gradient. *Primitive:* implicit-differentiation through AC-PF solver as a differentiable feasibility projection.

**R2. Differentiable combinatorial restoration.** [F #9; C #10] No restoration paper backpropagates through Steiner-arborescence / MIP layers. *Primitive:* Berthet-perturbed Steiner solver; IntOpt / MIPLearn dual-decomposition gradients; cvxpylayers for SOCP.

**R3. Inrush- and frequency-nadir-aware sequencing.** [F #2; F #11] Static load models are universal; cold-load pickup, transformer inrush, GFM droop, virtual inertia are at most linearised. *Primitive:* analytic frequency-nadir bound as a QP layer; GFM cluster with droop / virtual inertia as logical MILP constraints.

**R4. Deep POMDP with topology-attention belief.** [F #4; C #4] Restoration is naturally a POMDP with optimal-stopping structure (when to re-close a tie under uncertain fault location), but no surveyed paper formalises it as such. *Primitive:* Bayesian fault filtering, particle critic, Whittle index, Graph-Set-Transformer belief.

**R5. Restoration foundation model.** [F #6, #13; C #7] No transferable pretrained policy; per-feeder overfit is endemic. *Primitive:* graph transformer pretrained on synthetic feeder + damage scenarios with masked-switch-and-status BERT-style pretext.

**R6. Robust min-max-min with crew-adaptive adversary.** [F #5] Existing DRO microgrid formation handles load/DER ambiguity but not crew-adaptive adversarial damage. *Primitive:* three-level robust opt via C&CG with affine decision rules.

**R7. Axiomatic fairness / equity.** [F #7] No work guarantees envy-freeness, max-min, proportional fairness across census tracts. *Primitive:* Lorenz-dominance constraint inside MILP; Shapley fairness on restoration order.

**R8. Cyber-physical product-graph routing.** [F #8] Joint feasibility-coupled co-optimisation of cyber and physical paths is open. *Primitive:* Cartesian product of cyber + physical graphs; single Steiner / shortest-path on product.

**R9. Multi-energy with tight conic couplings.** [F #12] Multi-stage stochastic gas-electricity exists but uses linearised Weymouth. *Primitive:* tight conic relaxations with sequencing.

**R10. Open benchmark.** [F #15] No PGLib-equivalent. *Primitive:* feeder + damage generator + crew/MESS pool + IBR cranking + ENS/GWh/restoration-time/equity harness.

**R11. Algebraic-scheduling for crew dispatch.** [E #5] Tropical/max-plus algebra maps exactly to synchronise-and-energise restoration; unrealised. *Primitive:* max-plus eigenvalue = critical-path time; max-plus row balancing = crew load distribution.

**R12. Coalition mechanism-design for inter-utility crew coordination.** [E #8; F #14] Cooperative MARL optimises team return; combination with individual-rational / credit-assignment mechanism design is open. *Primitive:* core-selecting / VCG-on-restoration coalitions.

### 2.3 Breakthrough programme for restoration

**M8. Differentiable-Steiner + AC-Projection + Frequency-Nadir-QP Restoration Policy (DiffStAC-Fn).** [Fuses F.bd #1 #2 #3, C.bd #2; closes R1 + R2 + R3.] Architecture: GNN policy → Berthet-perturbed Steiner-arborescence layer (radial feasibility, differentiable) → Newton-corrected AC-PF projection layer (voltage feasibility, implicit gradients) → analytic frequency-nadir QP layer (transient feasibility, KKT gradients). Each layer is differentiable; the composed policy is trained end-to-end with PPO using implicit-function-theorem gradients. *Risk:* triple-nested implicit gradients have high condition number; mitigate with proximal regularisation.

**M9. Deep POMDP Restoration with Topology-Attention Belief and Bayesian Fault Filter.** [Fuses F.bd #4, C.bd #1 #9; closes R4 + R5.] Belief over post-event topology maintained by Graph-Set-Transformer with MoCo-style contrastive update; particle critic (R2D2-style); Dreamer-world-model rollouts for 100× planning; optimal-stopping head decides "energise next branch vs. wait for diagnosis". *Risk:* belief-state collapse under high observation noise.

**M10. Restoration Foundation Model (Restore-FM).** [Fuses F.bd #6, C.bd #4; closes R5 + R10.] Pretrain on ≥10⁴ synthetic feeders × damage scenarios with three SSL pretexts: (i) masked switch-status BERT, (ii) contrastive damage / no-damage pairs, (iii) next-energisation-step autoregressive. Fine-tune per utility. Distribute the pretraining corpus as the open benchmark (closes R10). *Risk:* sim-to-real gap; mitigate with adversarial domain adaptation and HIL validation.

**M11. Robust Min-Max-Min Restoration with Crew-Adaptive Adversary.** [Fuses F.bd #5, R6; closes R6.] Three-level: operator commits switching plan → adversary commits damage (crew-aware) → operator adapts recourse with affine decision rules → solved via column-and-constraint generation. *Risk:* C&CG scaling beyond 50 buses; mitigate with Benders + ML cut-selection.

**M12. Cyber-Physical Product-Graph Steiner Restoration.** [Fuses F.bd #8; closes R8.] Construct Cartesian product `G_cyber × G_phys`; solve a single weighted Steiner-tree / shortest-path on the product graph with simultaneous cyber + physical feasibility constraints. Gives joint cyber-physical recovery sequence. *Risk:* product graph blows up; mitigate with abstraction-refinement (CEGAR-style).

**M13. Lorenz-Equity-Constrained Restoration MILP with Causal Counterfactuals.** [Fuses F.bd #7, D.bd #5; closes R7 + G10.] Encode equity as Lorenz-dominance constraint inside chance-constrained MILP. Use do-calculus to evaluate counterfactual restoration orderings under structural causal model. Produce equitable, causal-aware restoration plans. *Risk:* identifiability conditions for causal effects may fail; characterise pre-deployment.

**M14. Max-Plus Crew Scheduler with Persistent-Homology Damage Assessment.** [Fuses E.bd #5, E.bd #4; closes R11 + R4.] Encode restoration as max-plus state-space `x(k+1) = A ⊗ x(k)`; eigenvalue = critical-path time; row-balancing = crew load. Couple with PH-based damage detection from PMU streams. Produces optimal cyclic re-energisation rhythm. *Risk:* max-plus and PH literatures have not been bridged; engineering effort substantial but mathematically clean.

**M15. Online VCG with Rolling-Horizon AC-OPF for Multi-Utility Crew Coordination.** [Fuses E.bd #9, F.bd #6; closes R12 + G9.] Incentive-compatible online auction for crew assignment; allocation rule is a learned AC-OPF + restoration surrogate; payments via dual prices computed by autodiff. *Risk:* incentive compatibility requires monotone allocation rules; learned surrogates may violate this — need verifier.

### 2.4 What probably *won't* work for restoration

- **Single-feeder RL fine-tuning at deployment time.** Sample complexity is too high; transferable pretrained policies (M10) are mandatory.
- **DC-only relaxations with post-hoc AC validation.** [F gap #1] explicitly notes voltage violations are routinely missed; AC-feasibility projection (M8) is non-negotiable.
- **Restoration without learned belief over faults.** Open-loop sequencing wastes crew time on diagnosed branches; belief-state POMDP (M9) is the right framing.
- **LLM-only restoration recommendation.** [F #13] LLMs do not natively satisfy AC feasibility; useful only as planner *over* verified MILP/MARL sub-tools.

---

## 3. Cross-Task Synergies — Projects That Solve Both PF and Restoration

Two of the seven PF moonshots and three of the seven restoration moonshots share architectural backbones. Combining them yields three *meta-projects* with disproportionate impact.

### 3.1 Meta-Project Σ1: Unified Sheaf-DEQ Backbone with Restoration Head

Combine **M1 (PE-U-DEQ-PF)** and **M2 (Sheaf-FM)** as the shared encoder, with task-specific heads:
- PF prediction head: linear projection to `(V, θ)`.
- KKT-aware OPF head [M3].
- Restoration policy head [M8 + M10]: differentiable Steiner + AC-projection + frequency-nadir QP, fine-tuned from Sheaf-FM pretraining.

The sheaf-Laplacian backbone gives algebraic generalisation of Y-bus; the DEQ fixed-point gives nodal-balance feasibility; the heads are interchangeable. *Direct extension of the host project (PIGNN-Attn-LS):* replace Y-bus implicit pre-conditioner with sheaf-Laplacian; replace residual line-search with DEQ-Anderson; reuse LoRA+PHead for per-grid adaptation.

### 3.2 Meta-Project Σ2: Manifold-Conformal + Persistent-Homology Risk Layer

Combine **M4 (Manifold CP)** with **M7 (PH + do-calculus)**:
- Online PMU stream → PH time-series → cascade probability.
- Forward predictions from Σ1 → manifold-CP joint prediction region → conformal-risk-controlled feasibility certificate.
- Counterfactual restoration ordering via do-calculus over the PF causal DAG (closes G10 + R7 + R8 simultaneously).

Yields a *risk-aware decision layer* that wraps any PF prediction or restoration policy with finite-sample, distribution-free, topology-adaptive guarantees.

### 3.3 Meta-Project Σ3: Game-Theoretic Coordination Layer for TSO–DSO + Crew

Combine **M15 (online VCG)** with mean-field-control bilevel TSO–DSO (E.bd #6) and no-regret AC-OPF (E.bd #8):
- TSO sets price signals as Stackelberg leader.
- Continuum of DSOs solve MFC with AC-PF constraints (Σ1 as DSO inner solver).
- Crews allocated by online VCG with rolling-horizon restoration plans (M10 as allocator).
- No-regret learning gives sub-linear regret against social cost.

A single coordination architecture spanning prediction (Σ1), risk (Σ2), and incentives (Σ3) — three layers of a *production-ready* grid intelligence stack.

---

## 4. Connection to the Host Project (PIGNN-Attn-LS + LoRA+PHead)

The host project sits at the **intersection** of soft-constraint PINN, edge-aware self-attention GNN, and parameter-efficient transfer (Chunks A, B). Its current architecture handles four of the twelve identified PF gaps adequately (edge attention, line-search-corrected fixed point, LoRA transfer, dataset-size scaling). The remaining eight gaps — G1, G2, G3, G4, G5, G6, G7, G8 — are *natural next steps* from PIGNN-Attn-LS.

**Closest single-step extension:** replace the line-search post-hoc correction with a **Permutation-Equivariant Deep-Equilibrium layer (M1)** whose fixed point is the AC-PF solution. Empirically this should eliminate the post-hoc correction step (which is currently a runtime cost) and provide certified contractive convergence. This is the highest-leverage 6-month project for the host group.

**Twelve-month moonshot:** add **Sheaf-Laplacian backbone (M2)** with **KKT-aware complementarity head (M3)** and **manifold conformal layer (M4)** — yielding a model that is (a) algebraically rigorous, (b) KKT-feasible by construction, (c) finite-sample-distribution-free in UQ, and (d) directly extensible to OPF and restoration via Σ1.

**Twenty-four-month programme:** the full Σ1 + Σ2 + Σ3 stack would be a complete grid-intelligence platform. With 2–3 PhD students, this is feasible.

---

## 5. Five Concrete Project Proposals (Ranked by Leverage / Effort)

These are the projects we recommend the host group pursue, ordered by expected reward-per-effort.

### Project P1. PE-DEQ-PF (M1) — *6-month sprint*

- **Aim:** Replace line-search post-correction in PIGNN-Attn-LS with permutation-equivariant Deep-Equilibrium layer whose fixed point is the AC-PF solution.
- **Mathematical primitives:** Banach-contractive MPNN, Anderson acceleration, implicit-function-theorem gradients.
- **Falsifiable claim:** Zero KCL residual at inference + zero-shot bus-permutation invariance + ≥10% improvement on N-1 contingency RMSE vs. PIGNN-Attn-LS.
- **Risk:** Contractivity may fail near voltage-collapse; mitigate with input-convex parametrisation.
- **Venues:** ICLR / NeurIPS / IEEE TPWRS.

### Project P2. KKT-PIGNN with Fischer–Burmeister Complementarity (M3) — *9-month*

- **Aim:** Add complementary-slackness enforcement on AC-PF inequalities via Fischer–Burmeister projection layers.
- **Mathematical primitives:** Smoothed Fischer–Burmeister, augmented Lagrangian, dual-output GNN.
- **Falsifiable claim:** Zero inequality violations on test set (CANOS-PF, PowerGraph) without post-hoc projection.
- **Venues:** NeurIPS / IEEE TPWRS / Power Systems Computation Conference.

### Project P3. Manifold-CP + Conformal-Risk-Control PF Wrapper (M4) — *6-month*

- **Aim:** Stack a manifold-conformal layer on PIGNN-Attn-LS giving joint distribution-free prediction regions on the AC-PF feasibility manifold.
- **Mathematical primitives:** Riemannian distance on PF manifold, conformal risk control scalar.
- **Falsifiable claim:** Empirical coverage ≥(1 − α) on N-1 perturbations, joint over all bus voltages, with calibration set size ≤500.
- **Venues:** ICML / NeurIPS / JMLR.

### Project P4. Sheaf-Laplacian Foundation Grid Model (M2) — *18-month*

- **Aim:** Pretrain a sheaf-GNN on ≥10⁵ synthetic grids with masked-admittance + N-1 contrastive SSL; linear-probe with PF / OPF / restoration heads.
- **Mathematical primitives:** Cellular sheaves with admittance-valued restriction maps; sheaf Laplacian; masked-feature / contrastive pretraining.
- **Falsifiable claim:** Linear-probe PF RMSE on unseen IEEE grids ≤ from-scratch GNN trained on the same grid with full supervision.
- **Risk:** Sheaf training has not been demonstrated at 10⁵-grid scale; this is the biggest engineering risk in the programme.
- **Venues:** ICLR / NeurIPS / Nature Machine Intelligence.

### Project P5. DiffStAC-Fn Restoration Policy (M8) — *12-month*

- **Aim:** End-to-end differentiable restoration policy: GNN → Berthet-perturbed Steiner-arborescence → AC-PF projection → frequency-nadir QP → PPO training.
- **Mathematical primitives:** Perturbed combinatorial solvers, implicit AC-PF gradients, KKT-gradient QP.
- **Falsifiable claim:** Zero AC-PF violations and zero frequency-nadir violations during exploration on IEEE 123-bus / DOE restoration scenarios.
- **Venues:** IEEE TPWRS / Nature Communications / NeurIPS Climate workshop.

---

## 6. Hard Problems / Risk Map

| Problem | Why hard | Mitigation |
|---|---|---|
| Sheaf-GNN at 10⁵-grid scale | Training instability; no precedent | Start with HetSheaf [Barbero+2024]; scale gradually; release intermediate checkpoints |
| DEQ contractivity near voltage collapse | Banach contractivity fails | Input-convex parametrisation; spectral regularisation on message function |
| Manifold-CP computational cost | Riemannian distance is expensive | Local linearisation of PF Jacobian; approximate nearest-neighbour |
| Triple-nested implicit gradients (M8) | High condition number | Proximal regularisation; staggered training |
| Sim-to-real gap for foundation models | HIL validation rare | Adversarial domain adaptation; multi-fidelity training [Khayambashi+2024] |
| Persistent-homology latency | PH computation slow | Streaming approximate persistence [Cohen-Steiner et al.] |
| Mechanism-design incentive compatibility | Learned allocation may break monotonicity | Verifier-based post-hoc certification |
| Adversarial robustness certification | α-CROWN scales poorly on graphs | Lipschitz-bounded GNN parametrisation; tighter IBP bounds |

---

## 7. References

The full bibliography (~260 entries) is distributed across the six research chunks. Quick-access landmark papers cited above:

- **Host project:** PIGNN-Attn-LS [arxiv:2509.22458]; LoRA+PHead [arxiv:2602.18227]; Dataset-size [arxiv:2602.19667].
- **PINN-PF:** KCLNet [2506.12902]; FRMNet; MPA-DNN [2510.09349]; FPL-OPF [2604.23548]; Powerformer [2401.02771]; Flow-Attentional GNN [2506.06127]; PowerGraph; CANOS [Piloto+2024].
- **Foundation:** UGCN [2509.08672]; PowerModelsGAT-AI [2603.16879]; PowerGraph-LLM [2501.07639].
- **Counter-examples:** Parker 2026 adversarial [2602.17975]; BOOST-RPF [2603.21977]; Scaling laws [2601.02706].
- **Sheaf / topology:** Bodnar 2022; HetSheaf [Barbero+2024]; Hernández-García 2025 Betti-resilience.
- **Conformal:** Mollaali 2024; Alcántara 2026 stratified CP; SPLICE 2026; Gibbs & Candès ACI.
- **DRO / Bayesian / GP:** Brock 2025 KL-DRJCC; Zhou 2024–25 WJCC; Pareek/Deka/Mitrovic 2023–25 GP-PF; Ding 2025 robust GNN-RSE.
- **Causal / IB:** Ghosh 2024 [2410.19179]; Speckhard 2025 [2505.11517]; Tang 2025 transfer-entropy.
- **RL OPF / restoration:** Wu 2024 PD-PPO; Sayed 2023 projection; Feng 2023/2026 Lyapunov; Jacquemart 2024 Nature Comms; Hosseini 2023 hierarchical.
- **Restoration MILP / robust:** Liang 2025 IET RPG; Zhu 2025 IJEPES DRO; Shi 2024 DRO; Zhang 2024 multi-stage gas-electric; Huang 2025 frequency-nadir IBR.
- **Game theory:** Bo 2025 MFG-EV; Jiang/Bolognani/Belgioioso 2025 Stackelberg; Bauer 2024 Shapley-OPF; SurroShap 2025; Chen/Monshizadeh 2024 aggregative games.

For full citation details and access, see the per-domain chunks. The chunks also contain knowledge-base source labels (e.g. `paper-pinn-gnn-edge-attention`, `gnn-pf-survey`, `arxiv-restoration-rl`) that can be queried via `ctx_search` if downstream agents need follow-up.

---

## Appendix: One-Line Summary

> **Six surveyed fields converge on a single architectural recommendation: a sheaf-Laplacian / Y-bus-equivariant Deep-Equilibrium GNN with KKT-aware complementarity heads and manifold-conformal calibration — pretrained as a foundation grid model on synthetic grid populations and fine-tuned via LoRA-style adaptation — solves the highest-leverage open problems in both AC power flow prediction and grid restoration path finding.**
