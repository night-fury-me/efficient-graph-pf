# D — Information Theory & Statistics for Power Flow Prediction and Grid Restoration

**Scope:** 2023–2026 work on information-theoretic and statistical foundations relevant to power flow (PF) prediction, OPF, state estimation, and grid restoration. Indexed sources (queryable via `ctx_search`):
`arxiv-conformal-power-system`, `arxiv-bayesian-pf-uq`, `arxiv-gp-pf-direct`, `arxiv-wasserstein-opf-direct`, `arxiv-dr-cc-power`, `arxiv-rse-power`, `arxiv-evt-power`, `arxiv-te-grid`, `arxiv-conformal-load`, `arxiv-ib-se`, `arxiv-laplace-nn-uq`, `arxiv-dro-opf-search`, `arxiv-ensemble-load`, `arxiv-causal-discovery-power`, `arxiv-mutual-info-power-grid`, `arxiv-causal-power-cascading`, `arxiv-info-bottleneck-power`, `arxiv-conformal-risk-control`, `arxiv-adaptive-conformal-online`, `arxiv-moment-based-dro`, `arxiv-deep-gp-time-series`.

---

## Landscape — Information Theory

### 1. Information bottleneck (IB) for grid state estimation
The IB principle compresses input X into a representation Z that retains minimal sufficient information about a target Y by minimising the Lagrangian `I(X;Z) − β·I(Z;Y)`. While the IB literature on graphs is large and growing (66+ arXiv hits for "information bottleneck" + "graph neural network"), explicit application to *power-grid* state estimation is essentially absent — the closest grid-targeted work is **Information-Theoretic Grid Topology Reconstruction using Low-Precision Smart Meter Data** [Speckhard, 2025; arXiv:2505.11517], which uses mutual-information estimation to determine minimal measurement precision/sampling for reliable topology recovery. General IB-GNN advances such as disentangled-IB explainers (ST-TGExplainer [arXiv:2605.19822]) and joint source–channel coding for semantic communication [arXiv:2511.07826] are transferable but untested on PMU/SCADA pipelines.

### 2. Mutual-information estimators (MINE, InfoNCE, CLUB) on grid data
MINE/InfoNCE/CLUB are off-the-shelf in self-supervised representation learning but the grid-specific application is sparse. **Information-Theoretic Grid Topology Reconstruction** [Speckhard, 2025] is the lone direct application uncovered for low-precision smart-meter pipelines. Contrastive learning has been used implicitly inside PINN/GNN-PF backbones (see survey D), but none of the surveyed PF papers report `I(Z;V)` curves or MINE-style bounds.

### 3. Rate-distortion analysis of PMU/SCADA compression
A single relevant 2024–25 paper: **Information-Theoretic Grid Topology Reconstruction** [Speckhard, 2025; arXiv:2505.11517] derives precision/sampling thresholds (an implicit rate-distortion analysis) for smart-meter measurements that still allow topology recovery. Otherwise, rate-distortion is essentially unused for compressing PMU streams under PF-residual fidelity constraints. The semantic-communication literature (e.g. variable-length JSCC, arXiv:2511.07826) provides reusable formal machinery.

### 4. Entropy-regularised OPF and maximum-entropy load forecasting
Although entropy-regularised optimisation thrives in RL and OT, OPF papers using *maximum-entropy* priors over uncertain renewables are rare; the most directly relevant 2025 work is **Distributionally Robust Joint Chance-Constrained Optimal Power Flow using Relative Entropy** [Brock, Zhang, Lavaei, Sojoudi, 2025; arXiv:2501.03543], which uses KL-ambiguity sets — a Bayesian/relative-entropy formulation of DR-CC-OPF. Maximum-entropy load forecasting tied to physical inertia priors remains unexplored.

### 5. Information-theoretic detection of false-data injection (FDIA)
FDIA detection is dominated by deep-learning classifiers (recursive variational AE, spectral GNN, random-forest variants, all 2024–25). Truly *information-theoretic* detectors (likelihood-ratio, KL-divergence between attack/no-attack distributions, MI-based detectors) are surprisingly thin: **Locational FDIA Detection Using Recursive Variational** [IEEE IoT 2025] and **Cubature Kalman Filter as a Robust State Estimator Against Model Uncertainty and Cyber Attacks** [Tasooji & Khodadadi, 2025; arXiv:2503.21070] sit closest, but treat the problem as nonlinear filtering rather than channel-coding.

### 6. Channel-coding / error-correcting perspective on robust SE
Untouched in the power-systems literature for the 2023–26 window. The conceptual link — measurements = a noisy channel; state = code-word — has been articulated in classical communications but no recent arXiv paper instantiates it for power grids. **Power System Robust State Estimation As a Layer** [Ding et al., 2025; arXiv:2511.22836] embeds RSE as a differentiable layer with learnable weights — adjacent in spirit but not coding-theoretic.

### 7. Causal information flow (transfer entropy, Granger causality) on cascading failures
This is one of the few information-theory subareas with a 2024–25 grid-specific paper: **Identification of Pressure Points in Modern Power Systems Using Transfer Entropy** [Tang, Liu, Anderson, Srikrishnan, 2025; arXiv:2508.08513; Cell Reports Sustainability] uses transfer entropy on simulated New-York-State grid trajectories to find components whose utilisation patterns are *early predictors* of downstream shortages. Complementary causal-graph efforts: **Cascading Failure Prediction via Causal Inference** [Ghosh, Dwivedi, Tajer, Yeo, Gifford, 2024; arXiv:2410.19179] learns a directed latent graph; **A Causal-Guided Multimodal LLM for Power-System Time Series** [Zhou et al., 2025; arXiv:2511.07777] couples a physics-statistics causal-discovery module with an LLM backbone; **Carbon-NeuGC** [CAC 2024, DOI 10.1109/CAC63892.2024.10864688] applies neural Granger causality to power-system carbon attribution. **Granger Causality for Prediction in Dynamic Mode Decomposition** [EPSR 2023, DOI 10.1016/j.epsr.2023.109865] uses Granger inside DMD for power-system mode prediction.

---

## Landscape — Statistics & UQ

### 8. Conformal prediction (CP) for PF uncertainty
CP for PF has emerged as a clear 2024–26 thread:
- **Conformalized Prediction of Post-Fault Voltage Trajectories Using Pre-trained and Finetuned Attention-Driven Neural Operators** [Mollaali, Zufferey, Constante-Flores, Moya, Li, Lin, 2024; arXiv:2410.24162] — QAF-DeepONet with split-CP on voltage trajectories.
- **Trustworthiness Layer for Foundation Models in Power Systems: Application to N-k Contingency Screening** [Alcántara & Chatzivasileiadis, 2026; arXiv:2602.07995] — stratified CP partitioned by contingency severity and grid sub-graph.
- **SPLICE: Latent Diffusion over JEPA Embeddings for Conformal Time-Series Inpainting** [Zinflou, 2026; arXiv:2605.00126] — adaptive CP (ACI) wrapping JEPA-encoded daily load segments; 93–95 % empirical coverage on 13 load datasets.
- **Dual-Splitting Conformal Prediction for Multi-Step Time Series Forecasting** [Yu et al., 2025; arXiv:2503.21251] — multi-step CP for time-series; tested on resource-scheduling tasks.
- **Conformalized Quantum DeepONet Ensembles** [Matlia, Moya, Lin, 2026; arXiv:2605.00330] — ensemble + adaptive CP on power-system dynamics.
General CP theory (50+ "conformal risk control" papers, 28+ "adaptive conformal inference" papers) provides ready-to-port machinery; ACI [Gibbs & Candès], conformal risk control [Angelopoulos et al.] and online localized CP [arXiv:2605.05497] are all unexplored on PF outputs.

### 9. Bayesian neural networks (BNN) for PF
- **Bayesian Quantum Neural Network for Renewable-Rich Power Flow** [Zhu, Zhu, Bu, 2024; arXiv:2410.22062] — BNN with quantum-circuit weights for PF on renewable-heavy networks; reports tighter uncertainty bands than deterministic baselines.
- **DAE-Aware Bayesian Inference for Joint Generator-Network Parameter Estimation** [Albustami, Taha, Mahadevan, 2026; arXiv:2604.15686] — joint Bayesian posterior over generator inertia/damping + branch impedances on IEEE 9- and 39-bus DAE models.
- **Bayesian Transformer for Probabilistic Load Forecasting in Smart Grids** [Debnath, Mia, 2026; arXiv:2603.07899] — MC-Dropout + variational FFN + stochastic Bayesian attention; PJM/ERCOT/ENTSO-E benchmarks; CRPS 0.0289 (7.4 % better than deep ensembles).
- **Bayesian Deep Neural Networks for Spatio-Temporal Probabilistic OPF** [2023; OpenAlex] — multi-task BNN for OPF outputs.
Laplace approximation literature (37 recent arXiv) — **Self-Supervised Laplace Approximation** [Rodemann et al., 2026; arXiv:2605.12208], **Sub-network Laplace Approximations** [Raha et al., 2026; arXiv:2605.09075], **Tubular Riemannian Laplace Approximations for BNNs** [arXiv:2512.24381], **Improving the Linearized Laplace via Quadratic Approximations** [arXiv:2602.03394] — yet no Laplace-PF paper exists.

### 10. Distributionally robust optimisation (DRO) for OPF
Wasserstein-DRO dominates 2023–25 OPF UQ:
- **Distributionally Robust Joint Chance-Constrained OPF using Relative Entropy** [Brock, Zhang, Lavaei, Sojoudi, 2025; arXiv:2501.03543] — KL-ambiguity DR-CC-OPF.
- **FICA: Faster Inner Convex Approximation of Chance Constrained Grid Dispatch with Decision-Coupled Uncertainty** [Zhou, Yang, Morstyn, 2025; arXiv:2506.18806] — Wasserstein-DRJCC with AGC factors.
- **Strengthened & Faster Linear Approximation to Joint Chance Constraints with Wasserstein Ambiguity** [Zhou, Xia, Yang, Morstyn, 2024; arXiv:2412.12992].
- **Distributionally Robust OPF with Uncertain Renewable Output** [Yang, Song, Zhao, 2023; arXiv:2306.14053].
- **Distributionally Robust Joint Planning of Coastal Distribution Network and PV-Storage-EV Stations** [Gao, Wang, Chen, Shen, 2025; arXiv:2511.09321] — tri-layer DRO with ambiguity sets capturing correlated uncertainties.
- **Distributionally Robust Frequency-Constrained Microgrid Scheduling** [Yang et al., 2024; arXiv:2401.03381].
- **Prescribing Decision Conservativeness in Two-Stage Power Markets** [Liang, Li, Liu, Dvorkin, 2024; arXiv:2412.10554] — end-to-end calibration of wind forecast under DR-OPF.
- **Multiple Joint Chance Constraints Approximation** [Wen et al., 2024; arXiv:2404.01167].
Moment-based DRO and kernel-DRO advances exist (37 + papers) but are largely outside the grid domain — porting kernel-DRO / Sinkhorn-DRO to OPF is a clear opportunity.

### 11. Gaussian-process (GP) surrogates for PF and contingency screening
A small but mature thread:
- **Stability-Constrained AC OPF — A Gaussian-Process-Based Approach** [Di Vito, Sundar, Fioretto, Deka, 2025; arXiv:2507.23094] — GP surrogate for the *generator-dynamics stability constraint* embedded in AC OPF.
- **Power Flow Approximations for Multiphase Distribution Networks using Gaussian Processes** [Glover, Pareek, Deka, Dubey, 2025; arXiv:2504.21260].
- **Data-Driven Stochastic AC-OPF using Gaussian Processes** [Mitrovic, 2024; arXiv:2402.11365].
- **Data-Efficient Strategies for Probabilistic Voltage Envelopes (PVE) under Network Contingencies** [Pareek, Deka, Misra, 2023; arXiv:2310.00763] — network-aware Vertex-Degree-Kernel GP.
- **Learning Power Flow with Confidence: A Probabilistic Guarantee Framework for Voltage Risk** [Pareek, Misra, Deka, 2023; arXiv:2308.07867].
- **GP CC-OPF** [Mitrovic, Kundacina, Lukashevich, Vorobev, Terzija, Maximov, 2023; arXiv:2302.08454].
- **Decoupled-Value Attention for PFNs: GP Inference for Physical Equations** [Sharma, Singh, Pareek, 2025; arXiv:2509.20950] — Prior-Data Fitted Networks (transformer-based GP surrogate) for physical systems including PF.
Sparse GPs and manifold GPs on grid graphs are absent from arXiv (`"sparse Gaussian process" + "power system"` → 0 hits).

### 12. Extreme-value statistics (EVT) for rare contingencies / N-k
- **Ensemble-Based Peak-Demand Probability-Density Forecasting with Application to Risk-Aware Power System Scheduling** [Yu, Tang, 2025; arXiv:2506.01358] — ensemble of tree learners on covariate space, fitting local GEV; demonstrates 38 % capacity-reduction on PJM.
- **Bidding in Ancillary Service Markets: An Analytical Approach Using Extreme Value Theory** [Herstad, Kazempour, Mitridati, Zwart, 2024; arXiv:2412.02308] — EVT for the 90 % reserve-bid reliability mandate in Denmark.
- **The Scaling Behaviours in Achieving High Reliability via Chance-Constrained Optimization** [Deo, Murthy, 2025; arXiv:2504.07728] — EVT-flavoured scaling laws for CC-OPF reliability targets.
N-k explicit EVT modelling for unseen contingencies remains under-explored.

### 13. Robust statistics for outlier-resistant state estimation
- **Power System Robust State Estimation As a Layer: A Novel End-to-end Learning Approach** [Ding, Shi, Duan, Zhao, Ruan, Zhao, Xu, 2025; arXiv:2511.22836] — RSE embedded as differentiable NN layer with *learnable* measurement weights; physically-consistent hybrid loss.
- **Cubature Kalman Filter as a Robust State Estimator Against Model Uncertainty and Cyber Attacks in Power Systems** [Tasooji, Khodadadi, 2025; arXiv:2503.21070].
M-estimator/LTS classical thread is older; explicit modern leverage-analysis with NN-based RSE is still embryonic.

### 14. Causal inference for what-if analyses of grid interventions
- **Cascading Failure Prediction via Causal Inference** [Ghosh, Dwivedi, Tajer, Yeo, Gifford, 2024; arXiv:2410.19179] — latent directed graph for transmission-line cascades.
- **A Causal-Guided Multimodal LLM** [Zhou et al., 2025; arXiv:2511.07777] — physics-statistics causal-discovery embedded inside an LLM for power-system time-series.
- **Identification of Pressure Points Using Transfer Entropy** [Tang et al., 2025; arXiv:2508.08513] — explicitly disclaims causal interpretation but identifies bottlenecks consistent with known causal pathways.
- **Carbon-NeuGC** [CAC 2024] — neural Granger causality on power-system carbon flows.
Do-calculus on grid causal graphs (counterfactual: "what if line L were upgraded?") is unrealised; existing work is associational/predictive.

---

## Research gaps (12)

1. **Conformal prediction for vector-valued nodal voltages treats components independently.** No method outputs a *joint* prediction region calibrated under the AC PF feasibility manifold. Manifold-conformal-prediction (`conformal prediction on a sub-manifold of R^{2n}`) for `{V_i, θ_i}` would be a first.

2. **Conformal risk control for PF constraint satisfaction is unstudied.** Conformal-risk-control machinery (50+ papers, none on PF) could turn "Pr[|V_i − V̂_i| ≤ ε] ≥ 1 − α" into "Pr[ max_i g_i(V̂) ≤ 0 ] ≥ 1 − α" by selecting a single per-instance scalar threshold for *all* AC-PF inequality constraints.

3. **No published Bayesian-NN PF surrogate uses Laplace approximation despite a mature 2024–26 Laplace literature** (37+ papers including sub-network Laplace, tubular Riemannian Laplace, self-supervised Laplace). Last-layer Laplace on a fast PF GNN would give principled epistemic uncertainty in O(1) extra cost.

4. **Information-bottleneck training of GNN-PF surrogates is absent.** No paper has reported the `I(X;Z) − β·I(Z;V)` Pareto frontier for SCADA→voltage encoders, despite the well-established IB-GNN catalogue.

5. **Rate-distortion analysis of PMU streams under PF-residual constraints does not exist.** Speckhard 2025 [2505.11517] starts the discussion for smart-meter topology, but no work poses the trilemma "compress measurements at rate R, accept distortion D, bound PF-residual ε" for the full PMU pipeline.

6. **Mutual-information estimators (MINE, InfoNCE, CLUB) are not used as auxiliary losses on grid data.** Self-supervised contrastive pre-training on grid topologies is unstudied, although successfully demonstrated on molecules and traffic graphs.

7. **No information-theoretic channel-coding view of state estimation.** Treating measurements as a noisy channel with structured codewords (PF-feasible states) — and importing low-density parity-check-like decoders — is entirely open.

8. **DRO + GP surrogate is missing.** Existing GP-CC-OPF assumes a parametric noise model; combining GP epistemic variance with a Wasserstein ambiguity set over forecasts (kernel-DRO over the GP posterior) is unexplored.

9. **N-k contingency screening lacks extreme-value modelling.** EVT work (Yu 2025, Herstad 2024) targets peak demand and reserve markets; no paper fits a GEV/GPD over post-N-k violation severity over the full contingency tree.

10. **Causal do-calculus has not been instantiated on AC PF DAGs.** Ghosh 2024 [2410.19179] learns a latent cascade graph but does *not* exploit Pearl's do-operator for counterfactual restoration ("what if line L_47 were re-energised first?"). Pearl-style intervention reasoning fused with restoration MDPs is missing.

11. **Adaptive conformal inference under topology drift is unstudied.** ACI [Gibbs & Candès] adapts to *temporal* covariate shift but not to *graph-structural* shift (line outage, switching). For grid restoration where topology changes mid-stream, structure-adaptive CP is needed.

12. **Robust statistics + deep RSE rarely uses M-estimator / LTS theory.** Ding 2025 [2511.22836] makes weights learnable but does not analyse breakdown-point or leverage. A breakdown-point-aware loss for GNN-RSE under bad-data leverage points is open.

---

## Breakthrough directions (8) — each tied to a specific mathematical object

1. **Rate-distortion-feasibility trilemma for SCADA compression.**
   Object: a single Lagrangian `L = I(X;Z) + λ_D · D(X, X̂(Z)) + λ_F · E[max_g g(PF(X̂))_+ ]` that compresses measurements `X → Z`, reconstructs `X̂`, and penalises *PF residual / inequality violation*. The achievable surface in `(R, D, ε)`-space generalises Shannon rate-distortion to physics-feasible reconstruction.

2. **Conformal prediction on the AC-PF feasibility manifold.**
   Object: a conformity score `s(V, V̂) = d_g(V, V̂)` measured along the *Riemannian distance on the PF manifold* `M = {(V, θ) : P − Re(V ⊙ (Y V)*) = 0, Q − Im(...) = 0}`. Calibration gives joint prediction sets that respect Kirchhoff/Ohm.

3. **Information-bottleneck PF surrogate.**
   Object: `min_q I_q(SCADA; Z) − β · I_q(Z; V_true)` trained with InfoNCE/CLUB lower/upper bounds. Predicts not just `V̂` but the *minimum sufficient representation* of measurements for PF prediction — yields physical insight into which buses carry information.

4. **Wasserstein-DRO over GP posteriors for OPF.**
   Object: ambiguity set `B_ε^W(μ̂_GP)` = Wasserstein-ball around the GP-posterior predictive of renewable forecasts; chance-constrained OPF inside this ball. Combines GP epistemic quantification with kernel-DRO worst-case guarantees.

5. **Do-calculus on PF causal graphs for restoration scheduling.**
   Object: a structural causal model `M = (V, U, F)` on switching actions `do(s_t = 1)` with potential outcomes `Y_{do(s)}(t)`. Counterfactual restoration policy `π*(s_1, …, s_T) = argmax E[ load_served | do(s_1) … do(s_T) ]` evaluated under topological identifiability conditions.

6. **Last-layer Laplace approximation for GNN-PF.**
   Object: posterior `p(W_last | D) ≈ N(W̃_MAP, H^{-1})` over last linear layer of a GNN-PF backbone, propagated through PF residual to give calibrated voltage variances `Σ_V`. Almost-free epistemic UQ on top of any pretrained GNN-PF.

7. **Conformal risk control for joint PF constraint satisfaction.**
   Object: a single calibrated scalar `λ̂ = inf{λ : Pr[L(V̂ + λ·s) ≤ α] ≥ 1 − δ}` where `L` is the worst-case AC-PF inequality residual. Yields *one* control knob giving a finite-sample, distribution-free guarantee that the PF surrogate's outputs feasibility-bound the true PF solution.

8. **Extreme-value model for N-k contingency cascade severity.**
   Object: a generalised Pareto distribution fit to peaks-over-threshold of the loss `L_k = unserved-energy(N − k)` over Monte-Carlo contingency samples; tail-index `ξ` directly tied to *system fragility*. Coupled with PF surrogate, gives `O(MC)`-cheap N-k risk envelopes.

---

## Bibliography (appendix)

### Conformal prediction and adaptive CP (PF / load / power)
- Mollaali A., Zufferey G., Constante-Flores G., Moya C., Li C., Lin G. (2024). *Conformalized Prediction of Post-Fault Voltage Trajectories Using Pre-trained and Finetuned Attention-Driven Neural Operators.* arXiv:2410.24162.
- Alcántara A., Chatzivasileiadis S. (2026). *Trustworthiness Layer for Foundation Models in Power Systems: Application to N-k Contingency Screening.* arXiv:2602.07995.
- Zinflou A. (2026). *SPLICE: Latent Diffusion over JEPA Embeddings for Conformal Time-Series Inpainting.* arXiv:2605.00126.
- Yu Q., Cao Z., Wang R., Yang Z., Deng L., Hu M., Luo Y., Zhou X. (2025). *Dual-Splitting Conformal Prediction for Multi-Step Time Series Forecasting.* arXiv:2503.21251; Applied Soft Computing 2025.
- Matlia P., Moya C., Lin G. (2026). *Conformalized Quantum DeepONet Ensembles for Scalable Operator Learning with Distribution-Free Uncertainty.* arXiv:2605.00330.

### Bayesian deep learning for PF / load
- Zhu Z., Zhu S., Bu S. (2024). *Bayesian Quantum Neural Network for Renewable-Rich Power Flow with Training Efficiency and Generalization Capability Improvements.* arXiv:2410.22062.
- Albustami A. A., Taha A. F., Mahadevan S. (2026). *DAE-Aware Bayesian Inference for Joint Generator-Network Parameter Estimation.* arXiv:2604.15686.
- Debnath S., Mia M. U. (2026). *Bayesian Transformer for Probabilistic Load Forecasting in Smart Grids.* arXiv:2603.07899.
- Rodemann J., Marquard A., Augustin T., Caprio M. (2026). *Self-Supervised Laplace Approximation for Bayesian Uncertainty Quantification.* arXiv:2605.12208.
- Raha S., Khare K., Patra R. K. (2026). *Optimality of Sub-network Laplace Approximations.* arXiv:2605.09075.

### Gaussian processes for PF / OPF
- Di Vito V., Sundar K., Fioretto F., Deka D. (2025). *Stability-Constrained AC Optimal Power Flow — A Gaussian Process-Based Approach.* arXiv:2507.23094.
- Glover D., Pareek P., Deka D., Dubey A. (2025). *Power Flow Approximations for Multiphase Distribution Networks using Gaussian Processes.* arXiv:2504.21260.
- Mitrovic M. (2024). *Data-Driven Stochastic AC-OPF using Gaussian Processes.* arXiv:2402.11365.
- Pareek P., Deka D., Misra S. (2023). *Data-Efficient Strategies for Probabilistic Voltage Envelopes under Network Contingencies.* arXiv:2310.00763.
- Pareek P., Misra S., Deka D. (2023). *Learning Power Flow with Confidence: A Probabilistic Guarantee Framework for Voltage Risk.* arXiv:2308.07867.
- Mitrovic M., Kundacina O., Lukashevich A., Vorobev P., Terzija V., Maximov Y. (2023). *GP CC-OPF: Gaussian Process based optimization tool for Chance-Constrained Optimal Power Flow.* arXiv:2302.08454.
- Sharma K., Singh S., Pareek P. (2025). *Decoupled-Value Attention for Prior-Data Fitted Networks: GP Inference for Physical Equations.* arXiv:2509.20950.

### Distributionally robust OPF / chance-constrained OPF
- Brock E., Zhang H., Lavaei J., Sojoudi S. (2025). *Distributionally Robust Joint Chance-Constrained Optimal Power Flow using Relative Entropy.* arXiv:2501.03543.
- Zhou Y., Yang H., Morstyn T. (2025). *FICA: Faster Inner Convex Approximation of Chance Constrained Grid Dispatch with Decision-Coupled Uncertainty.* arXiv:2506.18806.
- Zhou Y., Xia Y., Yang H., Morstyn T. (2024). *Strengthened and Faster Linear Approximation to Joint Chance Constraints with Wasserstein Ambiguity.* arXiv:2412.12992.
- Yang J., Song J., Zhao C. (2023). *Distributionally Robust Optimal Power Flow with Uncertain Renewable Energy Output.* arXiv:2306.14053.
- Gao W., Wang Y., Chen W., Shen X. (2025). *Distributionally Robust Joint Planning of Coastal Distribution Network and PV-Storage-EV Stations.* arXiv:2511.09321.
- Yang L., Yang H., Cao X., Guan X. (2024). *Distributionally Robust Frequency-Constrained Microgrid Scheduling Towards Seamless Islanding.* arXiv:2401.03381.
- Liang Z., Li Q., Liu A., Dvorkin Y. (2024). *Prescribing Decision Conservativeness in Two-Stage Power Markets: A Distributionally Robust End-to-End Approach.* arXiv:2412.10554.
- Wen Y., Guo Y., Hu Z., Hug G. (2024). *Multiple Joint Chance Constraints Approximation for Uncertainty Modeling in Dispatch Problems.* arXiv:2404.01167.
- Liu S., Yang B., Li X., Yang X., Wang Z., Zhu D., Guan X. (2024). *Optimal Hardening Strategy for Electricity-Hydrogen Networks with Hydrogen Leakage Risk Control.* arXiv:2410.20475.

### Robust state estimation
- Ding Y., Shi W., Duan M., Zhao Y., Ruan J., Zhao J., Xu Z. (2025). *Power System Robust State Estimation As a Layer: A Novel End-to-end Learning Approach.* arXiv:2511.22836.
- Tasooji T. K., Khodadadi S. (2025). *Cubature Kalman Filter as a Robust State Estimator Against Model Uncertainty and Cyber Attacks in Power Systems.* arXiv:2503.21070.

### Extreme value statistics for power systems
- Yu B., Tang W. (2025). *Ensemble-Based Peak Demand Probability Density Forecasting with Application to Risk-Aware Power System Scheduling.* arXiv:2506.01358.
- Herstad T. R., Kazempour J., Mitridati L., Zwart B. (2024). *Bidding in Ancillary Service Markets: An Analytical Approach Using Extreme Value Theory.* arXiv:2412.02308.
- Deo A., Murthy K. (2025). *The Scaling Behaviours in Achieving High Reliability via Chance-Constrained Optimization.* arXiv:2504.07728.

### Information theory on grids (IB / MI / transfer entropy / Granger)
- Speckhard D. T. (2025). *Information-Theoretic Grid Topology Reconstruction using Low-Precision Smart Meter Data.* arXiv:2505.11517.
- Tang K., Liu M. V., Anderson C. L., Srikrishnan V. (2025). *Identification of Pressure Points in Modern Power Systems using Transfer Entropy.* arXiv:2508.08513; Cell Reports Sustainability, DOI 10.1016/j.crsus.2026.100660.
- Ghosh S. S., Dwivedi A., Tajer A., Yeo K., Gifford W. M. (2024). *Cascading Failure Prediction via Causal Inference.* arXiv:2410.19179.
- Zhou Z., Li Y., Yu X., Liu R., Guo Z., Yan Z., Chow M.-Y., Yang Y., Xu Y. (2025). *A Causal-Guided Multimodal Large Language Model for Generalized Power System Time-Series Data Analytics.* arXiv:2511.07777.
- *Carbon-NeuGC: Neural Granger Causality Based Attribution Analysis of Power System Carbon* (2024). CAC 2024, DOI 10.1109/CAC63892.2024.10864688.
- *Granger Causality for Prediction in Dynamic Mode Decomposition: Application to Power Systems* (2023). EPSR, DOI 10.1016/j.epsr.2023.109865.

### FDIA detection (information-adjacent)
- *Locational False Data Injection Attack Detection in Smart Grid Using Recursive Variational AE* (2025). IEEE IoT J., DOI 10.1109/JIOT.2025.3526672.
- *Enhancing Detection of False Data Injection Attacks in Smart Grid Using Spectral Graph Neural Networks* (2025). IEEE TII, DOI 10.1109/TII.2025.3545044.
- *Detection and Defense Against Multi-Point FDIA of Load Frequency Control* (2025). IEEE TSG, DOI 10.1109/TSG.2025.3578985.

### Generic methodological references (porting candidates)
- Gibbs I., Candès E. (2021/2023). *Adaptive Conformal Inference under Distribution Shift.* (Foundational ACI work referenced throughout SPLICE and 28+ recent papers.)
- Angelopoulos A. N., Bates S., Fisch A., Lei L., Schuster T. *Conformal Risk Control.* (Foundational; 50+ follow-up papers in 2024–26.)
- ST-TGExplainer (2026), arXiv:2605.19822 — disentangled IB GNN explainer (transferable to grid topologies).
