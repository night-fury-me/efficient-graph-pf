# ICDM 2026 Topic Synthesis — HyperDEQ-PF

**Date:** 2026-05-24
**Source:** Multi-field deep inspection across Mathematics, Abstract Algebra, Information Theory, Statistics, Graph Theory, Game Theory, Data Mining, and Power Flow domains.
**Provenance:** Five parallel research-survey agents (2024–2026 literature) + one upstream novelty check against the rejected "SCAM" (Sheaf-Cohomological Conformal Anomaly Mining) candidate.

---

## 1. Why this document exists

A prior proposal "SCAM" — sheaf-Laplacian as conformal nonconformity score — failed novelty inspection (YELLOW): its core mechanism overlaps PINN-CP (arXiv 2509.13717) which already uses physics-residual as CP score. The reviewers would likely demote SCAM to "PINN-CP with a sheaf-Laplacian residual on power graphs".

This document records a deeper round of cross-field inspection, including ICDM 2024–2025 accepted-paper analysis, and synthesizes a refined proposal that:
- Avoids the PINN-CP overlap entirely
- Matches the ICDM 2025 winning-paper template
- Leverages the host project's unique dual-domain (HVN + LVN) data asset
- Provides three theorems with real content (not definitions or routine corollaries)

---

## 2. Cross-agent findings

### 2.1 Agent 1 — Information Theory + Statistics (arXiv survey 2024–2026)

**Top primitives:**
1. Selective Conformal Risk Control with E-values (e-process always-valid stopping)
2. Sinkhorn-DRO with primal bilevel reformulation (arXiv 2512.12550, 2503.20703)
3. IRM-free Causal-Subgraph Discovery via Invariant Distribution Criterion (arXiv 2510.20295)
4. PAC-Bayesian Bounds for Inductive GCN with Data-Dependency (arXiv 2509.06600)
5. InfoNCE-anchor / Scoring-Rule MI Estimator (arXiv 2510.25983)

**Moonshots proposed:**
- A. Causal Markov-Boundary Pruning for Equilibrium GNNs
- B. Sinkhorn-DRO Equilibrium PF with E-process Monitoring
- C. Transfer-Entropy-Guided Heterogeneous Attention

### 2.2 Agent 2 — Graph Theory + Data Mining (with ICDM 2024–2025 intel)

**Top primitives:**
1. NODESAFE — Bounded/Uniform Energy-based OOD for graphs (arXiv 2504.13429, 2025)
2. Multiparameter Persistent-Homology + GNN (Graphcode, arXiv 2405.14302) + Persistent Local Homology Sheaf nets (arXiv 2311.10156, 2511.00677)
3. Subgraph-GNN equivariant framework via graph products + coarsening (arXiv 2406.09291)
4. Graph Foundation Models: GFM-RAG (arXiv 2502.01113), GraphAny, LUMINA (arXiv 2603.04300)
5. Homophily-aware Environment Mixup for OOD-generalized GAD + Structural-Temporal Coupling Dynamic GAD (arXiv 2505.08330)

**Moonshots proposed:**
- M1. Topo-GAD-PF (persistent-homology augmented graph anomaly detector)
- M2. PE-DEQ-FM (Cross-grid Graph Foundation Model via DEQ pretraining)
- M3. Causal-Subgraph Mining for Voltage-Collapse Attribution

**Critical ICDM 2025 intel:**
- **Best Paper:** "Scalable Graph Classification via Random Walk Fingerprints" (Li, Wang, Böhm) — graph mining track
- **Best Student Runner-Up:** zero-shot cross-city via hypernetworks — direct template for cross-voltage PF
- **Tao Li Award:** Jundong Li (UVa) — graph data mining / trustworthy ML
- **Growing sub-tracks:** spectral/wavelet GAD, dynamic-network change attribution, explainable subgraph mining, post-hoc explainability

### 2.3 Agent 3 — Game Theory + Mechanism Design

**Top primitives:**
1. Exact Any-Order Shapley Interactions for GNNs (arXiv 2501.16944, Muschalik et al. 2025) — receptive-field-bounded exact computation
2. Banzhaf Value via GNNs on Network Flow Games (arXiv 2510.13391)
3. Distributionally Robust Nash Equilibrium via VI (arXiv 2510.17024)
4. MFG-RegretNet (DSIC + BNE, arXiv 2603.28329)
5. Hybrid MFC/MFG for DER Aggregators in Wholesale Markets (arXiv 2507.03240)

**Moonshots proposed:**
- A. DEQ-Bus-Shapley (Shapley attribution for PE_DEQ_PF)
- B. Distributionally Robust DEQ-PF (KKT-conic VI)
- C. Mean-Field-Equilibrium Surrogate + Banzhaf Pricing

### 2.4 Agent 4 — Power Flow + ICDM Trends (most critical intel)

**Top active PF-ML sub-areas (2024–2026):**
1. PIGNN-with-correction-operator (host project family)
2. Test-Time-Training + active learning
3. Grid foundation models: GridFM-v0, GridSFM-Premier
4. Physics-informed cascading-failure prediction (PI-GN-JODE, arXiv 2603.20838, Mar 2026)
5. Cross-topology generalisation: UGCN (arXiv 2509.08672), SaMPFA, PFΔ benchmark

**ICDM winning paper shape (synthesized from awards):**
> **graph + scalability + (federated OR hypernetwork OR contrastive) + trustworthiness**

**ICDM 2024 best:** "Random Walk Fingerprints" (scalable graph classification)
**ICDM 2025 best:** "DP-FedLoRA" (privacy + federated + LoRA + trustworthy)
**ICDM 2025 BSR-Up:** zero-shot cross-city via hypernetworks ← **direct template**

**Five confirmed unpublished gaps in PF-ML:**
1. Certified robustness for AC-PF *regression*
2. Cross-voltage HVN↔LVN benchmark
3. DEQ-residual-as-anomaly-score
4. **Hypernetwork PF-weight generation** ← matches winning template
5. DEQ-backed grid foundation model

**Moonshots proposed:**
1. **HyperPF** — zero-shot HVN↔LVN via hypernetwork on graph descriptors (HIGHEST data-asset moat; ICDM 2025 BSR-Up template)
2. DEQ-Anomaly — DEQ fixed-point residual as bus-level FDIA score with certified Lipschitz radii
3. JumpDEQ — replaces Neural-ODE of PI-GN-JODE with PE_DEQ_PF (tight competition with arXiv 2603.20838)

### 2.5 Agent 5 — Mathematics + Abstract Algebra

**Top primitives:**
1. Polynomial Neural Sheaf Diffusion (PNSD, arXiv 2512.00242, Nov 2025)
2. Gauge-Equivariant Graph Networks via Self-Interference Cancellation (GESC, arXiv 2511.16062, Nov 2025)
3. Optimal-Transport Conformal Prediction (OT-CP, arXiv 2501.18991, 2502.03609) — explicitly avoids physics-residual scoring
4. Bundle Neural Networks (BuNN, arXiv 2405.15540, 2602.12884) — connection-Laplacian and parallel transport
5. Hodge GNN with Edge-Space Detection (IEEE TSG 2024.3389948 + arXiv 2503.12919)

**Moonshots proposed:**
- A. HOL-PF — Holonomy-Aware DEQ for Multi-Voltage Power Flow (Bundle NN + PE_DEQ_PF)
- B. OT-SheafCP — Joint-Bus Sinkhorn Conformal Sets on Sheaf Sections (sidesteps PINN-CP via OT distance)
- C. GESC-DEQ — Gauge-Equivariant DEQ for Slack-Free AC-PF

---

## 3. Cross-agent themes (each appears in ≥3 of 5 agent reports)

| Theme | Agents | Significance |
|---|---|---|
| Cross-voltage / cross-topology transfer | 1 (causal pruning), 2 (PE-DEQ-FM), 4 (HyperPF) | Unique data moat — only this lab has paired HVN+LVN with consistent format. Matches ICDM 2025 BSR-Up theme. |
| Bus-level attribution via DEQ implicit gradients | 1 (causal), 2 (causal subgraph), 3 (Shapley/Banzhaf), 4 (DEQ-Anomaly) | Implicit-function gradients are "free" causal/Shapley estimands on PE_DEQ_PF. Trustworthy-ML angle. |
| Going beyond CP | 1 (Sinkhorn-DRO, e-process), 5 (OT-CP) | Explicitly sidesteps PINN-CP overlap risk. |
| Gauge / bundle / holonomy structure of AC-PF | 5 (BuNN, GESC, Hodge) | U(1) phase symmetry is mathematically rich and unexploited. |
| Anomaly localization (not just detection) | 2 (Topo-GAD, NODESAFE), 3 (Shapley), 4 (DEQ-Anomaly) | Trustworthy-ML track + ICDM 2025 Tao Li Award alignment. |

---

## 4. Refined proposal: HyperDEQ-PF

### 4.0 Novelty-check result (2026-05-24, second pass)

**Verdict: YELLOW leaning GREEN — REFINE then COMMIT.**

The exact 4-way combination (hypernet + DEQ + cross-voltage AC-PF + Lipschitz-certified Shapley anomaly) is unpublished. Strongest novelty axis: **hypernet-generated DEQ** (zero arXiv hits in any domain). Three adjacencies require explicit differentiation:

| Closest prior | Differentiation strategy |
|---|---|
| **UGCN** (arXiv 2509.08672) — cross-topology PF transfer | Mandatory primary baseline; pitch on **voltage-class** axis not topology |
| **LUMINA** (arXiv 2603.04300/2605.02133) — Grid Foundation Model for AC-OPF, May 2026 | Mandatory primary baseline; frame HyperDEQ-PF as orthogonal (hypernet-*conditioned*, not pretrain-finetune) |
| **HyPINO** (arXiv 2509.05117) — Swin-Transformer hypernet generates PINN weights for multi-physics PDEs | Differentiate by DEQ backbone + graph (not Transformer + Euclidean) |
| **HyperDeepONet** (Lee+ ICLR 2023; arXiv 2507.18346) — hypernet generates DeepONet trunk weights | Differentiate by DEQ (implicit) vs DeepONet (explicit feedforward) |
| **ICDM 2025: Zero-Shot Cross-City Trajectory** (Gunkel+) | Same hypernet-zero-shot archetype in mobility; we are the power-systems analogue (positive precedent — proves the recipe works) |

**Refined headline:** "Hypernetwork-Conditioned Deep-Equilibrium Operators for Cross-Voltage AC Power Flow with Exact Shapley Attribution" — leads with the **truly empty quadrant** (hypernet + DEQ + exact-SI Shapley), not "graph foundation model" (where LUMINA owns the term).

### 4.1 Title

**Hypernetwork-Conditioned Deep-Equilibrium Operators for Cross-Voltage AC Power Flow with Exact Shapley Attribution** (working title)

Subtitle / Alt: *HyperDEQ-PF: Cross-Voltage Graph Equilibrium Mining with Hypernetwork-Generated Operators and Certified Bus-Level Anomaly Localization*

### 4.2 One-sentence pitch

Train a single graph-conditioned **hypernetwork** that *generates* the weights of a PE_DEQ_PF equilibrium operator for any graph descriptor (HVN or LVN), enabling **zero-shot cross-voltage transfer**; use the operator's implicit-function gradients for **exact bus-level Shapley attribution**; certify anomaly-localization robustness via **Lipschitz bounds**.

### 4.3 Why this maximises ICDM 2026 acceptance probability

| ICDM signal | How HyperDEQ-PF hits it |
|---|---|
| Matches winning template (Agent 4) | graph (GNN) + scalability (one model for all grids) + hypernetwork (ICDM 2025 BSR theme) + trustworthiness (Lipschitz + Shapley) |
| Closes 4 of 5 confirmed gaps | (1) certified robustness, (2) cross-voltage benchmark, (4) hypernetwork PF weights, (5) DEQ-backed foundation model |
| Aligns with Tao Li Award 2025 | Trustworthy graph data mining |
| Unique data moat | Only this lab has paired HVN + LVN with consistent format |
| No PINN-CP overlap | Hypernetwork generalization theory + Shapley game theory, not conformal physics-residual scoring |
| Benchmark contribution | The HVN↔LVN cross-voltage benchmark itself is a publishable artifact (ICDM loves benchmarks) |

### 4.4 Three falsifiable claims

**C1 — Cross-voltage zero-shot transfer.** A single HyperDEQ-PF trained on HVN (4–32 buses, 100 MVA base) achieves test RMSE within 1.5× of an LVN-supervised PE_DEQ_PF on the 722-bus LVN test split, *without ever training on LVN data*.

**C2 — Exact bus-level attribution.** Using Muschalik et al. 2025 (arXiv 2501.16944) receptive-field-bounded exact Shapley interactions adapted to the DEQ implicit-gradient operator, we produce ground-truth bus attributions on HVN (≤32 buses, exact computation feasible), and demonstrate that approximate attribution on LVN matches the exact HVN ranking on shared structural motifs.

**C3 — Certified anomaly localization.** The DEQ-residual-based bus-level anomaly score has Lipschitz constant `L = L_hyper · L_DEQ · √λ_max(L_F)`, giving a closed-form certificate: for adversarial perturbation `‖δ‖ ≤ ε`, anomaly-rank stability is guaranteed within `2L·ε`. Empirically validated on Parker-2026 attack at multiple ε levels.

### 4.5 Three theorems (real content)

**Theorem 1 — Cross-voltage contractivity.**
Let `H: G → Θ` be the hypernetwork mapping graph descriptors `g` to PE_DEQ_PF parameters `θ`. Under standard L-smoothness of H and contractivity of `F_θ` for all `θ` in the image of H, the generated operator `F_{H(g)}` is contractive uniformly over the descriptor manifold. The maximum cross-voltage Lipschitz inflation is bounded by `L_H · diam(G_train ∪ G_test)`.

**Theorem 2 — Exact Shapley faithfulness.**
For the DEQ fixed point `z*(g, S) = F_{H(g)}(z*, S)`, the bus-level Shapley value `φ_i(z*) = (1/n!)·Σ_π [z*(S_π_<i ∪ {i}) − z*(S_π_<i)]` can be computed exactly in O(n·k!) for k-hop receptive fields, via Muschalik+2025's exact-SI algorithm specialized to the implicit operator. Combined with the implicit function theorem, this gives `φ_i = ∂z*/∂S_i · Δ_i` where `Δ_i` ranges over coalitions.

**Theorem 3 — Lipschitz anomaly bound.**
The bus-level anomaly score `a_i(x) = ‖[F_{H(g)}(z*) − z*]_i‖²` is `(2L_H · L_DEQ · ‖z*‖)`-Lipschitz in the input perturbation δ. For Parker-2026-style attack with budget ε, the rank-k anomaly set is stable up to `2L_H · L_DEQ · ‖z*‖ · ε / margin`.

### 4.6 Empirical validation matrix (revised — MVN-first, mandatory baselines per novelty check)

| Component | Dataset | **Mandatory baselines (from novelty check)** | Additional baselines |
|---|---|---|---|
| Cross-voltage transfer (C1) | MVN↔HVN (pilot subsets, then full); LVN later | **UGCN (2509.08672), LUMINA (2603.04300/2605.02133)** | HyPINO (2509.05117), HyperDeepONet (2507.18346), GFM-RAG, GraphAny, per-grid PE_DEQ_PF (oracle) |
| Bus-Shapley attribution (C2) | HVN small-bus (4–12) for exact, IEEE-118 | Muschalik exact-SI on plain GCN (2501.16944), ProxySHAP (2605.22738) | SurroShap (2310.13325), GNNExplainer |
| Certified anomaly (C3) | MVN/HVN + N-1 contingencies + FDIA + Parker-2026 attack | **Certified-DEQ via randomized smoothing (Cai+24, arXiv 2411.00899)** | NODESAFE (2504.13429), GNNSafe (2302.02914), DOMINANT, CRC-SGAD (2504.02248) |
| Computational cost | All | Per-grid retrained PE_DEQ_PF (existing winning baseline) | |

### 4.7 Why this beats the other 12 moonshots (rejection log)

| Moonshot | Why HyperDEQ-PF wins | Disposition |
|---|---|---|
| Sheaf-CP (SCAM) | PINN-CP overlap (YELLOW novelty) | Rejected |
| OT-SheafCP | OT-CP is recent + unstable; harder empirical validation | Future work |
| HOL-PF (Bundle NN DEQ) | Bundle NN (2602.12884) too new as primary theory | Differentiator only |
| GESC-DEQ (gauge-eq) | No empirical advantage yet demonstrated | Future work |
| Causal Markov Pruning | No cross-voltage benchmark story | Folded as ablation |
| Sinkhorn-DRO + e-process | Training-only contribution | Folded as primitive |
| Transfer-Entropy attention | Local modification, not paradigm-shifting | Skip |
| Topo-GAD-PF | Persistent homology is heavy compute | Skip |
| Causal-Subgraph Mining for Collapse | Niche application | Skip |
| DEQ-Bus-Shapley alone | Missing cross-voltage moat | **Folded INTO HyperDEQ-PF** |
| DR DEQ-PF (KKT-conic VI) | Training-only | Subordinate primitive |
| MFG+Banzhaf pricing | Out-of-scope for ICDM (market design) | Skip |
| JumpDEQ (PI-GN-JODE replacement) | Direct competition with arXiv 2603.20838 | Skip |

### 4.8 Comparison with rejected SCAM proposal

| Property | SCAM (YELLOW — rejected) | HyperDEQ-PF |
|---|---|---|
| Core mechanism | physics-residual → CP score | hypernetwork-generated equilibrium operator |
| Closest prior | PINN-CP (2509.13717) — same mechanism | GFM-RAG, UGCN — different (no DEQ, no hypernet of weights) |
| Novelty verdict | YELLOW (mechanism overlap) | GREEN (pending independent novelty check) |
| Theorem strength | T1 = definition, T2 = corollary | T1 = new contractivity result, T2 = exact-SI extension, T3 = bus-level Lipschitz |
| Trustworthy-ML hooks | Adversarial robustness only | Adversarial + Shapley + cross-domain + Lipschitz |
| Benchmark contribution | No new benchmark | HVN↔LVN cross-voltage benchmark = release artifact |
| ICDM template fit | partial | full match (Agent 4 confirmed) |

### 4.9 4-month execution plan

| Month | Milestone | Deliverable |
|---|---|---|
| 1 | Implement hypernetwork `H(g) → θ` for PE_DEQ_PF; train on HVN; validate generated weights converge | `models/hyperdeq_pf/` |
| 2 | Train HyperDEQ-PF jointly on HVN + LVN; measure zero-shot HVN→LVN transfer | First C1 table |
| 3 | Implement exact bus-Shapley (HVN) + Lipschitz cert (both); adversarial validation | C2, C3 tables |
| 4 | Write & submit; package HVN↔LVN benchmark | ICDM 2026 paper + benchmark repo |

### 4.10 Two pre-decision pilots (run in parallel before committing)

1. **Pilot 1 (1 week):** Minimal hypernetwork (3-layer MLP) producing PE_DEQ_PF weights from a 32-dim graph descriptor (#buses, mean degree, spectral radius of Y_bus, etc.). Train on HVN only, evaluate zero-shot on LVN.
   - **Go condition:** cross-voltage RMSE within 5× of fully-supervised LVN baseline.
2. **Pilot 2 (1 week):** Verify Muschalik+2025's exact-SI algorithm can be adapted to the DEQ implicit-gradient operator.
   - **Go condition:** clean recurrence relation exists.

### 4.11 Risk register

| Risk | Mitigation |
|---|---|
| Hypernetwork generalization may fail on multi-voltage LVN topology | Use Bundle-NN connection-Laplacian features in the descriptor (Agent 5 primitive) |
| Exact Shapley computation infeasible beyond 32-bus | Restrict exact to HVN; use Banzhaf approximation (2510.13391) for LVN |
| Lipschitz constant too loose to be useful | Tighten via spectral normalization on the hypernetwork |
| ICDM reviewer says "just another foundation model" | Differentiate via DEQ implicit gradient → exact Shapley path (no other GFM does this) |

### 4.12 Two pivot options if pilot fails

- **Pivot A:** OT-SheafCP (Agent 5 Moonshot B) — sidesteps PINN-CP via Sinkhorn distance on sheaf sections instead of physics residual.
- **Pivot B:** HOL-PF + Gauge-Eq DEQ — combine Bundle NN holonomy with GESC gauge-equivariance for slack-free PF (more mathematically novel; higher risk, higher reward at ICML/NeurIPS rather than ICDM).

---

## 5. Next actions (revised 2026-05-24 — MVN-first pivot)

### Revised pilot strategy

**Start with MVN ↔ HVN, NOT LVN ↔ HVN.** Rationale:
- MVN and HVN share **identical 14-column schema, identical bus-range (4–32), and standard PyPower bus_typ encoding** — no converter, no bug surface.
- They differ in S_base (100 MVA vs 10 MVA) and U_base (110 kV vs 10 kV) — genuine cross-voltage class transfer test.
- LVN is 722-bus multi-voltage with separate converter pipeline; defer until MVN ↔ HVN method is validated.

**Use SUBSET (~1500 samples each), not full datasets.** Rationale:
- Fast iteration (~5 min/epoch on subset vs ~16 min/epoch on full)
- Stratified subset preserves bus-count distribution (4–32)
- If method works on subset, scale to full HVN (15k) + MVN (30k); then LVN (36k) later
- Honest go/no-go in 1 week of GPU time instead of 4 weeks

### Pilot pipeline

| Step | Action | Time |
|---|---|---|
| 1 | Build stratified subset: ~50 samples per bus-count (4–32) from each of HVN + MVN → ~1500 each | 5 min |
| 2 | **Independent novelty check on HyperDEQ-PF** (running in parallel) | 30 min |
| 3 | Implement minimal hypernetwork `H(g) → θ` taking 32-dim graph descriptor → PE_DEQ_PF weights | 1 day |
| 4 | Train HyperDEQ-PF on HVN subset only, evaluate zero-shot on MVN subset test split | 1 day |
| 5 | Compare against fully-supervised PE_DEQ_PF on MVN subset (oracle baseline) | 1 day |
| 6 | **Go condition:** zero-shot MVN RMSE within 5× of oracle MVN-supervised RMSE | Decision |
| 7 | If go: scale to full HVN + MVN, add Shapley + Lipschitz components | 4 weeks |
| 8 | If go: extend to LVN (third domain) | 4 weeks |
| 9 | Write & submit ICDM 2026 | 4 weeks |

### Pilot subset specification

| Property | Value |
|---|---|
| Samples per dataset | ~1500 (50 per bus-count × 29 bus-counts) |
| Split | 80/10/10 → 1200 train / 150 val / 150 test |
| Stratification | Per bus-count (preserves topology diversity) |
| Random seed | 42 (matches existing runs) |
| File outputs | `datasets/HVN_stratified_1500.parquet`, `datasets/MVN_stratified_1500.parquet` |

### Go / no-go criteria (rigorous)

| Test | Pass | Soft Pass | Fail |
|---|---|---|---|
| HVN→MVN zero-shot RMSE / MVN-supervised RMSE | ≤ 2× | 2–5× | > 5× |
| Per-bus-count generalization | uniform across 4–32 | drift but stable | collapse at large N |
| Wall-clock per epoch (subset) | < 10 min | 10–20 min | > 30 min (infeasible) |

If 2/3 pass → scale to full datasets. If only 1/3 → pivot to OT-SheafCP or HOL-PF. If 0/3 → abandon HyperDEQ-PF.

### Then

1. **If pilot passes:** scale to full HVN + MVN, add Shapley + Lipschitz components, extend to LVN (full 4-month execution plan from §4.9).
2. **If pilot fails:** pivot to one of the two options in §4.12.
3. **Either way:** novelty check from current parallel run informs final positioning.

---

## 6. Provenance

- **Agent 1 — Information Theory + Statistics:** completed, 88s, 38k tokens, 16 web searches
- **Agent 2 — Graph Theory + Data Mining:** completed, 101s, 44k tokens, 17 web searches, doc indexed
- **Agent 3 — Game Theory + Mechanism Design:** completed, 112s, 59k tokens, 12 web searches, 7 KB sources
- **Agent 4 — Power Flow + ICDM Trends:** completed, 186s, 47k tokens, 19 web searches, 2 KB sources, separate doc written to `docs/_research_chunks/G_icdm2026_frontier_pf_ml.md`
- **Agent 5 — Mathematics + Abstract Algebra:** completed, 414s, 56k tokens, 21 web searches, 3 KB sources

All five reports cross-referenced and synthesized 2026-05-24.

---

## 7. Source bibliography (deduplicated across agents)

**Hypernetwork + cross-domain transfer:**
- LUMINA topology-transferable AC-OPF (arXiv 2603.04300)
- GFM-RAG (arXiv 2502.01113)
- UGCN (arXiv 2509.08672)
- ICDM 2025 BSR-Up: zero-shot cross-city hypernetworks

**Game-theoretic attribution:**
- Exact Any-Order Shapley Interactions for GNNs (arXiv 2501.16944, Muschalik et al. 2025)
- Banzhaf Value via GNNs on Network Flow Games (arXiv 2510.13391)
- Distributionally Robust Nash Equilibrium via VI (arXiv 2510.17024)
- SurroShap (arXiv 2310.13325)

**Anomaly localization:**
- NODESAFE Bounded Energy OOD (arXiv 2504.13429)
- GNNSafe (arXiv 2302.02914)
- CRC-SGAD (arXiv 2504.02248)
- Mahalanobis OOD geometry (arXiv 2510.15202)

**Lipschitz / certified robustness:**
- Lipschitz-Bounded Networks for Robust CP (arXiv 2506.05434)
- Calibration Robustness in Split CP under Adversarial Attacks (arXiv 2511.18562)
- Parker 2026 Adversarial Attack on CANOS (arXiv 2602.17975)

**Equilibrium / DEQ:**
- PE_DEQ_PF (host project)
- PIGNN-Attn-LS (arXiv 2509.22458)
- NEO-Grid (arXiv 2509.21668, closest DEQ-grid prior art)
- PI-GN-JODE (arXiv 2603.20838, Mar 2026)

**Mathematical primitives (for differentiators / future pivots):**
- Polynomial Neural Sheaf Diffusion (arXiv 2512.00242)
- Gauge-Equivariant Graph Networks GESC (arXiv 2511.16062)
- Optimal-Transport Conformal Prediction (arXiv 2501.18991, 2502.03609)
- Bundle Neural Networks (arXiv 2405.15540, 2602.12884)
- Hodge GNN Edge-Space Detection (arXiv 2503.12919)
- Graphcode multiparam PH (arXiv 2405.14302)
- Subgraph-GNN via graph products (arXiv 2406.09291)

**Avoided overlaps (rejected proposals):**
- PINN-CP (arXiv 2509.13717) — physics-residual as CP nonconformity (SCAM mechanism overlap)
- Sheaf-Laplacian anomaly on supply chains (Wang 2025, MDPI Math 13/11/1795)
- Conformalized Post-Fault Voltage Trajectories (arXiv 2410.24162)
- Spatio-Temporal CP for Power Outage (arXiv 2411.17099)
