# G. ICDM 2026 Frontier: Power-Flow ML × Data-Mining Intersection
_Compiled 2026-05-24. Sources: arXiv (2024-Q2 → 2026-Q2), ICDM 2024/2025 award lists, IEEE Xplore, NeurIPS DBT 2025._

## 1. Top 5 Active Sub-Areas in Power-Flow ML (2024-2026)

| # | Sub-area | Key 2024-2026 papers | Status vs. host project |
|---|---|---|---|
| 1 | **Physics-informed GNN PF surrogates with attention / line-search correction** | PIGNN-Attn-LS (2509.22458, host), Topology-Aware GGNN (2507.02078), Towards Generalization for AC-OPF (2510.06860) | Host is SOTA; trend is post-hoc correction operators. |
| 2 | **Test-Time Training & active learning for PF** | TTT-for-AC-PF (2511.22343), Constraint-Informed Active Learning ACOPF (2511.06248), Sample-Efficient 3-phase (2305.14799) | Untouched by host; ripe for DEQ-based TTT. |
| 3 | **Foundation models for the grid** | GridFM-v0 (IBM/EPRI), GridSFM-Premier (Microsoft, 80k buses), Weather-FM-for-Grid (2509.25268), Augmented Pre-trained GNNs (ACM e-Energy 2025) | No DEQ-backed FM exists; explicit-layer only. |
| 4 | **Cascading-failure & dynamics prediction** | PI-GN-JODE (2603.20838, Mar 2026), Cascading Blackout Severity GNN (2403.15363), Power Failure Cascade (2404.16134) | Adjacent; could reuse PE_DEQ_PF as the per-step state predictor. |
| 5 | **Cross-topology / multi-scale generalisation** | UGCN (2509.08672), SaMPFA local-topology slicing (2601.01387), PFΔ benchmark (2510.22048), PGLearn (2505.22825) | Host has unique HVN+LVN paired data — cross-voltage transfer benchmark is *unbuilt*. |

## 2. Top 5 ICDM 2024-2025 Award / Theme Signals

| # | Paper / Signal | Theme | Relevance |
|---|---|---|---|
| 1 | **ICDM 2024 Best Paper** — "Scalable Graph Classification via Random Walk Fingerprints" (Li, Wang, Böhm) | Probabilistic graph fingerprints, scalability | Anything *scalable + graph + provable* wins. |
| 2 | **ICDM 2025 Best Paper** — "DP-FedLoRA: Privacy-Enhanced Federated Fine-Tuning for On-Device LLMs" (Xu et al.) | Federated LoRA + DP for LLMs | Cross-utility federated PF surrogates with DP could ride this wave. |
| 3 | **ICDM 2025 Best Student Paper** — "HyHG: Temporal Hypergraph Contrastive Learning for Biomedical Hypothesis Generation" | Temporal hypergraphs + contrastive SSL | Power grids = temporal hypergraphs (transformers, 3-windings). |
| 4 | **ICDM 2025 Best Student RU** — "Zero-Shot Cross-City Trajectory Prediction Using Hypernetworks" | Hypernetwork-based zero-shot domain transfer | Direct parallel: zero-shot HVN→LVN via hypernetwork on graph descriptors. |
| 5 | **Workshop signal** — Open-World Anomaly Detection track; "Heterogeneous Graph Anomaly Detection with Graph Wavelet Transformer" | Heterogeneous + open-world + wavelet | Grid anomaly localisation with bus-type heterogeneity fits perfectly. |

ICDM "shape" 2024-2025: graph + scalability + (federated OR hypernetwork OR contrastive) + trustworthiness (DP / certified / open-world). Anomaly detection on graphs is the most reliable acceptance lane.

## 3. Three Candidate ICDM 2026 Moonshots (Intersection: unpublished, ICDM-track-fit, leverages host assets)

### Moonshot A — **DEQ-PF as a Certified Anomaly Localiser for Bus-Level FDIA**
*Track: Trustworthy ML / Graph Mining.*
**Pitch.** Use PE_DEQ_PF's fixed-point residual ‖F(x*) − x*‖ on each bus as an *intrinsic* anomaly score for false-data-injection. Derive certified per-bus detection radii from the model's Lipschitz constant (host already has K-robust contractive DEQ). Compare to SHAP-LUNAR (2025), spatial-temporal transformer FDIA, ARMA-GNN detectors.
**Why ICDM:** combines (i) graph anomaly localisation (hot track), (ii) certified robustness for *regression* (open gap — all certified GNN work is classification), (iii) data mining of grid telemetry.
**Assets used:** PE_DEQ_PF, contractive DEQ, HVN+LVN datasets, Newton-Raphson ground truth.
**Differentiator:** First DEQ fixed-point residual as anomaly signal; first certified bus-level detection radii for AC PF regression.

### Moonshot B — **HyperPF: Zero-Shot Cross-Voltage Transfer via Hypernetworks on Graph Descriptors**
*Track: Graph Mining / Transfer Learning.*
**Pitch.** A hypernetwork ingests graph descriptors (R/X distributions, voltage class, base impedance) and emits the PE_DEQ_PF weights for that voltage tier. Pretrain on HVN (4-32 buses transmission), zero-shot to LVN (722 buses, multi-voltage distribution) and vice-versa. Direct echo of ICDM 2025 Best Student RU (zero-shot cross-city via hypernetwork).
**Why ICDM:** explicit cross-domain zero-shot + scalable graph + concrete benchmark. Host's HVN/LVN pair is the unique data asset — *no public cross-voltage PF benchmark exists*.
**Assets used:** HVN + LVN (paired by Newton-Raphson), PE_DEQ_PF backbone, bus_type/S_base normalisation knowledge.
**Differentiator:** First HVN↔LVN zero-shot benchmark; release as PF-X-Voltage benchmark dataset (ICDM loves benchmark releases).

### Moonshot C — **JumpDEQ: Physics-Informed Fixed-Point Jump Equations for Cascading-Failure Mining**
*Track: Temporal Graph Mining / Anomaly Detection.*
**Pitch.** Generalise PI-GN-JODE (2603.20838, Mar 2026) by replacing the Neural-ODE continuous block with the PE_DEQ_PF fixed-point operator at every cascade round; relay trips become *jumps in the fixed-point map*. Mine PMU/SCADA streams to predict (i) per-bus failure probability, (ii) the next equilibrium voltage. Anchors on data mining of temporal cascade graphs.
**Why ICDM:** brings DEQ (NeurIPS-style) into the data-mining cascade-prediction line, beats PI-GN-JODE on equilibrium-fidelity, and frames it as *temporal graph anomaly mining*. Open-world cascade detection workshop is the natural venue if not main track.
**Assets used:** PE_DEQ_PF, K-robust contractive DEQ (gives provable convergence even under jumps), HVN topology library.
**Differentiator:** First fixed-point-as-jump-operator formulation; closed-loop physics consistency between rounds.

## 4. Quick Risk Triage
- **A (Anomaly + Certified)** — Lowest novelty risk, highest fit with ICDM trustworthy-ML track. **Pick if conservative.**
- **B (HyperPF)** — Highest data-asset moat (only host has HVN+LVN paired). **Pick if you want a benchmark-paper acceptance.**
- **C (JumpDEQ)** — Highest scientific upside but competes with PI-GN-JODE which already landed Mar 2026; needs 6 months of work. **Pick if you want a top-paper shot.**

## 5. Key Confirmed Gaps (none published as of 2026-05)
1. Certified robustness for AC PF *regression* (all certified GNN work is classification).
2. Cross-voltage HVN↔LVN benchmark dataset.
3. DEQ fixed-point residual as anomaly score in any domain (incl. grids).
4. Hypernetwork-driven PF surrogate weight generation.
5. DEQ-backed grid foundation model (GridFM-v0 / GridSFM-Premier are explicit-layer only).

## 6. Source Inventory (arXiv IDs)
2406.07234, 2502.05702, 2503.22721, 2505.22825, 2507.02078, 2508.01951, 2509.08672, 2509.22458, 2509.25268, 2510.06860, 2510.22048, 2511.06248, 2511.22343, 2601.01387, 2602.17975, 2603.20838, 2603.21977, 2104.11846, 2110.08956, 2504.03065, 2510.03638, 2503.01140, 2106.14342, 2304.11663, 2403.15363, 2404.16134, 2503.00567, 2503.02890.
ICDM awards: stonybrook.edu/~icdm2025, icdm2024.org, dm.cs.univie.ac.at, icdm.zhonghuapu.com/Awards.
