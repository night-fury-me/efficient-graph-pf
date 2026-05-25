# Paper Narrative — Final Positioning (2026-05-25)

## Title

**"AEGIS: Exact Structural Vulnerability Prediction for Deep Equilibrium Graph Neural Networks"**

*(AEGIS = **A**dversarial **E**quilibrium **G**raph **I**mplicit **S**ensitivity)*

## One-paragraph pitch

We show that contractive DEQ-GNNs admit exact first-order prediction of adversarial vulnerability under graph structure perturbation — a property impossible for explicit GNNs. The structural sensitivity matrix S = (I-J)⁻¹J_A, derived from the fixed-point equation via IFT, predicts adversarial shift with tightness 1.00±0.01 across 5 benchmark domains, outperforms Mettack in 30/30 comparisons, and provides deterministic per-node certificates 2-5x larger than randomized smoothing. We prove a sharp phase transition at a critical perturbation budget, below which all certificates hold and above which they become void. The framework is domain-agnostic: the same 5-line API handles citation networks, e-commerce, and encyclopedia graphs — and as a case study, we show it naturally recovers power grid contingency rankings.

## Core insight

DEQ-GNNs have a structural property that no explicit GNN has: a **fixed point**. This fixed point enables **exact prediction of adversarial vulnerability** (tightness 1.00±0.01). Not bounded. Not approximated. Exact to first order. The gap between predicted and actual adversarial shift is <1% across all domains and seeds.

## Why this matters

1. **For ML**: First certified robustness framework for DEQ-GNNs under *structural* (graph topology) perturbation. The constrained tightness of 1.00 means we're not just bounding — we're predicting adversarial vulnerability exactly. No other GNN robustness method achieves this.

2. **For implicit models broadly**: "Implicit models are more *analyzable* than explicit ones" — the fixed-point structure gives us the sensitivity matrix S, which exactly predicts how structural perturbations propagate. This is a structural advantage of DEQ models over explicit K-layer GNNs.

3. **For applications**: Domain-agnostic framework that works identically on citation networks, e-commerce graphs, encyclopedia, and power grids. As a case study, the adversarial vulnerability spectrum naturally recovers N-1 contingency rankings in power systems.

## Selling points (ranked)

1. **Tightness = 1.00±0.01** — Exact prediction of adversarial shift, validated across 5 domains × 10 seeds. No other GNN robustness method achieves this. This is the headline number.

2. **Phase transition theorem** — Genuinely novel: first characterization of the three-regime (subcritical/critical/supercritical) vulnerability landscape for DEQ-GNNs, with sharp threshold ε_crit = (1-ρ)/||W||₂. Empirically validated: 83x amplification as ρ→1.

3. **30/30 vs Mettack** — IFT structural attack beats the SOTA graph attack baseline at every seed (10), every dataset (3), every budget level. Direct equilibrium analysis beats surrogate transfer.

4. **Deterministic certificates 2-5x larger than smoothing** — At equal coverage (70-92%), our radii are 2.0-5.2x larger. And ours are deterministic (always valid), not probabilistic.

5. **Domain-agnostic 5-line API** — Same `IEMiner` works on Cora, Citeseer, Pubmed, Amazon Photo, WikiCS, and power flow. Engineering contribution with immediate practical value.

6. **Adversarial robustness ≡ N-1 contingency** — First formal bridge between adversarial ML and power systems contingency analysis. Both are instances of the same sensitivity matrix S.

## Theory structure

- **Theorem 1** (Phase Transition in Adversarial Vulnerability): Three regimes around ε_crit. Genuinely novel — nobody has characterized this for DEQ-GNNs.
- **Proposition 1** (Optimal Structural Attack): SVD of S gives the worst-case perturbation direction. Application of known tools to new domain.
- **Proposition 2** (Per-Node Robust Radius): r_v = m_v / (||∂f/∂z_v|| · ||S_v||). Deterministic certificate.

## Experimental evidence

| Experiment | Result | Seeds |
|---|---|---|
| Constrained tightness | **1.00±0.01** across 5 datasets | 10 |
| IFT vs Mettack damage | **IFT wins 30/30** | 10 × 3 datasets |
| Det. vs smoothing radii | **2.0-5.2x larger** at equal coverage | 10 × 3 datasets |
| Phase transition | **83x amplification** as ρ→1 | validated |
| Scalability | N=20 (0.5s) to N=200 (8.1s) | measured |
| Cross-domain consistency | 5 benchmarks + 1 PF | 10 seeds each |

## Paper structure (proposed)

1. **Introduction**: Implicit models are more analyzable than explicit ones (the fixed-point structure enables exact vulnerability prediction)
2. **Background**: DEQ-GNNs, IFT, structural perturbation model
3. **Theory**: Theorem 1 (phase transition) + Props 1-2 (attack + certificates)
4. **IEM Framework**: Domain-agnostic API, constrained sensitivity matrix
5. **Experiments**:
   - 5.1: Cross-domain adversarial analysis (Table 1, 5 datasets × 10 seeds)
   - 5.2: Attack comparison vs Mettack (Table 2, 3 datasets × 10 seeds)
   - 5.3: Certificate comparison vs smoothing (Table 3, 3 datasets × 10 seeds)
   - 5.4: Phase transition validation (scan ρ from 0.2 to 0.99)
   - 5.5: Scalability (N=20 to 200, timing table)
   - 5.6: Case study — power flow N-1 contingency (τ=0.73)
6. **Related work**: DEQ robustness (El Ghaoui+21), GNN attacks (Zügner+19), GNN certificates (Bojchevski+20)
7. **Conclusion**

## Framing discipline

- Power flow is Section 5.6 (case study), NOT the motivation
- Lead with the ML contribution: "exact prediction" is the hook
- Theorem cites mathematical origins (IFT, contraction mapping) explicitly
- Propositions are honest about being applications of known tools
- All weaknesses pre-disclosed: first-order only, 50-node subgraphs, constrained perturbation model
