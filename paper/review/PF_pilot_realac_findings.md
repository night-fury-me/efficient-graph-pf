# PF Pilot — AEGIS vs REAL PandaPower AC N-1 (findings)

**Date:** 2026-05-31 · **Script:** `scripts/pilot_aegis_realac_n1.py` · **Status:** pilot (case14+case57, 3 seeds, nominal load)

## Motivation
The headline PF table (`tab:ieee`) scores AEGIS against `greedy_structural_attack` — the **surrogate's own** edge-deletion (ℓ2 shift of the 64-dim hidden state), **not** independent AC physics. The paper describes this as "ℓ2 voltage-angle deviation per AC contingency" and compares the resulting AEGIS τ against a LODF τ that *is* computed vs real PandaPower AC N-1 (apples-to-oranges). This pilot computes the **honest** number: AEGIS (and graph-centrality baselines) scored against `true_n1_severity` (real `pp.runpp`, line out-of-service, ℓ2 of ΔV,Δθ) — the same real-AC truth R2_05/R2_06 use for PI/LODF.

## Method
Train ContractiveGCN-PF exactly as `exp_topk_precision_ieee.py`; compute AEGIS S_c per-edge ranking; score it + degree / edge-betweenness / current-flow-betweenness centrality against real AC N-1. **Mapping verified**: surrogate-edge ↔ pp-line bus-pair overlap = **1.00** on both cases (so τ is trustworthy).

## Results (vs REAL AC N-1, Kendall τ / P@10, 3 seeds)

| Case | Method | τ | P@10 |
|---|---|---|---|
| case14 | **AEGIS** | **+0.56 ± 0.07** | **0.80** |
| case14 | degree | +0.53 | **0.90** |
| case14 | edge-betweenness | −0.19 | 0.60 |
| case14 | current-flow-betw. | −0.31 | 0.60 |
| case57 | **AEGIS** | **−0.16 ± 0.03** | **0.10** |
| case57 | degree | −0.28 | 0.10 |
| case57 | edge-betweenness | +0.09 | 0.50 |
| case57 | current-flow-betw. | +0.16 | 0.60 |

**Reference:** headline AEGIS-vs-**surrogate** τ = case14 **+0.42**, case57 **+0.67**. LODF/PI vs real-AC ≈ **+0.30–0.33**.

## Debugging (per "unexpected results are usually bugs")
Ruled out artifacts: overlap=1.00, **0 parallel lines** on both cases. case14 clean (0 islanding). On **case57, 5/63 outages island the grid** (sev=∞) and are the **most-severe** contingencies — all **low-degree radial lines** (deg-sum 4–8: (34,35),(24,29),(31,32),(33,34),(36,37)). AEGIS and degree rank high-flow lines and miss these → negative τ; betweenness/current-flow detect bridge/radial lines → positive. The case57 negative is **genuine**.

## Conclusion
1. **The PF physics-recovery framing is refuted.** Against real AC N-1, AEGIS goes from the headline surrogate-vs-surrogate **+0.67 → −0.16** (case57); on case14 it holds (+0.56) but only **ties degree** (+0.53, and degree wins P@10 0.90 vs 0.80).
2. **AEGIS does not beat trivial topology on real AC N-1** in either case. The surrogate τ measured **linearization quality** (S_c vs the surrogate's own nonlinear response), not physics recovery.
3. **Fundamental limitation:** islanding (radial-line removal = discrete graph disconnection) is invisible to AEGIS's continuous first-order resolvent sensitivity, yet islanding events are the most-severe real N-1 contingencies on meshed grids. This is structural, not a tuning gap.

## Implications for the paper
- The claims "recovers brute-force N-1 critical-line severity," "ℓ2 voltage-angle deviation per AC contingency," and "AEGIS matches or leads LODF against the AC voltage-angle truth" are **not defensible** as written.
- Honest fallback (REFRAME/CONCEDE): the valid result is "S_c recovers the **surrogate's own** N-1 ranking from one query" — i.e. the model-auditing claim, not cross-domain physics. The PF case study should be rescoped to that, or cut/appendixed.
- This is the pilot doing its job: re-grounding would **refute** the headline, so it is not a strengthening experiment. Found before a power-systems reviewer would have.

## Improvement attempt (Pilot 2): can augmenting AEGIS fix it? — NO
`scripts/pilot_hybrid_islanding.py` tested S_c + a topological islanding term (bridge edges ranked by island size). Result: **insufficient, and the deeper limitation is fundamental.**

Diagnostic (`/tmp/diag_div57.py`) on case57's 5 inf-severity contingencies — **none are flat-start artifacts** (none converge even with DC-init + 50 iters):
- **1 true islanding** (graph bridge, line 31–32) — the bridge term catches this one.
- **4 true voltage collapse** (lines 24–29, 33–34, 34–35, 36–37) — AC PF has **no solution** post-contingency.

The bridge augmentation moved case57 only −0.16→−0.12 (caught 1/5). The 4 voltage-collapse events are the killer, and they expose a **structural** problem:

> **A contractive equilibrium surrogate (κ<1, the paper's assumption A3) ALWAYS converges to a unique fixed point by construction. It cannot represent "no stable operating point exists" (voltage collapse) — the (I−J_z) it inverts is non-singular by design.** Voltage collapse is precisely the loss of fixed-point existence, the most severe class of real N-1 contingency on stressed grids.

So the very property that makes AEGIS's theory work (contraction → resolvent → ε_crit phase transition) is what disqualifies it from real PF contingency screening. The two cannot be reconciled without abandoning the contractive core. Additionally, even on the *finite* (flow-redistribution) contingencies, S_c only **ties degree centrality** (case14: ~0.45–0.56 vs 0.53), so it does not clearly add value over trivial topology there either.

**Net:** improving AEGIS does not rescue the physics-recovery claim. Rescope to model-auditing (S_c recovers the surrogate's own ranking) or cut.

## Caveats / to confirm in a full run
3 seeds, 2 cases, nominal operating point (surrogate eval load differs). Direction is robust (case57 strongly negative; case14 ties degree, operating-point-independent). A full run (case14/30/57/118, 10 seeds, matched operating points, + LODF/PI columns) would quantify but is very unlikely to reverse direction.
