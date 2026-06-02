# Interp gating smoke — does the resolvent mechanistically decode an implicit GNN? (seed 42)

**VERDICT: WEAK.** The κ<1 IGNN cleanly learns reachability (no contraction wall), and the
resolvent is *causally faithful in direction* and *exact for small interventions*, but it does
**not beat its black-box / adjacency baselines** on the recovery gates (C1, C2) and its
first-order intervention prediction (C3) has ~50% magnitude error for the discrete edge deletion
that actually changes the algorithm output. This gate does **not** support the breakthrough pivot
on connected-components; it confirms the support-law near-definitionality the scout warned of.

Task: train IGNN_Kappa to execute source-reachability / connected-components on planted-partition
graphs (N=60, 2–4 components), then test whether (I−J_z)⁻¹ / S_c decodes the known algorithm.

Script: `scripts/exp_interp_smoke.py`  ·  CSV: `results/exp_interp_smoke.csv`

## Gate numbers (8 held-out graphs, seed 42)

| Gate | Metric | Value | Read |
|---|---|---|---|
| **G1** | ρ(J_z) eval | **0.836 ± 0.003** (κ<1 ✓) | contraction holds |
| **G1** | test acc (held-in) | **1.000** | algorithm learned |
| **G1** | node acc (held-out graphs) | **0.996 ± 0.007** | generalizes |
| **C1** | resolvent-gain AUC | **1.000 ± 0.000** | recovers component… |
| **C1** | black-box input-grad AUC | **1.000 ± 0.000** | …but baseline ties it |
| **C2** | eigen-mode align (IGNN) | **0.989 ± 0.003** | modes = comp indicators… |
| **C2** | eigen-mode align (Â control) | **0.988 ± 0.003** | …but raw Â ties it (W/φ′ adds nothing) |
| **C3** | predict-vs-resolve cosine (pure-weight) | **0.969 ± 0.003** | direction faithful |
| **C3** | predict-vs-resolve rel-err (pure-weight) | **0.542 ± 0.140** | magnitude off ~50% |
| **C3** | predict-vs-resolve cosine (real renorm delete) | **0.482** | barely correlated for real edge removal |
| **C3** | label-flipping bridges found | 4 / 8 graphs | sparse |

## Differentiable paths / assumptions used (all verified)
- IGNN fixed point z*=φ(Â z* Wᵀ + U(X)), φ=relu. State Jacobian J_z = diag(vec mask)·kron(Â,W);
  input Jacobian J_x = ∂F/∂X = diag(vec mask)·kron(I_N, U). Both row-major over (N,hidden).
- C1 gain g_u = ‖∂z*_u/∂x_s‖ from block of (I−J_z)⁻¹ J_x at the source one-hot column.
  Baseline = ‖∂P(reach)_u/∂x_s‖ via autograd through the full unrolled solve.
- C3 predicted Δz* = S_c[:,bridge]·(−Â_ij) (first-order, S=(I−J_z)⁻¹J_A, J_A finite-diff per iem
  convention); re-solved Δz* from full nonlinear equilibrium. Reported BOTH pure-weight deletion
  (matches S_c's perturbation model) and fully renormalized deletion (the real intervention).
- C2 top-k J_z eigvecs (by resolvent gain 1/|1−λ|) reduced to node-space norm profiles vs unit
  component-indicator basis; Â's own eigvecs as the trivial-spectral control.
- Double precision throughout the linear algebra.

## Self-checks (both PASS)
- **S1** BFS vs union-find reachability labels agree exactly (20 graphs).
- **S2** resolvent block (I−J_z)⁻¹J_x[:,src] vs autograd ∂z*/∂x_s through the differentiable
  solve: **rel-err 4.4e-16, cosine 1.000** (bit-exact). J_x Kronecker form also matched
  finite-diff ∂F/∂X to 3e-11.

## Debug-before-accepting (the WEAK result is real, not a bug)
Scaling test on a label-flipping bridge — predicted vs re-solved Δz* at fractions t of the full
deletion:

| t | ‖pred‖ | ‖real‖ | cos | rel-err |
|---|---|---|---|---|
| 0.02 | 0.0084 | 0.0083 | 1.000 | 0.007 |
| 0.10 | 0.0420 | 0.0406 | 1.000 | 0.038 |
| 0.30 | 0.1259 | 0.1144 | 0.996 | 0.136 |
| 0.60 | 0.2518 | 0.2084 | 0.989 | 0.264 |
| 1.00 | 0.4196 | 0.3024 | 0.971 | **0.480** |

First-order prediction is **essentially exact for small δ** and the error grows smoothly with |δ|
→ this is genuine second-order nonlinearity of a *full* bridge deletion (Â_ij≈0.27 is a large
perturbation; relu mask flips + degree renorm), not a sign/scale bug. S_c is correct.

## Interpretation (honest)
1. **No contraction wall** for CC/reachability: κ<1 is *compatible* with executing this algorithm,
   because reachability is a support computation the message-passing support law
   (Âᵏ)_uv=0 for k<d(u,v) handles natively. (Different algorithms — e.g. shortest-path with a
   near-critical relaxation front — could still wall; untested.)
2. **C1 and C2 are near-definitional and do not beat baselines.** Resolvent gain separates
   reachable/unreachable perfectly *because* cross-component gain is structurally zero — but a
   plain input-gradient does the same (AUC 1.000 = 1.000), and the eigen-modes are just the
   adjacency's component-indicator spectral structure (IGNN 0.989 ≈ Â 0.988). The learned W/φ′
   adds nothing detectable. This is "S_c with an interpretability sticker," exactly the scout's
   relabeling risk.
3. **C3 is the only differentiator and it is only partial:** the resolvent gets the *direction* of
   the equilibrium response right (cos≈0.97) and is exact infinitesimally, but for the actual
   discrete intervention (delete the bridge, renormalize) it is off ~50% in magnitude and only
   cos 0.48 — i.e. first-order linear response is not a faithful predictor of the realized output
   change for the very edges that matter algorithmically.

## Recommendation
Do **not** invest in the connected-components resolvent-decoding direction on this evidence: the
recovery gates collapse to baselines and the causal-intervention gate is only directionally
faithful. If the pivot is pursued, the scout's own ranking points elsewhere —
(a) an algorithm whose causal structure is **not** reducible to adjacency support (single-source
shortest-path / Bellman-Ford, where the resolvent must track shortest-path edges, not just
connectivity, so C1/C2 can beat adjacency), and (b) a higher-order / re-linearized intervention
predictor (or honest reporting that only the infinitesimal response is exact). The κ<1↔expressivity
tension did **not** bite for CC; it must be re-probed on a near-critical algorithm before any
"algorithm lives near criticality" claim.
