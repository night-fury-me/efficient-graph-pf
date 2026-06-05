# Opt-B — analytic matrix-free operator (IGNN), fixes Pubmed OOM

**Status:** ALL 5 STEPS DONE and verified (2026-06-05). See
`results/exp1/opt_b_analytic_findings.md` for results. Analytic path == autograd to
machine precision (Cora fp64/fp32), 19× faster; full-graph Pubmed σ₁ runs at 6.29 GB
(autograd OOMs at 24 GB); EXP-1 Pubmed defense un-dropped (50 models, 10 seeds, peak
6.3 GB) with a clean cap frontier in `results/exp1/exp1_pubmed.csv`. This file is the
original self-contained plan.

## Why
- The matrix-free operator `ScalableSensitivity` (`iem/scalable.py`) applies `J_z`/`J_A` via
  **autograd** JVP/VJP (`_jvp_Jz`, `_vjp_Jz`, `_structural_jvp`, `_structural_vjp`). At ρ→1 the
  Neumann depth reaches 3000, so each matvec is up to 3000 autograd passes; and `_structural_vjp`
  builds an **N×N backward graph** that OOM'd full-graph Pubmed (>24 GB) — the reason Pubmed was
  dropped from EXP-1's defense.
- Replacing the autograd applications with the IGNN's **closed-form** Jacobian applications removes
  the backward graph (→ no OOM) and is fast at any N.

## Why this is high-stakes (handle with care)
`ScalableSensitivity` feeds every σ₁ reading in the paper. A vec-convention or spectral-scaling slip
is invisible but corrupts every reported σ₁. **Gate on `allclose` across the full rSVD before trusting.**

## Operator form
IGNN operator `F(Z) = ReLU(Â · _W_eff(Z) + X_proj)`, with `_W_eff(Z) = Z @ Wᵀ` (W = effective,
spectral-normalized weight; ignn_cora.IGNN). At the equilibrium `Z = z_star` (N×d), the active mask is
`φ′ = (z_star > 0)` (since `z_star = ReLU(preact)` ⇒ `z_star>0 ⇔ preact>0`).

## Derived analytic applications (all matmuls, no autograd → no backward graph)
Let `V = reshape(v, (N,d))`, `U = reshape(u, (N,d))`, `Z = z_star` (N×d):
- `J_z · v   = φ′ ⊙ (Â · V · Wᵀ)`
- `J_zᵀ · u  = Â · (φ′ ⊙ U) · W`
- `J_A · δA  = φ′ ⊙ (δA · Z · Wᵀ)`          (δA is N×N; result flattened to D)
- `J_Aᵀ · u  = (φ′ ⊙ U) · W · Zᵀ`           (result is the N×N gradient w.r.t. A)

Row-major (`reshape(-1)`) vec convention; `Â` symmetric. The constant `X_proj` term drops out of `J_z`
(it does not depend on z) but NOT of the equilibrium itself.

## Steps
1. **Expose W.** Add an opt-in analytic path to `ScalableSensitivity` (constructor kwarg `ignn_weight=W`
   or a subclass). Extract the *effective* W incl. spectral scaling — verify numerically that
   `_W_eff(Z) == Z @ Wᵀ` for the trained model (the spectral-norm scale must be folded in).
2. **Implement** the 4 analytic methods behind the flag; keep the autograd path as fallback.
3. **Verify allclose** vs the autograd ops ACROSS the full rSVD (`top_k_svd` → σ₁, v₁), not just one
   JVP, on a Cora 50-node subgraph AND a mid-size full graph. Tolerance ~1e-4.
4. **Pubmed re-test.** Run the full-graph Pubmed σ₁-analysis on the 24 GB 4090 — confirm the OOM is
   gone and σ₁/κ are sane; cross-check against the EXP-1 cluster Cora/Citeseer behaviour.
5. **If clean:** un-drop Pubmed from EXP-1's defense (re-run the 10 Pubmed lipschitz_cap shards that
   OOM'd), and update `results/exp1/exp1_defense_baseline_findings.md` (remove the Pubmed-dropped note).

## Watch-list
- Row-major Kronecker/vec convention (verified for the dense jacrev fix; re-confirm here).
- Spectral-norm scaling: W must be the *effective* weight used in the forward, not the raw `model.W.weight`.
- `_estimate_rho` uses `_jvp_Jz` — if you swap to analytic, it benefits too (and must still match).

## Related (already done this session, for context)
- Dense path: `_compute_structural_jacobian` J_z now uses `torch.func.jacrev` (6.3s→0.16s; verified allclose).
- Opt-A: explicit-GNN `S_K` autodiff (`compute_explicit_sensitivity_ad` in `scripts/exp_tau_all_datasets.py`),
  verified for GCN/GIN/SAGE/APPNP; GAT keeps FD (attention mask).
