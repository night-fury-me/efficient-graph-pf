# Step A — adopt the c=0.9 IGNN recipe as the repo default (2026-05-31)

**Goal.** Make the validated `c=0.9` IGNN recipe (hard spectral cap ‖W‖₂=0.9 via
analytic, differentiable SVD rescale; dropout 0.5; forward max_iter=100/tol=1e-6;
cosine LR, 400 ep) the repo default so training yields ~80%-accurate, genuinely
contractive (κ=‖J_z‖₂<1) models, WITHOUT breaking the sensitivity pipeline.

**Status: DONE.** All four verification bars pass. Numbers below are real (RTX 4090,
`.venv/bin/python`, public Planetoid splits, no torch reinstall).

## Files changed (2 files, transparent to sensitivity math)

### 1. `iem/examples/ignn_cora.py` — `IGNN` class
- `__init__(..., c: float|None = 0.9, dropout: float = 0.5, spectral_norm=True)`.
  Old positional signature `IGNN(nf, hidden, nc)` still works (new args defaulted),
  so every existing caller silently inherits the c=0.9 model.
- New `_W_eff(Z)`: when a hard cap `c` is set, rescale W to ‖W_eff‖₂≤c using the
  analytic 2-norm `torch.linalg.matrix_norm(W, ord=2)` and `clamp(c/‖W‖₂, max=1.0)`
  (scale DOWN only). Differentiable; the rescale stays in the autograd graph.
- `operator(Z, ctx)` unchanged in interface (`ctx["A_hat"]`, `ctx["X_proj"]`); it
  now calls `_W_eff(Z)` instead of `self.W(Z)`. The cap lives INSIDE `operator`,
  so `ScalableSensitivity` / `structural_sensitivity_matrix` differentiate the
  actual capped operator with zero special-casing.
- `forward(..., max_iter=100, tol=1e-6, train_dropout=False)`: dropout applied to
  Z* before the head only when `train_dropout=True`. Also fixed a latent off-by-one
  at convergence (`Z = Z_new` on break; `Z_star = Z`) so z* is the converged iterate.
- Legacy ‖W‖=1 model reproducible via `c=None` (re-applies the `spectral_norm`
  parametrization).

### 2. `scripts/revision_R2/_common.py` — `train_ignn`
- Recipe baked into defaults: `epochs=400, hidden=64, c=0.9, dropout=0.5,
  lr=0.01, wd=5e-4, fwd_iter=100, fwd_tol=1e-6, cosine=True`. Cosine LR schedule;
  trains with `train_dropout=(dropout>0)`.
- Backward compatible: `train_ignn(X, A, y, mask, nf, nc, dev, seed)` unchanged;
  `epochs`/`hidden` remain optional; recipe knobs are extra optional kwargs
  (`c=None` recovers the legacy model).

`iem/scalable.py` carries an UNRELATED, pre-existing change (vectorized
upper-triangular edge extraction — identical ordering, perf only); the sensitivity
MATH (B1–B4) was deliberately NOT touched here.

## (a)+(b) Accuracy (5 seeds) + contractivity κ=‖J_z‖₂  (power iteration on JᵀJ at z*)

κ is the TRUE A3 factor (ReLU mask included), measured by differentiating
`model.operator` at the fixed point — which also proves the cap is transparent to
autograd. ‖Â‖₂≈1.0 for all three, so κ≈c=0.9 as designed.

| Dataset  | Test acc (mean ± sd) | κ mean ± sd | κ max | ‖Â‖₂ |
|----------|----------------------|-------------|-------|------|
| Cora     | 80.56 ± 0.47 %       | 0.8986 ± 0.0002 | 0.8990 | 1.0003 |
| Citeseer | 69.60 ± 0.60 %       | 0.8984 ± 0.0002 | 0.8986 | 1.0000 |
| Pubmed   | 79.12 ± 0.40 %       | 0.8876 ± 0.0022 | 0.8902 | 1.0001 |

Per-seed Cora: 80.40 / 80.90 / 80.50 / 81.10 / 79.90 %.
Per-seed Citeseer: 68.90 / 70.50 / 69.50 / 69.80 / 69.30 %.
Per-seed Pubmed: 78.80 / 79.10 / 79.70 / 79.30 / 78.70 %.
All κ < 1 → genuinely contractive (assumption A3 holds strictly). Accuracy matches
the validated target (Cora ~80, Citeseer ~69–70, Pubmed ~79) with sd < 2.

## (c) Sensitivity pipeline intact

- Fresh c=0.9 Cora model → `ScalableSensitivity` built via the
  `exp_fullgraph_attack_table.build_op` path (matrix-free op + trustworthy ρ via
  Rayleigh power-iteration; rebuild w/ neumann_terms=3000 if ρ near boundary).
  Built N=2708, edges=5278, rho(J_z)=0.8882 (no rebuild — below the 0.98
  boundary). Five sampled edge vulnerabilities v_ij = ‖S_c[:,k]‖ finite
  (13.8–44.9); σ₁(S_c) [matrix-free] = 341.80, finite. ALL FINITE/SENSIBLE = True.
- `verify_core_implementation.py`: **8/10 hard checks PASS** (unchanged, no
  regression). Only check 5 (rel-err 0.2936 ≈ 1−1/√2, the B1 √2 basis-norm bug)
  and check 8 (rel-err 0.4849, the B2 σ₁(S_c)-vs-σ₁(S) bug) fail; NOT fixed here
  (separate B1–B4 step). Checks 0–4a and 7a/7b/7c all PASS on the new c=0.9 model:
  z* is a fixed point at 4e-15; J_z/J_A/S match finite-difference (the cap is
  transparent to the sensitivity Jacobians); matrix-free == dense.

## (d) Downstream smoke test

`scripts/exp_fullgraph_attack_table.py` imports clean; its `train_ignn` builds the
new model (c=0.9, dropout=0.5) automatically. Single (Cora, seed 42) `run_single`
ran in 47s, all-finite: ρ(J_z)=0.890, neumann_K=110, σ₁=820.66; SVD attack
dmg=61.20 (11 flips), Cls-PGD dmg=29.81 (11 flips), Random dmg=6.26 (1 flip) — the
one-query SVD attack dominates the baselines, as the method predicts. DOWNSTREAM_OK.

## Verification artifacts
- `scripts/revision_R2/_verify_stepA.py` (Cora/Citeseer acc+κ + sensitivity helpers)
- `scripts/revision_R2/_verify_stepA_rest.py` (Pubmed acc+κ + sensitivity pipeline)
- `scripts/revision_R2/_verify_stepA_downstream.py` (downstream smoke)
- JSON: `_verify_stepA_results.json`, `_verify_stepA_rest_results.json`
