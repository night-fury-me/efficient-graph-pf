# Bug audit — `exp_aegis_regularized_training.py` (σ₁(S_c) regularized defense)

**Verdict: implementation is CORRECT. No bug invalidates the frontier.** The
result (σ₁ ↓, attack flips → 0, cert_frac peaks then declines, accuracy cost)
is real. Two minor robustness nits + one design recommendation below.

## What I verified

### 1. Matrix-free σ₁(S_c) machinery — VALIDATED to ground truth (2 ways)
- **Dense-SVD probe** (`scripts/_probe_aegis_sigma1.py`, tiny IGNN): exact
  `(I−J_z)⁻¹ J_A P_c` formed densely, `svdvals[0]` = **0.665431** vs
  `aegis_sigma1` = **0.665431** → **0.0000% error**. Singular gap was only 1.51,
  so power iteration landing on σ₁ proves the `Sc_rmatvec` **adjoint is correct**
  (a wrong adjoint misses σ₁ at a small gap).
- **Run's own sanity** (real Cora baseline, `.log`): analysis randomized-SVD
  σ₁ = **334.7396** vs `aegis_sigma1` = **334.6143** → **0.04% agreement**;
  grad-z* identical; ‖dσ₁/dW‖ = 2877 (differentiable).
- Hand-checked the autograd transposes: `vjp_Jz` (reverse VJP), `jvp_Jz` /
  `Sc_matvec` (forward-over-reverse double-grad through `dummy`), `P_c`↔`P_c^T`
  (sym placement vs (i,j)+(j,i) sum). All correct.

### 2. The frontier is not an artifact — five concerns cleared
- **σ₁ collapse is genuine reshaping, NOT ‖W‖→0.** At λ=0.003, σ₁ drops 80×
  (335→4) while **‖J_z‖₂ is unchanged (0.896→0.898)**. Trivial weight shrinkage
  would drag ‖J_z‖ down with it; it doesn't. The penalty reshapes the J_A→z*
  composition, not the weight scale.
- **Penalized quantity == measured quantity** (0.04% / 0.00%) — the σ₁ column is
  measured by the *independent* analysis path, and it equals what the penalty
  optimizes. The frontier compares one consistent operator.
- **Attack is not stale-v₁.** Line 513 recomputes `analysis_sigma1(model)` fresh
  per λ, so `attack_damage` uses each regularized model's *own* v₁. The 0-flips
  is real robustness against the re-optimized attack.
- **λ=0 baseline is honest** — `if lam > 0.0` (L377) genuinely disables the
  penalty (not multiply-by-zero).
- **895s for λ=0 is not a bug** — λ=0 reuses `base_model` (L505), so its
  `train_s` is the exhaustive full-graph certify over 2138 nodes (~15 min, per
  docstring), not redundant training.

### 3. Measurement wrappers reuse already-verified paper code
`analysis_sigma1`→`build_op`/`svd_direction` (four-quadrant table),
`attack_damage`→`apply_perturbation`/`measure_attack` (SVD≥PGD table),
`certified_fraction`→`certify_fullgraph` (certify pilot, 0 breaches). Thin
wrappers over code already validated elsewhere; transitively sound.

## Minor issues (non-blocking — do NOT affect the run)
1. **dtype fragility** (L159): `torch.zeros(N, model.hidden, device=X.device)`
   ignores dtype → crashes under float64. Harmless (real run is all-float32).
   One-line fix: add `dtype=X.dtype`.
2. **Approximate training gradient**: `detach_zstar=False` detaches z* inside the
   J_z JVP/VJP but keeps grad at the J_A evaluation point — a valid *descent*
   direction (σ₁ provably falls), not the exact IFT gradient. Fine for a
   regularizer; the measured σ₁ is independent. Footnote it if we claim
   "gradient descent on σ₁."
3. **Redundant double forward solve** per step (CE forward + penalty's own
   Picard). ~2× slower; not incorrect. Could reuse z*.

## Design recommendation (improves the result, not a bug)
The penalty is **raw σ₁ ≈ O(100s)** with ‖dσ₁/dW‖ ≈ 2877, so even λ=0.003 gives
λ·‖∇‖ ≈ 8.6 — comparable to the CE gradient. **This is why accuracy cliffs from
78%→63% at the first nonzero λ** (and why the agent is refining at λ∈[3e-4,2e-3]).
A principled fix: penalize **log σ₁** or **σ₁/σ₁_baseline** (scale-free) → a
smoother, more controllable frontier without the cliff. Worth a follow-up grid
before the cluster sweep.

## Bottom line
The defense is real and the code is sound. Safe to (a) take the low-λ refinement
as the headline frontier, and (b) queue the multi-seed/multi-dataset sweep — but
consider switching to a normalized (log) penalty first to kill the accuracy cliff.
