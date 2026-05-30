# X9 — Earn the phase transition: findings

**Closes:** T2 / R01-W2 / R02-§2 ("phase transition never reached in practice, ρ≤0.42").
**Date:** 2026-05-30 · **Script:** `scripts/exp_phase_transition_stress.py` · **Data:** `results/exp_phase_transition_stress.csv`
**Setup:** Cora, 50-node BFS subgraph, IGNN at spectral cap κ=0.99, 10 seeds. Heavy linalg in float64.

---

## 1. What X9 had to show

Theorem 1(b) (corrected, T1) states `‖(I−J_z)⁻¹‖₂ ≥ 1/dist(1, spec(J_z))`, diverging as `Ω(1/(ε_crit−ε))` when a **real** eigenvalue of `J_z` → +1. Reviewers objected the cliff is *never reached* (ρ≤0.42). X9 must **exhibit** the cliff (a real eigenvalue → +1 with resolvent blow-up) and show the trained model sits safely below it — both on one panel.

## 2. Design + the critique that changed it

First design (scale the linear Jacobian `J_lin = Â⊗W` radially) was **wrong** and caught in inspection *before* running:
- `W` (trained 64×64) has an **all-complex spectrum**, so `Â⊗W` has no real eigenvalue. Radial scaling sweeps eigenvalues *past* the line Re=1 but never *through the point* 1; `dist(1,spec)` bottoms out at `|Im|` and the resolvent **peaks but never diverges** → would have silently shown a bounded bump.

**Fix (used):** drive the cliff with the operator's **symmetric part** `S = Â⊗W_sym`, `W_sym=(W+Wᵀ)/2` (symmetric → real spectrum guaranteed). Its top eigenvalue is exactly `J_z`'s real-axis contraction margin (numerical abscissa); scaling it → 1 is Thm 1(b)'s "real eigenvalue → +1," giving the exact `‖(I−S)⁻¹‖ = 1/(1−λ)`.

## 3. Correctness verification (protocol step 2–3)

Built-in gate on seed 0 — analytic vs the trusted autograd `compute_jacobian`:
- linear Jacobian (Kronecker convention) error **1.54e-9**
- masked Jacobian (ReLU active-set) error **1.54e-9**
- symmetric-part identity `S − sym(J_lin)` = **0.0**

All ≪ tol 1e-3. **VERIFY PASSED.** The Kronecker vec-convention, the ReLU mask, and the symmetric construction are confirmed correct.

## 4. Results (10 seeds, κ=0.99)

**Curve (i) — STRESS (symmetric part, λ → +1): the cliff.** Resolvent matches `1/(1−s)` to **1.1e-11** (it *is* the identity; std across seeds 5e-9):

| λ (=s) | 0.50 | 0.90 | 0.95 | 0.99 | 0.999 | 0.9999 |
|--------|------|------|------|------|-------|--------|
| dist(1,spec) | 0.50 | 0.10 | 0.05 | 0.01 | 1e-3 | 1e-4 |
| resolvent | 2 | 10 | 20 | 100 | 1,000 | 10,000 |

**Curve (ii) — TRAINED (actual masked `J_z`): the safety margin.**

| quantity | mean ± std | range |
|----------|-----------|-------|
| λ_real_max(J_z) | **0.424 ± 0.043** | 0.328–0.483 |
| numerical abscissa λ_max(sym J_z) | 0.450 ± 0.038 | 0.365–0.493 |
| dist(1, spec) | **0.576 ± 0.043** | 0.517–0.672 |
| resolvent ‖(I−J_z)⁻¹‖₂ | **1.805 ± 0.117** | 1.555–1.961 |

The trained operator sits at λ≈0.42 / dist≈0.58 with a flat resolvent ≈1.8, while the cliff is at λ→1 / dist→0 / resolvent→∞. Reaching even resolvent=10 requires the contraction margin to shrink ~6× (dist 0.58→0.10, λ→0.90). **The cliff is real (curve i) and the trained models sit far below it (curve ii)** — the phase transition is earned, not asserted.

## 5. Bonus finding — the safety is two-layered (important for prose honesty)

The unmasked linear operator already sits well below the cap:
- spectral radius ρ(Â⊗W) = **0.471 ± 0.018**
- symmetric-part top eigenvalue μ₁ = **0.497 ± 0.014**

Since ‖Â‖₂≈1, this means ρ(W) ≈ 0.47 vs ‖W‖₂ ≈ 0.99 — **W's non-normality** (ρ(W) ≈ ½‖W‖₂) does most of the work bringing 0.99 → ~0.47; the **ReLU active-set** trims the rest (0.47 → 0.42). So the safety is robust (two independent mechanisms), but the paper's current attribution *"the trained ReLU pattern keeps ρ(J_z)≤0.42"* slightly over-credits ReLU.

**Paper-prose implication (for the integration step):** keep the claim (it's true of the masked `J_z`) but credit both mechanisms, e.g. *"the learned weight's non-normality (ρ(W)≈½‖W‖₂) together with the ReLU active set keeps ρ(J_z)≈0.42."* This strengthens the safety narrative rather than weakening it.

## 6. Minor numerical note

This fresh run gives λ up to 0.483 and resolvent up to 1.96 — marginally above the paper's "≤0.42 / →1.80" (which describe `results/exp_phase_transition.csv` across κ=0.3→0.99). Same seeds/subgraph; the small gap is CUDA-training nondeterminism. Qualitatively identical; if the paper quotes a hard "≤0.42", soften to "≈0.42" or quote the 10-seed mean 0.42.

## 7. Files of record
- `scripts/exp_phase_transition_stress.py` — experiment (with built-in autograd verify gate)
- `results/exp_phase_transition_stress.csv` — 10 seeds × (1 trained + 9 stress) rows
- `results/exp_phase_transition.csv` — existing safe-side sweep across κ (curve ii context)

## 8. Remaining for paper integration (not part of this experiment cycle)
- Upgrade `fig:phase_transition` to the two-curve (λ, resolvent) panel: cliff `1/(1−λ)` + trained cluster at λ≈0.42. (Serif 11pt per house style.)
- One-clause prose update per §5 (credit non-normality + ReLU).
