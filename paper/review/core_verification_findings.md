# Core implementation verification (2026-05-31)

Three independent checks: (A) adversarial audit of `iem/adversarial.py` (autograd probes), (B) adversarial audit of `iem/scalable.py` (dense cross-checks), (C) my FD harness `scripts/verify_core_implementation.py` (perturb→reconverge ground truth on a 12-node real-IGNN Cora subgraph). They agree.

## VERIFIED CORRECT (the method is sound)
- `J_z = diag(phi')(A_hat⊗W)` — matches autograd exactly; correct Kronecker order, phi' on correct side (FD 2e-9).
- `J_A = dF/dvec(A)` — row-major `i*N+j`, consistent everywhere (FD 7e-11).
- **`S = (I−J_z)^{-1} J_A` — the IFT sensitivity, the heart of AEGIS — CORRECT** (FD 9e-5; transposed/`I+J_z` variants differ by ~0.03, ruled out).
- **Transfer bridge `d_k = w_k·v_k` — CORRECT, ratio 0.987 (range 0.97–1.00)** — the τ=0.99 backbone rests on sound math.
- Matrix-free `matvec` / `top_k_svd` / `edge_vulnerability` == dense `S_c` to ~1e-7 (at ρ≈0.96).
- `per_node_robust_radius`, `greedy_structural_attack`, `extract_ego_subgraph`, smoothing cert — all correct.

## REAL BUGS FOUND
**B1 — `constrained_sensitivity_matrix` omits the `1/√2` basis normalization.** `S_c[:,k]=S[:,iN+j]+S[:,jN+i]` is built on basis `(e_i e_j^T+e_j e_i^T)` (Frobenius norm √2), not the unit `b_k`. So `σ₁(S_c)` overstates "max shift per unit ‖δA‖_F" by **√2** (FD check 5 fails at rel-err 0.293 = 1−1/√2). **Affects magnitude only**: reported `σ₁(S_c)`, the "tight constrained bound," the optimal-attack `max_first_order_shift`. **Does NOT affect**: per-edge `v_ij` RANKING (√2 cancels across columns), the transfer `d_k=w_k·v_k` (uses the same √2 scaling, self-consistent), per-node radii (margin-based), ε_crit/theory.

**B2 — `optimal_structural_attack.max_first_order_shift` uses unconstrained `σ₁(S)`** but returns a symmetrized/edge-supported direction whose actual shift is `σ₁(S_c)`-bounded (FD check 8: 0.16 gap). Overstates the achievable one-query shift. Fix: report `σ₁(S_c)` (post-B1) consistently.

**B3 — `scalable._estimate_rho` overshoots (can exceed 1) on non-normal `J_z`**: true ρ 0.96→est 1.077, 0.99→est 1.05. Reports wrong ρ and silently pins Neumann depth K=cap=500.

**B4 — Neumann K=500 under-truncates at ρ≳0.98** (dense-vs-matvec at ρ=0.99: 10.8% error at K=500, 9e-7 at K=2000). At ρ≈0.96 it's exact. **Danger zone is full-graph results at ρ≳0.98 — i.e. Amazon Photo (paper says ρ≈1.00).**

**B5 (latent) — `node_sensitivity_norms` clamps Hutchinson probes to |E|** (downward-biased; no reported figure uses it).

## Impact on paper claims
| Claim | Status |
|---|---|
| Transfer τ=0.99 (rankings, ρ<0.98 datasets) | **SAFE** (ranking; √2 cancels) |
| Amazon Photo +0.996 (ρ≈1.00) | **RE-VERIFY** (B3/B4 high-ρ truncation) |
| Tightness ≈1.01 | survives as a ratio; ε mislabeled by √2 |
| AtkAdv 3.2–4.1× (measured displacement ratio) | likely SAFE (verify) |
| σ₁(S_c) values (e.g. fig:sc_heatmap σ₁=41.2) | **overstated √2 — fix** |
| ε_crit, theorem, per-node radii | SAFE |
| Fraud case study (v_ij ranking, transfer) | SAFE |
| Full-graph defense dilution (σ₁ RATIO) | SAFE (√2 cancels in ratio) |

## Fixes (next iteration)
1. `constrained_sensitivity_matrix`: `col = (S[:,iN+j]+S[:,jN+i])/√2` (+ matching `/√2` in `scalable._edges_to_delta_A`/matvec so dense==matrix-free).
2. `optimal_structural_attack`: report `σ₁(S_c)` consistently.
3. `scalable._estimate_rho`: Rayleigh-quotient spectral-radius estimate; raise Neumann cap to ≥3000; re-verify Amazon Photo.
4. Re-run `verify_core_implementation.py` → all 10 checks must pass. Then re-derive σ₁ figures + re-confirm Amazon Photo.
