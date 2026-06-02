# Step B — B1–B4 sensitivity-code fixes + √2 convention reconciliation (2026-05-31)

**Status: DONE. `verify_core_implementation.py` → 10/10 hard checks PASS.**
Transfer bridge (check 6) preserved and provably invariant to the B-fixes.

---

## 1. The convention decision (edge-weight; transfer-safe). Rescaling REJECTED.

**Decision: `S_c` columns use the natural EDGE-WEIGHT parametrization**

    column_k = S[:, iN+j] + S[:, jN+i]      (UN-normalized; NO 1/√2)

so that:

- `v_k = ‖column_k‖` is the equilibrium shift **per unit edge-weight** (δc_k = 1).
  The per-edge rankings `{v_ij}` and the transfer bridge **d_k = w_k · v_k**
  (Prop. transfer; verify_core check 6 ≈ 0.97) are CORRECT in these units and stay
  byte-for-byte unchanged.
- `σ₁(S_c)` is the **max shift per unit ‖c‖** (edge-weight L2 norm), the quantity
  the matrix-free rSVD / dense SVD already return.
- A symmetric edge perturbation has `‖δA‖_F = √2 · ‖c‖`. Therefore the
  **threat-model (‖δA‖_F-budgeted) bound = σ₁(S_c)/√2**. This is now exposed as a
  *separate* field, not by rescaling the operator.

### Why rescaling `S_c` by 1/√2 was REJECTED

The earlier parked plan (see `core_verification_findings.md`, old `rerun_plan.md`
"B1") proposed dividing `S_c` columns by √2 (and matching `/√2` in
`scalable.matvec`/`_edges_to_delta_A`) to make `σ₁(S_c)` a per-Frobenius quantity.
**That corrupts the transfer bridge `d_k = w_k · v_k`** — the paper's headline
(τ median +0.99; Amazon Photo +0.996). With rescaling, `v_k → v_k/√2`, so the
predicted single-edge-removal damage `w_k·v_k` would be wrong by 1/√2 against the
*reconverged* ground truth `d_k`. verify_core check 6 would drift from ≈0.97 to
≈0.68. The bridge is finite-difference-anchored; we must not move it.

**Resolution: keep the operator in edge-weight units (bridge intact); report the
per-Frobenius bound as the derived quantity `σ₁(S_c)/√2` wherever the *threat
model* (`‖δA‖_F ≤ ε`) is the budget.** Single source of truth, zero bridge risk.

---

## 2. Code diffs (B2 / B3 / B4)

### B2 — `iem/adversarial.py :: optimal_structural_attack`
The function returned a symmetric, edge-supported direction (`sym(reshape(v₁))`)
but reported `max_first_order_shift` / `sigma_1` from the **unconstrained** σ₁(S),
which over-states the achievable shift of the returned (feasible) direction.
**Fix:** report the **constrained** σ₁(S_c) — consistent with the direction — and
expose the per-Frobenius bound. Direction unchanged.

    sigma_1                 = σ₁(S_c)              # was σ₁(S) (unconstrained)
    max_first_order_shift   = ε · σ₁(S_c)          # was ε · σ₁(S)
    sigma_1_per_fro         = σ₁(S_c)/√2           # NEW: ‖δA‖_F-budgeted bound
    max_shift_per_fro       = ε · σ₁(S_c)/√2       # NEW
    sigma_1_unconstrained   = σ₁(S)                # kept for reference
    # vulnerability spectrum now read off S_c columns (constrained_sensitivity_matrix)

### B3 — `iem/scalable.py :: _estimate_rho`
Was: ~30 power-iters returning the operator 2-norm `‖J_z v‖` — an UPPER bound on ρ
that can exceed 1, overshoot a true ρ≈0.96/0.99, and then silently pin the Neumann
depth at the cap (under-truncating).
**Fix:** power-iterate (n_iter 30→**150**) and return the **Rayleigh-quotient**
spectral radius `|⟨v, J_z v⟩|` (sign-aware), mirroring `rho_rayleigh` in
`scripts/exp_fullgraph_attack_table.py`. Does NOT overshoot.

### B4 — `iem/scalable.py :: _adaptive_neumann_depth`
Raised the truncation **cap 500 → 3000** (adaptive depth selection unchanged; only
the ceiling). At ρ=0.99, K=500 gives resolvent error ~ρ^K/(1−ρ) ≈ 70% relative;
K=3000 keeps high-ρ graphs (Amazon Photo, ρ≈1) accurate.

### Harness — `scripts/verify_core_implementation.py` (made self-consistent)
- **check 5:** FD probe changed from unit-Frobenius `b_k=(e_i e_j+e_j e_i)/√2` to
  **unit edge-weight `b_k=(e_i e_j+e_j e_i)` (δc_k=1)**; expects ‖Δz‖/h ≈ v_ij[k].
  This tests what `v_ij` actually is under the edge-weight convention → PASS.
- **check 8:** now compares `atk["sigma_1"]` to dense **σ₁(S_c)** (passes via B2).
- **check 8b (new informational):** verifies `atk["sigma_1_per_fro"] = σ₁(S_c)/√2`
  exactly, and reports the random-search max-‖Δz‖ over symmetric edge δA with
  ‖δA‖_F=1 (the per-Frobenius interpretation; a lower bound that under-samples the
  SVD optimum, same as check 4b).
- **checks 4a and 6: UNTOUCHED** (verified byte-identical vs HEAD).

(B1 as originally scoped — the `/√2` rescale of `constrained_sensitivity_matrix`
and `scalable.matvec` — is intentionally NOT applied; superseded by the
edge-weight decision above. `iem/scalable.py` matvec / `_edges_to_delta_A` /
`edge_vulnerability` normalization is unchanged, as required.)

---

## 3. verify_core 10/10 breakdown (`.venv/bin/python`, Cora, N=12 ego-subgraph)

    Subgraph N=12 hidden=64 |edges|=11  D=768
    [PASS] 0. z* is a fixed point  ‖F(z*)-z*‖/‖z*‖        rel-err=4.37e-15 (tol 1e-09)
    [PASS] 1. J_z @ v   vs  dF/dz finite-diff             rel-err=6.66e-10 (tol 1e-04)
    [PASS] 2. J_A @ vec(dA)  vs  dF/dA finite-diff        rel-err=4.51e-11 (tol 1e-04)
    [PASS] 3. S @ vec(dA)  vs  dz*/dA finite-diff         rel-err=1.56e-04 (tol 5e-03)
    [PASS] 4a. S_c[:,k] == S[:,iN+j]+S[:,jN+i]            rel-err=0.00e+00 (tol 1e-09)
    [ -- ] 4b. σ₁(S_c)=12.1233  random-search max=7.7836  (best<=σ₁: True)
    [PASS] 5. v_k == FD single-edge shift (unit edge-weight) rel-err=4.28e-04 (tol 5e-03)
    [ -- ] 6. transfer d_k/(w_k v_k): mean=0.966 range=[0.937,0.986]  (->1 bridge)
    [PASS] 7a. ScalableSensitivity.matvec vs dense S_c@v  rel-err=4.74e-07 (tol 1e-03)
    [PASS] 7b. matrix-free σ₁ vs dense σ₁(S_c)            rel-err=1.04e-07 (tol 5e-03)
    [PASS] 7c. edge_vulnerability vs dense ‖S_c[:,k]‖     rel-err=2.81e-07 (tol 5e-03)
    [PASS] 8. optimal_structural_attack σ₁(S_c) vs dense  rel-err=0.00e+00 (tol 5e-03)
    [ -- ] 8b. σ₁(S_c)/√2=8.5724  atk.sigma_1_per_fro=8.5724 (relerr 0.0e+00)
               random-search max-per-Fro=7.7836 (relerr 9.2e-2)  [per-‖δA‖_F bound]
    RESULT: 10/10 hard checks PASSED  -- core numerics VERIFIED

**check 6 (transfer bridge) DEBUG — fully resolved.** First observation: 0.966
here vs the historical "0.987" in `core_verification_findings.md`. Investigated to
ground truth (no hand-waving):
- My edited libs are deterministic: σ₁=12.1233, check 6 = 0.966 on two repeats.
- Swapping the two libs back to **HEAD** *in the same working dir / same
  `datasets/`* gives the **identical** σ₁=12.1233 and check 6 = **0.966** (only
  check 8 differs — it correctly FAILS at HEAD because B2 is absent). ⇒ **B2/B3/B4
  do NOT change check 6 or σ₁(S_c); the bridge is invariant to step B** (also true
  by code path: check 6 calls neither `optimal_structural_attack` nor
  `ScalableSensitivity`).
- The "0.987" reproduced only inside a throwaway `git worktree` at HEAD, whose
  `DATA_ROOT = PROJ_ROOT/datasets` resolved to a *different* directory and
  re-downloaded Cora with a different node/edge order (different ego subgraph). It
  is a data-path artifact of that worktree, not a code effect.
- Under the **adopted c=0.9 IGNN recipe** (Step A, `ignn_cora.py`) + current
  committed `datasets/`, the genuine reproducible bridge value is **0.966 (range
  [0.937, 0.986])** — well within the first-order regime (→1). Preserved exactly.

**ScalableSensitivity ρ sanity (full-graph c=0.9 Cora, N=2708, |E|=5278):**

    model.c = 0.9 | fixed _estimate_rho ρ = 0.889727 |
    rho_rayleigh (200 it) = 0.894189 | |diff| = 4.5e-3 | ρ < 1 (no overshoot) = True |
    adaptive neumann_K = 119 (cap 3000)   →  SANITY: PASS

The fixed `_estimate_rho` now agrees with the reference Rayleigh estimate and stays
below 1 (old code would have reported ‖J_z v‖ ≳ ρ, risking the ≥1 pin).

---

## 4. PAPER numbers that MUST change downstream — FLAG for integration (do NOT edit paper now)

The paper currently (theory.tex:67) states the **unit-Frobenius** convention
(`b_k=(e_i e_j+e_j e_i)/√2`, `‖b_k‖_F=1`, "σ₁(S_c) consistent with ‖δÂ‖_F=ε"). The
**code computes σ₁(S_c) in edge-weight units** = √2 × that per-Frobenius value.
Reconcile by adopting the edge-weight convention in the paper and reporting the
budget bound as **σ₁(S_c)/√2**:

| Location | Current | Change to | Note |
|---|---|---|---|
| `theory.tex:64` **prop:attack** | shift `ε·σ₁(S_c) ≤ ε·σ₁(S)` (claimed under `‖δA‖_F≤ε`) | **`ε·σ₁(S_c)/√2`** for the `‖δA‖_F≤ε` budget | σ₁(S_c) is per-edge-weight; divide by √2 for the Frobenius budget |
| `theory.tex:67` **S_c basis prose** | `b_k=(e_i e_j+e_j e_i)/√2`, `‖b_k‖_F=1`, "σ₁(S_c) consistent with `‖δÂ‖_F=ε`" | restate as **edge-weight basis** `b_k=(e_i e_j+e_j e_i)` (δc_k=1, `‖b_k‖_F=√2`); note **per-‖δA‖_F bound = σ₁(S_c)/√2** | makes prose match the code & verify_core |
| `framework.tex:30` **fig:sc_heatmap caption** | `σ₁ = 41.2` | **`σ₁ = 29.1`** (= 41.185/√2) **if** reported as the `‖δA‖_F`-budget bound | the figure's 41.185 is edge-weight units |
| `figures/data/sc_meta.tex:5` macro `\schmSigmaOne` | `41.185` | **`29.12`** (= 41.185/√2) **if** the figure reports the per-Frobenius bound | auto-generated; regen via `build_figure_data.py` OR /√2 |
| `fig_sc_heatmap.tex:177` sidebar `$\sigma_1=\schmSigmaOne$` | follows macro | follows macro (29.12) | same source |
| Any **"tight constrained bound" σ₁ values / `max_first_order_shift`** quoted as a `‖δA‖_F`-budget shift | `ε·σ₁` | **`ε·σ₁/√2`** | wherever the budget is Frobenius |

**MUST stay UNCHANGED (edge-weight quantities — do NOT /√2):**
- `{v_ij}` per-edge vulnerability **rankings** (√2 cancels) — fraud case, tab:explicit τ.
- transfer **d_k = w_k·v_k** (Prop. transfer); τ heatmap (+0.99 median, +0.996 Amazon).
- Tightness ratios, AtkAdv (damage **ratios** — √2 cancels), per-node radii r_v
  (margin-based), ε_crit, ρ/κ, the singular-**gap** 0.39–0.50 (ratio).
- The σ₁-agreement claim "dense vs matrix-free within 0.03%" (both edge-weight).

> Recommended framing for integration: state σ₁(S_c) is per-edge-weight, then write
> the threat-model bound once as `ε·σ₁(S_c)/√2`. The figure can keep 41.2 (labelled
> "per edge-weight") OR show 29.1 (labelled "per ‖δA‖_F"); pick one and label it.
> The task brief asks for 29.1.

---

## 5. Files modified (this step)

- `iem/adversarial.py` — B2 (`optimal_structural_attack` constrained σ₁ + per-Fro fields).
- `iem/scalable.py` — B3 (`_estimate_rho` Rayleigh, 150 it), B4 (cap 500→3000).
- `scripts/verify_core_implementation.py` — check 5 (unit edge-weight), check 8/8b
  (σ₁(S_c) + per-Fro informational), header/docstring convention note.

(Not mine, pre-existing Step-A working-tree state: `iem/examples/ignn_cora.py`,
`scripts/revision_R2/_common.py` — the c=0.9 recipe. Left untouched.)
