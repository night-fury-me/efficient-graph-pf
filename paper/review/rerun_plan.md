# AEGIS c=0.9 full re-run plan (2026-05-31)

**Decision:** adopt the c=0.9 recipe (fixes the non-reproducible accuracy table:
real code gives 62% Cora vs the paper's claimed 77.5%; c=0.9 → 80% + genuine κ<1)
AND fold in the parked B1–B4 sensitivity-code fixes (right time — re-running with
known √2/ρ bugs would bake them in). Then re-run the suite + update the paper.
Confirmed numbers in `ignn_accuracy_findings.md`, `confirm_ignn_fix.py`.

## Foundation (correctness-critical; verify each before proceeding)
- [x] **A. c=0.9 model/training recipe.** DONE (2026-05-31, see
      `stepA_c09_adoption.md`). IGNN hard-cap ‖W‖₂=0.9 (analytic-SVD rescale,
      differentiable) + dropout 0.5 + forward it=100/tol 1e-6; train_ignn cap=0.9,
      dropout=0.5, cosine LR, 400 ep. Verified (RTX 4090, 5 seeds, public splits):
      Cora 80.56±0.47 / Citeseer 69.60±0.60 / Pubmed 79.12±0.40 %, all κ=‖J_z‖₂<1
      (max 0.8990/0.8986/0.8902). Sensitivity pipeline intact (ScalableSensitivity
      v_ij + σ₁ finite; verify_core still 8/10, same checks 5&8 fail). Downstream
      `exp_fullgraph_attack_table` (Cora,1seed) runs clean. Both edits backward
      compatible (`c=None` recovers legacy ‖W‖=1).
- [x] **B. B1–B4 code fixes.** DONE (2026-05-31, see `stepB_b1b4_findings.md`).
      **CONVENTION DECISION:** adopted the EDGE-WEIGHT parametrization of `S_c`
      (`column_k = S[:,iN+j]+S[:,jN+i]`, NO `/√2`); the original "B1 = `/√2` on
      columns + matching `/√2` in scalable" plan was **REJECTED** — it corrupts the
      transfer bridge `d_k=w_k·v_k` (check 6 would drift 0.97→0.68). The
      per-‖δA‖_F budget bound is reported as the derived `σ₁(S_c)/√2` instead.
      Applied: B2 (`optimal_structural_attack` reports σ₁(S_c) + new
      `sigma_1_per_fro=σ₁(S_c)/√2`, direction unchanged); B3 (`_estimate_rho` →
      Rayleigh-quotient `|⟨v,J_z v⟩|`, 150 it, no overshoot); B4 (Neumann cap
      500→3000). `scalable` matvec/`_edges_to_delta_A`/`edge_vulnerability`
      normalization UNCHANGED. verify_core check 5 → unit edge-weight probe; check
      8 → σ₁(S_c); new check 8b (σ₁(S_c)/√2 informational). **verify_core: 10/10
      PASS.** check 6 bridge = 0.966 (range [0.937,0.986]), proven INVARIANT to
      B2/B3/B4 (HEAD libs give identical 0.966 in same datasets dir). ρ sanity
      (full c=0.9 Cora): _estimate_rho 0.8897 ≈ rho_rayleigh 0.8942, <1, K=119.
      **DOWNSTREAM PAPER NUMBERS TO CHANGE (flagged in `stepB_b1b4_findings.md`
      §4, NOT yet edited):** prop:attack shift `ε·σ₁ → ε·σ₁/√2`; fig:sc_heatmap +
      `sc_meta.tex \schmSigmaOne` `41.185 → 29.12` (/√2); theory.tex:67 S_c basis
      prose → edge-weight; any "tight constrained bound" σ₁ as a ‖δA‖_F-budget
      → /√2. UNCHANGED: {v_ij} rankings, transfer τ, tightness/AtkAdv ratios, r_v,
      ε_crit, singular gap.

## Re-run experiments (under fixed model + code)
- [ ] C1. `tab:cross_domain` (Test Acc / κ / ε_crit / AtkAdv / Cov%; report both
      subgraph κ and full-graph κ to fix the regime ambiguity).
- [ ] C2. Transfer τ 39-cell (`tab:explicit` / `fig:tau_heatmap`).
- [ ] C3. Four-quadrant (`tab:attack_full`), subgraph + full-graph.
- [ ] C4. Defense / delocalization (`sec:defense_ablation`) + Stackelberg-lite.
- [ ] C5. Certify experiment — re-validate non-vacuity + full-graph soundness on c=0.9.
- [ ] C6. Fraud case study (`fig:fraud_case`).
- [ ] C7. Phase-transition fig + Amazon Photo +0.996 (B3/B4-sensitive; high-ρ).
- [ ] C8. σ₁ figures / scalability (`fig:sc_heatmap` σ₁ — B1-sensitive).

## Integrate extensions + paper
- [ ] D1. AEGIS-Certify theorem + section.   - [ ] D2. AEGIS-Universal prop + RL demo.
- [ ] D3. Stackelberg-lite + spectral-delocalization note.   - [ ] D4. Spine reframe.
- [ ] D5. Update all tables/figures/numbers; fit 10pp.

**Status:** A DONE, B DONE (verify_core 10/10; convention = edge-weight,
transfer-safe; paper-number changes flagged in `stepB_b1b4_findings.md` §4 for the
integration step D5). Next: C (re-run experiments under fixed model + code).
