# R1 — Diagnostic-only code release: record

**Closes:** R1 ("no runnable code in reviewed version"; reviewer §6/Rec4).
**Date:** 2026-05-30.

---

## 1. What R1 requires
Ship the **diagnostic-only** path unconditionally — the per-edge vulnerability scores `{v_ij}` and
per-node first-order radii `{r_v}` — since those scores alone cannot reconstruct a perturbation.
Keep the **attack-direction synthesis** (`δÂ*`, Algorithm 1's direction step / `optimal_structural_attack`)
gated per the coordinated-disclosure protocol. The paper already discloses this in `conclusion.tex`
(disclosure protocol, item iii).

## 2. What was delivered
- **`iem.adversarial.diagnostic_analysis(F, model, z_star, ctx, logits, labels)`** — a clean API that
  returns `{v_ij}`, `{r_v}`, `sigma_1`, `rho`, `eps_crit`. It **does not call** `optimal_structural_attack`
  and never returns the SVD direction `δÂ*`. By construction the released path produces only scalar
  scores and radii.
- **`scripts/aegis_diagnose.py`** — a runnable, self-contained, anonymized demo (Cora, 50-node ego-subgraph,
  exact dense path). Verified output:
  ```
  AEGIS diagnostics  (Cora, 50-node ego-subgraph, 61 edges)
    rho(J_z) = 0.2320    sigma_1(S_c) = 16.1357    eps_crit = 0.7680
    Top-10 most vulnerable edges (largest v_ij): edge (1,49) v_ij=6.49 ...
    Per-node first-order radii r_v: median = 0.1664, min = 0.0118
    [diagnostic-only: no attack direction synthesised (gated per disclosure protocol).]
  ```
  (Note: `r_v` uses the corrected min-over-classes radius — X5/consistency fix.)
- The demo intentionally **does not import** `optimal_structural_attack`, so running it cannot synthesise
  a perturbation.

## 3. Anonymization
Scan of `iem/` and `scripts/` `.py` for author / institution / email leaks: **none found** (only false
positives such as `model_name`/`case_name`). The released files (`diagnostic_analysis`, `aegis_diagnose.py`,
the `iem.examples.ignn_cora` IGNN they use) carry no author identifiers.

## 4. Public-release bundling guidance (for camera-ready / repo publish)
The working repo keeps `optimal_structural_attack` because the experiments need it. For the **public
diagnostic-only bundle**, include:
- `iem/adversarial.py` functions: `diagnostic_analysis`, `structural_sensitivity_matrix`,
  `constrained_sensitivity_matrix`, `per_node_robust_radius`, `extract_ego_subgraph`,
  `critical_perturbation_budget`, `extract_W_spectral_norm`, `_compute_structural_jacobian`;
- `iem/certify.py` (spectral_radius), `iem/ift.py` (compute_jacobian), an example model + loader;
- `scripts/aegis_diagnose.py`.
**Exclude** `optimal_structural_attack` and Algorithm 1's `δÂ*` direction step (gated).

## 5. Paper note
A `framework.tex` one-liner pointing to the runnable code was drafted but **dropped** — it pushed the
build to 11 pp and is redundant with the existing `conclusion.tex` disclosure ("the diagnostic-only path
… released unconditionally"). The Algorithm 1 radius line was already updated to the corrected
min-over-classes form (X5). Paper builds clean at 10 pp.

## 6. Files of record
- `iem/adversarial.py` — `diagnostic_analysis` (new)
- `scripts/aegis_diagnose.py` — runnable diagnostic-only demo (new)

## 7. Status
**R1 closed:** the diagnostic-only path is implemented, runnable, verified, and anonymized; the attack
synthesis stays gated; public-release bundling is documented above.
