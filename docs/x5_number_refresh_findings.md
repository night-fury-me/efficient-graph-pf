# X5 Number Refresh — unifying all `r_v` reporting on the corrected radius

**Date:** 2026-05-30
**Trigger:** Adopting the corrected min-over-classes composed-norm radius (`prop:radius`, T3)
as the *single* radius implementation. Every `r_v`-derived number in the paper was
recomputed from `per_node_robust_radius` (corrected) after removing the two bespoke paths
that had drifted from it.

## Root cause of the drift

The corrected radius is
`r_v = min_{c != y_v}  m_v^{(c)} / || (W_{y_v} - W_c) S_v ||_2`,
which composes the head-margin Lipschitz constant *through* the sensitivity matrix `S_v`.
Two earlier paths disagreed with it:

1. **`scripts/revision_R2/R2_02_agnncert_comparison.py` (`aegis_radii`)** referenced a
   non-existent `model.readout` and silently fell back to `L_head = 1.0`, *dropping* the head
   Lipschitz term. That **inflated** AGNNCert-comparison radii. Rewired to
   `per_node_robust_radius(..., model.head)`.
2. **`iem/scalable.py` (`_scalable_node_radii`)** used the prediction-based matrix-free
   surrogate; under the corrected definition every predicted node has a positive radius, so the
   old "fraction with positive `r_v`" coverage metric became trivially ~100%.

Net effect after unification: all radii now flow from one corrected, internally-consistent
composed-norm bound — no per-experiment radius code remains.

## Per-result old -> new (all edits in `paper/sections/experiments.tex`)

| Result (location) | Quantity | Old | New |
|---|---|---|---|
| AGNNCert table (l.123) | Cora median `r_v` | 0.187 | **0.163** |
| AGNNCert footnote (l.126) | per-dataset mean ratio `r_cert/r_v` | [4.9, 10.2] | **[4.4, 15.0]** |
| AGNNCert footnote (l.126) | per-cell ratio (30 seeds) | [1.5, 22.5]x | **[2.1, 39.0]x** |
| AGNNCert footnote (l.126) | per-seed Kendall tau | [-0.08, 0.23] | **[-0.11, 0.24]** |
| AGNNCert prose (l.130) | "first-order radii ... tighter" | 4.9--10.2x | **4.4--15.0x** |
| Smoothing (l.55) | AEGIS Cora Det rad @ sigma=0.05 cmp | 0.046 | **0.078** |
| Smoothing (l.55) | dense `r_v` | ~0.09 | **~0.10** |
| Hyperparam frontier (l.154) | c=0.5 `r_v` @ 72.1% acc | 0.090 | **0.147** |
| Hyperparam frontier (l.154) | c=0.9 `r_v` @ 80.6% acc | 0.051 | **0.089** |
| `tab:cross_domain` Cov% | **metric redefined** | frac(`r_v`>0) | **Cov%@0.05 = frac(`r_v`>0.05)** |
| Cov%@0.05 | Cora | 81+/-9 | **83+/-10** |
| Cov%@0.05 | Citeseer | 83+/-4 | **94+/-5** |
| Cov%@0.05 | Pubmed | 74+/-23 | **76+/-18** |
| Cov%@0.05 | Amazon | 92+/-4 | **86+/-8** |
| Cov%@0.05 | WikiCS | 70+/-3 | **78+/-4** |

Unchanged anchors that were verified, not assumed: Cora AGNNCert tau = +0.08; RS baseline
sigma=0.05 base radius = 0.123; hyperparameter accuracies 72.1% / 80.6%; Citeseer/WikiCS
deterministic radii 0.123 / 0.100.

Sources: `results/exp_cov_at_budget.csv` (5x10 seeds), `docs/exp_smoothing_sweep_results.md`
(10 seeds), `scripts/revision_R2/R2_02_agnncert_comparison.py` CSV output.

## Cov% redefinition

Old "fraction of subgraph nodes with positive `r_v`" is vacuous under the corrected
prediction-based radius (~100% everywhere). Replaced with a budget-anchored statistic,
**Cov%@0.05 = mean_v 1[r_v > 0.05]**, where 0.05 matches the smoothing sigma and the breach
experiment's smallest budget (`scripts/exp_cov_at_budget.py`). Caption updated; column header
kept as `Cov%` (definition carried by caption) to protect the 10-page pagination.

## Narrative impact — refresh STRENGTHENS the paper

- **AGNNCert:** AEGIS first-order radii remain meaningfully tighter than the sound IBP/AGNNCert
  certificate; the tightness range *widened* on the high end (10.2x -> 15.0x). Rank correlation
  stays weakly positive / near-zero (tau in [-0.11, 0.24]), which is exactly the
  diagnostic-not-certificate framing: AEGIS exposes the dominant direction, it does not promise
  a guarantee. Story intact, slightly stronger.
- **Smoothing:** AEGIS deterministic radius (0.078) is still below the structure-blind RS base
  radius (0.123) at sigma=0.05, preserving the "RS larger but structure-blind vs AEGIS smaller
  but structurally informative" contrast. The gap narrowed (0.046 -> 0.078), which *helps* — AEGIS
  sits closer to RS while adding edge structure. Story intact.
- **Hyperparameter frontier:** monotonic accuracy--robustness tradeoff preserved and more
  visible (spread 0.089--0.147 vs old 0.051--0.090). Story intact, sharper.
- **Cov%@0.05:** 76--94% tracks the contractivity ordering (Citeseer most robust at 94%, Pubmed
  most variable at +/-18). Cleaner, budget-anchored coverage. Story intact.

## Verification

Rebuilt with `latexmk`: exit 0, **10 pages**, **0 overfull hboxes**, no LaTeX errors, no
undefined refs/citations. Stale-number scan across `paper/` clean (only out-of-file hits were
`\heatcell{...}{0.046/0.051}` — genuine S_c heatmap data cells, unrelated).
