# AEGIS-Stackelberg pilot — CUT SHORT (Cora only, 3 seeds)

Run halted by user at 1h22m (slow: ~9–32 min/seed). **Cora is complete (3 seeds);
Citeseer seed-42 ran (`best_r=1`) but was not flushed to CSV; WikiCS never ran.**
Script `scripts/exp_stackelberg_coverage.py`; CSVs `results/stackelberg_coverage_*.csv`.
Conventions verified correct (unit-basis √2; masked-operator σ₁ = σ₁(S_c[:,E∖M])).

## VERDICT (two-part, honest)

### (A) The "subspace-portfolio breakthrough" does NOT survive
- **`best_r=1` on all 3 Cora seeds** (and Citeseer seed-42). The rank-1 selection
  `portfolio_r1` (≈ the existing `v_ij` score) is the best portfolio; **r=5 and r=10
  are strictly WORSE** (residual B=50: r1 14.30 < r5 15.48 < r10 16.19).
- The proposal conflated **edge-delocalization** (real: v₁ spans ~40–89 edges) with
  **mode-delocalization** (would need σ₁≈σ₂≈…). The spectrum is **seed-dependent, not
  uniformly flat**: σ₂/σ₁ = 0.99 (seed 42), 0.54 (137), 0.45 (271). Rank-1 already
  captures the defense; spreading budget over modes only dilutes it.
- ⇒ The submodular-coverage *portfolio* theorem is not supported. (Coverage is only
  **weakly submodular**: greedy beats top-B by ~2% mean, sets differ 31/36 — neither a
  clean modular "top-B is exact" nor a strong (1−1/e) story.)

### (B) A MODEST active defense DOES survive — and answers reviewers' C-1
- S_c-hardening (= `v_ij` / `portfolio_r1`) at B=50 edges: **−25% worst-case σ₁** and
  **−11% real reconverged damage** (1.83→1.62), **beating all centrality nulls**
  (degree −15%, cfb modest, betweenness ~0%) **and random (~0%)**. This is a positive
  centrality-null comparison on the *defense* side (the reviewers' C-1 complaint).
- **Certified residual floor holds:** residual saturates at σ_{r+1} within FP tolerance
  — i.e. AEGIS certifies a sound post-hardening worst-case bound σ_{r+1}·ε.
- Defense is weak at small budgets (4–12% at B≤20), strengthening only by B=50 —
  consistent with (and a spectral explanation of) the paper's existing delocalization.

## Numbers (Cora, mean / 3 seeds; clean σ₁=19.05)
| method | B=50 residual | σ₁ reduction | real damage @B=20 |
|---|---|---|---|
| portfolio_r1 (≈v_ij) | 14.30 | **25.0%** | **1.62 (−11%)** |
| greedy (ref) | 14.94 | 21.6% | 1.61 |
| v_ij | 15.21 | 20.2% | 1.68 |
| cent_degree | 16.16 | 15.2% | 1.70 |
| random | 18.99 | 0.3% | — |
| none | 19.05 | — | 1.83 |

## Honest pivot (recommendation)
**Downgrade AEGIS-Stackelberg from co-headline "breakthrough" to a supporting result:**
"an S_c-derived hardening defense that beats centrality nulls (answers C-1) with a
certified residual floor σ_{r+1}·ε; the optimal hardening set is the rank-1 v_ij score
itself (no portfolio gain)." **Drop** the submodular-portfolio theorem. The Core
package's new *headline* is therefore **AEGIS-Certify** (validated, strong) + **AEGIS-
Universal** (breadth) + the spine reframe; Stackelberg-lite rides along as a C-1 answer.

## Caveat / untested
Cora-only. The **high-κ regime (WikiCS, κ≈0.93)** is untested — near criticality the
resolvent may concentrate the spectrum (one dominant mode), the one place a portfolio
gain could still appear. Weak prior, but the only open door if we want to revisit the
portfolio angle.
