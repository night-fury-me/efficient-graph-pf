# σ₁ regularizer: log vs raw penalty — head-to-head (Cora, seed 42, 150 ep, certify@400)

**Verdict: the two penalties trace the SAME robustness–accuracy frontier.** At
matched σ₁ the accuracy is nearly identical; the penalty form only changes the
λ→σ₁ mapping (log needs ~10× larger λ to reach the same point). **Log did NOT
remove the accuracy cliff — it relocated it.** The tradeoff is intrinsic to the
model/problem, not the parameterization. **Headline = RAW** (better behaved);
**LOG is a confirming ablation** (the frontier is penalty-invariant).

## Matched-σ₁ alignment (read accuracy / cert / damage at equal worst-case σ₁)

| σ₁ level | penalty | λ | acc | cert_frac | atk_dmg | flips | ‖J_z‖₂ |
|---:|:--|---:|---:|---:|---:|---:|---:|
| 334.7 | both | 0.0 | 0.784 | 0.41 | 26.9 | 10 | 0.896 |
| ~175 | LOG | 0.001 | 0.779 | 0.43 | 22.1 | 3 | 0.896 |
| ~84 | LOG | 0.003 | 0.771 | 0.69 | 7.30 | 2 | 0.896 |
| ~32 | RAW | 0.0003 | 0.738 | 0.845 | 3.37 | 3 | 0.890 |
| **~11** | **RAW** | **0.001** | **0.679** | **0.908** | 1.07 | 0 | 0.888 |
| **~11** | **LOG** | **0.01** | **0.682** | **0.835** | 1.19 | 0 | 0.886 |
| ~0.3 | RAW | 0.03 | 0.541 | 0.695 | 0.027 | 0 | 0.863 |
| ~0.3 | LOG | 0.03 | 0.539 | 0.81 | 0.035 | 0 | 0.881 |

**The clean matched pair (σ₁≈11):** RAW acc 0.679 vs LOG acc 0.682 — identical to
within noise, same attack damage (1.07 vs 1.19), and both keep ‖J_z‖₂≈0.887 (genuine
S_c reshaping, not ‖W‖ collapse — consistent with the bug audit). RAW certifies
somewhat more here (0.908 vs 0.835); LOG certifies more at σ₁≈0.3 (0.81 vs 0.70).
The cert_frac gaps are within single-seed/400-sample noise + the margin-collapse
non-monotonicity — neither penalty dominates on certification.

## Why RAW is the headline (despite the identical frontier)

1. **Bounded below.** RAW penalizes σ₁≥0 (a floor). LOG penalizes log σ₁ → −∞, an
   unbounded reward: in the smoke, LOG λ=1.0 drove σ₁→0.02 and **collapsed acc to
   0.22** (CE never fell). LOG only works in a narrow small-λ band; RAW cannot
   blow up this way.
2. **Finer resolution in the usable band.** RAW's grid landed densely at acc
   0.74–0.64 (λ=3e-4…2e-3), giving a clean operating-point choice. LOG's grid
   jumped acc 0.771→0.682 across one λ step (0.003→0.01) — it skips the 0.70–0.75
   band entirely.
3. **No "scale-free" payoff.** The hoped-for gentler descent did not appear — the
   cliff is intrinsic, so log's only effect is to make λ harder to tune.

## Recommended operating points (RAW)

- **λ=0.0003** — acc 0.738 (**−4.6**), σ₁ 31.8 (**10.5× ↓**), cert 0.41→**0.845**,
  attack damage 26.9→3.37 (**8× ↓**), flips 10→3. *The "modest cost" headline.*
- **λ=0.001** — acc 0.679 (−10.5), σ₁ 11.0 (30× ↓), cert **0.908** (peak), **0 flips**.
  *The "kills the attack" point.*

## What goes in the paper

- **Main:** the RAW frontier + λ=0.0003 operating point as the working defense.
- **Supplement / ablation:** log vs raw matched-σ₁ table → "the frontier is
  invariant to penalty parameterization; the accuracy–sensitivity tradeoff is a
  property of the model, not the regularizer" (this *strengthens* the claim — it is
  not a tuning artifact). Note log's unbounded-below collapse as the reason raw is
  preferred.

## Cluster sweep
Queue the **RAW** penalty multi-seed (×10) / multi-dataset (Cora, Citeseer)
sweep at λ∈{0, 3e-4, 1e-3, 3e-3, 1e-2} once the current campaign drains. Do NOT
sweep log (ablation only).
