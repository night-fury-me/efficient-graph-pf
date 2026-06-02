# Regularizer (#1) — multi-seed validation (10 seeds, cluster regsweep)

Raw σ₁(S_c) penalty, Cora, c=0.9 IGNN, 150 epochs, cert_sample=400, seeds
[42,137,271,314,1729,2718,3141,5772,6561,9999]. Source: cluster `regsweep_s*`
→ `results/aegis_regularized_training_raw_s*.csv` (10/10 jobs DONE). Mean±std.

| λ | acc | σ₁(S_c) | cert_frac | attack_dmg | flips |
|---|---|---|---|---|---|
| 0.0    | 0.781±0.018 | 319.3±59.2 | 0.403±0.083 | 27.76±6.31 | 10.8±5.6 |
| 0.0003 | 0.739±0.008 | 32.6±1.7   | 0.823±0.033 | 3.12±0.18  | 2.3±1.3 |
| 0.001  | 0.690±0.007 | 10.7±0.6   | 0.892±0.022 | 1.06±0.07  | 1.0±1.2 |
| 0.003  | 0.619±0.006 | 3.9±0.2    | 0.923±0.015 | 0.38±0.03  | 0.2±0.4 |
| 0.01   | 0.564±0.006 | 0.8±0.1    | 0.864±0.037 | 0.08±0.01  | 0.0±0.0 |

**Verdict: the single-seed headline HOLDS across 10 seeds, tight error bars.**
- σ₁ and attack damage strictly monotone ↓ (319→0.8; 27.8→0.08), small std.
- cert_frac peaks **0.923±0.015 at λ=0.003** then declines — the margin-collapse
  non-monotonicity, now confirmed multi-seed (not single-seed noise).
- **Operating point λ=0.0003: −4.2 acc points (0.781→0.739±0.008) buys 10× lower
  σ₁ (319→32.6±1.7), cert_frac 0.40→0.823±0.033, 9× less attack damage (27.8→3.1),
  flips 10.8→2.3.** Reproducible (acc std ±0.008).
- λ=0.001: σ₁ 10.7±0.6, cert 0.892±0.022, ~1 flip; λ=0.003: ~0 flips.

Seed-42 (the local head-to-head run) was representative: 0.738/31.8/0.845 vs
multiseed 0.739±0.008 / 32.6±1.7 / 0.823±0.033 (within std). **The σ₁-regularizer —
the paper's working defense — is statistically validated.** Raw penalty is the
headline (see [[regularizer_log_vs_raw]]); log is the penalty-invariance ablation.

Cluster note: campaign completed 2026-06-01 07:07 (291 done / 31 failed of 322);
scheduler exited normally. 11 of the failures are stale originals superseded by the
`_rr` requeues (all succeeded); 20 are genuine all-seed failures in `amzfg` + `stack`.
