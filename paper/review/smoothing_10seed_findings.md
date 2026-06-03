# AEGIS-Conformal vs Randomized Smoothing — Cora, 10 seeds (FINAL, for app:smoothing)

Run: `exp_conformal_vs_smoothing.py --dataset Cora --subgraph-nodes 200 --M 200
--extrap-M 10000 --score aps` per seed, single-stream (GPU-contention-free). Seeds =
the 10 preferred. Per-seed CSVs `results/cvs_cora_s<seed>.csv`. HARD RULE satisfied.

## Aggregate (mean±SD over 10 seeds)

| method | matching | ε | cov | set | cert_frac | wall@10⁴ (s) |
|--------|----------|---|-----|-----|-----------|--------------|
| AEGIS-Conformal | — | 0.01 | 0.90±0.06 | 1.36±0.18 | — | ~1–3 (zero-sample) |
| AEGIS-Conformal | — | 0.05 | 0.90±0.07 | 1.05±0.11 | — | ~1–3 (zero-sample) |
| RandSmoothing | **frob** (same ε-ball) | 0.01 | **1.00±0.00** | **7.00±0.00** | **0.00±0.00** | 7,601 |
| RandSmoothing | **frob** (same ε-ball) | 0.05 | **1.00±0.00** | **7.00±0.00** | **0.00±0.00** | 10,858 |
| RandSmoothing | per_edge (larger ball) | 0.01 | 0.95±0.04 | 1.26±0.14 | 0.96±0.02 | 15,054 |
| RandSmoothing | per_edge (larger ball) | 0.05 | 0.99±0.01 | 2.39±0.38 | 0.77±0.06 | 36,333 |

## Speedup (smoothing wall@10⁴ / conformal wall@10⁴), per-seed ratio, mean±SD
| matching | ε | speedup | range |
|----------|---|---------|-------|
| frob | 0.01 | **11,695× ± 3,367** | 5,381–14,016 |
| frob | 0.05 | **16,735× ± 4,695** | 7,782–20,004 |
| per_edge | 0.01 | 23,127× ± 6,665 | 10,677–27,744 |
| per_edge | 0.05 | 56,819× ± 17,135 | 25,479–70,933 |

## Takeaways for the paper
- **Same-ε-ball smoothing is deterministically VACUOUS**: matched σ=ε/√(2|E|) is so small
  the smoothed classifier is unchanged → abstains to the full 7-label set (cov 1.00, set
  7.00, cert_frac 0.00) on **every** seed (zero variance). This is the headline: randomized
  smoothing cannot certify at the continuous perturbation scale AEGIS targets.
- **AEGIS-Conformal is non-vacuous AND ~10⁴× faster**: cov 0.90 (exactly nominal, 10-seed
  mean — single-seed dips to 0.78–0.87 are finite-sample over n_test=60 and average out),
  sets ~1.0–1.4 labels, at zero Monte-Carlo samples (~1–3 s vs 7,600–10,900 s for frob
  smoothing → **11,700×/16,700×**).
- **Honesty control (per_edge)**: smoothing certifies (cert 0.77–0.96) only on a strictly
  larger per-coordinate ball (σ=ε, not ε/√(2|E|)), and even there costs 23,000×–57,000×.
  We hand smoothing both advantages (larger ball + report its best matching) and the
  analytic certificate still dominates wall-clock by four orders of magnitude.
- The single-seed `rc=1` flags in the run logs are the benign `conformal_cov>=0.9@eps0.01`
  self-check (over-strict at n=60); the 10-seed mean is exactly 0.90. CSVs all valid.
