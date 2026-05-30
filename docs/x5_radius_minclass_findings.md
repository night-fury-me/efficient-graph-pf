# X5 — Min-over-classes radius recompute: findings

**Closes:** T3 / corrected `prop:radius`.
**Date:** 2026-05-30 · **Script:** `scripts/exp_radius_minclass.py` · **Data:** `results/exp_radius_minclass.csv`
**Setup:** Cora + Citeseer, 50-node BFS subgraph, IGNN (κ=0.9), 10 seeds each, ε∈{0.05,0.10,0.20}.

---

## 1. The gap X5 targets

`prop:radius` (corrected, T3) is the **min-over-classes composed-norm** radius
`r_v = min_{c≠y_v} m_v^{(c)} / ‖(W_{y_v}−W_c) S_v‖₂`, but the **implementation**
(`iem/adversarial.py:per_node_robust_radius`) and **Algorithm 1** (`framework.tex:27`) still use
the **runner-up surrogate** `r̂_v = m_v / (‖W_{y_v}−W_{c*}‖₂ · ‖S_v‖₂)` (runner-up only, *product*
norm). X5 recomputes both (prediction-based), reports the shift, and re-checks the breach
guarantee under the corrected radius.

## 2. Result — the corrected radius is ~3× LARGER, and still valid

| dataset | median r_v surrogate → minclass | mean r_v | total breaches | false-safe (sur / min) |
|---------|-------------------------------|----------|----------------|------------------------|
| Cora | 0.0331 → **0.0926** (+180%) | 0.0448 → 0.1189 | 34 | **0 / 0** |
| Citeseer | 0.0413 → **0.1354** (+228%) | 0.0497 → 0.1491 | 15 | **0 / 0** |

- **~3× larger.** The composed norm `‖(W_p−W_c)S_v‖ ≤ ‖W_p−W_c‖·‖S_v‖` (Cauchy–Schwarz) is ~⅓ of the
  product on average here, so the surrogate **under-reported** r_v. The product-norm conservatism
  dominates the runner-up-swap "optimism" the theory flags (which bites only at specific nodes).
- **Breach guarantee survives.** Across all 20 seed-runs × 3 ε, **false-safe = 0** for *both* radii:
  every node that actually flipped had ε ≥ its own r_v. The larger radius does **not** over-claim —
  safe nodes get bigger radii, but vulnerable (breaching) nodes still get tiny radii < ε. So the
  corrected radius **discriminates better** (large for safe, small for vulnerable).

## 3. Interpretation — this is a strengthening, not a liability

The corrected radius is the *accurate* first-order distance-to-boundary; the surrogate was a loose
(mostly conservative) approximation. A larger, still-valid certificate is unambiguously better:
- Paper r_v numbers (surrogate-based) **under-state** the certified radius by ~3×.
- The AGNNCert comparison improves: `r_cert/r_v` tightens from the reported ~[4.9, 10.2] toward ~⅓ of
  that (≈[1.6, 3.4]) — AEGIS's first-order radius sits **closer** to the sound IBP certificate.

## 4. Caveats (honest)
- The breach false-safe test probes the **S_c-optimal direction** (the paper's own breach methodology),
  not all perturbations; the first-order *guarantee* itself is `prop:radius`. false-safe=0 is supporting
  evidence, consistent with how the paper validates `r_v`.
- These are subgraph medians at κ=0.9; they do **not** reproduce the paper's exact reported figures
  (e.g. "Cora med. r_v = 0.187" in the AGNNCert row, or "0.046 at σ=0.05") which use their own
  contexts. The robust, context-independent finding is the **relative** one: corrected ≈ 3× surrogate,
  breach guarantee intact.

## 5. Verification (protocol step 3)
- composed ≤ product (Cauchy–Schwarz) holds per node by construction (ratio ~⅓ measured).
- false-safe = 0 across 20 seed-runs × 3 ε for both surrogate and min-over-classes radius.
- prediction-based (argmax) margins → all positive, theory-correct (radius preserves the argmax).

## 6. Files of record
- `scripts/exp_radius_minclass.py` — recompute (both radii) + breach false-safe re-validation
- `results/exp_radius_minclass.csv`

## 7. Paper integration
**Consistency fix DONE (2026-05-30, option a):**
1. ✅ `iem/adversarial.py:per_node_robust_radius` rewritten to the min-over-classes composed-norm form (now the default; `runner_up_surrogate=True` reproduces the old form). Verified to match this experiment's `both_radii` to **0.00e+00** (surrogate to 7e-9).
2. ✅ **Algorithm 1** (`framework.tex:27`) updated to `r_v ← min_{c≠y_v} m_v^{(c)}/‖(W_{y_v}−W_c)S_{c,v}‖`; paper rebuilds clean at 10 pp.

**Deferred (option 2 — number refresh):** the reported r_v figures (med r_v, `r_cert/r_v`, 0.046/0.09) are surrogate-based and are now **conservative under-estimates**. Any radius experiment re-run now yields the corrected (~3× larger) values **by default**; pass `runner_up_surrogate=True` to reproduce the paper's exact figures. Refreshing them in-context (a strengthening: larger radius, AGNNCert ratio tightens from ~7.6× toward ~2.5×) remains a go/no-go.

## 8. Status
**T3/X5 closed.** Corrected radius ≈ 3× larger, breach guarantee holds (false-safe=0). **Consistency fix applied** — impl + Algorithm 1 now match `prop:radius` (verified, build clean at 10 pp). The number refresh is deferred; until then the code default produces the corrected values and `runner_up_surrogate=True` reproduces the paper's reported figures.
