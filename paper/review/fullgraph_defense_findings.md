# Full-graph defense re-run — findings (2026-05-31)

**Question:** does the "42±8% / 61±7%" edge-protection result (subgraph N=50, `R2_13`) hold on the full graph? Reviewers (R2/R4) flagged it lives on 50-node subgraphs.

**Scripts:** `scripts/exp_fullgraph_defense.py` (σ₁ metric, agent-built) and `scripts/exp_fullgraph_defense_disp.py` (R2_13's exact reconverged-displacement metric, ε=0.10). Library fix in `iem/scalable.py` (vectorized edge-list build, 156s→<1s, edge order bit-identical, σ₁ unchanged — benign 150× speedup).

## Result (Cora full, N=2708, |E|=5278, 10 seeds) — cross-validated by TWO metrics

| metric | k=5 top-v | k=10 top-v | random k=5/10 |
|---|---|---|---|
| σ₁(S_c) | 2.5±2.1% | 4.8±3.1% | 0.2% / 0.1% |
| displacement (R2_13 exact) | 2.4±1.8% | 4.6±2.9% | 0.2% / 0.1% |

**Subgraph reference (R2_13):** 42%/61% top-v, 11%/18% random.

## Verified conclusions
1. **The 42%/61% is a small-graph artifact.** Full-graph absolute reduction is ~2.4–4.6% — ~15× smaller. Confirmed by σ₁ AND displacement metrics independently (not a metric confound).
2. **Relative advantage HOLDS / amplifies:** top-v beats random ~12–46× (vs ~4× on subgraphs).
3. **Adaptive erosion is WORSE full-graph:** adaptive−nonadaptive gap = −0.9pp (k=5) / −1.9pp (k=10), vs the subgraph's "within 0.1pp." The adaptive attacker recovers a meaningful fraction of the (small) defense gain — so "survives adaptive recomputation" does **not** hold cleanly full-graph.
4. **Root cause (metric-independent):** the top adversarial mode `v_1` is delocalized — participation ratio 41–89 edges, 15–107 edges for 50–90% of its energy. Masking 5–10 of ~5278 edges excises a negligible slice. k-sweep: reduction reaches subgraph-scale only at k≈100–400 (2–8% of |E|). On the 48-edge subgraph, k=5 was already 10% of edges.

## Recommendation
Do NOT keep "42%/61%" as a headline (abstract). **Reframe the defense around the delocalization finding** — a genuine structural insight: *adversarial vulnerability in GNNs is delocalized over tens of edges, so few-edge protection is fundamentally limited; the per-edge ranking still concentrates risk far better than chance (12–46× vs random).* Report the honest full-graph numbers in §defense_ablation; drop the 42% from abstract + fraud-case-study caption. The experiment is sound (unlike PF) — only the magnitude/framing changes.
