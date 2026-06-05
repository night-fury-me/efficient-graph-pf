# Baseline Audit — "512-query black-box search"

**Auditor verdict: WEAK-BASELINE (chosen-weak / strawman). Confirms R2 §2a.**

## 1. File location

- **Implementation:** `scripts/exp_x7_independent_attack.py` (Exp X7, "AEGIS one-query
  direction vs genuinely INDEPENDENT attackers"). 257 lines.
- **Paper claim at stake:** `paper/sections/experiments.tex:35` — *"a separately trained
  surrogate (no shared gradients) recovers 99% of the one-query equilibrium damage
  (cos=0.99), versus 44±4% for a 512-query black-box search."* (R2 cites this as "L43";
  same sentence, stale line number — the live location is L35.)
- **Result artifact:** `results/x7_independent_attack.csv` (column `dmg_blackbox`,
  `blackbox_frac`).
- **Methodology review:** `paper/review/R2_methodology.md` §2a [MAJOR] and Issue Register
  **M1**.

## 2. What the algorithm ACTUALLY is (file:line)

It is **pure i.i.d. random search ("Black-box RS"): best-of-M random unit directions.**
Not NES, not SimBA, not Bandits-TD, not Square-Attack — no principled black-box optimizer.

The entire attacker is this loop:

```python
# exp_x7_independent_attack.py:156-165
g = torch.Generator(device=A_sub.device).manual_seed(seed * 7919 + 1)   # L157
dmg_bb, flips_bb = -1.0, 0                                               # L158
for _ in range(M):                          # M = 512  (defined L64)     # L159
    v = torch.randn(n_edges, generator=g, device=A_sub.device)          # L160  i.i.d. Gaussian
    v = v / v.norm() * EPS                                              # L161  normalise to eps-ball
    A_v = apply_perturbation(A_sub, edge_list, v)                       # L162
    d, fl = measure_attack(target, Z_sub, ctx_sub, A_v, preds_clean)    # L163  REAL reconverged shift
    if d > dmg_bb:                                                      # L164  keep arg-max
        dmg_bb, flips_bb = d, fl                                        # L165
```

Self-described in the docstring (L9-10): *"Black-box RS : M queries -- best-of-M random
unit directions ... no gradients, no S_c."* and the constant comment (L64):
`M = 512  # black-box random-search query budget (strong independent baseline)` — the
"strong" label is the author's, not earned by the algorithm.

### Is the 512-query budget used efficiently? NO.

- **Non-adaptive / zero feedback.** The PRNG `g` is seeded ONCE (L157) and is never
  re-seeded, biased, or warm-started from the observed damage `d`. The 512 draws are
  fully **i.i.d.**; the measured shift `d` only ever updates the running max (L164-165),
  never the *sampling distribution*. This is exactly "512 i.i.d. random tries" — no
  coordinate descent, no gradient estimation, no proposal adaptation. A genuine
  query-based attacker (NES/SimBA/Bandits/Square) reinvests each query's feedback to
  steer the next; this baseline does not.
- **Sampling geometry is the worst case for the claim.** Directions are uniform on the
  full `|E|`-dim sphere (`randn` → normalize, L160-161). The SVD attack direction is
  ~1-dimensional in an `|E|`-dim edge space, so the expected squared overlap of a random
  unit vector with it is ~`1/|E|`, and best-of-512 still concentrates near a small
  fraction of the optimum. The 44% is essentially the order-statistic of 512 uniform
  spherical draws projected onto a rank-1 optimum — a property of *random search in high
  dimension*, **not** evidence that the direction is query-hard.

### How "recovers 44%" is measured.

- Per seed: `blackbox_frac = dmg_bb / max(dmg_aegis, 1e-12)` (L185), where `dmg_aegis` is
  the **single** one-query SVD attack `eps * Vh[0]` (L148-150) and `dmg_bb` is the
  best-of-512 random damage. Both use the identical `apply_perturbation` →
  `measure_attack` path on the SAME 50-node ego-subgraph (L117-118) and the SAME
  `edge_list` coordinate space (L142-143 asserts alignment). Metric = REAL reconverged
  equilibrium shift `||Z_pert* - Z_clean*||` (non-circular). Mean over 10 seeds × datasets
  (L242) ⇒ the reported **44±4%**.
- Note: the paper's "cos=0.99 / 99%" half of the sentence is the *surrogate-transfer*
  attacker (X7's attacker #3, L167-176, best of surrogate-SVD± and surrogate-Cls-PGD),
  which is a credible independent attacker. Only the **512-query** half is weak.

## 3. GAPS / STRENGTH table

| Issue | Severity | file:line | Fix |
|---|---|---|---|
| Attacker is plain i.i.d. random search, not a principled query optimizer | MAJOR | `exp_x7_independent_attack.py:156-165` | Replace/augment with NES, SimBA, or Square-Attack-on-edge-simplex at the same 512-query budget |
| 512 queries used non-adaptively (PRNG seeded once L157; feedback `d` never steers sampling) | MAJOR | `exp_x7_independent_attack.py:157,159-165` | Use an estimator that reinvests each query (NES grad-estimate; SimBA coord flips; Square local moves) |
| Uniform full-sphere sampling vs rank-1 optimum ⇒ 44% is a dimensionality artifact, not query-hardness | MAJOR | `exp_x7_independent_attack.py:160-161` | A coordinate/greedy or structured proposal exploits sparsity of the optimum; report that number |
| Label "strong independent baseline" unjustified by the code | MINOR (framing) | `exp_x7_independent_attack.py:64`; docstring L9-10 | Drop "strong"; either upgrade the attacker or rename it "random-search lower bound" |
| Paper presents 44% rhetorically as "hard to find by query access" with no algorithm named | MAJOR | `experiments.tex:35` | Name the algorithm; report 44% against ≥1 principled black-box optimizer at 512 queries |
| Search confined to fixed 50-node ego-subgraph (shared with all X7 attackers) | MINOR (scope, shared) | `exp_x7_independent_attack.py:117-118` | Same subgraph-faithfulness caveat as rest of attack table (τ≈0.16); not unique to this baseline |
| **Strength (in favor):** budget-matched, same apply/measure path, REAL non-circular metric, 10 preferred seeds, 5 datasets | — (positive) | `exp_x7_independent_attack.py:16-18,66,142-143,185` | Keep; only the *attacker class* is weak, the harness is sound |

## 4. VERDICT

**WEAK-BASELINE (chosen-weak).** The "512-query black-box search" is **best-of-512 i.i.d.
uniform-random directions** (`exp_x7_independent_attack.py:156-165`) with **no adaptive
exploitation of query feedback**. The contrast in `experiments.tex:35` relies on this 44%
to argue the SVD direction "is hard to find by pure query access," but random search in an
`|E|`-dimensional space against a rank-1 optimum will *always* score low regardless of
direction-hardness — so the number measures the weakness of the attacker, not the security
of the method. R2 §2a [MAJOR] and Register M1 are **CONFIRMED**, not refuted. The harness
(budget match, real reconverged-shift metric, 10-seed × 5-dataset, aligned edge space) is
methodologically clean; the defect is solely the choice of attacker.

This is a **fairness defect, not a results defect** — and likely a *self-defeating* one:
the companion surrogate-transfer attacker (a stronger, gradient-free opponent) already
reaches 99% (cos=0.99), so a query optimizer landing between 44% and 99% would still leave
AEGIS's single query competitive. Upgrading the baseline most likely **strengthens** the
narrative.

## 5. Recommended stronger attacker

At the **same 512-query budget**, on the eps-Frobenius edge ball, report `dmg_bb` against at
least one of (in priority order):

1. **SimBA-style** orthogonal/coordinate query attack on the `edge_list` basis — cheapest
   drop-in: replace `v = randn` (L160) with sequential ± steps along sampled edge
   coordinates, keeping the move iff `measure_attack` increases. Adaptive, gradient-free,
   ideal for the sparse rank-1 optimum.
2. **NES** (antithetic Gaussian gradient estimate) — estimate `∇_δ ||Z_pert*−Z_clean*||`
   from ~`512/(2k)` query pairs, ascend on the eps-sphere. The canonical "principled
   black-box optimizer" R2 names.
3. **Square-Attack analogue on the edge simplex** — localized random square/edge-block
   updates accepted on improvement; strong, parallelizable, query-efficient.
4. (Optional) **Bandits-TD** with a topology/data-dependent prior — strongest but heaviest.

Minimum to satisfy the reviewer: add ONE of #1/#2 as a fourth column / variant in
`exp_x7_independent_attack.py` and update `experiments.tex:35` to name it.

## 6. Paper-number-at-risk

- **`44±4%`** — `experiments.tex:35` (the "512-query black-box search" figure). This is the
  number a reviewer will challenge as a strawman; it must be re-reported against a
  principled black-box optimizer at 512 queries or the sentence rewritten to stop implying
  query-hardness.
- Secondary exposure: the rhetorical force of the surrounding **"One query, not a search"**
  paragraph (`experiments.tex:35`) leans on this contrast. The `99% / cos=0.99` transfer
  number is NOT at risk (credible attacker). `tab:attack_full` "Random" column is the same
  family and inherits the caveat.
