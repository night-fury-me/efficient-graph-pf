# Audit: did the positioning radar (`fig_positioning_radar`) do justice to the competitors?

Date: 2026-06-04. Method: downloaded and read the five cited competitor papers
(Nettack, Mettack, Geisler GR-/PR-BCD, Schuchardt localized smoothing, Li–Wang AGNNCert)
and adjudicated each radar cell against the source text. Raw papers read in-sandbox;
exact quotes/numbers below.

## Radar scores under audit (0 = centre, 1 = outer ring)

| Axis | AEGIS | Attacks (Nettack/Mettack/GR-BCD) | Certifiers (loc. smoothing/AGNNCert) |
|---|---|---|---|
| Attack direction | 1.0 | 0.9 | 0.0 |
| Per-edge attribution | 1.0 | 0.8 | 0.0 |
| Large-budget damage | 0.5 | 1.0 | 0.0 |
| Per-node certificate | 0.65 | 0.0 | 1.0 |
| Label-free / no-retrain | 1.0 | 0.35 | 0.85 |
| Query efficiency | 1.0 | 0.3 | 0.1 |

## Verdict: NO on several cells. Six concrete problems.

### 1. Per-edge attribution: AEGIS 1.0 > attacks 0.8 is an inversion (HIGH severity)
All three attack papers compute an explicit per-edge attribution over *every* candidate
edge, then select under budget:
- Nettack: score function `s_struct(e)` per candidate, greedy pick of max log-prob change.
- Mettack: meta-gradient `∇_A` is a dense per-edge sensitivity field; explicit `S(u,v)=∇^meta·(−2a_uv+1)`.
- GR-BCD/PR-BCD: exact white-box per-edge gradient (Algo 2, `arg top-Δ(∇L)`); PR-BCD optimises a
  continuous per-edge probability vector `P∈[0,1]` (Algo 1).
The white-box gradient *is* the per-edge attribution ground truth. AEGIS is a label-free **proxy**
for it — and our own Table `tab:baselines` reports the AEGIS `S_c` ranking correlates only
**τ=+0.16 with GR-BCD on Cora** (τ=+0.69 Pubmed). Scoring the proxy above the oracle is internally
inconsistent. Attacks should be ≥ AEGIS here.

### 2. Attack direction: AEGIS 1.0 > attacks 0.9 is a milder inversion (MED)
PR-BCD optimises a continuous gradient direction over edges before discretising (Algo 1, L300–308);
the attacks *define* the attack direction. AEGIS is a continuous proxy, not the gold standard.

### 3. Query-efficiency rationale "50-step / 512-query" is fabricated (MED-HIGH)
The string "512-query" appears in **none** of the attack papers; it is imported from PGD/black-box
folklore. Real costs: Nettack greedy `Δ=d_v0+2` steps with O(1) incremental scoring (cheap);
Mettack zero queries but O(T·N²) with T≈100 unrolled inner-training steps; GR-/PR-BCD **50–500
gradient epochs** (PR-BCD global default 500; Cora 50+250). The 0.3 family score is defensible for
the gradient attacks but the *stated reason is wrong*, and Nettack-greedy is actually cheap.

### 4. Certifier query-efficiency 0.1 libels AGNNCert (HIGH)
AGNNCert is **deterministic, zero Monte-Carlo**: one certified prediction = base GNN on exactly
**T disjoint subgraphs**, with T=80 (node) / T=30 (graph) in their experiments. The rationale
"1e3–1e4 MC samples per node" is a *smoothing* cost that does not apply to AGNNCert at all.
AGNNCert should score ~0.8–0.9 on query efficiency, not 0.1. (Separately, localized smoothing
actually uses **5×10⁵ samples/node** for certification — App C.3 — *above* the stated "1e3–1e4" band.)

### 5. Certifier per-node certificate 1.0 misreads localized smoothing (HIGH)
Localized smoothing is a **collective** robustness certificate (joint over n predictions via an LP) —
literally its title/contribution — not per-node. AGNNCert *is* per-node (Thm 6 bound `M`). Scoring the
shared point 1.0 on a "per-node" axis is wrong for the smoothing half.

### 6. Each family averages methods with opposite profiles (structural, HIGH)
- Certifiers: localized smoothing (probabilistic, collective, 5×10⁵ samples/node, noise-augmented
  training) vs AGNNCert (deterministic, per-node, ~30–80 evals, subgraph-augmented training). Every
  shared cell is wrong for at least one of them.
- Attacks: cheap-greedy Nettack vs expensive gradient PR-BCD differ ~100× in cost.

## Where we were TOO GENEROUS to competitors (over-credit, not injustice)
- **Label-free / no-retrain = 0.85 for certifiers is too high.** Both retrain: localized needs
  noise-augmented base training; AGNNCert "trains the GNN classifier using both the training
  nodes/graphs and their generated subgraphs, whose labels are same as the training" (§4.1). Both
  consume training labels. Honest score ≈0.4 — which *widens* AEGIS's lead.
- **Attacks label-free/no-retrain = 0.35, rationale "poisoning needs labels/surrogate," is false for
  the headline GR-/PR-BCD**, which are **evasion, fixed-θ, no-retrain, surrogate-free** (Geisler L71,
  L73, L2258). They need true labels for the loss but do NOT retrain. The cell conflates two axes;
  "no-retrain" deserves ~1.0 for evasion attacks.

## Citation accuracy (verify)
- `schuchardt2023localized`: the paper read (arXiv 2210.16140) lists third author **Wollschläger**,
  not Scholten, and venue **ICLR 2023** (bib says ICML). Confirm before camera-ready.
- `li2025agnncert`: arXiv **2502.00765**, USENIX Security 2025 — confirmed correct.

## Recommended honest re-scores
- Attacks: per-edge → **1.0**, direction → **0.95–1.0** (concede the attack axes); AEGIS slightly
  below on both (it is a label-free proxy, and that value is already captured by the label-free/query
  axes). Keep large-budget 1.0.
- Query efficiency: keep attacks ~0.3 but rewrite the rationale to real costs (Δ-greedy / O(T·N²) /
  50–500 epochs); drop "512-query."
- Certifiers: **split the vertex** (or pick one representative per axis). At minimum, raise AGNNCert's
  query efficiency to ~0.85, relabel the certificate axis (per-node vs collective), and lower
  label-free/no-retrain to ~0.4.
- Net effect: the honest radar still shows AEGIS as the only method with mass on every axis (its
  real selling point is *coverage*, label-free, one query, not peak dominance on the attack axes),
  and it stops overclaiming exactly where reviewers can check.

## Resolution (applied 2026-06-04)

`fig_positioning_radar.tex` rebuilt as a **7-axis heptagon** under **frontier semantics** (each polygon
traces the best its thread achieves; stated in caption + comment). Changes vs the audited version:
- Conceded the attack axes: attacks now **1.0** on attack direction / per-edge / large-budget; AEGIS
  **0.9 / 0.9 / 0.5** (proxy, with the τ=0.16–0.99 caveat in the caption).
- Split the unscorable "label-free / no-retrain" conjunction into **Label-free** (AEGIS 1.0, certifiers
  0.9, attacks 0.1) and **No retraining** (AEGIS 1.0, evasion attacks 1.0, certifiers 0.2).
- Certifier query efficiency raised to **0.85** (AGNNCert deterministic, T=30–80 evals), no longer
  scored at the smoothing cost; caption notes localized smoothing is *collective* at ~1e5 samples/node.
- Removed the fabricated "50-step/512-query" and false "poisoning needs labels" rationales.
- **Defense axis: considered, then dropped (Option A).** A "model hardening" axis would (i) double-count
  the per-node-certificate axis for the certifier polygon (their robustness and certificate come from
  one mechanism) and (ii) score an axis we don't benchmark head-to-head; AEGIS's σ₁(S_c) defense already
  has its own main-text subsection (`sec:defense`, `tab:defense`). Radar kept to the comparable
  audit/certify/cost axes; caption + prose point to `sec:defense` for the defense pillar.
- Bib `schuchardt2023localized`: author Scholten→**Wollschläger**, venue ICML→**ICLR 2023**, exact title.
- Verified: standalone figure compiles + visually checked; full `aaai_aegis.tex` rebuild clean (no
  undefined refs/cites, `sec:defense` cref resolves, figure on p.20, bib renders "Wollschläger").

Outcome: AEGIS now visibly *loses* 4 of 7 axes to a specialist (and ties a 5th, No retraining) and wins
on coverage alone — the strawman optic is gone and every cell is defensible against a reader who checks
the source papers.
