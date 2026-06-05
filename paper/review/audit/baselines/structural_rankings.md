# Faithfulness Audit — Structural Edge-Ranking Baselines

Scope: the structural edge-ranking controls used against AEGIS — (1) Degree /
degree-proportional, (2) Edge-betweenness centrality, (3) Spectral ranking, and
(4) the Random baseline that serves as the AtkAdv denominator.

Date: 2026-06-05. Auditor: baseline-faithfulness pass.

---

## 1. Implementation locations

| Baseline | File | Function (line) |
|---|---|---|
| Degree-proportional (continuous) | `scripts/exp_attack_baselines.py` | `degree_proportional_attack` (130) |
| Spectral (continuous) | `scripts/exp_attack_baselines.py` | `spectral_heuristic_attack` (153) |
| Edge-betweenness (continuous) | `scripts/exp_attack_baselines.py` | `betweenness_centrality_attack` (177) |
| Random (continuous) | `scripts/exp_attack_baselines.py` | `random_attack` (208), invoked once at 293 |
| AtkAdv-vs-Random column | `scripts/exp_attack_baselines.py` | `run_single` (310–315), CSV col `atk_adv_vs_random` |
| Degree (discrete removal) | `scripts/revision_R2/R2_08_fullgraph_repro.py` | `degree_ranking` (77) |
| Random (discrete removal) | `scripts/revision_R2/R2_08_fullgraph_repro.py` | `main` (114–117), `aegis_over_random` (131) |
| Random (CORRECT reference) | `scripts/exp_greedy_topk_attack.py` | `N_RANDOM_SHUFFLES=5`, averaged (51, 234–244) |

Results artifact: `results/attack_baselines.csv` (150 rows = 3 datasets x 10 seeds
x 5 methods). All 10 preferred seeds present.

---

## 2. What each baseline computes (verified against source)

All four continuous baselines in `exp_attack_baselines.py` share the SAME pattern:
build a per-edge `raw_weights` vector, normalize it to L2 = `eps` (`EPS=0.01`),
then symmetric-fill `dA[i,j]=dA[j,i]=weight`. They all run on the SAME per-seed
artifacts: one 50-node BFS ego-subgraph `A_sub`, one shared `edge_list`, the same
fixed point `Z_sub`, the same `reconverge`. Damage = `||Z_pert - Z_sub||` (301).

- **Degree-proportional** (134–144): `deg = (|A_sub|>1e-10).sum(dim=1)` (degree from
  the nonzero adjacency pattern); per-edge weight = `max(d_i, d_j)`; normalized to
  L2=eps. Endpoint-degree function = **max-endpoint degree**. Standard and sensible.
- **Spectral** (156–168): leading eigenvector `v1` of `A_sub` via
  `torch.linalg.eigh` (largest eigenvalue, index -1); per-edge weight = `v1[i]*v1[j]`
  (leading-eigenvector outer-product magnitude, edge-restricted). This is the
  textbook spectral edge-importance. NOT degenerate: measured damage 0.08–0.26
  (mean), strictly below AEGIS but well above Random — a legitimately weaker attack.
- **Edge-betweenness** (182–193): builds `nx.Graph` from `edge_list`, calls the
  official **`networkx.edge_betweenness_centrality(G, normalized=True)`** (nx 3.4.2),
  maps each edge via `(i,j)` or `(j,i)` key with `.get(key, 0.0)` fallback. This is a
  PROPER edge-betweenness (not node-betweenness misapplied). Disconnected subgraphs
  are handled correctly by nx (per-component shortest paths; isolated edges -> small
  value, fallback 0.0).
- **Random** (208–219): one `torch.randn(len(edge_list))` draw, normalized to L2=eps,
  symmetric-filled. SINGLE draw per (dataset, seed) — see Section 4.

- **`R2_08` `degree_ranking`** (77–86): `deg=(A_hat>0).sum(dim=1)` (unweighted
  degree), edges ranked DESC by `max(deg[i],deg[j])`. Discrete top-k edge-REMOVAL
  ranking (edges zeroed, 89–94). Standard endpoint-degree ranking; correct.
- **`R2_08` Random** (114–117): `rng.shuffle` of the edge list, seed-derived, single
  permutation; `aegis_over_random = d_a / max(d_r,1e-9)` (131). Single-draw.

### Budget / normalization consistency (apples-to-apples): PASS
AEGIS (`aegis_svd_attack`, 122) uses `weights = eps * Vh_c[0]`, and `Vh_c[0]` is a
unit-norm right singular vector, so `||weights||_2 = eps`. Every baseline also
normalizes its `raw_weights` to L2=eps and uses the identical symmetric fill.
Therefore ALL five methods have `||dA||_F = sqrt(2)*eps` (same Frobenius budget,
same edge support). Budget is consistent. Same subgraph, edge_list, fixed point,
and reconverge loop for all methods => same-target evaluation holds.

---

## 3. GAPS / CORRECTNESS table

| # | Issue | Severity | File:line | Fix |
|---|---|---|---|---|
| 1 | Random denominator is a SINGLE draw (one `torch.randn`), not averaged over multiple draws. Per-seed AtkAdv divides by one noisy Gaussian; mean-of-ratios is a biased estimator of E[AEGIS]/E[Random] (Jensen). Random damage CoV across seeds = 20% Cora / 16% Citeseer / 24% WikiCS — material noise injected into the denominator. | MAJOR (for any vs-Random number sourced here) | `exp_attack_baselines.py:208,293,310-313` | Average over >=5 draws inside each seed before forming the ratio, mirroring `exp_greedy_topk_attack.py:236` (`N_RANDOM_SHUFFLES=5`). Then report ratio-of-means or mean of per-seed (mean-random) ratios. |
| 2 | `R2_08` Random uses a single shuffle as denominator (same defect as #1) for the full-graph `aegis_over_random` repro. | MAJOR (within R2_08 only) | `R2_08_fullgraph_repro.py:114-117,131` | Average cumulative damage over >=5 shuffles per (dataset,seed,k). |
| 3 | Internal inconsistency: the repo's OWN flagship script `exp_greedy_topk_attack.py` averages Random over 5 shuffles (the correct method), but `exp_attack_baselines.py` and `R2_08` do not. The Random methodology is not uniform across scripts. | MODERATE | three scripts (see table 1) | Standardize on the 5-shuffle averaged Random everywhere; delete or fix the single-draw paths. |
| 4 | `R2_08` docstring claims "AEGIS vs degree/spectral/betweenness" but only degree + random are implemented; spectral and betweenness are absent. | MINOR (scope/doc) | `R2_08_fullgraph_repro.py:3` vs body | Either add the two missing baselines or correct the docstring to "AEGIS vs degree". |
| 5 | The CSV column is named `atk_adv_vs_random`, but the "+6–148%" paper claim is AEGIS/baseline (per-method), NOT AEGIS/Random. The single column name invites mis-sourcing the headline number to the (defective) Random denominator. | MINOR (naming/provenance) | `exp_attack_baselines.py:313,367` | Also emit per-baseline ratio columns (`aegis_over_degree`, `_spectral`, `_betweenness`) so the "+6–148%" claim is traceable to a stored column. |

No degeneracy found in Spectral; no node-vs-edge betweenness confusion; degree
convention is explicit (max-endpoint). Centralities use the official `networkx`
implementation. Hand-rolled spectral/degree pieces are correct.

---

## 4. Is Random a FAIR denominator? — special note

**Partly. The denominator is correctly NORMALIZED but incorrectly ESTIMATED.**

- Same budget / same support / same target as the attacks: YES. `random_attack`
  is normalized to L2=eps and symmetric-filled exactly like the structured attacks,
  on the same `A_sub`/`edge_list`/`Z_sub` (Section 2). So Random is not biased by a
  different normalization — a common failure mode that is ABSENT here.
- Averaged over multiple draws: **NO.** It is a single draw per seed in BOTH
  `exp_attack_baselines.py` (line 293) and `R2_08` (line 114-117). The CONTRAST
  script `exp_greedy_topk_attack.py` does average (5 shuffles), proving the team
  knows the correct method and that these two scripts deviate from it.
- Consequence: any AtkAdv whose denominator is the `exp_attack_baselines.py`
  Random (or the `R2_08` Random) carries 16–24% denominator noise per seed and a
  mean-of-ratios bias. With 10 seeds the mean is reasonably stabilized, but it is
  not the clean E[AEGIS]/E[Random] the caption implies, and per-seed std is inflated.

**Crucial scoping result (recomputed from `results/attack_baselines.csv`):**
The headline "+6–148% AtkAdv" beating degree/edge-betweenness/spectral
(experiments.tex:54) is computed as **AEGIS_damage / baseline_damage per seed**
(each baseline is its OWN denominator), confirmed by recomputation:
- Degree: +6.2% (Cora), +8.1% (Citeseer), +2.3% (WikiCS)
- Betweenness: +6.5%, +8.7%, +2.4%
- Spectral: +48.6%, +57.1%, +146.9%  -> the "+148%" endpoint
- (the "+6" endpoint = Cora-degree 6.2%)

=> **The single-draw Random does NOT enter the "+6–148%" claim.** That claim is
robust to issues #1/#2. Random only enters the `atk_adv_vs_random` CSV column and
the `tab:cross_domain` AtkAdv (3.2–4.1x). The recomputed single-draw AEGIS/Random
means (3.50/4.23/4.02) do NOT exactly match the table (3.6/4.1/3.8), which indicates
**tab:cross_domain's AtkAdv is sourced from the 5-shuffle-averaged matrix-free
pipeline, not from this single-draw script** — i.e., the PUBLISHED AtkAdv numbers
are on the methodologically sound (averaged) path. This should be confirmed by the
owner of the cross_domain/greedy pipeline, but the evidence points that way.

---

## 5. Verdicts (per baseline)

| Baseline | Verdict | Note |
|---|---|---|
| Degree-proportional (`degree_proportional_attack`) | **FAITHFUL** | Max-endpoint degree, correct L2=eps budget. |
| Edge-betweenness (`betweenness_centrality_attack`) | **FAITHFUL** | Official `nx.edge_betweenness_centrality`, proper edge-betweenness. |
| Spectral (`spectral_heuristic_attack`) | **FAITHFUL** | Leading-eigenvector outer product; non-degenerate. |
| Random (`random_attack`, single draw) | **MINOR-GAPS** | Correct normalization/budget; defective by single-draw estimation (#1). Same-budget so not a biased scale, but a noisy/Jensen-biased estimator of the denominator. |
| `R2_08 degree_ranking` | **FAITHFUL** | Standard max-endpoint degree removal ranking. |
| `R2_08` Random (single shuffle) | **MINOR-GAPS** | Same single-draw defect (#2), repro-only. |

Overall: the three STRUCTURAL centralities are FAITHFUL and standard. The Random
control is MINOR-GAPS (right budget, wrong number of draws), and the gap is
isolated to vs-Random numbers, NOT the degree/betweenness/spectral comparison.

---

## 6. Paper numbers at risk

- **experiments.tex:54 — "+6–148% AtkAdv beats degree, edge-betweenness, spectral":
  NOT AT RISK.** Independently reproduced from the CSV as AEGIS/baseline ratios;
  Random is not the denominator for this claim.
- **tab:cross_domain AtkAdv (3.2–4.1x, AEGIS/random) and experiments.tex:14:
  AT RISK ONLY IF sourced from the single-draw script.** Caption (experiments.tex:18)
  defines AtkAdv = AEGIS/random damage. The numbers do NOT match this script's
  single-draw output (3.50/4.23/4.02), suggesting they come from the 5-shuffle
  averaged pipeline (sound). ACTION: confirm provenance; if any cross_domain AtkAdv
  ever traces to `exp_attack_baselines.py`'s `atk_adv_vs_random`, recompute with
  averaged Random.
- **`results/attack_baselines.csv` `atk_adv_vs_random` column: NOT PUBLICATION-GRADE
  as-is** (single-draw); fine as a diagnostic, must be averaged before any figure/table
  cites it.
- **R2_08 `aegis_over_random` full-graph repro: low-stakes**, but should average
  shuffles before being quoted as a robustness number.

### Recommended single change with highest leverage
Add a `N_RANDOM_DRAWS=5` loop around `random_attack` in
`exp_attack_baselines.py:293` (and `R2_08:114-117`), averaging damage before the
ratio — matching `exp_greedy_topk_attack.py`. This closes #1/#2/#3 at once and
makes every vs-Random number defensible without touching the safe "+6–148%" claim.
