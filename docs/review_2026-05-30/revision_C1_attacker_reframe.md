# Revision C1 — Re-aim the attacker claim (DONE, data-grounded)

**Concern (CRITICAL; Devil's Advocate + Domain).** The headline "one-query SVD matches/dominates a 50-step PGD" leaned on (a) **Shift-PGD**, which the paper itself labels "solver validation, not an independent baseline" (it uses AEGIS's own IFT gradients on the equilibrium-shift objective the SVD direction maximises *by construction*); (b) **equilibrium shift** as the metric, while prediction flips are tiny; and an independent scalable attacker (GR-BCD) can beat AEGIS at larger budgets. Verb "dominates" overclaimed.

**Strategy (honest, not undersold):** REFRAME the axis from "best attacker" → "one-query *diagnostic* that recovers the same first-order direction, non-circularly, at a fraction of the cost"; FIX with a cost-normalized axis + flip reporting computed from existing data; CONCEDE (costlessly) that it is not a peak attacker.

## Evidence computed from `results/full_attack_table.csv` (90 rows; no GPU rerun)

| Dataset (ε=0.10) | SVD dmg | Cls-PGD dmg | Shift-PGD/SVD | flips (max, any method) | **SVD damage-per-query vs Cls-PGD** |
|---|---|---|---|---|---|
| Cora | 3.70 | 2.51 | 0.82 | 1.8% | **73.7×** |
| Citeseer | 4.63 | 2.97 | 0.73 | 1.4% | **78.0×** |
| WikiCS | 2.10 | 0.67 | 0.92 | 0.6% | **156.4×** |

- **Cost-per-query (the honest "one-query" value, hardware-independent):** SVD = 1 query; PGD = 50 steps. SVD delivers **74–156×** the equilibrium damage *per query*.
- **Flips are tiny for every method** (0–1.8% at ε=0.10 on these datasets) → a regime property, not an AEGIS weakness. (Paper's breach-rate figure already shows higher flips at ε=0.20 / Pubmed.)
- **Regime check (refutes "wins only at tiny ε"):** mean Cls-PGD/SVD damage ratio = 0.65 (ε=0.01) → 0.60 (ε=0.05) → 0.55 (ε=0.10). The SVD lead on equilibrium shift **does not shrink** across the tested range; it slightly widens. So the lead is *not* a tiny-ε artifact — statable honestly.
- **Non-circular evidence already in the body (now foregrounded):** gradient-independent transfer recovers 99% (cos=0.99); 512-query black-box only 44±4%. This is the answer to "circular gradient artifact," so the reframe leads with it.

## Edits applied

**Abstract** (`sections/abstract.tex`) — length-neutral (9→9 words):
- *Before:* "…while the one-query SVD direction **matches 50-step PGD attackers**."
- *After:* "…while one query **recovers the direction 50-step PGD finds**."

**§Four-Quadrant** (`sections/experiments.tex`) — paragraph rewritten:
- Removed the verb "**dominates**" and the closing "a one-query guarantee that empirically **matches iterative attackers**."
- Now: leads with the *by-construction* concession (Prop. attack) → non-circular evidence (99% / 44%) → **cost axis (74–156× damage per query; Cls-PGD needs 50× the cost; Shift-PGD = solver validation)** → flips 0–1.8% for all methods → explicit concession: "a one-query **diagnostic** … not a peak attacker; budget-heavy attackers (GR-BCD, `tab:baselines`) can exceed it at larger budgets."

## Not changed / notes
- `tab:attack_full` left intact (prose carries the new cost axis; avoids table reflow at 10pp).
- Line 36 "beats Mettack/heuristics" claim is already *qualified* to the early-warning k∈{1..5} small-budget regime — consistent with the new "GR-BCD exceeds at larger budgets" concession; left as-is.
- **Optional future bulletproofing (needs GPU rerun, not required):** add per-method wall-clock/FLOPs to `exp_full_attack_table.py` for a runtime cost axis. The per-query axis above is already hardware-independent and sufficient.
