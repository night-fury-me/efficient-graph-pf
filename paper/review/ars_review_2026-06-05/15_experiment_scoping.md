# Experiment Scoping — the 3 needs-data review comments

These are the only review items that cannot be addressed by editing (P1-7, P1-8, P1-10). All text comments
(P0, P1-6/9, P2-11/12/14/15) are done and committed. Below: exact configs, metrics, success criteria, and
the strategic framing for each possible outcome (FIX / DEFEND / REFRAME / CONCEDE).

## Standing protocol (apply to all three)
- **Seeds:** the 10 preferred seeds `[42,137,271,314,1729,2718,3141,5772,6561,9999]`. No fewer-seed result.
- **Model:** IGNN of `eq:ignn_operator`, `W in R^{64x64}` spectral-normalized, ReLU, Adam lr 0.01; spectral cap `c=0.9` (the accuracy recipe) unless noted.
- **Datasets:** Cora, Citeseer, Pubmed, Amazon Photo, WikiCS, Amazon Fraud (public splits).
- **Protocol per experiment:** implement -> critique-inspect for bugs -> verify on one seed -> run all seeds -> write `<name>_findings.md` -> next. One experiment at a time. Debug unexpected/bad results deeply before accepting.
- **Page note:** the body is at 7pp with zero slack; new results land in the **appendix** (figures/tables), with at most a one-clause pointer in the body funded by compression.

---
## EXP-1 (P1-8): Matched spectral-norm / robust-backbone defense baseline
**Answers:** "Is the `sigma_1(S_c)` defense beating generic Lipschitz regularization, or is it just that?" Plus the review's "defense is Cora-only" flag.

**Hypothesis.** Penalizing `sigma_1(S_c)` (the structural-sensitivity operator) yields a better certified-robustness / accuracy frontier than penalizing the generic weight norm `||W||_2`, because it targets the exact operator that governs the perturbation response, not just the layer's Lipschitz constant.

**Design.**
- **Defenses (train each to a matched clean-accuracy grid):**
  1. `sigma_1(S_c)` penalty (ours), `lambda in {0, 3e-4, 1e-3, 3e-3, 1e-2}` (the `tab:defense` grid).
  2. **Spectral-norm-on-`W` penalty** (generic Lipschitz control), matched `lambda` grid retuned to span the same accuracy range.
  3. RobustGCN-lite + GNNGuard-lite under the `c<=0.9` cap (already referenced in `app:explicit`; promote to numbers here).
- **Datasets:** Cora + **Citeseer + Pubmed** (de-Cora-specific the defense; the review's specific complaint). 10 seeds, `c=0.9`.
- **Metrics at each `(defense, lambda)`:** clean accuracy; AEGIS attack damage `||DeltaZ*||` at `eps=0.10`; AEGIS-Conformal certified fraction at `eps=0.05`; GR-BCD damage at a matched budget (the independent-attacker check, shared with EXP-2).
- **Runs:** 3 defenses x 5 lambda x 3 datasets x 10 seeds ~= 450 training runs + eval. IGNN is small/fast.

**Success (and how to frame it).**
- *Win (expected):* at matched accuracy, the `sigma_1(S_c)` penalty gives a higher certified fraction and lower attack damage than the spectral-`W` penalty -> **FIX/DEFEND**: "the coupled operator is the right thing to penalize; generic Lipschitz control is strictly weaker per accuracy point."
- *Tie:* **REFRAME** -> "matches generic Lipschitz regularization on robustness while being the same operator that audits and certifies (which the generic penalty is not)." Do not oversell; the coupling/audit story carries it.
- *Loss:* **CONCEDE** -> demote the defense from a pillar to a corollary; the audit + cheap-conformal contributions stand. (Memory: debug deeply first; a loss here is likely a tuning/bug issue.)
- **Don't undersell:** keep the headline that the *same* operator does all three; the baseline only tests the defense axis in isolation.

**Lands in:** expand `tab:defense` to 3 datasets + the baseline rows (appendix `app:ablations`), one body clause.
**Compute:** ~a few GPU-hours (IGNN training is cheap; the certificate eval is `N=200` dense).

---
## EXP-2 (P1-7): Unification-value ablation (answers the Devil's Advocate's strongest point, DA-M1)
**Answers:** "Is the unification substantive, or just the fact that the same Jacobian appears in three formulas (so the -0.65 anticorrelation is definitional)?"

**Hypothesis.** The coupling is operational, not notational: a *single* training knob (`sigma_1(S_c)` penalty) coherently moves all three capabilities **and an independent adversary's damage**, more efficiently than assembling three off-the-shelf tools. The independent-attacker transfer is the non-definitional part: `sigma_1(S_c)` is not GR-BCD's objective, so if penalizing it also blunts GR-BCD, it captures a real transferable vulnerability axis.

**Design (reuses EXP-1's trained models; adds two measurements).**
- **(a) One knob, three coherent effects.** Over the `sigma_1(S_c)` lambda-sweep, plot {AEGIS attack damage, AEGIS-Conformal certified fraction, **GR-BCD damage (independent attacker)**} vs lambda on >=3 datasets. Claim to establish: all three fall together monotonically. The GR-BCD curve is the key: it shows the knob is not self-referential.
- **(b) One query vs three pipelines (compute + completeness).** Tabulate what you get and what it costs: AEGIS = one rSVD query -> {attack direction, per-edge ranking, per-node radii, certificate input} in ~1 s; the *union* = 50-step PGD attack + randomized smoothing certificate (`~10^4` samples) + a separately-trained defense. Report total wall-clock and which artifacts each path yields (the union's attack names no edge; its certificate names no edge; its defense surfaces no per-edge sensitivity).
- **(c) Efficiency frontier (the decisive plot).** Certified-fraction-vs-accuracy frontier: `sigma_1(S_c)` penalty vs spectral-`W` penalty (from EXP-1). If `sigma_1(S_c)` dominates, the coupled operator is demonstrably the right control variable.

**Success / framing.**
- *Win:* the `sigma_1(S_c)` knob blunts the independent GR-BCD attack and dominates the spectral-`W` frontier -> **the unification is substantive**, a clean rebuttal to DA-M1.
- *Partial (knob helps AEGIS attack + certificate but not GR-BCD):* **REFRAME** -> the value is "all three from one query at `10^2`-`10^4`x below the union's compute," a cost/completeness claim that holds regardless. State honestly that the cross-attack transfer is partial.
- **Don't undersell:** even the weakest outcome keeps the "one closed-form query yields the full triad, the union needs three pipelines" claim, which is real and quantifiable.

**Lands in:** a unification-ablation figure + the compute table in `app:ablations`; one or two sentences in the intro/`sec:defense` rebutting "the coupling is definitional."
**Compute:** marginal over EXP-1 (adds GR-BCD eval on the already-trained models).

---
## EXP-3 (P1-10): SOTA structural-attack head-to-head expansion
**Answers:** "The GR-BCD/PR-BCD comparison is two rows; show the full picture including where AEGIS loses."

**Hypothesis.** AEGIS's label-free one-query proxy recovers most of the gold-standard structural attacker's edge selection in the early-warning (small-budget) regime across datasets, and GR-BCD/PR-BCD dominate raw damage only at the large budgets they target (already conceded; now quantified).

**Design.**
- **Attackers:** GR-BCD and PR-BCD (`geisler2021robustness`, BCD family) on **all 6 datasets**, budget sweep `k in {1,2,5,10,20,50}` (currently only Pubmed `k=10`, Cora `k=5`).
- **Metrics per (dataset, k):** (i) Kendall `tau` between AEGIS's `A_ij v_ij` ranking and GR-BCD's selected-edge set; (ii) raw damage of AEGIS-one-query vs GR-BCD vs PR-BCD (equilibrium damage and prediction flips). 10 seeds.
- **Threat-model positioning table (NO compute, can draft now):** a table contrasting what each certifier/attacker targets, AEGIS (continuous Frobenius `eps`-ball, per-node + per-edge) vs AGNNCert (`li2025agnncert`, bounded *discrete* edit count, majority vote) vs convex relaxation (`zugner2019certifiable`) vs collective smoothing (`schuchardt2021collective`). Makes AEGIS's distinct threat model explicit; this is the cleanest part and is text-only.
- **Runs:** 2 attackers x 6 datasets x 6 budgets x 10 seeds ~= 720 attack runs (PR-BCD at large `k` on Amazon Photo / Pubmed is the slow part).

**Success / framing.**
- Report the early-warning win **and** the large-budget loss in the same plot. **DEFEND** the audit framing (label-free, one query, early-warning) without claiming to beat a label-aware iterative attacker at its own large-budget game. Memory: bulletproof over hand-waving, so show the loss explicitly rather than caveating it in prose.

**Lands in:** expand `tab:baselines` (currently 2 rows) into a full table + a budget-sweep figure, both in `app:baselines`; the threat-model table in `app:baselines`. One body clause.
**Compute:** several GPU-hours (PR-BCD large-budget on the bigger graphs dominates).

---
## Suggested order and shared infrastructure
1. **EXP-1 (defense baseline)** first: it trains the models EXP-2 reuses, and answers the concrete "Cora-only" flag.
2. **EXP-2 (unification value)** next: adds the GR-BCD transfer + compute table on EXP-1's models; this is the highest-stakes (rebuts DA-M1).
3. **EXP-3 (attack expansion)** independently, any time; its **threat-model table is text-only and can be drafted immediately** if you want that part in now.

Each writes a `*_findings.md` before the next. Flag bugs / unexpected results for deep debugging (do not accept a weak defense result without ruling out a tuning bug). All page-budget-affecting additions go to the appendix.
