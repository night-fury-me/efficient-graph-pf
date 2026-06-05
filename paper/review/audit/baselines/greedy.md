# Baseline audit: label-aware Greedy (reference upper bound)

**Scope.** The sequential edge-removal "Greedy" oracle that AEGIS is measured against in
`fig:greedy_topk`. Greedy is our (near-)oracle reference upper bound, not a published method.
"Faithful" here = a CORRECT, STRONG sequential greedy whose framing matches the paper.

**Date.** 2026-06-05 · **Auditor pass:** verification protocol (checks 1-4).

---

## 1. Implementation locations

| Role | File:line | Objective |
|------|-----------|-----------|
| Greedy that FEEDS the figure | `scripts/exp_greedy_topk_attack.py:173` (`greedy_sequential_removal`) | `‖Z_pert − Z_clean‖₂` (equilibrium shift) |
| Per-method scorer (AEGIS/Degree/Random/Greedy) | `scripts/exp_greedy_topk_attack.py:152` (`measure_cumulative_damage`) | same `‖Z_pert − Z_clean‖₂` |
| AEGIS static ranking compared against it | `scripts/exp_greedy_topk_attack.py:114` (`get_edge_rankings`, S_c column-norm) | S_c, then scored by shift |
| Figure renderer | `scripts/figures/make_fig_greedy_curves.py` (reads `results/greedy_topk_attack.csv`) | — |
| Library brute-force (single-edge, NOT used by this figure) | `iem/adversarial.py:692` (`greedy_structural_attack`) | `‖Z_new − z_clean‖₂` (shift) |

The figure is produced *entirely* by `exp_greedy_topk_attack.py` → CSV → `make_fig_greedy_curves.py`.
`iem/adversarial.py::greedy_structural_attack` is a separate single-edge brute force used by the PF /
tightness / Mettack scripts; it also uses the shift objective and **no labels**. There is **no
flip-based or loss-based greedy anywhere** in the repo (the function directly below it,
`adversarial.py:734`, is `randomized_smoothing_certificate`, unrelated).

---

## 2. The greedy algorithm as implemented (`exp_greedy_topk_attack.py:173-203`)

```
A_current = A_sub.clone();  remaining = all edge indices
for step in 1..k:
    best = argmax over edge in remaining of:
        A_test = A_current with edge (i,j) zeroed (symmetric)
        Z_test = reconverge(model, Z_clean, A_test)      # fixed-point solve
        dmg    = ‖Z_test − Z_clean‖₂                      # <-- objective
    append best to order; remaining.remove(best)
    A_current = A_current with best edge zeroed           # <-- state carried forward
    damages.append(best_damage)
```

### Check 1 — Truly sequential? **YES.**
At every one of the k steps it re-scans *all remaining* candidate edges, re-solves the equilibrium
for each candidate on top of the already-removed edges (`A_current`), takes the argmax marginal
damage (`:191`), commits it, and carries the mutated `A_current` into the next step (`:198-200`).
This is a genuine sequential greedy, **not** a static top-k of a one-shot score. The oracle is **not**
understated on its own metric.

*Sub-note (benign):* each `reconverge` warm-starts from `Z_clean` rather than the running perturbed
state (`:189`). For a contractive IGNN operator the fixed point is unique, so the converged `Z_test`
is init-independent; warm-start affects only iteration count, not the value. This does **not** weaken
the oracle. (Same pattern in `measure_cumulative_damage:166`, so AEGIS and Greedy are treated
identically — no differential bias.)

### Check 2 — Truly "label-aware" / oracle? **NO. This is the core finding.**
The objective is `‖Z_pert − Z_clean‖₂` (`:190`), the **label-free equilibrium-shift** signal.
Labels (`data["y"]`) appear in this file ONLY to *train* the IGNN (`:88, :96, :104`); they are never
read by the attack. Greedy uses **no true labels, no classification loss, no prediction flips**.

This is precisely the signal AEGIS's S_c ranker is built to approximate. So:
- The caption phrase "**label-aware** Greedy ... AEGIS recovers 54-67% ... **with no label access**"
  is **factually wrong**: *neither* method has label access. The contrast it sets up does not exist.
- The script docstring's claim of "no shared optimization pathway" / "closes the circular attack
  evaluation critique" (`:9-11`) is **overstated**: Greedy and AEGIS optimize the *same* objective
  (shift); AEGIS is the cheap closed-form ranker, Greedy the exhaustive search, for one target.

### Check 3 — Same metric AEGIS is scored on? **YES (and this is the saving grace).**
AEGIS, Degree, Random, and Greedy are all scored by the identical `measure_cumulative_damage`
→ `‖Z_pert − Z_clean‖₂`. So the "54-67% of Greedy" ratio is a legitimate apples-to-apples ratio
**on the shift metric**. The number is real; only its *label-aware* labelling is wrong.

### Check 4 — Candidate-set / budget / target parity. **YES.**
All four methods share: the same 50-node ego subgraph (`extract_ego_subgraph(..., max_nodes=50)`,
`:123`), the same `edge_list` candidate set (existing edges of `A_sub`), the same budget
`k = min(10, n_edges)` (`:216`), the same `Z_clean`/`ctx_sub`, the same 10 seeds. Parity holds.

---

## 3. GAPS

| # | Issue | Severity | File:line | Fix |
|---|-------|----------|-----------|-----|
| G1 | Caption calls it "**label-aware** Greedy" and says AEGIS wins "**with no label access**", implying Greedy *uses* labels. The impl uses only the label-free shift `‖ΔZ*‖`. Misframes the oracle and the contrast. | **HIGH** (claim integrity) | `paper/sections/experiments.tex:30`; impl `exp_greedy_topk_attack.py:190` | Rename to "Greedy shift-oracle" / "brute-force equilibrium-shift Greedy"; drop "with no label access" (it does not distinguish the methods). OR, to keep "label-aware", re-implement the objective as true prediction-flip count / cls-loss (a real label-aware oracle) and re-run. |
| G2 | Docstring: "fully black-box ... no shared optimization pathway ... closes the circular attack evaluation critique" — but AEGIS and Greedy share the shift objective. | MED (provenance/over-claim) | `exp_greedy_topk_attack.py:9-11` | Reword: AEGIS = closed-form ranker, Greedy = exhaustive search, *same* shift objective; the non-circular evidence is the *continuous→discrete τ-transfer* fig, not this one. |
| G3 | Greedy is single-target (one ego subgraph / one `Z_clean`), shift-only; it is the strongest greedy *for the shift metric* but is NOT an oracle for end-task harm (accuracy/flips). If any prose elevates it to "worst-case damage", that overclaims. | LOW (interpretation) | `exp_greedy_topk_attack.py:206-232` | Caption already says "ℓ₂ damage" — keep damage framed as shift, never as accuracy. |
| G4 (non-issue, documented) | `reconverge` warm-starts from `Z_clean`, not running state. | NONE | `:166, :189` | No action: unique fixed point ⇒ value-identical; applied symmetrically to all methods. |

---

## 4. VERDICT

**MINOR-GAPS** (strong correct oracle, but **MISLABELED**).

Justification:
- As a *sequential greedy* it is **faithful and strong**: true per-step argmax with re-evaluation
  and committed state, full candidate scan, parity with AEGIS on subgraph/budget/seeds, and scored
  on the *same* metric AEGIS is scored on. The "54-67% of Greedy" / "matches on Citeseer-WikiCS"
  comparison is **methodologically sound on the equilibrium-shift metric** — the headline number is
  trustworthy and is NOT a weak-oracle artefact.
- It is, however, **not label-aware**. It optimizes the label-free `‖ΔZ*‖` — the very signal AEGIS
  uses. The figure caption's "label-aware ... with no label access" framing is incorrect and, if a
  reviewer checks the code, damaging to credibility (it reads as a manufactured asymmetry that isn't
  there). This is a *labelling/framing* defect, not a *strength* defect: the oracle is not weakened,
  so this is MINOR-GAPS rather than WEAK-OR-MISLABELED — but G1 is HIGH-severity and must be fixed
  before submission (one-line caption edit, or upgrade the objective to genuine label-aware flips
  and re-run).

**Paper number at risk:** `fig:greedy_topk` (caption, `experiments.tex:30`) — the "AEGIS matches
label-aware Greedy on Citeseer/WikiCS and recovers **54-67%** of its Cora damage at k=5-10 with no
label access" claim. The *quantitative* 54-67% / match result survives (apples-to-apples shift
ratio). The words **"label-aware"** and **"with no label access"** do not, and must be corrected.
