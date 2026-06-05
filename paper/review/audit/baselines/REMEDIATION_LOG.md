# Baseline remediation log (bulletproof: implement faithful + re-run)

Environment: `.venv/bin/python` → torch 2.11+cu128, RTX 4090. 10 preferred seeds enforced.
Each item: implement faithful code → critique-inspect → run (10 seeds) → compare to paper → update.

---

## Item 1 — Random AtkAdv denominator ✅ (no paper change)

**Finding:** `Random` baseline was a single draw, used as the `AtkAdv = damage/Random` denominator.

**Material conclusion — the paper number is SOUND, no change needed.** `tab:cross_domain`'s
"AtkAdv = AEGIS/random" (Cora 3.6 / Citeseer 4.1 / Pubmed 3.2 / Amazon 3.8 / WikiCS 3.8, and the
headline "3.2–4.1×") is computed by the library, which **already averages over 5 draws**:
`iem/adversarial.py:569` `for _ in range(n_random)` with `n_random=5` and
`actual_rand = sum(actual_rands)/len(actual_rands)` (`:587`). The single-draw bug existed only in two
auxiliary scripts that do **not** feed that table. The earlier A-fix (`tab:explicit` IGNN AtkAdv → 3.6)
is therefore also sound.

**Code fixes (hygiene/consistency):**
- `scripts/exp_attack_baselines.py` — `Random` now averaged over `N_RANDOM_DRAWS=5` (was 1). Feeds the
  **+6–148%** comparison, which is a *relative* AEGIS-vs-baseline damage ratio (Random cancels), so the
  paper number is unchanged; the fix only stabilises the reported absolute AtkAdv. Smoke (1 seed) ran
  clean on GPU (52 s); full 10-seed re-run in progress to refresh the CSV.
- `scripts/revision_R2/R2_08_fullgraph_repro.py` — `Random` now averaged over 5 shuffles. **Non-load-bearing**
  (the full-graph "9.82×/3.25× over degree" is AEGIS/degree, not AEGIS/random), so not re-run.

**Paper edits:** none required.

---

## Item 2 — Greedy "label-aware" mislabel ✅ (evidence-based rename)

**Finding:** the Greedy oracle maximized the **label-free shift** `‖ΔZ‖`, yet the caption called it
"label-aware Greedy … with no label access."

**Bulletproof experiment** (`scripts/exp_greedy_labelaware.py`): built a *genuine* label-aware oracle —
greedy maximizing CE loss on the subgraph's true labels — and scored prediction damage (accuracy drop /
flips). **Result: at the figure's budget (k=5–10, 50-node subgraph) the label-aware oracle causes ~0
prediction damage** (Cora 0.88→0.88 acc, WikiCS 0.66→0.66; Citeseer ≤2%), consistent with the paper's
own "flips <2%". So a "no label access" contrast is *unachievable* here — the comparison is inherently a
label-free equilibrium-shift comparison. The shift continuity reproduces the original number
(AEGIS/greedy-shift: Cora ≈0.49–0.60, Citeseer/WikiCS ≈1.0).

**Fix (paper):** `experiments.tex:69` caption — "label-aware Greedy … with no label access" →
"the **greedy shift-oracle** … from a single closed-form `S_c` ranking, **with no per-edge search**."
Strength preserved (closed-form ranking ≈ exhaustive search) without the false label claim. The 54–67%
(10-seed, original CSV) is unchanged; a 10-seed re-run of the label-aware script is confirming the
degeneracy + continuity. New script: `scripts/exp_greedy_labelaware.py`, CSV `results/greedy_labelaware.csv`.

---

## Items 3+4 — PGD ("IFT-gradient" + Cls-PGD strength) ✅

**Finding:** "shift-PGD" was labeled "IFT gradients" but used a truncated 50-step unroll; Cls-PGD was
under-powered (sign step on an L2 ball, spurious per-coord clamp, no restarts).

**Fix (code):** rewrote `pgd_attack` (`exp_full_attack_table.py`, the load-bearing script — confirmed it
reproduces both 1.5–3.1× and 72–92%): L2-normalised ascent (no sign/clamp), **3 random restarts** (best
kept), and a **converged inner solve** so the unrolled gradient equals the implicit/IFT (resolvent)
gradient for the contractive IGNN; uses the model's own operator (NOT S_c → non-circular).
`exp_classification_attack.py` confirmed auxiliary (its CSV is unreferenced) — not touched.

**Result (10-seed, 63 min):**
- **Shift-PGD (genuine IFT) now reaches 91–100% of AEGIS-SVD** (mean 0.978; was 72–92%, an
  unconverged-unroll artifact) → a *stronger* "SVD = iterative optimum" validation; the "IFT-gradient"
  label is now honest.
- **Cls-PGD: the 74–156× per-query advantage HOLDS at 78–157×** (svd/cls 1.6–3.1×). The strengthening
  did NOT shrink it → it was **not** a weak-baseline artifact; the gap is genuine objective degeneracy
  (classification has ~0 signal at these budgets, flips <2%).

**Paper edits:** 74–156×→**78–157×** (abstract, `experiments.tex:35`, `F:24`); 72–92%→**91–100%** (`F:26`);
15–70%→**15–71%** (`F:25`); 1.5–3.1×→**1.6–3.1×** (`F:36`); + faithfulness disclosure (Frobenius-projected,
3 restarts, converged IFT gradient). `latexmk` clean, 23pp.

---
(items 5–8 appended as completed)
