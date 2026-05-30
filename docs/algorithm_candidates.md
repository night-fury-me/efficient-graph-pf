# Algorithm-box candidates for the AEGIS paper

**Current state:** `grep -rn '\begin{algorithm}' paper/` returns **0 hits**.
A method paper with no boxed pseudocode is a yellow flag at ICML/NeurIPS/TPAMI
("where exactly is the algorithm?"). The framework section currently spends
~1.5 columns of prose ("Stage 1 … Stage 4") describing what is essentially one
procedure — a strong sign that a boxed algorithm would compress and clarify.

Ranked by paper value (impact × space saved × reviewer-defensibility).

---

## Tier 1 — Should appear as formal `Algorithm` boxes in main text

### Algorithm 1 — AEGIS Vulnerability Analysis (matrix-free pipeline)
- **Code:** `iem/scalable.py :: scalable_adversarial_analysis` (and
  `auto_adversarial_analysis` dispatcher); dense fallback is
  `iem/adversarial.py :: full_adversarial_analysis` (L781).
- **Paper home:** Section "Framework" (replaces Stage 1–4 prose).
- **Why it belongs:** it is **the** headline method. Inputs (`F, z*, ctx, ε, k`)
  and outputs (`σ₁, v₁, {v_ij}, {r_v}`) are well-defined; the four stages map
  cleanly to ~20 lines of pseudocode. Currently scattered across four `\textbf{}`
  paragraphs in `framework.tex` lines 17–65.
- **Replaces / compresses:** `framework.tex` subsections "Stage 1" through
  "Stage 4" — net length neutral but far easier to cite (`Alg.~1, line 5`).

### Algorithm 2 — Constrained Sensitivity & Maximally Sensitive Direction
- **Code:** `iem/adversarial.py :: constrained_sensitivity_matrix` (L171)
  + `optimal_structural_attack` (L240).
- **Paper home:** Either inside Algorithm 1 as a sub-procedure, **or** as a
  separate short box right after Prop. 1.
- **Why it belongs:** the symmetric edge-pairing
  `[S_c]_{:,k} = S_{:,iN+j} + S_{:,jN+i}` is the mathematical novelty backing
  Prop. 1 (constrained tightness). Currently described in prose only;
  reviewers checking the proof of Prop. 1 will need to map the math to the
  code, and a 6-line algorithm box closes that gap.
- **Cost:** ~6–10 lines; can be compactly embedded as Algorithm 2.

---

## Tier 2 — Should appear as an Algorithm box in the Case Study

### Algorithm 3 — AEGIS for N-1 Contingency Screening (and brute-force baseline)
- **Code:** `iem/examples/contractive_pf.py :: _iem_n1` (L149) and
  `_brute_force_n1` (L129).
- **Paper home:** `sections/case_study.tex`, alongside the existing
  brute-force-vs-AEGIS prose.
- **Why it belongs:** the case study's central claim is that AEGIS reduces
  N-1 screening from `O(|E|)` full power-flow solves to **one** IFT analysis
  plus `O(|E|)` cheap operator queries. That asymptotic claim *is* an
  algorithmic contribution and should be visible as side-by-side pseudocode
  (brute-force vs. AEGIS) — currently it lives only in prose. Power-systems
  reviewers will want this for direct comparison with classical LODF/N-1
  screeners.
- **Suggested format:** "Algorithm 3a (brute force) / 3b (AEGIS)" side-by-side
  to make the `K full solves` → `1 Jacobian + K cheap probes` contrast
  unmissable.

---

## Tier 3 — Appendix only (do not put in main text)

| Candidate | Code | Verdict |
|---|---|---|
| Per-node robust radius | `adversarial.py::per_node_robust_radius` (L350) | Already a closed-form in Prop:radius. Box is redundant. |
| Phase-transition scan | `adversarial.py::phase_transition_scan` (L570) | Empirical validation procedure, not a contribution. Keep as prose in experiments. |
| Greedy edge-removal | `adversarial.py::greedy_structural_attack` (L658) | Standard top-k brute force from prior literature. A box would draw a reviewer's eye to a non-contribution. |
| Adaptive PGD (IFT) | `exp_adaptive_attack.py::adaptive_pgd_attack` (L51) | Standard PGD adapted to IFT gradients. **One** appendix box is worth it — pre-empts the "did you use a strong enough adaptive attack?" reviewer question. |
| Critical-budget computation | `adversarial.py::critical_perturbation_budget` (L302) | A 1-line formula `(1-ρ)/‖W‖₂`. Not a box; it's an equation. |

---

## Recommended action

1. **Add Algorithm 1 to `framework.tex`** — single box replacing the four
   `\textbf{Stage…}` paragraphs. Highest ROI item.
2. **Add Algorithm 2** as a short box adjacent to Prop. 1 (or as inset inside
   Algorithm 1).
3. **Add side-by-side Algorithms 3a/3b** to `case_study.tex` to make the
   N-1 speedup unambiguous.
4. **One appendix box** for adaptive PGD as a defensive measure.

No new algorithmic content needs to be invented — every candidate above is
already implemented, tested, and referenced in the manuscript. The work is
purely formalization for reviewer legibility.
