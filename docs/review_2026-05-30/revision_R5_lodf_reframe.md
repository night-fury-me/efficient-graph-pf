# Revision R5 — "competitive with industry LODF" reframe (DONE)

**Concern (R3 W1):** "competitive with industry LODF" is apples-to-oranges. LODF is exact, closed-form, ~1–2 orders faster; on LODF's own fair target AEGIS is competitive-not-dominant.

**Verified numbers (already in `case_study.tex` — no rerun):**
- P@10 = 0.66–0.81 on case14–118 (`tab:ieee`: 0.74/0.68/0.66/0.81); τ = +0.37 to +0.62.
- LODF τ = 0.44–0.58; AEGIS matches/leads (0.62–0.67 on case57/118, Wilcoxon p<0.01; overlaps on case14/30).
- **On LODF's fair thermal-overload target: AEGIS P@10 = 0.60 (case57), "competitive but not dominant."** ✓ (already conceded, line 40)
- **Binary adjacency beats admittance-weighting: P@10 = 0.81 vs 0.27 (case118)** — the genuine non-obvious finding (line 40). ✓
- Runtime (footnote): LODF <0.13 s; AEGIS 2–23 s → LODF ~15–180× faster. ✓ (R3's "~150×")

**Strategy (REFRAME + CONCEDE-that-builds-the-niche):** position `S_c` as a model-agnostic, label-free diagnostic that audits a *learned* GNN-PF surrogate — recovering N-1 rankings without contingency labels or the admittance matrix exact LODF needs. Complementary to LODF (which needs the physical model), not a rival. Keep the binary-beats-admittance result and the "competitive but not dominant" concession in the body (both already honest in line 40).

**Edits applied:**
- **Abstract:** "…recovers N-1 rankings (P@10 0.66–0.81), **competitive with industry LODF**." → "…recovers N-1 rankings (P@10 0.66–0.81) **from a learned surrogate, without the admittance matrix exact LODF requires.**"
- **`case_study.tex` intro (line 6):** "…(P@10 0.66–0.81), **competitive with industry LODF screening**." → "…(P@10 0.66–0.81) **without access to the admittance matrix that exact LODF screening requires.**"
- Line 40 (Baselines) left intact — it already concedes "competitive but not dominant" and foregrounds binary-beats-admittance.

**Not changed / note:** binary-beats-admittance (0.81 vs 0.27) stays in the Baselines paragraph (clearly stated, italicised). If you want it foregrounded into the case-study intro as the lead result, that's a small follow-up. Builds at 10pp.
