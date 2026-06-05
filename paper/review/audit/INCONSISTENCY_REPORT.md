# AEGIS paper — contradiction & staleness audit (body + appendix)

Date: 2026-06-05. Canonical build: `aaai_aegis.tex`. Method: three extractor agents built
schema-locked claims ledgers (`ledger_body.md`, `ledger_appendix_theory.md`,
`ledger_appendix_experiments.md`); parent cross-checked shared quantities and verified every
hard finding against the source line. A contradiction = same quantity, two different values.

Legend: 🔴 hard numeric contradiction · 🟠 unsupported/likely-stale, reconcile · 🟡 terminology/cosmetic ·
🔵 clarity (not a contradiction) · ✅ checked, clean.

---

## RESOLUTION LOG (2026-06-05) — A, B, C fixed against the latest data

All three resolved by tracing each contested number to its authoritative data file and using the **most recent** measurement.

- **C ✅ FIXED.** Authoritative source: `results/aegis_conformal_summary.csv` (10-seed, `gate_worstcase_cov` @ε=0.05 = 0.98/0.98/0.97/0.96/1.00/0.98/0.95/0.95 → **0.95–1.00**), which matches `tab:conformal` exactly. Appendix E's "0.92–0.98" (both occurrences, `E_conformal.tex`) and the body prose floor "0.94" were stale → updated to **0.95–1.00** (`experiments.tex` + `E_conformal.tex`).
- **A ✅ FIXED.** The latest IGNN-Cora attack-advantage measurement is `attack_baselines.csv` (May 26; `atk_adv_vs_random` ≈ 3.6–3.8), corroborating `tab:cross_domain` (3.6±.6) and the body's 3.2–4.1×. `tab:explicit`'s 7.6× came from the earlier (May 25) explicit-extension run on a non-canonical IGNN instance (81.6% subgraph acc). → `tab:explicit` IGNN AtkAdv **7.6× → 3.6×** (`F_experiments.tex:306`). NB: now equals GCN-2's 3.6± .6 — a real coincidence, not a copy-paste.
- **B ✅ FIXED.** Latest GAT data: `tau_all_datasets.csv` / `sparse_gat_findings.md` (Jun 5) give GAT-2/Cora unweighted τ=+0.54 (matches the table) and weighted τ=+0.32. The suite table row (1.01 / 2.1× / +.54) is the canonical, latest-corroborated value; the prose's 0.99 / 4.4× / +0.56 / +0.61 (from the May-27 standalone `gat_standard_comparison.txt`) was stale → prose updated to **1.01 / 2.1× / τ +0.54**, and the unweighted-vs-weighted pair **(+0.61 vs +0.56) → (+0.54 vs +0.32)** (`F_experiments.tex:320–329`).

- **D ✅ FIXED.** "~6% accuracy" was unsupported (no unconstrained baseline in any table) and overstated — `tab:explicit` shows IGNN at 77.5% vs the explicit architectures' 76.6–82.2%. Re-anchored to the real, traceable gap: `conclusion.tex:8` "the ε_crit track costs ~6% accuracy" → **"trails the best explicit architecture by ~5 points (`\cref{tab:explicit}`)"** (IGNN 77.5 vs APPNP 82.2 ≈ 4.7).
- **E ✅ FIXED.** The identical [1.19,2.47] band is an *identity*, not a copy-paste: both η and g_W are governed by κ₂(V_W) (under (A4), J_z=Â⊗W with Â symmetric ⇒ nonnormality of J_z = that of W). Stated explicitly at `D_boundary.tex:42` — the κ₂(V_W) certificate "also governs the ReLU slack η (`\cref{rem:eta_relu}`), so the two share this band".
- **F ✅ FIXED.** `experiments.tex:74` "brute-force **N-1 removal**" → "brute-force **single-edge removal**" (now uniform with the other 9 occurrences; power-systems jargon removed).
- **G ✅ FIXED.** `F_experiments.tex:5` stale header comment — "two verified orphan figures (fig_attack_comparison, fig_positioning)" → "attack-comparison figure (fig_attack_comparison; fig_positioning lives in the intro)".
- **H ✅ FIXED.** Unified the GAT naming to the dominant **GAT†**: the two stray "GAT-2†" (fig:tau_heatmap caption `experiments.tex:79`, prose `F_experiments.tex:328`) → "GAT†".

Verification: `latexmk` exit 0, 23 pages, no errors / no undefined references (both new `\cref`s resolve).

All of A–H resolved. Open items from the original audit that were **clarity-only (🔵, not contradictions)** and left as-is by design: I (two τ references — +0.98 vs ~0.5, correctly scoped to different baselines) and J (+0.90 fraud Δτ vs brute-force, not GR-BCD's +0.27).

---

## 🔴 A. IGNN / Cora "AtkAdv" is 3.6× in the body but 7.6× in the appendix
- **Body** `tab:cross_domain` — `sections/experiments.tex:25`: IGNN, Cora, **AtkAdv = 3.6±.6×**.
  Caption (`:18`) defines "AtkAdv is AEGIS/random damage." Body headline (`:14`) reports the
  derived range **"3.2–4.1× attack advantage over random."**
- **Appendix** `tab:explicit` — `sections/appendix/F_experiments.tex:306`: IGNN, Cora,
  **AtkAdv = 7.6±0.5×**. Caption (`:297`) defines "AtkAdv: AEGIS/random damage."
- Identical metric, identical model, identical dataset → 3.6× vs 7.6× (factor 2.1). The appendix
  value also sits **outside** the body's stated 3.2–4.1× range, so a reader moving body→appendix
  hits a direct conflict. Accuracy (77.5±1.7) matches across both tables, which makes this look
  like a number that was re-synced everywhere except AtkAdv.
- **Fix:** rerun/confirm which IGNN-Cora AtkAdv is current; update the stale table. If the two are
  computed on different bases (subgraph vs full-graph, different ε), state that explicitly — right
  now nothing signals a difference.

## 🔴 B. GAT† row: `tab:explicit` table cells disagree with the prose directly below it
- **Table** `F_experiments.tex:311`: GAT† — Tight. **1.01±.01**, AtkAdv **2.1±0.1×**, τ **+.54±.06**.
- **Prose** `F_experiments.tex:318–321` ("GAT† scope"): GAT† — tightness **0.99**, AtkAdv **4.4×**,
  τ **+0.56**.
- AtkAdv 2.1× vs 4.4× (factor ~2) and tightness 1.01 vs 0.99 are the same quantities for the same
  model and conflict. τ is murkier still: table +.54 (caption: *unweighted* v_ij), prose +0.56
  ("AtkAdv 4.4×, τ=+0.56"), and `F:328` says unweighted = **+0.61** vs weighted +0.56 — so the
  table's +.54 matches **neither** prose value.
- **Fix:** pick the current GAT† numbers and make the row and the three prose mentions agree; keep
  the unweighted-vs-edge-weighted τ split explicit (table is unweighted, headline is edge-weighted).

## 🔴 C. AEGIS-Conformal gate at ε=0.05: body says 0.94–1.00, appendix E says 0.92–0.98
- **Body prose** `sections/experiments.tex` (sec:certify): gate "turns conservative
  **(0.94–1.00)** at ε=0.05."
- **Body table** `tab:conformal` ε=0.05 gate column (bold): Cora 0.98/0.98, Citeseer 0.97/0.96,
  Pubmed **1.00**/0.98, WikiCS **0.95**/0.95 → empirical range **0.95–1.00**.
- **Appendix E** `sections/appendix/E_conformal.tex:146` and again `:235`: "the empirical gate is
  conservative **(0.92–0.98 at ε=0.05)**", "0.92–0.98 at ε=0.05, zero equilibrium divergence across
  all 4138 gate nodes."
- Appendix E's 0.92–0.98 matches neither the table (0.95–1.00) nor the body prose (0.94–1.00): its
  upper bound 0.98 contradicts Pubmed's 1.00 and its lower 0.92 contradicts WikiCS's 0.95. **E is
  stale** (predates the current `tab:conformal`).
- Minor sub-issue: body prose lower bound "0.94" is itself 0.01 below the table min (0.95).
- **Fix:** update both E_conformal mentions to 0.95–1.00 (and align the body prose 0.94→0.95).

---

## 🟠 D. "~6% accuracy" cost of the ε_crit track is unsupported by any shown number
- Only at `sections/conclusion.tex:8`: "the ε_crit track costs ~6% accuracy." No table establishes
  it. The c-cap ablation (`F_experiments.tex:236–237`) shows c=0.5→72.1%, c=0.9→80.6%; main
  IGNN-Cora is 77.5%. None of these differences is ~6% against a shown unconstrained baseline.
  (Older drafts said ~5%; the 5% is fully gone, so the number has drifted.)
- **Fix:** cite the establishing measurement (likely 80.6% at c=0.9 vs an unconstrained ~86%, which
  is not currently in any table) or soften/qualify the claim.

## 🟠 E. η and g_W carry the byte-identical band [1.19, 2.47]
- η ("nonnormality slack"): `B_sensitivity.tex:142`, `theory.tex:37`.
- g_W = ‖W‖₂/ρ(W): `D_boundary.tex:41`, `:225`.
- Two distinct constants reported with the same value to 3 sig figs, both labeled "nonnormality."
  The theory bounds both by W's eigenvector conditioning (η≤κ(V_W), g_W≤κ₂(V_W)), so they *may*
  coincide numerically — but as written it reads as a copy-paste/conflation.
- **Fix:** confirm they were measured independently; if they are the same number, say so; if not,
  the coincidence is worth a word so a reviewer doesn't read it as a duplicated cell.

---

## 🟡 F. "N-1 removal" — lone power-systems phrasing
- `sections/experiments.tex:74`: "predicts brute-force **N-1 removal**." Every other occurrence
  (9 total across body + appendix: abstract, case_study ×2, C_rankings ×2, F ×2, theory) says
  **"single-edge removal."** Rename for consistency and to complete the model-auditing reframe.

## 🟡 G. Stale comment in F_experiments.tex
- Header comment `F_experiments.tex:4–5` claims it defines `fig_positioning`; that figure is
  actually defined in `introduction.tex:10–12`. Comment-only (invisible in PDF); fix or delete.

## 🟡 H. GAT naming drift
- "GAT†" (`tab:explicit`) vs "GAT-2†" (`F:328`, `fig:tau_heatmap` caption). Pick one.

---

## 🔵 Clarity risks (correct but conflatable)
- **I. Two headline τ's for "rank agreement."** τ=+0.98 is vs brute-force single-edge removal
  (`fig:tau_heatmap`); τ≈0.35–0.81 / "mean ≈0.5" is vs the GR-BCD optimizer (`tab:baselines`,
  `tab:da_decomp`). Both correct; always label the reference so they aren't read as inconsistent.
- **J. Δτ "+0.90" on fraud** is vs brute-force single-edge removal (`fig:tau_heatmap`); the fraud
  Δτ vs GR-BCD in `tab:da_decomp` is only +0.27 (`F:175`). Never present +0.90 as the GR-BCD value.

---

## ✅ Checked and clean (NOT issues)
- **Power-flow "voltage collapse" (`conclusion.tex:8`)** is the deliberate model-auditing *limitation*,
  paired with `introduction.tex:6` "we audit the model, not the grid's physics." Coherent, not stale.
- **10-seed rule:** clean everywhere; seed list `[42,137,271,314,1729,2718,3141,5772,6561,9999]`
  matches exactly (`F:364`). No seed-42-only / log-penalty ablation lingering.
- **Counts:** 6 datasets / 7 architectures / 4 domains / 420 runs consistent body↔appendix; the old
  "9 datasets / 330 / 390 runs" strings are fully purged (grep: none).
- **Bracket theorem:** ε_crit ≤ ε_break ≤ (C/β)ε_crit with C=g_W(1+κ)/(1−κ) identical in body
  (`theory.tex:43–46`) and appendix (`D_boundary.tex:91–95`). Proven 10–16× vs measured 2–9×
  consistent across body, theory, and `tab:constants`.
- **Headlines consistent:** 74–156×, 99%/44±4%/512-query, −0.65 (10/10 seeds), N=7,650, σ₁ match
  0.03% @N=200, α=0.1.
- **IGNN-Cora accuracy 77.5** consistent across `tab:cross_domain` and `tab:explicit`; the 80.6% is a
  separate c=0.9 robustness-backbone ablation point, not a competing "the" number.
- **Greedy "54–67%"** (attack recovery vs label-aware Greedy, `fig:greedy_topk`) and **"42–61%"**
  (σ₁(S_c) edge-protection defense on a subgraph, `F:158`) are different experiments, not a clash.
- **fig:positioning** is defined (intro), not an orphan.
- **Amazon Fraud** is present and load-bearing in both body (case study) and appendix F — the case
  study and its supporting `tab:baselines`/`tab:da_decomp` rows agree; not dropped.

## Priority order for fixing
1. A, B, C (hard numeric contradictions — a careful reviewer will catch any of them).
2. D, E (reconcile/cite before submission).
3. F, G, H, I, J (cleanup; low risk but cheap).
