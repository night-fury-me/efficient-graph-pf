# AEGIS Paper Reorganization Plan

## Context

The AEGIS paper (AAAI-27, 7 body pages + unlimited appendix; canonical file `paper/aaai_aegis.tex`, which `\input`s `paper/sections/*.tex`) compiles cleanly but does not read as one coherent story. Diagnosed from the actual page geometry (via `.aux` newlabel page map + `pdfinfo`):

- **Two competing structures fight each other.** The method/theory is organized one way, with the *certify* leg (AEGIS-Conformal, `sec:conformal`) and the *defend* leg (`sec:defense`) buried **inside** the Theory section, while Experiments uses a *separate* 4-claim layout (contractivity / one-query attack / transfer / fraud). The abstract promises "one object → audits, certifies, defends," but the body never uses that spine as its organizing principle. This double structure is the root of the "no coherent story" complaint.
- **Theory is ~3 of 7 body pages** (spans p2→p5); **Experiments is only ~2** (p5→p7). Theory is statement→proof with little intuition (dry); Experiments shows only ~1 table + 2 plots in-body.
- **The method algorithm `alg:aegis` and nearly every empirical exhibit live in the 10-page appendix** (p9–18): the smoothing head-to-head, the baseline comparison, the four-quadrant attack table, per-architecture detail. The *defend* leg has **no table anywhere**.
- 106 over/underfull boxes — the body is packed to the margins, which is *why* material got exiled.

**Decisions (confirmed with the author):** (1) **Full restructure** around one audit/certify/defend spine. (2) **Surface existing results only** — pure reorganization, zero new compute; the 10-seed rule means any number promoted into the body must already be 10-seed-validated. (3) The two-sided bracket theorem (`thm:cf2s`) becomes a **supporting callout** — the S_c tool stays the headline, the bracket keeps a one-line numbered presence with its full statement/proof in the appendix.

**Intended outcome:** the same three words — **audit / certify / defend** — run through the intro bullets, the operator interface, the theory, and the experiments, each leg carrying visible in-body empirical weight; theory compressed ~3pp→~1.3pp (with the operator/algorithm content moved up into a preceding Construction section); the method algorithm and key tables promoted into the body; net body ≤7pp.

**Construction before Theory (author preference).** The body presents the AEGIS *operator/construction* (§3) **before** the *theorems* (§4): a tool paper reads better method-first — define what $S_c$ is and how one query yields the three diagnostics, then prove the guarantees (safety boundary, certified radius, conformal coverage) that justify them. This also removes a defect in the current draft, where the Theory section invokes $S_c$ and its construction before the operator/algorithm are ever defined.

---

## Target body structure (budget targets; reconciled to ≤7pp in Phase 3)

| # | Section (file) | Purpose | Budget |
|---|---|---|---|
| 1 | **Introduction** (`introduction.tex`) | Keep; retune contribution bullet (2) so the *tool* leads and the bracket reads as supporting | 1.0 |
| 2 | **Background & Threat Model** (`background.tex`) | IGNN/IFT setup; symmetric, edge-only, continuous `‖δÂ‖_F≤ε` (soft buffer: can trim to 0.5) | 0.6 |
| 3 | **The AEGIS Operator: One Object, Three Diagnostics** (`framework.tex`) | Define `S_c` + `P_c`; matrix-free routing; **`alg:aegis` promoted in-body**; `fig:pipeline`; the boxed "one query → attack / ranking / radius" reading | **1.2** |
| 4 | **Theory: Safety Boundary & Certificates** (`theory.tex`) | `phase_transition` (formal) + `radius` (formal) + the AEGIS-Conformal score-shift equation; bracket callout + transfer/explicit demoted | **1.3** |
| 5 | **Experiments: Audit / Certify / Defend** (`experiments.tex`) | Three subsections mirroring the spine, each anchored by an in-body exhibit | **2.3** |
| 6 | **Case Study: Fraud** (`case_study.tex`) | One-query real-domain breach (soft buffer: can trim to 0.2) | 0.3 |
| 7 | **Related Work** (`related_work.tex`) | Keep as-is (already good) | 0.5 |
| 8 | **Conclusion** (`conclusion.tex`) | Keep (limitations + ethics disclosure) | 0.4 |

Two moves fix "no coherent story": (i) **Construction (§3) now precedes Theory (§4)** so the operator is defined before it is reasoned about; (ii) the **rebuild of Experiments into Audit/Certify/Defend** plus relocating the certify/defend material out of Theory, so the same spine appears in §1, §3, §4, §5. Nominal budgets sum to ~7.6; Phase 3 reconciles to ≤7pp via the Background/Case-Study buffers and overfull-box cleanup.

---

## Section split + content triage

With Construction (§3) before Theory (§4), the operational "what it is / how you compute it" content moves up, and Theory becomes purely the guarantees that justify the readings.

**§3 The AEGIS Operator (`framework.tex`)** — the operational content:
- `S_c` + `P_c` definition (moved up from the `theory.tex` construction paragraph; pull the minimal notation it needs from `background.tex`)
- matrix-free operator + cost (already here)
- **`alg:aegis` promoted** from `app:algorithm`; `fig:pipeline`
- the **three-readings box**: attack direction = leading right-singular vector of `S_c` (`prop:attack` folded in inline); edge ranking = `S_c` column norms; per-node radius `r_v` (named here; formal bound proved in §4)

**§4 Theory (`theory.tex`)** — the guarantees:

| Result (label) | Decision | De-dry / handling |
|---|---|---|
| `thm:phase_transition` (ε_crit 3-regime) | **KEEP FORMAL in §4** | Backbone of contribution (2). Add one intuition sentence: *"Below ε_crit no perturbation can break contraction; the same operator that proves safety also points at the worst attack."* |
| `prop:radius` (per-node `r_v` bound) | **KEEP FORMAL in §4** | The formal radius guarantee. §3 names `r_v` as the third reading; §4 proves the bound. |
| `prop:attack` (SVD-optimal direction) | **DEMOTE → folded into the §3 reading box** | Inline in §3: *"the leading right-singular vector of S_c is the first-order-optimal attack (Prop. A.x)."* Full statement to appendix. |
| S_c construction paragraph | **MOVE → §3** | Operational — belongs with the operator, not the theorems. |
| `thm:cf2s` (two-sided bracket) | **SUPPORTING CALLOUT in §4** | Per decision (3): keep a one-line **numbered** theorem statement in-body with the empirical hook (*"the norm certificate understates the true breaking budget by 2–9× across 10 seeds"*); full statement `thm:cf2s_full` + proof stay in `app:bracket`. Do **not** silently drop — reviewers track named contributions. |
| `obs:eta_bound`, `rem:eta_relu` | **APPENDIX-ONLY** | Reference only as the "norm-vs-radius gap g_W" inside the bracket callout. |
| `sec:conformal` subsec | **SPLIT** | Keep the closed-form score-shift equation (`L_1^c ε + C_v ε²`) + ε_crit as one displayed equation in §4 (the certify *mechanism*); **move `tab:conformal` to §5.2 Certify** (the certify *evidence*). Use identical "AEGIS-Conformal" phrasing + explicit forward-ref to §5.2. |
| `sec:defense` subsec (prose-only) | **MOVE → §5.3 Defend** | The σ1-penalty is contribution (3); give it a table (below). |
| `prop:transfer` (cont→discrete) | **DEMOTE → sentence in §4** | *"continuous S_c rankings transfer to discrete edge flips (Prop. A.x; τ=0.99 over 390 runs, §5)."* Full statement to appendix. |
| `prop:explicit` (K-layer GNNs) | **APPENDIX-ONLY + pointer** | Half-sentence: *"S_c extends to K-layer explicit GNNs (App. X), covering 6 of our 7 architectures."* |

Net: §3 ≈ **1.2pp** (operator + algorithm + reading box); §4 ≈ **1.3pp** (2 full formal results + 1 displayed certify equation + 4 demotion sentences).

---

## Promotion + merges (appendix → body)

| Exhibit | From | → To | Action |
|---|---|---|---|
| `alg:aegis` | `app:algorithm` | **§3 Construction** | Promote in full, single-column (~0.4pp). Fixes complaint 6; non-negotiable. |
| `tab:attack_full` + `tab:baselines` | appendix | **§5.1 Audit** | **Merge into one Audit table**: rows = AEGIS / PGD / GR-BCD / PR-BCD; cols = misclassification, queries, wall-clock, 74–156× per-query ratio. |
| `tab:conformal` + `tab:smoothing` | theory / appendix | **§5.2 Certify** | **Merge into one Certify table**: coverage @ 0.90 nominal + gate; smoothing vacuous (full label set); ~10⁴× cheaper. |
| **(new) `tab:defense`** | progress notes | **§5.3 Defend** | Build from existing validated numbers: σ1 335→32.6±1.7, coverage 0.82±0.03, clean acc 73.9±0.8% (~5% cost), attack/defense corr −0.65±0.12. **Pre-flight: confirm these are 10-seed before tabling** (see Risks). |
| `fig:breach`, `fig:tau_heatmap` | already in-body | **§5.1 Audit** | Keep. |
| `fig:pipeline` | already in-body | **§3 Construction** | Keep here (the spine overview), not Experiments. |
| `fig:sc_heatmap`, `fig:greedy_topk`, phase/scalability, fraud diagram | appendix | **stay** | Budget protection — only the three merged/new tables + algorithm come forward. |

The two merges are **mandatory** (3 tables, not 5) to keep the body ≤7pp.

---

## Experiments reorganization (`sections/experiments.tex`)

Replace the 4-claim layout with three subsections; the old `sec:cross_domain` / `tab:cross_domain` is absorbed as the **setup paragraph** of 5.1 (this is what removes the competing decomposition).

| Subsec | Claim | Anchor exhibit | Support |
|---|---|---|---|
| **5.1 Audit** | One query finds the worst attack; beats gradient/structural baselines | `fig:breach` + **merged Audit table** | contractivity/`tab:cross_domain` folded in as setup; `fig:tau_heatmap` (τ=0.99, 390 runs) backs the demoted transfer claim |
| **5.2 Certify** | Distribution-free certificate is sound (gate holds at nominal) and non-vacuous where smoothing degenerates | **merged Certify table** | one line tying back to the §4 ε_crit closed form |
| **5.3 Defend** | σ1(S_c) penalty trades clean acc for certified robustness; attack/defense couple | **new `tab:defense`** | corr −0.65±0.12; note the trade is *certified margin*, not small-budget accuracy |

---

## Execution sequence (phased, recompile between phases)

**Phase 1 — high-impact moves (fixes complaints 1, 3, 5, 6).** Swap the `\input` order in `aaai_aegis.tex` so `framework.tex` precedes `theory.tex`, and move the S_c construction paragraph from `theory.tex` into `framework.tex`; promote `alg:aegis` into `framework.tex`; rebuild `experiments.tex` into Audit/Certify/Defend; create `tab:defense`; do the two table merges; relocate `tab:conformal` out of theory. Recompile, re-measure pages.

**Phase 2 — theory compression (fixes complaints 2, 4).** Apply the triage in `theory.tex`: demote bracket→callout, attack→inline, transfer→sentence; push `eta_bound`/`eta_relu`/`explicit` to appendix; split conformal (keep score-shift eq, move table); add the intuition sentences and the boxed S_c "reading." Recompile, re-measure.

**Phase 3 — page-budget cleanup.** Fix overfull boxes; trim the Background / Case-Study soft buffers if needed to land ≤7pp; resolve any undefined refs from the moves; final read-through that each of audit/certify/defend has ≥1 in-body exhibit.

Landing Phase 1 first means complaints 1/3/5/6 are resolved before the riskier proof-prominence edits, and the page budget is re-measured before theory is touched.

---

## Risks

- **R1 — Theory looks thin.** Mitigate: keep **two** full formal results (`phase_transition` + fused S_c/`radius`) + the displayed certify equation; appendix carries `thm:cf2s_full`, all proofs, and honest-status remarks, so rigor is signposted.
- **R2 — Promotions re-blow the page budget.** Mitigate: the two merges are mandatory; algorithm single-column; recompile + re-measure between phases; Background/Case-Study are release valves.
- **R3 — Bracket loses prominence.** Resolved per decision (3) as a supporting callout, but keep it **numbered** and with the 2–9× empirical hook so reviewers tracking contributions still find it.
- **R4 — Split conformal reads disjoint.** Mitigate: identical "AEGIS-Conformal" naming + explicit forward-reference from the §3 equation to §5.2.
- **R5 — Defense table vs the 10-seed rule.** "Surface existing only" forbids new runs; if the validated defense numbers are **not** 10-seed, do **not** fabricate a body table — keep defend as prose + an appendix note and flag for a later 10-seed run. Confirm seed count in `paper/review/AEGIS_PROGRESS.md` / the defense findings file before building `tab:defense`.

---

## Verification

1. **Compile clean:** `cd paper && latexmk -pdf aaai_aegis.tex` (or `pdflatex → bibtex → pdflatex ×2`); 0 errors.
2. **Page budget:** `pdfinfo aaai_aegis.pdf | grep Pages`, then the `.aux` page map (`grep newlabel ... aaai_aegis.aux`) to confirm `sec:conclusion` lands by p7 and the appendix starts only after references.
3. **No undefined refs:** `grep -E "undefined|LaTeX Warning: Reference|\?\?" aaai_aegis.log` returns nothing (the moves change many `\cref` targets).
4. **Overfull boxes:** `grep -c "Overfull \\\\hbox" aaai_aegis.log` — should not exceed the current count; trim in Phase 3.
5. **Spine check (manual read):** Construction (§3) precedes Theory (§4); each of §5.1/5.2/5.3 has ≥1 in-body table/figure; `alg:aegis` is in §3; §3+§4 combined ≤~2.5pp; the words audit/certify/defend appear in §1, §3, §4, §5.

## Critical files
- `paper/sections/theory.tex` — compress per triage
- `paper/sections/experiments.tex` — rebuild into Audit/Certify/Defend
- `paper/sections/framework.tex` — promote `alg:aegis`
- `paper/sections/appendix.tex` — receive demoted statements/proofs; source of promoted tables/algorithm
- `paper/sections/introduction.tex` — retune contribution bullet (2)
- `paper/aaai_aegis.tex` — top-level (no structural change expected)
- `paper/review/AEGIS_PROGRESS.md` — confirm 10-seed status of defense numbers before tabling
