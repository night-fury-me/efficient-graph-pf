# AEGIS Revision Plan — Addressing External Reviews 01 & 02 (Strategic Edition)

**Target venue:** ICDM 2026 Research Track · **Hard constraint:** 10 IEEE two-column pages *including references and appendices*.
**Current state (updated 2026-05-30):** evidence program complete — `paper/aegis.pdf` compiles cleanly at **10 pages** (latexmk exit 0; 0 errors / 0 undefined refs / 0 overfull). Per-item completion status in the **Progress Log** below. The R2 round (R2_01–R2_10, `docs/r2_experiments_full_report.md`) closed **X1/X2/X3** (commit `2d4e7c2`); **X9/X4/X5** ran in `3e44c41` with the X5 r_v refresh in `dd8eeb2`; **R1** shipped in `470cba2`; and **X7** (the last, optional item) closed in `fcd7698`. All P2 experiments and R1 are run, integrated, and committed; **E3/E5** done; overview diagram (C1) corrected, re-exported, re-embedded. No evidence experiments remain.
**Sources:** `docs/external-review-01.md` (R01: *borderline / weak-reject*), `docs/external-review-02.md` (R02: *borderline / weak-accept, major revision*).

---

## Strategic posture (read this first)

We address every reviewer concern, but **honesty is a scalpel, not a sledgehammer**. Excessive hedging would win the review battle and lose the acceptance war: a paper that apologizes for its own contribution reads as weak. **Precision sells; hedging doesn't.** Every reviewer point is sorted into one of four moves — and only the last weakens us:

1. **FIX & FLEX (genuine errors).** Correct them, then present the correction as *more* rigor. The Theorem 1(b) fix is not a retreat — "we prove *exactly when* the resolvent blows up and that trained models *provably* avoid it" is a **bolder** claim than the buggy one.
2. **DEFEND WITH EVIDENCE (contested-but-true).** Where a reviewer doubts a claim, close it by *running the experiment that confirms it* — never by softening the wording. (rSVD fidelity at N=7,650; independent attack baseline; full-graph τ; false-safe rate; *exhibiting* the phase transition under stress.) This is the "bulletproof over hand-waving" path.
3. **REFRAME WITH PRECISION (scope, don't shrink).** "Globally optimal *first-order hidden-state* direction" is bold *and* exact. The three-diagnostics-from-one-object thesis and the N=7,650 scaling stay front and center; the IFT lineage gets one confident clause, not an apology.
4. **CONCEDE only true liabilities.** case300, anonymity (ref 51), typos, literally-universal phrasing. Surgical and minimal.

**Keep the bold narrative and the memorable hooks where we can earn or precisely define them.** Spend honesty only where it *preempts an attack*, not where it dilutes the sell.

### Locked decisions (author-confirmed)

- **"One-Query" stays in the title** — defined precisely up front as *one matrix-free construction of `S_c` that yields all three diagnostics* (vs. running three separate methods). Keep the hook, remove the objection. (Item N2.)
- **"Phase transition" stays** — and we **earn it** with a stress-test that exhibits an actual sharp resolvent blow-up (Item T2 / Exp X9). Trained-model safety then reads as *"a real cliff exists and our models sit safely below it."*
- **Power-flow stays as a differentiator** — defended cheaply: drop case300, compare LODF on its native DC metric, reframe as a *learned AC-surrogate that recovers N-1 rankings competitively with industry LODF*. No retrain. (Item P1.)

---

## Progress log — completion status (updated 2026-05-30)

**Build:** `aegis.pdf` compiles clean (latexmk exit 0; 0 errors, 0 undefined refs, 0 overfull) at **10 pages**. Theory verified across **two adversarial `ml-theory-reviewer` rounds** (caught and fixed 2 real bugs). This table doubles as the response-letter trail.

| ID | Status | What was done | Where |
|----|--------|---------------|-------|
| **C3** | ✅ done | Ref [51] is the authors' **published ICASSP 2026** paper → cited in **third person** (normal published-work citation; *not* blinded — blinding a searchable published title is counterproductive). Author block already `Anonymous`; no funding/URL/PDF-metadata leak. "our institutional" → "an institutional". | `aegis.bib`, `aegis.bbl`, `conclusion.tex` |
| **T1** | ✅ done | Invalid Neumann *lower* bound → unconditional eigenvalue bound `1/minᵢ|1−λᵢ|`; divergence conditional on an eigenvalue→1 (real-positive/Perron case); `ε_crit` recast as conservative contraction radius, slack tied to η. `diag(−s,0)` counterexample inline. | `theory.tex` Thm 1(b)+proof |
| **T3** | ✅ done | Radius → `min_{c≠yᵥ} m_v^{(c)}/‖(W_{yᵥ}−W_c)Sᵥ‖₂` (composed norm, Cauchy–Schwarz); runner-up form demoted to a possibly-optimistic surrogate; resolves runner-up-swap minor. | `theory.tex` `prop:radius` |
| **T4** | ✅ theory · ⏳ X4 | Deletion = fixed-normalization masking; recompute-norm gap `O(dᵢ⁻¹)`; forward-ref to `sec:cross_domain` for empirical agreement (X4). | `theory.tex` `prop:transfer` |
| **R3** | ✅ done | `L_J ≤ ‖W‖₂²` corrected to `‖W‖₂²‖z*‖` (dropped equilibrium-norm restored); `‖z*‖` bounded via fixed point + `‖Â‖‖W‖<1`; remainder summed over linear regions. *(2nd-review catch.)* | `theory.tex` `prop:transfer` proof |
| **T2** | ✅ math/framing · ⏳ X9 | "Phase transition" kept; math fixed via T1. Empirical *earning* (stress-test exhibiting blow-up) is X9. | `theory.tex`, `experiments.tex` |
| **N1** | ✅ done | Contribution (1) owns the IFT lineage (Koh–Liang/gould) + `P_c` + matrix-free + "no prior object yields all three at once"; `related_work.tex` carried the fuller positioning. | `introduction.tex`, `related_work.tex` |
| **N2** | ✅ done | "One query" defined in intro = one matrix-free `S_c` construction (one rSVD over the resolvent), not three analyses. Title unchanged (hook kept). | `introduction.tex`, title |
| **C4/E1/A1** | ✅ done | Abstract reframed precise-but-bold: scoped "first-order", retired "no prior method"/"maximally sensitive", "continuous-edge-weight message passing" (A1), "competitive with LODF", pile-up + tiered-disclosure removed. Hooks kept. | `abstract.tex` |
| **E2** | ✅ reframe · ✅ X7 proven | Shift-PGD flagged as solver-validation; "matches 50-step PGD" not "recovers only 72–92%"; SVD optimal for the first-order hidden-state objective. X7 run: gradient-independent transfer recovers 99% (cos=0.99, model-intrinsic), 512-query black-box only 44% — direction is not a circular gradient artifact. | `abstract.tex`, `experiments.tex` |
| **E4** | ✅ done | GR-BCD framed honestly: "label-free one-pass diagnostic, not a budget-optimal attacker — competitive, not dominant." | `experiments.tex` |
| **P1** | ✅ done | case300 removed (prose + table row + footnotes); LODF on native-DC/thermal-overload metric + "competitive, not dominant"; "learned physics" → "topology-driven flow concentration"; secondary baselines → footnote. Differentiator preserved. | `case_study.tex` |
| **C1** | ✅ done | Diagram corrected in `aegis.drawio`, **re-exported & re-embedded** (paper force-rebuilt: 10 pp, clean): typos (B4/B5), PyTorch (B6), τ=0.37–0.62 (A1; case300 number dropped), P_c caption → 'One analytical object, three diagnostics' (A2), r_v → min-over-classes (A3), 7×5 datasets (C7), S_K subtitle (D8). Verified at 300 dpi. | `figures/aegis.drawio`, `aegis--overview--diagram.pdf` |
| **C2** | ✅ already clean | Verified: shared theorem counter; refs are `\cref` (auto-typed) or correct `Thm.`/`Prop.`; zero Remark/Prop-as-theorem refs. No edit needed. | — |
| **Compression** | ✅ done | Abstract + experiments (redundant lead-ins, duplicated envelope numbers, GR-BCD, radius/defense/robust-backbone prose) + theory wording → **11→10 pages**, clean. | abstract/experiments/theory |
| **E3 / X2** | ✅ done | Full-graph result (R2_08) promoted to a bolded `\textbf{Full-graph scale.}` run-in item: AtkAdv over degree-proportional **9.82×** (Citeseer), **3.25×** (Cora) at k=10 vs ≈1.1× on subgraphs. Kept as a prose headline (not a standalone table) to hold 10 pp — net offset by folding the 1.8%/τ=0.16 coverage point in from "Subgraph size". | `experiments.tex` |
| **E5** | ✅ done (Option B) | Coverage-clarification sentence added to `\textbf{Setup.}`: "the 9 datasets and 7 architectures are cumulative; all 7 archs only in Tab. V + Fig. 7's 33 cells; power flow the 4 IEEE cases." Closes the count-mismatch (R02-Rec7). Offset by removing two redundancies (duplicated subgraph def; 1.8%/τ=0.16 now in E3 item). Full coverage *table* (Option A) parked unless a reviewer asks; scope + LaTeX draft in `docs/e5_coverage_table_scope.md`. | `experiments.tex` |
| **R1** | ✅ done | Diagnostic-only code path released (`diagnostic_analysis`, v_ij/r_v, attack synthesis gated); commit `470cba2`, `docs/r1_diagnostic_release.md`. | repo, `framework.tex` |
| **X1 / R2** | ✅ done | rSVD/matrix-free fidelity closed by **R2_04**: Halko bound on rank-6 rSVD; σ₁ matrix-free vs dense within 0.03% at N=200 (dense infeasible beyond, >24 GB); κ²⁰⁰ truncation residual ∈[10⁻¹⁰⁵,10⁻⁴⁸] across the suite incl. Amazon N=7,650. Integrated. | `experiments.tex` §Scalability |
| **X3 / R4** | ✅ done | Breach/false-safe quantified by **R2_03** (per-dataset breach rates + 95% CIs; Mettack sign test p=1.06×10⁻⁴³). Breach figure upgraded 2026-05-30: bands ±1 SD → **95% CI** (Student t, 9 df, verified to match R2_03 to 1e-16); median diamond now defined via a legend proxy. | `experiments.tex`, `fig_breach_rate.pdf` |
| **X9** | ✅ done | *Earned the phase transition*: symmetric-part construction drives a real eigenvalue→+1, empirical blow-up matches Ω(1/(ε_crit−ε)). Commit `3e44c41`, `docs/x9_phase_transition_stress_findings.md`. | `experiments.tex` |
| **X4** | ✅ done | Fixed- vs recomputed-normalization deletion validated (T4); scope-only defense, threat-model + fixed-norm qualifier. Commit `3e44c41`, `docs/x4_deletion_normalization_findings.md`. | `experiments.tex` |
| **X5** | ✅ done | Min-over-classes radius (corrected T3) implemented + all r_v paper numbers refreshed on the unified impl. Commits `3e44c41`/`dd8eeb2`, `docs/x5_radius_minclass_findings.md` + `docs/x5_number_refresh_findings.md`. | `experiments.tex` |
| **X7** *(opt.)* | ✅ done | Independent attack run: one-query SVD direction beats 512-query black-box 2.3× (44% reached) and ties/leads transfer from an independent surrogate (99%, cos=0.99 → model-intrinsic), 50/50 cells. Commit `fcd7698`, `docs/x7_independent_attack_findings.md`. | `experiments.tex` |

**✅ Fig. 1 (C1) done:** diagram corrected, re-exported, and re-embedded (verified at 300 dpi; paper rebuilt clean at 10 pp).
**Page budget:** at 10 pages with **zero slack** — confirmed 2026-05-30 that any net +1 line tips 10→11 pp (nonlinear float/reference repack). The E3 and E5 additions were each held at 10 pp by removing equivalent redundancy; any remaining additions (e.g. X9 figure/sentences) must likewise be offset (§6).
**Remaining to ship:** none — all P2 evidence experiments (**X1, X2, X3, X4, X5, X9, X7**) and **R1** are run, integrated, and committed. Paper holds at **10 pp / 0 overfull**. X7 (the last item) closed 2026-05-30 via commit `fcd7698`.

---

## 0. How to read this plan

Each item lists: **what reviewers said → concrete change → file/anchor → move-type → new experiment? → page delta → effort → risk if skipped.**
Priorities: **P0** = correctness/desk-reject blocker; **P1** = high-leverage reframing; **P2** = evidence experiments; **P3** = polish.

File map (`paper/sections/`): `abstract.tex` (7 ln), `introduction.tex` (10), `background.tex` (25), `theory.tex` (136), `framework.tex` (43), `experiments.tex` (195), `case_study.tex` (46), `related_work.tex` (12), `conclusion.tex` (10). Title: `paper/aegis.tex:43`.

---

## 1. Consolidated issue map (deduplicated; tagged by strategic move)

| ID | Issue | R01 | R02 | Sev | Move | Target |
|----|-------|:--:|:--:|:--:|------|--------|
| **T1** | Thm 1(b) "Neumann *lower* bound 1/(1−‖J′‖)" invalid without spectral alignment | W1 (wrong) | praises (right) | P0 | **FIX & flex** | `theory.tex` |
| **T2** | "Phase transition" never reached in practice (ρ≤0.42) | W2 | §2 | P0 | **DEFEND (earn it)** + FIX | `theory.tex`, `experiments.tex` |
| **T3** | Radius uses only runner-up c\*, not min-over-classes | W5 | minor (c\* swaps) | P0 | **FIX & flex** | `theory.tex` |
| **T4** | Transfer prop ignores degree-renormalization of incident edges | W3 | — | P0 | **FIX + DEFEND** | `theory.tex`, `experiments.tex` |
| **N1** | Novelty thinner than framed (S = standard IFT sensitivity) | implied | §1 | P1 | **REFRAME (confident)** | `abstract.tex`, `introduction.tex`, `related_work.tex` |
| **N2** | "One-Query" hook misleading | W4 | §4 | P1 | **REFRAME (keep+define)** | `aegis.tex:43`, abstract/intro |
| **E1** | Transfer leads with +0.998; true range +0.16…+0.998; sign test weak | (subgraph) | §3 | P1 | **REFRAME + DEFEND** | `abstract.tex`, `experiments.tex` |
| **E2** | PGD comparison favors AEGIS (Shift-PGD shares AEGIS gradients) | Exp3 | §4 | P1 | **DEFEND + reframe** | `abstract.tex`, `experiments.tex` |
| **E3** | Too much main evidence on 50-node BFS subgraphs | Exp1 | §3 | P2 | **DEFEND (promote full-graph)** | `experiments.tex` |
| **E4** | GR-BCD framing uneven (worse on Cora) | Exp2 | — | P1 | **REFRAME (as strength)** | `experiments.tex` |
| **E5** | Headline counts vs per-table coverage mismatch | — | minor/Rec7 | P1 | **REFRAME (add table)** | `experiments.tex` |
| **R1** | No runnable code in reviewed version | Clarity | §6/Rec4 | P1 | **DEFEND (release)** | repo, `framework.tex` |
| **R2** | rSVD fidelity at scale asserted, not measured | — | §6/Rec6 | P2 | **DEFEND (measure)** | `experiments.tex` |
| **R3** | L_J ≤ ‖W‖₂² under-justified | — | §6 | P2 | **FIX** | `theory.tex` |
| **R4** | r_v breach claim tautological; false-safe rate unquantified | (cert) | minor | P2 | **DEFEND (measure)** | `experiments.tex` |
| **P1** | Power-flow attackable (uniform load, ℓ₂-angle ≠ N-1, LODF cross-metric, case300) | W6 | §5 | P1 | **DEFEND cheaply + CONCEDE case300** | `case_study.tex` |
| **A1** | GAT† modified arch sold as coverage win | — | §7 | P1 | **REFRAME (precise scope)** | `abstract.tex`, `framework.tex` |
| **C1** | Typos / equation OCR artifacts | Clarity | minor | P3 | **FIX** | Fig.1 src, compile |
| **C2** | Numbering ("theorem 6"→Remark; "theorem 4"→Prop) | Clarity | — | P3 | **FIX** | cross-refs |
| **C3** | Triple-blind: ref [51] near authors' own PF work | Clarity | — | P0 | **FIX (desk-reject)** | `aegis.bib` |
| **C4** | Abstract overloaded | Clarity | Rec1 | P1 | **REFRAME (precise-but-bold)** | `abstract.tex` |

---

## 2. Theory keystone — fix the math, then *earn* the phase transition (T1 + T2)

**The disagreement.** R01: the divergence in Thm 1(b) rests on `‖(I−J′)⁻¹‖₂ ≳ 1/(1−‖J′‖₂)`, an invalid lower bound. R02: praises it as correct. **R01 is right.** Standard Neumann gives an *upper* bound; the same RHS as a *lower* bound is false:
> Counterexample: `M = diag(−s,0)`, `‖M‖₂=s`, yet `‖(I−M)⁻¹‖₂ = 1 < 1/(1−s)`.

**The fix (strength, not retreat).** Replace the norm bound with the **unconditional eigenvalue bound**
> `‖(I−J′_z)⁻¹‖₂ ≥ ρ((I−J′_z)⁻¹) = 1 / min_i |1 − λ_i(J′_z)| = 1/dist(1, spec(J′_z))`,

true for *every* matrix. Then state regime **(b)** precisely: divergence `Ω(1/(ε_crit−ε))` holds *when a real eigenvalue `λ(ε)` of `J′_z(ε)` approaches +1* (not merely `‖J′_z‖₂→1`), at a rate set by the inverse spectral gap. Keep `ε_crit=(1−κ)/‖W‖₂` verbatim, **relabeled** as a *sufficient contraction-preservation radius* (the valid upper-bound certificate).

**Why this is the single highest-leverage move (one fix → four wins):**
1. **T1:** the bound is now unconditionally true; the conditional clause is honest about *when* blow-up happens.
2. **R01/R02 disagreement:** resolved in our favor — no normality needed for the *eigenvalue* bound (R02's praise survives), R01's objection removed.
3. **R02 §2 ("never reached"):** the thing that fails to occur is *an eigenvalue reaching +1* — and the data (`ρ(J_z)≤0.42`) shows eigenvalues stay far from 1. This **explains** the conservatism instead of apologizing for it.
4. **Sells the theory harder:** we now claim a *precise characterization of the danger condition* plus a *provable safety margin*.

**Earn the phase transition (Exp X9 — the DEFEND move that keeps the hook).** Add a stress-test that *exhibits* the transition rather than asserting it:
- Along an adversarial path that aligns `δÂ` with the binding **eigen**-direction (not just the top singular vector of `Â`), or by relaxing the spectral cap so `κ→1`, drive a real eigenvalue of `J′_z` toward +1 and plot **two curves on one panel**: (i) a stressed model whose `min_i|1−λ_i|→0` with `‖(I−J′_z)⁻¹‖` blowing up as `Ω(1/(ε_crit−ε))` — the **demonstrated** phase transition; (ii) the trained model, eigenvalue bounded away from 1, resolvent flat — the **2–4× safety margin**.
- This single figure earns "phase transition" (curve i) *and* showcases trained-model safety (curve ii). Reuse/extend `fig_phase_transition`; ~0.5 day.

**Net for §2:** Theorem 1 becomes *rigorous and demonstrated*; the "phase transition" and "ε_crit" hooks stay, now bulletproof. Subsection title "Phase Transition and Scalability" is retained (earned).

---

## 3. Priority work items

### P0 — correctness & desk-reject blockers

- **C3 — Anonymity (do first, 1 hr).** Audit `aegis.bib` ref [51] + all self-cites; cite own prior PF work in third person; anonymize released code/data. *Risk: triple-blind desk reject.*
- **T1 — Eigenvalue bound + conditional divergence** (`theory.tex`, `thm:phase_transition` regime b). Per §2. *FIX & flex. +2 ln. 0.5 day.*
- **T3 — Min-over-classes radius** (`theory.tex`, `prop:radius`). Headline form `r_v = min_{c≠y_v} (f_{y_v}−f_c)/‖(W_{y_v}−W_c)S_v‖₂`; present the current c\*-product form as a valid *conservative approximation* (note `‖(W_{y_v}−W_c)S_v‖₂ ≤ ‖W_{y_v}−W_c‖₂‖S_v‖₂`, so it already lower-bounds). Absorbs R02's "runner-up swaps" minor. *FIX & flex. +2 ln. 0.5 day (recompute via X5).*
- **T4 — Continuous-to-discrete under renormalization** (`theory.tex`, `prop:transfer`). Define the discrete op as **fixed-normalization edge masking** (D held fixed), then *validate* (Exp X4) that fixed- vs recomputed-normalization deletion give near-identical rankings. *FIX + DEFEND. +1 ln + reuse table. 1 day.*
- **R3 — L_J ≤ ‖W‖₂² sentence** (`theory.tex`). Single activation flip ⇒ rank-1 change to `diag(φ′)`; near-simultaneous flips non-generic; bound used only for *ordering*. *FIX. +1 ln. 2 hr.*

### P1 — defend & reframe (cheap, kills the most objections)

- **C4 + N1 + N2 + E1 + A1 — Abstract** (`abstract.tex`). Full spec in §5. Precise-but-bold: keep hooks, scope optimality, lead with the strong transfer number, own the IFT lineage in one clause, fix arch-coverage wording. *REFRAME. ~0 net. 0.5 day.*
- **N1 — Novelty, asserted confidently** (`related_work.tex`, `introduction.tex`). One clause: *"`S_c` = equilibrium IFT sensitivity `(I−J_z)⁻¹J_A` specialized to **structural** (edge) perturbation via the edge-subspace projection `P_c`, made computable at N=7,650 by matrix-free routing — prior IFT work [22,55] targets feature/weight perturbation and does not yield structural edge diagnostics at scale."* Position as **first to unify three structural diagnostics in one object**, not as "incremental." *REFRAME. +1 ln (offset by trimming 1 RW sentence). 2 hr.*
- **N2 — Title hook kept + defined.** Title unchanged; add one sentence in abstract+intro defining "one query" = one matrix-free `S_c` construction yielding all three diagnostics. *REFRAME. +1 ln. 1 hr.*
- **E1 — Transfer story, lead strong** (`abstract.tex`, `experiments.tex`). Headline stays **+0.998 at full-graph scale**; immediately contextualize: low cells are a *diagnosed, resolved* subgraph-coverage artifact (50-node BFS = ~1.8% of edges), not a method failure — they recover at full scale. Report the sign test as *sign* agreement (29/33) and let X2/X3 magnitude carry consistency. *REFRAME + DEFEND. ~0 net. 2 hr.*
- **E2 — PGD honesty as a setup for strength** (`experiments.tex`, `abstract.tex`). State plainly: SVD direction is optimal *for the first-order hidden-state objective*; Shift-PGD shares AEGIS gradients (already flagged), so 72–92% validates the linearized solve, not superiority over a real adversary. Move "72–92%" out of the abstract as a *strength* claim; foreground Cls-PGD prediction-flip *parity*. *DEFEND + reframe. ~0 net. 3 hr.* **Recommended evidence (Exp X7):** add a genuinely independent transfer/black-box attack; if AEGIS's direction still leads, the claim becomes *stronger*, not weaker.
- **E4 — GR-BCD framed as a strength** (`experiments.tex`). R01's honest sentence: label-free equilibrium-sensitivity diagnostic, competitive on Pubmed (τ=+0.69), weaker on Cora under *direct attack damage* — because that is *not its objective*; its value is one-pass ranking + radii + direction. *REFRAME. +1 ln. 1 hr.*
- **E5 — Experimental-coverage table** (`experiments.tex`). One compact table mapping the 9 datasets × 7 architectures to each experiment (tightness=5; heatmap=33 cells×10 seeds=330 runs; four-quadrant=3; PF=5 IEEE). Kills the count discrepancy in one move. *REFRAME. +5 ln (compact) — offset in §6. 0.5 day.*
- **R1 — Release diagnostic-only code now.** The `r_v`/`v_ij` path cannot synthesize perturbations → not under the tiered gate. Ship it (anonymized); keep Alg. 1 steps 8–9 gated. One clarifying sentence in `framework.tex`. *DEFEND. +1 ln. 0.5 day.*
- **P1 — Power-flow, defended as a differentiator** (`case_study.tex`):
  - **Drop case300 entirely** (the one true liability — 22.6° RMSE invites "why is this here"). *CONCEDE.*
  - Compare **LODF on its native DC line-flow metric** (clean win where it holds) and clearly qualify any cross-metric number.
  - Confident reframe: *"a learned AC-surrogate whose first-order sensitivity recovers N-1 critical-line rankings, competitive with industry LODF on IEEE case14–118"* — **not** "operator-ready N-1 screening" (overclaim) and **not** "just preliminary" (underclaim).
  - Replace the unfalsifiable "emerges from learned physics" with *"consistent with topology-driven flow concentration."*
  - Compress secondary numbers (PI, PTDF, N-2 recovery, admittance-vs-binary) into one footnote.
  - *DEFEND cheaply. Net −4 ln (case300 + compression). No retrain. 0.5 day.*

### P2 — evidence experiments (consolidated in §5)

X1 (rSVD fidelity at N=7,650), X2 (promote full-graph to main table), X3 (false-safe rate), X4 (fixed- vs recomputed-norm ranking), X5 (recompute radius), X9 (earn the transition), X7 (independent attack, recommended).

### P3 — polish

- **C1** Fix Fig. 1 typos ("Vulnarability"→"Vulnerability", "perturnbation"→"perturbation"); recompile; scan `aegis.log` for overfull/missing-glyph warnings; verify no mangled inline equations (R02 noted OCR artifacts pp. 1–3).
- **C2** Cross-ref pass: "theorem 6"→Remark `rem:certificates`; "theorem 4"→Proposition; full `\cref` consistency.

---

## 4. What to KEEP — do not "fix" the honesty reviewers rewarded

Both reviews praised the self-policing. **Preserve:** AGNNCert decision-rule footnote (`tab:baselines`), the "Shift-PGD = solver-validation upper bound, not independent baseline" label, the κ-margin honesty, Remark `rem:certificates`, the GAT† dagger disclosure. The revision makes the **abstract/title agree with this honest body** — it does **not** dilute the body, and it does **not** add new hedges beyond the precise scoping above.

---

## 5. Consolidated evidence-experiment list

| # | Experiment | Closes | Effort | Output |
|---|-----------|--------|--------|--------|
| **X9** | **Earn the phase transition**: stress a model so a real eigenvalue → +1; exhibit `Ω(1/(ε_crit−ε))` blow-up vs flat trained curve (one panel) | T2, R01-W2, R02-§2 | 0.5 day | upgrade `fig_phase_transition` |
| **X1** | **rSVD fidelity at N=7,650**: spectral gap + proxy error (residual / σ̂₁ stability across oversampling) on Amazon Photo | R2 | 0.5–1 day | 1 panel / 2 sentences |
| **X2** | **Full-graph promotion**: Cora/Citeseer/Amazon full-graph runtime/mem/τ → main table; 50-node demoted to dense-vs-matrix-free validation | E3, R01-Exp1 | 1 day | re-tabulation, ~0 net |
| **X4** | **Fixed- vs recomputed-normalization deletion**: Kendall τ between rankings | T4 | 1 day | 1 row |
| **X5** | **Min-over-classes radius recompute** | T3 | 0.5 day | updates numbers |
| **X3** | **False-safe rate for r_v** (ε<r_v but flips) vs ε | R4 | 0.5 day | 1 sentence |
| **X7** *(recommended)* | **Independent direction baseline**: transfer/black-box attack vs SVD direction | E2 | 2–3 days | 1 row |

**Minimum viable set:** X9, X1, X2, X4, X5, X3 (each ≤1 day). X7 elevates the direction claim from "defended" to "proven."

---

## 6. Page-budget ledger (must net ≤ 0; power-flow is **kept**, so the budget is tighter)

| Addition | Δ | Cut | Δ |
|----------|--:|-----|--:|
| T1 eigenvalue lemma | +2 | case300 drop + PF secondary → footnote | −4 |
| T3 min-over-classes | +2 | duplicative envelope-ratio prose | −3 |
| T4 fixed-norm def | +1 | subgraph-size detail → validation note (full-graph promoted) | −3 |
| R3 L_J sentence | +1 | merge phase-transition/scalability lead-ins | −2 |
| N1 novelty clause | +1 | abstract tightening (universal phrasing, pile-up) | −2 |
| N2 "one-query" definition | +1 | robust-backbones secondary numbers compress | −2 |
| E4 GR-BCD sentence | +1 | RW 1 sentence (offset by N1) | −1 |
| E5 coverage table (compact) | +5 | Algorithm caption / misc | −1 |
| X1 rSVD sentences | +2 | | |
| X3 false-safe sentence | +1 | | |
| X9 earn-transition sentences (fig reused) | +2 | | |
| **Σ additions** | **+19** | **Σ cuts** | **−18** |

**Net +1 → close it** by making E5 a `\footnotesize` 4-row mini-table (−1) or trimming one defense-ablation number. **Contingency lever if it overflows: move one secondary table (e.g., `tab:tightness_eps` detail) to the released repo — NOT demote power-flow.** X7 (independent baseline), if added, needs ~+2 more lines — source from further PF-secondary compression.

**Realized (2026-05-29):** the theory bulletproofing actually added **~40** typeset lines, not the +7 budgeted above (eigenvalue derivation + counterexample, min-over-classes radius, corrected `L_J`). This pushed the build to 11 pages. Reclaimed via the **power-flow demotion** (case300 + footnoted secondaries), the **abstract reframe**, **experiments redundancy cuts**, and **theory-wording tightening**; the reference list then repacked off page 11 (nonlinear float effect). **Net result: clean 10-page build.** The pending **E5 coverage table** and **X1 rSVD sentences** still need offsetting before they land — the contingency lever above applies.

---

## 7. Abstract rewrite spec — precise *and* bold (`abstract.tex`)

**Lead bold, scope exactly, keep the hooks:**
- *"AEGIS extracts three actionable diagnostics — a globally optimal **first-order hidden-state** perturbation direction, per-edge equilibrium-sensitivity rankings, and per-node sensitivity radii — from a single matrix-free construction of one object, the constrained sensitivity matrix `S_c`, scaling to N=7,650 on one GPU."* (Defines "one query"; scopes optimality; keeps the thesis.)
- *"We prove a closed-form three-regime **phase transition** with critical budget `ε_crit=(1−κ)/‖W‖₂`, **exhibit** the predicted resolvent blow-up under adversarial stress, and show trained IGNNs operate with a **2–4× safety margin**."* (Earns the term; bold.)
- *"The continuous-to-discrete bridge holds at full-graph scale: edge-weighted ranking reaches **τ≈+0.998** against brute-force N-1 on Amazon Photo (N=7,650); the bridge is positive across 29/33 architecture–dataset cells, with the few cold cells traced to subgraph coverage and resolved at full scale."* (Leads strong, contextualizes honestly.)
- Architecture coverage: *"continuous-edge-weight message passing (GCN/SAGE/GIN/APPNP/IGNN and an edge-weighted GAT† variant)"* — no implied standard-GAT coverage.
- First-claim, made precise: replace "no prior method produces…" with *"AEGIS is the first to derive these three structural diagnostics from a single equilibrium-sensitivity object."*

**Remove:** "72–92% PGD recovery" as a *strength*; the bare unqualified universal claim. **Net length unchanged.**

---

## 8. Execution order (dependency-aware)

1. ✅ **C3 anonymity** — done (third-person ICASSP 2026 cite; build clean).
2. ✅ **Theory pass** `theory.tex` (T1 → T3 → T4 → R3) — done + 2 adversarial reviews.
3. ⏳ **Evidence experiments**: X9, X1, X2, X4, X5, X3 (X7 if time) — pending (need code/GPU).
4. ✅ **Power-flow defend** `case_study.tex` — done (case300 dropped, LODF native metric, confident reframe).
5. ✅ **Reframes**: abstract + one-query definition (N2) + novelty (N1) + GR-BCD (E4) + PGD (E2) — done.
6. ⏳ **E5 coverage table** + X-experiment numbers — pending (needs page offset).
7. ✅ / ⚠️ **Polish**: C2 verified clean, C1 *source* fixed, clean recompile at 10 pp — done **except the Fig. 1 re-export** (⚠️ author action).
8. ⏳ **Response letter**: map each reviewer point → change using §1 + the Progress Log; note (politely, with the eigenvalue bound) where R02 was mistaken on Thm 1(b). `/ars-revision-coach` can scaffold it.

**Outcome target:** move from "borderline, high-variance" to "likely weak-accept" by (a) fixing the one real theory bug and turning it into a stronger claim, (b) *earning* the phase-transition and one-query hooks instead of dropping them, (c) defending the power-grid differentiator while cutting only its one liability, and (d) replacing every overclaim with a precise, confident statement — all within 10 pages, without diluting the contribution.
