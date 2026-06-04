# Appendix Revision Plan — AEGIS (AAAI-27)

## Context

The technical appendix (`paper/sections/appendix.tex`, 666 lines, one file `\input` after `\appendix`
in `aaai_aegis.tex`) carries every proof, the full bracket theorem, the conformal derivation, the
detailed experiment tables, and the fraud-detector walk-through. It currently reads as terse,
notation-dense reference material: proofs compress steps into symbol-laden inline sentences (e.g.
`appendix.tex:65`, the `(b). From (a), d_{k1}-d_{k2} >= w_{k1}v_{k1}-...-C(w^2+w^2)` chain), notation is
defined nowhere centrally (no symbol table; ~24 core symbols introduced inline), and one proof defers
work (`prop:transfer` "omits cross terms as empirically lower-order"). The goal is to turn the appendix
into a genuinely *readable supplement* that lets a reader reconstruct the main paper's claims in full:
every proof derived step by step, intuition carried inside the derivation, inline notation pushed into
displays and a central symbol table, all symbols/equations defined once. We also add a dedicated
Table-of-Contents page styled like `paper/figures/image.png`, and fold in the verified orphan figures.

No page limit applies to the appendix; the main body (<=7 pp) and its theorem *statements* are **not**
touched. This is a rewrite of `sections/appendix.tex` plus a small ToC insertion in `aaai_aegis.tex`.

## Writing doctrine (applies to every paragraph rewritten)

- **Intuition embedded, never announced.** Each derivation step carries a short clause saying *why* it
  holds or what it means, fused into the sentence. No "Intuitively, ...", no standalone intuition
  paragraph. (Bad: "Intuitively the gain blows up." Good: "Each pass around the fixed-point loop
  re-injects a fraction `kappa` of the perturbation, so the total is the geometric sum `1/(1-kappa)`,
  which diverges as the operator nears the contraction boundary.")
- **Displays carry algebra; sentences carry reasons.** Any expression with >=2 binary relations becomes a
  numbered display. One idea per sentence.
- **Define once, reference after.** A new "Notation and Preliminaries" section holds the symbol table,
  standing assumptions, and abbreviations; downstream text references it instead of re-defining.
- **Concise and rigorous.** No throat-clearing, no em-dashes, no "leverage/delve/seamless/robustly"
  filler (per the academic-writing rule). Cut anything that does not help a reader understand a
  main-paper claim.
- **Restate-then-prove.** Each proof first restates the main-body statement (so the appendix is
  self-contained), then proves it.
- **Honesty preserved.** The explicitly-labelled conjecture (`rem:obs_o1`, two open gaps) and the
  caveat remarks (`rem:conf-caveats`, `rem:exchange-honesty`) stay as honest limitations — clarified,
  never upgraded to theorems or deleted.

## Target structure (coherent, best-practice; replaces the current A–F ordering)

A dedicated **ToC page** opens the appendix, then six lettered sections grouped by *what main-paper
claim they support*, so the reader moves objects -> sensitivity -> rankings -> boundary -> coverage ->
evidence:

- **A. Notation and Preliminaries** *(NEW)* — symbol table; standing assumptions (A1)–(A4) restated
  cleanly (from `appendix.tex:98–106` + the regime definition `ass:tight-v2`); abbreviation glossary
  (IFT, JVP/VJP, SVD, PGD, TPS, APS); a one-paragraph map of the core objects (operator `S`, constrained
  operator `S_c`, contraction `kappa`, budgets `eps_crit`/`eps_spec`, per-node radius `r_v`, conformity
  scores); and a short **"Matrix-free evaluation of S_c"** subsection (how `S_c` is applied via JVP/VJP
  without ever materialising it — relocate existing prose / any algorithm box here).
- **B. Equilibrium Sensitivity and the Phase Transition** — `thm:phase_transition`; nonnormality bound
  `obs:eta_bound` + ReLU remark `rem:eta_relu`; per-node radius `prop:radius`.
- **C. From Continuous Sensitivity to Discrete Rankings** — ranking transfer `prop:transfer` (close the
  cross-term gap, see below); explicit `K`-layer GNN sensitivity `prop:explicit`.
- **D. The Two-Sided Contraction Boundary** — regime definition; full bracket theorem `thm:cf2s_full`;
  constants table `tab:constants`; the honest nonlinear-break conjecture `rem:obs_o1` (two open gaps,
  kept as conjecture). **Place `fig_tightness_eps` here** (first-order tightness vs `eps` underpins the
  bracket's first-order basis).
- **E. The AEGIS-Conformal Coverage Bound** — conformity-score definition `def:conf-scores`;
  why-the-gradient-is-the-readout-gap remark `rem:margin-not-grad`; worst-case score-shift lemma
  `lem:score-shift`; robust-coverage theorem `thm:robust-cov`; honest-status remark
  `rem:exchange-honesty`.
- **F. Detailed Experimental Results and Reproducibility** — setup; four-quadrant attack comparison
  (`app:attack_full`) **+ `fig_attack_comparison`**; prior structural baselines (`tab:baselines`)
  **+ `fig_positioning`** as an extended-positioning subsection; certification vs randomized smoothing;
  phase transition & scalability (existing `fig_phase_transition`, `fig_scalability`); per-architecture
  explicit GNNs (`tab:explicit`); ablations & defense; fraud-detector walk-through (`app:fraud`);
  reproducibility (`app:repro`).

This is A–F (six sections), which also matches the lettering depth of `image.png`. **Label keys are
preserved** so main-body `\cref`s keep resolving (see Guardrails).

## ToC page — implementation (black, AAAI-clean, no forbidden packages)

`hyperref`/`xcolor` are forbidden/absent, so we reproduce `image.png`'s *layout* in black using the
standard `tocdepth`-toggle trick (no extra packages):

1. In `aaai_aegis.tex`, right after `\maketitle` (with the other float tweaks ~line 152), add
   `\setcounter{tocdepth}{-2}` so main-body `\section`/`\subsection` log **nothing** to the `.toc`.
2. Replace the `\appendix` / `\input{sections/appendix}` block (lines 196–197) with:
   - `\appendix`
   - `\clearpage`
   - title block: `{\noindent\Large\bfseries Appendix\par}` then a small skip;
     `\renewcommand{\contentsname}{Table of Contents}`;
   - `\setcounter{tocdepth}{2}` (re-enable logging for appendix sections);
   - a full-width rule `\noindent\rule{\linewidth}{0.8pt}` under the "Table of Contents" heading that
     `\tableofcontents` prints, plus a closing `\rule` after it (frames the list as in `image.png`);
   - `\tableofcontents`
   - `\clearpage`
   - `\input{sections/appendix}`
3. Two `pdflatex` passes are needed for page numbers to settle. The `article` default already renders
   top-level entries (A, B, …) bold without leaders and subsections (A.1) with dotted leaders + page
   numbers — exactly `image.png`'s pattern; `\appendix` supplies the letter numbering.

**Compile-time check / fallback:** confirm `aaai2026.sty` does not break `\tableofcontents`. If it does,
fall back to `titletoc`'s `\startcontents[app]…\printcontents[app]{}{0}{}` (CTAN-standard,
AAAI-compatible) for an appendix-only ToC, or a hand-built two-column list as a last resort. Do **not**
add `hyperref` or `xcolor`.

## Proof-by-proof rewrite specification

For each: restate the statement, then derive in explicit displayed steps with the intuition fused in.
Pull every multi-relation expression out of running text.

1. **`thm:phase_transition` (B).** Steps: fixed-point map and Jacobian `J_z`; IFT giving sensitivity
   `(I-J_z)^{-1} J_A`; Neumann series converging iff `kappa = ||J_z|| < 1`; geometric-sum gain
   `g_W/(1-kappa)`. Intuition: the loop re-injects a `kappa`-fraction each pass, so amplification is the
   geometric sum that diverges at the contraction boundary — the divergence *is* the transition.
2. **`obs:eta_bound` + `rem:eta_relu` (B).** Bound the nonnormality factor `eta` from the operator
   structure independent of the graph (all-active); ReLU remark notes general active patterns are
   handled empirically. Intuition: nonnormality inflates transient gain above the spectral radius. Keep
   the observation/empirical framing.
3. **`prop:radius` (B).** Steps: first-order change of the readout margin under an `eps`-perturbation
   along `S_c`; worst-case direction -> operator-norm denominator; certificate `prediction unchanged
   while eps < r_v`. Intuition: `r_v` is margin-you-have divided by how fast the worst perturbation eats
   it — distance to the nearest boundary in the metric the operator induces.
4. **`prop:transfer` (C) — close the Tier-B gap.** Currently bounds only the dominant IFT term and calls
   the cross terms "omitted, empirically lower-order." Rewrite to carry the **full** second-order
   remainder: (a) Taylor-expand discrete damage `d_k` in edge weight `w_k`; (b) first-order term =
   `w_k v_k` (the score); (c) bound the **entire** remainder `|R_k| <= L_J w_k^2` with
   `L_J <= ||W||^2 ||z*||` (state every term, no omission); (d) pairwise, the score gap exceeds the
   summed remainders so the order is preserved. Intuition: signal is `O(w_k)`, error `O(w_k^2)`, so for
   the small-weight edges of sparse graphs the linear order is reliable.
5. **`prop:explicit` (C).** Unroll `K` layers -> product Jacobian `S_K`; first-order Taylor -> shift
   bound `||Delta Z_K||_F <= sigma_1(S_K) ||delta A||_F`; show `sigma_1(S_K) <= ||J_A||(1-kappa^K)/
   (1-kappa)` converging to the IGNN bound as `K -> infinity`. Intuition: an explicit GNN has no fixed
   point, but unrolling gives the same structural Jacobian; the finite sum is the truncated geometric
   series.
6. **`thm:cf2s_full` (D).** Steps: (a) lower bound — norm certificate `eps_crit` is sufficient (no break
   below it); (b) upper bound — at `eps_spec` the all-active operator's leading eigenvalue hits 1,
   forcing a break; (c) bracket constant `C = g_W(1+kappa)/(1-kappa)` linking them, giving
   `eps_crit <= eps_break <= (C/beta) eps_crit`. Intuition: the certificate is conservative (operator
   norm = worst over all directions); the spectral budget is exact (actual unstable direction); their
   ratio is bounded by a constant, so the cheap screen is never more than a constant factor loose. Keep
   `tab:constants`.
7. **`rem:obs_o1` (D).** Keep as **conjecture**; state the two open gaps crisply (masked-operator
   spectral scaling; linear-to-nonlinear bifurcation) and the empirical support (`gamma ~ 1.02–1.06`,
   10 seeds). Do not upgrade.
8. **`lem:score-shift` (E).** Steps: margin decomposition; first-order margin shift under `eps` along
   `S_{c,v}`; translate to score shift via softmax monotonicity. Intuition: the score moves only through
   the margin, whose worst-case first-order drop is the same operator-norm geometry as `r_v`.
9. **`thm:robust-cov` (E).** Steps: inflate the conformal quantile by the worst-case score shift from
   the lemma; apply split-conformal coverage with shifted scores; conclude coverage `>= 1-alpha` up to
   `eps`. Intuition: pad the threshold by exactly the worst-case shift, so any in-budget perturbation
   still lands inside the slightly larger prediction set. Keep `rem:exchange-honesty`.

Also clean the **conformal definitions** (`def:conf-scores`, `eq:L1def`, `eq:Cvdef`): name TPS/APS
scores plainly and move the constant definitions into displays.

## Notation and Preliminaries (Section A) — content

- **Symbol table** (booktabs) covering the core set: `kappa, J_z, J_A, S, S_c, S_v, eps_crit, eps_spec,
  eps_reach, g_W, eta, beta, sigma_E, r_v, v_k, w_k, L_1^(c), C_v, score^TPS, score^APS, z*, C`.
  Each row: symbol, one-line meaning, where first used.
- **Standing assumptions (A1)–(A4)** (operator class, certified regime, contractivity `kappa<1`,
  all-active), restated once and referenced by tag thereafter.
- **Abbreviation glossary**: IFT, JVP/VJP, SVD, PGD, TPS, APS.
- **Core-objects paragraph**: the equilibrium sensitivity `(I-J_z)^{-1}J_A`, its constrained form `S_c`,
  the two budgets, the radius, the scores — one sentence each, so later sections reference rather than
  introduce them.

## Figures (decisions confirmed with the user)

**Include (3):**
- `fig_attack_comparison.pdf` -> Section F, `app:attack_full`. Values match `tab:attack_full`
  (`experiments.tex:47–49`) exactly (AEGIS 3.70/4.63/2.10 ±std; Cls-PGD 2.51/2.97/0.67; etc.). Caption:
  visual companion to the main-text table; equilibrium damage at `eps=0.10`, 10 seeds.
- `fig_tightness_eps.pdf` -> Section D. Datasets (Cora/Citeseer/Pubmed/WikiCS/Amazon Photo) are all in
  the suite. **Caption must distinguish** the *shift-prediction* tightness (actual/first-order ratio vs
  `eps`, up to ~1.39 at `eps=0.20`) from the *ranking* tightness quoted elsewhere (`0.99–1.02`), so the
  reader does not read a contradiction.
- `fig_positioning.pdf` -> Section F, new "Extended positioning" subsection. **Caption must map the
  baked-in legend numbers** `[1]–[5]` to real citations (`\citep{zugner2018adversarial}`,
  `zugner2019adversarial`, `geisler2021robustness`, `schuchardt2023localized`, `li2025agnncert` — all
  present in `aegis.bib` and already used).

**Exclude (2):** `fig_case_study_summary.pdf` and `fig_ieee14_case.pdf` are power-flow / N-1 / DC-LODF
results. The current paper has no power-grid content and the conclusion explicitly disclaims physics
("audits vulnerability, not physics … complements rather than replaces power-flow contingency
screening"). They were orphaned because that case study was rescoped out; including them would
contradict the narrative. Leave them on disk, unreferenced. (`fig_ieee14_case` also still carries
scaffold edges; its data file gives P@10=0.70, not the 0.74 baked into the `.tex` — a second reason.)

## Global notation-reduction tactics

- Lead with the symbol table (A); never redefine a symbol that A already defines.
- Convert every dense inline chain (the `appendix.tex:61`, `:65`, `:76`, `:79` style multi-relation
  sentences) into displayed `align`/`\[...\]` blocks with one prose clause of justification each.
- Replace symbolic connectives in prose with words ("because", "so", "which gives").
- Expand all abbreviations on first use (in A).

## Files to modify / create

**Split the appendix into one file per lettered section** (under a new `paper/sections/appendix/`
directory) for easy management. `sections/appendix.tex` becomes a thin **driver** that `\input`s the six
section files in order, so `aaai_aegis.tex`'s single `\input{sections/appendix}` line is unchanged.

- **`paper/sections/appendix.tex`** *(rewritten as a thin driver)* — only six `\input` lines
  (`\input{sections/appendix/A_preliminaries}` … `\input{sections/appendix/F_experiments}`) plus brief
  comments; no content lives here.
- **`paper/sections/appendix/A_preliminaries.tex`** *(NEW)* — symbol table, assumptions (A1)–(A4),
  abbreviation glossary, core-objects map, matrix-free `S_c` subsection.
- **`paper/sections/appendix/B_sensitivity.tex`** *(NEW)* — `thm:phase_transition`, `obs:eta_bound` +
  `rem:eta_relu`, `prop:radius`.
- **`paper/sections/appendix/C_rankings.tex`** *(NEW)* — `prop:transfer` (full remainder bound),
  `prop:explicit`.
- **`paper/sections/appendix/D_boundary.tex`** *(NEW)* — regime definition, `thm:cf2s_full`,
  `tab:constants`, conjecture `rem:obs_o1`; `fig_tightness_eps`.
- **`paper/sections/appendix/E_conformal.tex`** *(NEW)* — `def:conf-scores`, `rem:margin-not-grad`,
  `lem:score-shift`, `thm:robust-cov`, `rem:exchange-honesty`.
- **`paper/sections/appendix/F_experiments.tex`** *(NEW)* — detailed results + reproducibility;
  `fig_attack_comparison`, `fig_positioning`, existing `fig_phase_transition`/`fig_scalability`, tables
  (`tab:baselines`, `tab:explicit`), fraud walk-through.
- **`paper/aaai_aegis.tex`** — add `\setcounter{tocdepth}{-2}` after `\maketitle`; wrap the
  `\appendix … \input{sections/appendix}` region with the ToC block above (the `\input{sections/appendix}`
  line stays and now fans out to the six files). No new packages.
- **Graphics paths unchanged**: `\includegraphics` resolves relative to the main job directory
  (`paper/`), so `figures/...` still works from the new `sections/appendix/` files — no path edits.
- **No main-body section edits** (theory/experiments/framework/case_study/conclusion stay as-is); **no
  figure-source edits** (the three included figures already build to PDF — only `\includegraphics` +
  captions are added).

## Plan archival

As the **first implementation step**, copy this finalized plan into the repo at
`docs/appendix_revision_plan.md` (project `docs/` at the repo root, confirmed to exist), so the plan is
version-controlled alongside the paper.

## Guardrails / out of scope

- **Preserve all `\label` keys** referenced externally so `\cref`s keep resolving — in particular
  `thm:phase_transition, prop:radius, prop:transfer, prop:explicit, thm:cf2s, prop:attack,
  sec:explicit_extension, sec:transfer_explicit, sec:audit, sec:adaptive, app:fraud, app:baselines,
  app:phase_scal, app:ablations, app:explicit, tab:attack_full` and the appendix-internal `app:*`,
  `eq:*`, `thm:*`, `prop:*`, `lem:*`, `obs:*`, `rem:*`, `def:*` keys. If a section is renamed, keep its
  old `\label`.
- Do not alter theorem/proposition **statements** — only their proofs and surrounding prose.
- Keep `rem:obs_o1`, `rem:conf-caveats`, `rem:exchange-honesty` as honest limitations.
- No `hyperref`/`xcolor`; ToC stays black. Any figure work stays serif 11 pt (existing figures comply).
- All reported numbers stay 10-seed; introduce none that are not.

## Verification

1. **Compile** (LaTeX only — light, safe to run here): `cd paper && pdflatex -interaction=nonstopmode
   aaai_aegis && bibtex aaai_aegis && pdflatex aaai_aegis && pdflatex aaai_aegis`. Expect clean exit.
2. **ToC page**: opens on its own page; lists **only** appendix A–F with letter numbering, dotted
   leaders on subsections, page numbers, black text, "Appendix" + "Table of Contents" titles + framing
   rules — matching `image.png`'s layout. Confirm no main-body sections leak into it.
3. **Figures**: the three new `\includegraphics` render; no "undefined" floats; `fig_positioning`
   caption resolves `[1]–[5]`; the two PF figures remain unreferenced (grep the `.tex` and the `.log`).
4. **References**: grep `aaai_aegis.log` for "undefined" / "There were undefined references"; verify all
   main-body `\cref` targets still exist.
5. **Proof audit**: each proof restates its statement, derives in explicit displayed steps, embeds
   intuition (no "Intuitively," / standalone intuition paragraph), and contains no "omitted / easy to
   see / follows directly"; `prop:transfer` now carries the full remainder bound.
6. **Diff scope**: `git status` shows only `aaai_aegis.tex` (small), the rewritten driver
   `sections/appendix.tex`, the six new `sections/appendix/*.tex`, and `docs/appendix_revision_plan.md`;
   no main-body section files changed; `endofcontent` page (main-body length) unchanged.
7. Skim the rendered appendix PDF end-to-end for coherence and that it reads as a standalone supplement.
