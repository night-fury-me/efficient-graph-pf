# Plan: Surgical readability pass on the AEGIS main body (AAAI-27, 7pp hard limit)

## Context

The AEGIS paper (`paper/aaai_aegis.tex`, canonical AAAI-2026 build) "reads too dry and
dense" and sits exactly at the 7-page main-body limit (`\label{endofcontent}` resolves to
**page 7** in `aaai_aegis.aux`). The goal is a more coherent, easier-to-read paper that keeps
its academic tone and rigor, carries a single through-line (*one operator `S_c` → audit,
certify, defend*), defines every symbol at point of use, embeds intuition in the prose itself
(not "Intuitively…" sentences), and avoids LLM-style artifacts — all **without** resizing or
removing any figure/table and **without** spilling past page 7.

First-hand reading of the full main body changed the assessment versus the earlier sub-agent
audit: the paper is already tight, rigorous, and academic. Notation is largely defined at
first use (`sections/background.tex:6` sets the convention), there is essentially **one**
LLM-ism ("crucial"), and the strong through-line already exists. The "dry/dense" feeling comes
from a **small number of genuinely compressed hotspots**, not pervasive weak writing. The user
chose **Surgical** scope and **Length-neutral** edits accordingly: fix the hotspots, weave
intuition into existing sentences, verify notation, smooth the few terse seams — never grow the
body. Over-editing a strong, tight paper would risk degrading rigor or reintroducing the very
artifacts we must avoid.

## Pre-flight (the user's "before everything else" item): DONE — no action needed

Cross-checked the whole paper against `paper/review/radar_competitor_audit.md`. The audit's
six findings and citation fixes were **already applied today (2026-06-04)** and the paper is
**consistent**, so there is nothing to fix here:
- `figures/fig_positioning_radar.tex` carries the honest re-scores (attacks 1.0 on
  attack-direction/per-edge/large-budget; AEGIS 0.9/0.9/0.5; certifier query-efficiency 0.85;
  label-free split from no-retraining; fabricated "512-query/50-step" rationale removed).
- `sections/introduction.tex:11` caption reads "wins no axis outright but is the only method
  with nonzero mass on all seven" (frontier semantics) — matches the resolution.
- `aegis.bib`: `schuchardt2023localized` = Wollschläger / ICLR; `li2025agnncert` = USENIX 2025.
- The "512-query" / "50-step PGD" strings that remain (`sections/experiments.tex:35`) are
  AEGIS's **own** real baselines, not the competitor rationales the audit objected to.

This will be re-verified once at the end (single `grep`), but no edits are planned.

## Hard constraints (guardrails)

1. **Page 7 is the ceiling.** After every edit, recompile and confirm `endofcontent` stays on
   page ≤ 7. Never grow the body.
2. **No figure/table touched** in size or existence. Caption *prose* may be edited only if it
   stays length-neutral and does not change the float's height.
3. **Length-neutral per hotspot.** Intuition is bought by restructuring and by clawing words
   back from redundancy *within the same paragraph*, not by adding net length.
4. **No LLM artifacts.** No em-dashes as connectors, no "delve/leverage/seamless/showcase/
   underscore/crucial/pivotal/realm/intricate", no throat-clearing openers ("It is important
   to note", "Notably,", "Importantly,").
5. **Intuition is embedded**, woven into the causal logic of existing sentences — never a
   bolted-on "Intuitively," clause.
6. **One hotspot at a time**, compile-and-check between each (per the user's per-experiment
   protocol and the explicit "continuously monitor" instruction).

## Verification loop (run after EVERY hotspot edit)

From `paper/`:
```
latexmk -pdf -interaction=nonstopmode -halt-on-error aaai_aegis.tex   # route via ctx_execute if output floods
grep -E "endofcontent\}" aaai_aegis.aux        # expect {{8}{7}...}; the 2nd group (7) is the page — must be <= 7
```
Also confirm, from the build log / `.log`: "Output written on aaai_aegis.pdf" page count did
not jump; **no new** "Overfull \hbox" near the body; **no** "undefined" references/citations.
If an edit pushes `endofcontent` to 8, revert or tighten before moving on.

## Edit plan (the genuine hotspots — surgical, length-neutral)

Files live in `paper/sections/`. Listed in execution order (low-risk → high-value), each a
self-contained edit followed by the verification loop.

### 1. `experiments.tex:87` — the one LLM-ism (trivial)
"The **crucial** quantity is the \emph{gate}" → "The **key** quantity is the \emph{gate}".

### 2. `introduction.tex` — notation glossed at first use (tiny, for the "defined in context" rule)
Two symbols are used in the intro before their later definition; add a 2–3 word gloss inline:
- `:24` `\ecrit{=}(1{-}\kappa)/\norm{W}_2`: first `\kappa` → "contractivity factor `\kappa`".
- `:20`/`:24` first `P_c` → "(the edge restriction `P_c`)".
Net change ≈ +5 words; offset by trimming one redundant clause in the same long sentence (`:15`).

### 3. `conclusion.tex:8` — split the 5-clause limitations sentence
Current: five semicolon-joined clauses in one sentence. Rewrite as 2–3 short grouped sentences
(guarantee-scope / model-scope / cost), length-neutral. Candidate:
> "Three caveats bound the guarantees. The radii `r_v` are first-order thresholds, and
> AEGIS-Conformal currently forms `S_c` densely at `N=200`. On models, standard GAT and
> binary-mask architectures fall outside the framework (\cref{sec:explicit_extension}), and
> insertion attacks are scoped out (\cref{sec:background}); the `\ecrit` track also costs ~6%
> accuracy for its closed-form boundary."
Tune to exact length parity at edit time.

### 4. `theory.tex:29` — Theorem 1(b) critical regime (HIGH value): thread the causal "why"
The blow-up is currently asserted spectrally with no mechanism. Embed the cause — an eigenvalue
reaching 1 means `F` stops contracting along that mode, so the equilibrium is no longer pinned
and the resolvent gain diverges — at parity length. Candidate:
> "As `\varepsilon` grows, an eigenvalue of `J_z'` drifts toward 1; there `F` stops contracting
> along that mode, the equilibrium is no longer pinned, and the resolvent gain diverges,
> `\norm{(I-J_z')^{-1}}_2 \geq 1/\min_i\abs{1-\lambda_i(J_z')}` (driven by this vanishing gap to
> 1, not by `\norm{J_z'}_2\to1`). For normal `J_z'` with a dominant Perron eigenvalue the rate
> is `\Omega(1/(\ecrit-\varepsilon))`; in general `\ecrit` \emph{lower-bounds} the divergence
> threshold, the slack being the nonnormality `\eta` (`\eta\leq2.47` for ReLU; \cref{...})."
Claw back the few added words from the original parenthetical. Verify recompile + page.

### 5. `framework.tex:34` — "Matrix-free operator" paragraph (HIGHEST value)
The densest passage in the paper: it stacks apply-don't-form, the `O(Nd)` vs `O((Nd)^2)` cost,
Neumann series, JVP, and the dense fallback with no connective logic. Rewrite as a clear
*why-each-choice* flow at constant length, funding the added intuition by tightening the
redundant tail of the same paragraph ("which is what lets a single rSVD match iterative
attackers"). Candidate opening:
> "Scale rests on never forming `S_c`. Every reading we need only multiplies `S_c` by a vector,
> so we keep it as the operator `S_c v=(I-J_z)^{-1}J_A P_c v` and apply it in `O(Nd)` memory
> instead of storing the `O((Nd)^2)` dense matrix. The only hard factor is the resolvent, which
> contractivity (`\kappa<1`) lets us expand as a geometrically convergent Neumann series of
> depth `K` set by `\kappa` (`K\in[20,50]` for `\kappa<0.8`), each term a forward-mode JVP that
> applies `P_c` implicitly; only for `N\leq200` do we form `J_z,J_A` directly and take an exact
> dense SVD."
Hold total paragraph length constant (verify by recompile + page check).

### 6. Optional, only if budget allows after 1–5 (lowest priority, still length-neutral)
- `framework.tex:32`: the hidden-state-shift-vs-label rationale is already present but terse;
  smooth by one clause if and only if a word can be reclaimed nearby.
- `introduction.tex:15`: split the long three-thread sentence at a natural seam for breathing
  room (no new words).
- Leave the **abstract** essentially as-is (high-visibility, already well-built; density is
  expected there). Touch only if a clearly length-neutral smoothing presents itself.

## Two cross-cutting passes (run once, near the end)

- **Notation-definition verification.** Sweep the main body confirming every symbol is defined
  at or before first use. The only gaps found are `\kappa`/`P_c` in the intro (fixed in edit 2);
  confirm `\eta`, `\beta`, `g_W`, `S_c`, `S_v`, `\ecrit`, `r_v`, `\tau`, `\lambda` each resolve
  at their first main-body appearance. Add a ≤3-word gloss only where a genuine gap remains.
- **Artifact sweep.** `grep` the main-body sections for em-dashes and the banned word list;
  expect only the already-fixed "crucial". Fix any straggler length-neutrally.

## Out of scope

- Appendix prose (`sections/appendix*`), the bibliography content, and any figure/table
  geometry. The radar/citation items are already correct (pre-flight above).
- No new claims, numbers, or results — this is a writing pass only; all edits preserve meaning.

## Definition of done

All six numbered edits applied; both cross-cutting passes clean; `endofcontent` still on page
≤ 7 after the final build; no new overfull/undefined warnings; figures/tables untouched; a
final read confirms the through-line reads smoothly with intuition embedded and no LLM artifacts.

## Status: COMPLETED (2026-06-04)

Applied (5 edits, diff = 5 insertions / 5 deletions, all length-neutral):
1. `experiments.tex` — "crucial" → "key" (the one LLM-ism).
2. `introduction.tex` — glossed `\kappa` at first use ("in the contraction factor `\kappa{=}\norm{J_z}_2`").
3. `conclusion.tex` — split the 5-clause limitations sentence into three grouped sentences.
4. `theory.tex` — Thm 1(b) now leads with the mechanism ("an eigenvalue approaches 1, `F` stops
   contracting along that mode and the resolvent gain diverges…"), all math preserved verbatim.
5. `framework.tex` — "Matrix-free operator" paragraph rewritten so each choice carries its reason
   (never form `S_c` because every reading is a product `S_c v`; contractivity is what makes the
   Neumann expansion converge), words reclaimed from the redundant tail to hold length.

Edit 6 (optional smoothing) deliberately skipped: the candidates (`framework.tex:32`,
`introduction.tex:15`, abstract) were already coherent, and touching them risked bloat for
marginal gain under the chosen surgical scope.

Verification: every compile kept `endofcontent` on **page 7**; overfull hbox = 1 (unchanged
baseline), undefined refs/citations = 0; compile exit 0. Audit-consistency re-confirmed intact
(`fig_positioning_radar` re-scores, intro caption "wins no axis outright", bib
`schuchardt2023localized` = Wollschläger / ICLR). No figures or tables touched.
Changes left uncommitted in the working tree (no commit requested).

## Second pass: prose-level audit cross-check (2026-06-04)

The first pass checked the audit only against the radar figure + bib. A deeper cross-check of the
audit's *substantive facts about each competitor paper* against our **prose** characterizations
(intro / related work / appendix baselines) found four real contradictions, now fixed
(main-body fixes length-neutral; appendix unconstrained):

1. `introduction.tex:15` — attacks "return no continuous direction" was false (PR-BCD optimizes a
   continuous per-edge direction; the radar already concedes attacks 1.0 on direction). Rewrote the
   triad: attacks "rank edges by label-driven gradient search but return no per-node budget or
   certificate"; "smoothing and certifiers return per-node **or collective** certificates."
2. `introduction.tex:15` + `related_work.tex:8` — "per-node" certificate/robustness blanket was
   wrong for the **collective** certifiers (`schuchardt2023localized`, `schuchardt2021collective`).
   Fixed to "per-node or collective" (intro) and dropped the false "per-node" (related work →
   "certify robustness").
3. `introduction.tex:15` + `related_work.tex:6` — "**surrogate** gradients" is wrong for the
   headline GR-/PR-BCD (surrogate-free white-box evasion, Geisler). Dropped "surrogate".
4. `appendix/F_experiments.tex:43` — localized smoothing "needs 10^3–10^4 MC samples" contradicted
   both the audit (5×10^5) and line 83 of the same file (~10^5). Aligned to ~10^5.

Surviving "surrogate" mentions are AEGIS's own separately-trained-surrogate transfer test and the
"contractive surrogate" for power flow — legitimate, not competitor claims. Post-fix build: exit 0,
`endofcontent` page 7, overfull = 1, undefined = 0. Corrected intro triad is now internally
consistent and consistent with the honest 7-axis radar.
