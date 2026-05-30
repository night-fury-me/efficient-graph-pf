# Peer Review Report — EIC

## Reviewer Identity & Focus

Editor-in-Chief / Senior Area Chair for an ICDM/KDD/TKDE-class data-mining venue; graph-mining + trustworthy-ML generalist. This is an independent editorial review. I do **not** verify proof correctness line-by-line or adjudicate attack-literature minutiae; my lens is **scope fit, framing-level originality, significance, claim calibration, and the structure/page-budget tradeoff**. Per instruction I do not consult or speculate about other reviewers.

Paper: "AEGIS: One-Query Adversarial Diagnostics over the GNN Vulnerability Spectrum" (IEEE conference format, 10 pages, anonymized — `\author{Anonymous Author(s)}` in `aegis.tex`). Compiled `aegis.pdf` confirms exactly 10 pages.

---

## Recommendation + Confidence

**Recommendation: Major Revision** (weighted score 67.0 sits in the 65–79 Minor band, but two calibration/scope issues below are serious enough that I round the editorial decision down to Major Revision; see Detailed Comments).

**Confidence: 4 / 5.** I read all section sources (`abstract`, `introduction`, `background`, `theory`, `framework`, `experiments`, `case_study`, `related_work`, `conclusion`) and the main file in full. Confidence is not 5 only because I did not independently rerun experiments or verify proofs (out of my assigned scope).

---

## Summary Assessment (150–250 words)

AEGIS proposes a single analytical object, the *constrained sensitivity matrix* `\(S_c\)` (`\cref{sec:theory}`), from which three practitioner diagnostics are read in one matrix-free pass: the SVD-optimal perturbation direction (`\cref{prop:attack}`), per-edge sensitivity rankings (`\(v_{ij}\)` column norms), and per-node first-order radii (`\cref{prop:radius}`). It adds a closed-form three-regime "phase transition" for the contractive IGNN subclass (`\cref{thm:phase_transition}`, `\(\ecrit{=}(1{-}\kappa)/\norm{W}_2\)`), a continuous-to-discrete transfer bound (`\cref{prop:transfer}`), an explicit-GNN extension (`\cref{prop:explicit}`), and a power-flow case study (`\cref{sec:power_flow}`).

The work is well-organized, candid about limitations (the `\textbf{Limitations}` block in `conclusion.tex` is unusually honest), and the empirical breadth (9 datasets, 7 architectures, 330 runs) is real. The "one query → three diagnostics" packaging is a genuine and useful *consolidation*, and the matrix-free scaling to `\(N{=}7{,}650\)` is a concrete engineering contribution.

My editorial reservations are three. (1) **Originality is consolidation, not invention**: the paper itself states `\(S_c\)` "specialises equilibrium IFT sensitivity ... via the projection `\(P_c\)`" (`introduction.tex`, Contribution 1), and `\(P_c\)` is "the standard duplication-matrix reduction" (`theory.tex`). (2) **Two abstract claims are mis-calibrated relative to the body** ("matches 50-step PGD"; the headline `\(\tau{=}{+}0.996\)`). (3) **The 10-page budget is over-packed** (theory + six experiment subsections + a power case study), which costs clarity. None is fatal; all are fixable in revision.

---

## Strengths

1. **A real, useful unification for practitioners (S1).** The central pitch — that a deployed-GNN auditor wants three things at once (a direction, an edge ranking, a per-node budget) and that no prior single object yields all three — is well-motivated in `introduction.tex` and cleanly mapped to `\(S_c\)` in `\cref{sec:theory}`. The four-stage pipeline (`\cref{fig:pipeline}`) and `\cref{alg:aegis}` make the construction reproducible at the algorithm level. For an ICDM audience that values *deployable diagnostics*, this consolidation has practical pull.

2. **Honest, well-scoped claim hedging in the body (S2).** Where it matters most, the body is careful: `\(r_v\)` is repeatedly labeled a "first-order sensitivity threshold, not a probabilistic/sound certificate" (`\cref{rem:certificates}`, `related_work.tex`, and the `\cref{tab:baselines}` footnote with an explicit decision rule). The AGNNCert comparison is framed as *complementary*, not superior. The GAT limitation (`\partial Z/\partial A_{ij}=0` on existing edges → `\(S_c\)` undefined for binary-mask attention) is disclosed in `\cref{sec:explicit_extension}`. This is the kind of calibration EIC review rewards.

3. **Genuine scalability engineering (S3).** Removing the `\(O((Nd)^3)\)` dense solve via a truncated-Neumann/JVP matrix-free `\(S_c v\)` (`framework.tex`, "Matrix-free operator") and reaching `\(N{=}7{,}650\)` at 365 s / 5.5 GB (`\cref{fig:scalability}`), with `\(\sigma_1\)` agreement to 0.03% at `\(N{=}200\)` (`\cref{sec:scalability}`), is a concrete, checkable systems contribution rather than a paper-only claim.

4. **Adaptive-attack discipline and a falsifiable defense result (S4).** The defense ablation (`\cref{sec:defense_ablation}`) follows sober-look methodology: an adaptive attacker that recomputes `\(S_c\)` on the masked graph matches the non-adaptive one within 0.1 pp, attributed to the 43% spectral gap of `\(S_c\)`. Reporting that the defense survives recomputation (rather than only static masking) directly answers the failure mode `\cite{mujkanovic2022defenses}` flags.

5. **Breadth of evaluation across domains (S5).** 9 datasets / 7 architectures / 4 domains, plus the explicit-GNN transfer heatmap (`\cref{fig:tau_heatmap}`, 330 runs, 29/33 positive cells, sign test `\(p{<}10^{-5}\)`). Even where signals are weak (GCN-2 on sparse citation graphs), the framework "correctly flags" it via low dataset-level `\(\tau\)` — a sign of a self-aware evaluation.

---

## Weaknesses

### W1 — Framing-level originality is *consolidation of standard IFT sensitivity*, and the paper says so itself
- **Problem.** The headline conceptual claim ("one object → three diagnostics") rests on `\(S_c\)`, which the paper explicitly describes as a *specialisation* of known machinery: "`\(S_c\)` specialises equilibrium IFT sensitivity `\cite{koh2017understanding,gould2021deep}` to *structural* edge perturbations via the projection `\(P_c\)`" (`introduction.tex`, Contribution 1), and "The `\(N^2 \to |E|\)` projection `\(P_c\)` is the standard duplication-matrix reduction `\cite{magnus2019matrix}`" (`theory.tex`). The three "diagnostics" are then standard reads of any sensitivity matrix: leading right singular vector (direction), column norms (ranking), and a margin/Lipschitz ratio (radius). So the novelty is the *act of bundling* + the structural projection + the matrix-free implementation, not a new sensitivity theory.
- **Why it matters.** For an originality-weighted decision, "we are the first to compute three already-known quantities from one already-known operator and call it a spectrum" is a *significance/engineering* contribution, not a conceptual breakthrough. The title word "spectrum" and the framing risk overselling.
- **Suggestion.** Re-title or re-frame Contribution 1 to foreground what is actually new: the **edge-supported symmetric projection `\(P_c\)` as the object that makes IFT sensitivity a usable *structural* diagnostic**, and the matrix-free `\(S_c v\)` that makes it scale. State plainly (one sentence, early) that the per-component reads are standard SVD/margin operations; the unification and scaling are the contribution. This *strengthens* credibility rather than weakening the paper.
- **Severity: Major.**

### W2 — Two abstract claims are over-stated relative to the body ("matches 50-step PGD"; headline `\(\tau=+0.996\)`)
- **Problem (a) — "matches 50-step PGD".** The abstract (`abstract.tex`) states "the one-query SVD direction matches 50-step PGD attackers." The body (`\cref{sec:adaptive}`/`\cref{tab:attack_full}`) shows the SVD direction *beats* a 50-step IFT-gradient PGD, which "recovers only 72–92% of its damage." Critically, the paper itself flags that Shift-PGD is "solver validation, not an independent baseline" (`\cref{tab:attack_full}` caption) because it uses AEGIS's *own* IFT gradients — so "matches PGD" is comparing AEGIS to a degraded validator of itself, not to a strong independent attacker. The genuinely independent test (transfer from a separately-trained surrogate, `\(\cos{=}0.99\)`; 512-query black-box recovers 44%) is the *real* evidence and is more convincing; the "matches PGD" phrasing buries it under a weaker comparison.
- **Problem (b) — headline `\(\tau=+0.996\)`.** The abstract leads with `\(\tau{=}{+}0.996\)` against "brute-force N-1 on full-graph Amazon Photo." This is the **edge-weighted ranking `\(A_{ij}v_{ij}\)`** (the `\cref{prop:transfer}` first-order product), measured on a *single dataset* with *stratified top-`\(v_{ij}\)` ground-truth sampling* (`\cref{sec:explicit_extension}`). The same abstract's final sentence then reports the *power-grid* N-1 as `\(\tau{=}{+}0.37\)`–`\(0.62\)` / `\(P@10{=}0.66\)`–`\(0.81\)` (`case_study.tex`, `\cref{tab:ieee}`). Two different "N-1 vs `\(\tau\)`" numbers (0.996 and 0.37–0.62) appear in one abstract; a fast reader will conflate them, and the 0.996 — boosted by stratified sampling and being the raw `\(v_{ij}\)` ranking's best cell — reads as the method's typical accuracy when it is the best-case.
- **Why it matters.** Claim calibration is the dimension most scrutinized at this venue. Headlining a stratified-sampling best-case correlation and an asterisked self-validation "match" invites a reviewer to discount the whole results section.
- **Suggestion.** (i) Replace "matches 50-step PGD" with the defensible and stronger statement: the SVD direction is *recovered by a gradient-independent surrogate at `\(\cos{=}0.99\)`*, i.e. it is model-intrinsic, not a gradient artifact. (ii) For `\(\tau{=}{+}0.996\)`, state in-line that it is the edge-weighted `\(A_{ij}v_{ij}\)` ranking on Amazon Photo under stratified sampling, and report a typical/median full-graph `\(\tau\)` alongside it so the headline is representative.
- **Severity: Major.**

### W3 — The 10-page budget is over-packed, and clarity pays the price
- **Problem.** Into a strict 10-page IEEE-conf limit the paper fits: 1 theorem, 4 propositions, 1 observation, 2 remarks (8 theorem-like blocks), 7 figures, 5 tables, 1 algorithm, *and* six experiment subsections (`\cref{sec:cross_domain}` through `\cref{sec:explicit_extension}`) *plus* a power case study (`\cref{sec:power_flow}`). The experiments section in particular is a dense numeric stream ("3–10×", "149/150", "`\(p{<}10^{-43}\)`", "47–62% of edge pairs", "9.82×/3.25×") with several results pushed entirely into prose or footnotes (e.g. the PI/PTDF baselines live in a `\footnote` in `case_study.tex`).
- **Why it matters.** A reader cannot reconstruct *which experiment supports which claim* without effort: e.g. five distinct `\(\tau\)` regimes appear (subgraph `\(\approx0.16\)`, explicit-GNN cells, power `\(0.37\)`–`\(0.62\)`, Amazon `\(0.996\)`, GR-BCD `\(0.69\)`), and the reader must track which threat model and which ground-truth each uses. The theory–experiment-density tradeoff means neither the proofs (compressed into dense single paragraphs) nor the experiments get room to breathe.
- **Suggestion.** Pick a primary thrust. Either (a) move the explicit-GNN extension (`\cref{prop:explicit}`, `\cref{sec:explicit_extension}`) and/or the power case study to an appendix / companion and let the IGNN theory + diagnostics carry the 10 pages with full clarity; or (b) consolidate the six experiment subsections into 3–4 with a single "claim → table/figure" map at the top of `\cref{sec:experiments}` (the section already opens with a 4-claim list — make that list the literal organizing spine and label each result to it).
- **Severity: Major.**

### W4 — Scope-fit and significance for an ICDM *data-mining* audience are only partly established
- **Problem.** The contribution is squarely *graph-mining + trustworthy ML* (in scope), but the deepest technical content is equilibrium-network analysis (IGNN/DEQ resolvents, pseudospectra, conservative IFT) and the most quantitatively striking result is a *power-systems* case study benchmarked against LODF/PTDF. The closed-form `\(\ecrit\)` theory — the paper's main "theory" claim — applies *only* to the contractive IGNN subclass (stated repeatedly: `\cref{thm:phase_transition}`, "the closed-form `\(\ecrit\)` is available only for the contractive subclass" in `theory.tex`). IGNNs are a niche within the GNN-mining community.
- **Why it matters.** An ICDM PC will ask: how many readers run *implicit* GNNs in production? The broadly-applicable part (explicit-GNN `\(S_c\)` via finite differences, `\cref{prop:explicit}`) is exactly the part with *no* `\(\ecrit\)` guarantee and the weakest/most-variable `\(\tau\)` (e.g. GCN-2 `\(\tau{=}{-}0.04\)`, `\cref{tab:explicit}`). So the rigorous theory and the broad applicability do not overlap.
- **Suggestion.** Add 2–3 sentences in `\cref{sec:intro}` quantifying the *explicit-GNN* value proposition (the diagnostic is architecture-general even without the `\(\ecrit\)` certificate) and lead the significance argument with the matrix-free diagnostic on standard GCN/SAGE/GIN, treating IGNN `\(\ecrit\)` as the "bonus rigor" case. This realigns the paper's center of mass with the venue's median reader.
- **Severity: Minor → Major** (Major if the venue is strictly ICDM core; Minor for a TKDE-class venue with a robustness track).

### W5 — "One-query" framing is defensible but still slightly stretched
- **Problem.** `introduction.tex` correctly clarifies that "one query" means "one matrix-free `\(S_c\)` construction (a single randomized SVD over the equilibrium resolvent), not three separate analyses." But the title's "One-Query" and the abstract's "from a single object ... one query" still read, to a first-time reader, as one matvec. In reality one `\(S_c\)` build is a truncated Neumann series (depth `\(K{\in}[20,50]\)`, `framework.tex`) followed by a randomized SVD (`\(k{=}p{=}10\)`, 2 power iters) — i.e. dozens of JVPs/VJPs. The per-node radius and per-edge column norms are *additional* `\(O(|E|)\)` rmatvec passes (`\cref{alg:aegis}` lines 5–6).
- **Why it matters.** "One-query" is the paper's branding; if a reviewer reads it literally and finds it is "one construction comprising many matvecs," the framing looks like marketing.
- **Suggestion.** Keep "one query" but bind it everywhere to the in-paper definition ("one `\(S_c\)` construction, amortizing all three diagnostics"), and consider softening the *title* to e.g. "Single-Construction Adversarial Diagnostics ...". State the JVP/SVD cost once in the abstract or intro so the term cannot be read as one matvec.
- **Severity: Minor.**

---

## Detailed Comments

**On the editorial decision.** By the rubric the weighted average (67.0, below) maps to Minor Revision. I am recommending **Major Revision** because two of the issues (W1 framing and W2 calibration) touch the paper's *headline claims and title*, not just polish, and because W3 (over-packing) means a revision should make non-trivial structural cuts rather than sentence-level edits. These are the kinds of changes that warrant a second look, not an accept-on-faith. If the venue treats "Major/Minor" as a soft boundary, an argument for Minor is reasonable given how much the body *already* self-corrects the abstract's overreach — the fixes are largely a matter of pulling the body's honesty up into the abstract and title.

**On the abstract↔body consistency more broadly.** The body is consistently *more conservative* than the abstract, which is the safer direction but still a defect. Examples: abstract "matches 50-step PGD" vs body "PGD recovers only 72–92%" (`\cref{sec:adaptive}`); abstract "competitive with industry LODF" vs body "competitive but not dominant ... overlapping on case14/30" (`case_study.tex`). I recommend a single editorial pass to make the abstract's verbs match the body's verbs.

**On the power case study (`\cref{sec:power_flow}`).** This is the most *interesting* result for a general audience but also the most caveated: a narrow training envelope (uniform load scaling 70–130%, explicitly "does not cover seasonal peaks, dispatch shifts, or renewable ramps"), DC LODF benchmarked against AC voltage-angle truth, and `\(\tau{=}{+}0.37\)`–`\(0.62\)`. The honesty is commendable, but the case study is doing double duty (novelty hook *and* significance argument) on thin operational validation. I would either invest more (broader envelopes, as the authors themselves flag as the path to "operational-grade screening") or demote it to a shorter "transferability" vignette so it is not load-bearing for acceptance.

**On theory presentation.** `\cref{thm:phase_transition}` and `\cref{prop:transfer}` are compressed into very dense single paragraphs with the proofs inline. I did not verify them (out of scope), but editorially the regime-(b)/(c) discussion (normality, Perron mode, `\(\Omega(1/(\ecrit-\varepsilon))\)` rate, `\(\eta\)` slack) is hard to parse at this density; this is a symptom of W3. The `\(\mathrm{diag}(-s,0)\)` counterexample is a nice touch and should survive any compression.

**Reproducibility / artifacts.** `\cref{alg:aegis}` is concrete; hyperparameters (rSVD `\(k{=}p{=}10\)`, `\(n_\mathrm{iter}{=}2\)`, Neumann tol `\(10^{-6}\)`) are given. No code/data availability statement is visible in the sources I read (the `\textbf{Disclosure protocol}` in `conclusion.tex` gates attack code behind ethics review — reasonable, but a *diagnostic-only* artifact release should be promised concretely for camera-ready).

---

## Questions for Authors

1. **Calibration of the PGD comparison (W2a).** Given that Shift-PGD is explicitly "solver validation, not an independent baseline" (it uses AEGIS's own IFT gradients), on what *independent* strong attacker does the SVD direction "match"? Is the surrogate-transfer `\(\cos{=}0.99\)` result the intended evidence for the abstract's "matches 50-step PGD," and if so, why not state *that* in the abstract?

2. **Representativeness of `\(\tau=+0.996\)` (W2b).** What is the *median/typical* full-graph continuous-to-discrete `\(\tau\)` across datasets (not just the Amazon best cell), and how sensitive is the 0.996 figure to the "stratified top-`\(v_{ij}\)` ground-truth sampling"? Does it hold under uniform/random N-1 sampling?

3. **Scope/significance for explicit GNNs (W4).** For the architectures most ICDM readers use (GCN/SAGE/GIN, no `\(\ecrit\)` guarantee), what is the *practitioner-facing* claim you want remembered — and does the negative GCN-2 cell (`\(\tau{=}{-}0.04\)`) indicate a failure mode users must screen for before trusting `\(v_{ij}\)`?

4. **Insertion attacks (scope).** Edge *insertion* (Nettack-class) is deferred to "enlarge the basis to `\(\bar E \supseteq E\)`" (`background.tex`, `conclusion.tex`). Since insertion is the dominant structural-attack threat model in the GNN-robustness literature you cite, does scoping it out materially limit the "vulnerability spectrum" claim?

---

## Minor Issues

- **Title vs. content.** "Vulnerability Spectrum" overstates a single SVD spectrum of one operator; consider "sensitivity profile/diagnostics." (Ties to W1/W5.)
- **Two "N-1" meanings in the abstract.** "brute-force N-1 on Amazon Photo" (classification) and "recovers N-1 rankings on case14–118" (power) use the same term for different tasks in adjacent sentences — disambiguate.
- **Section ordering.** Related Work (`\cref{sec:related}`) appears *after* Experiments and the Case Study (`\input` order in `aegis.tex`: experiments → case_study → related_work → conclusion). This is legal but unusual for an IEEE-conf paper and can disorient reviewers; consider moving Related Work to after Introduction unless the late placement is a deliberate page-fit choice.
- **`\(\kappa\)` overloading.** `\(\kappa\)` is contractivity `\(\norm{J_z}_2\)` *and* `\(\kappa(V_W)\)` is a condition number in `\cref{obs:eta_bound}`; close in the text, easy to misread.
- **Footnote-buried baselines.** PI / PTDF baseline numbers in `case_study.tex` are in a footnote; for a results-bearing comparison these belong in `\cref{tab:ieee}` or its caption.
- **`\AEGIS` rendered `\textsc{Aegis}`** — fine, but the title uses all-caps "AEGIS"; ensure consistent casing in camera-ready.
- **Coverage non-uniformity** is disclosed ("all seven architectures appear only in `\cref{tab:explicit}`/`\cref{fig:tau_heatmap}`") — good; make sure the abstract's "7 architectures" is not read as applying to every experiment.

---

## Dimension Scores

| Dimension | Weight | Score (0–100) | Rationale |
|---|---:|---:|---|
| Originality | 20% | 58 | Useful *consolidation* + structural projection `\(P_c\)` + matrix-free scaling, but the core object is a self-described specialisation of standard IFT sensitivity with standard SVD/margin reads (W1). Not a conceptual breakthrough. |
| Methodological Rigor | 25% | 70 | Theory is clearly stated and assumptions audited (`\(\kappa\)`, `\(\eta\)` reported); adaptive-attack discipline (W/S4) is good. Docked for the self-validating PGD comparison framing (W2a) and theory compressed to near-illegibility (W3). |
| Evidence Sufficiency | 25% | 70 | Broad (330 runs, 9 datasets) and largely honest in-body. Docked for headline best-case `\(\tau{=}0.996\)` under stratified sampling presented as typical (W2b), thin power-flow operational validation, and coverage non-uniformity. |
| Argument Coherence | 15% | 66 | Claim→evidence chain is reconstructable but effortful; abstract verbs exceed body verbs; five distinct `\(\tau\)` regimes across different threat models strain the single-"spectrum" narrative (W2, W3). |
| Writing Quality | 15% | 72 | Clean, professional, candid limitations; but over-dense (8 theorem-likes + 12 floats in 10pp), footnote-buried results, and minor notation overloading (W3, minors). |

**Weighted average** = 0.20·58 + 0.25·70 + 0.25·70 + 0.15·66 + 0.15·72
= 11.6 + 17.5 + 17.5 + 9.9 + 10.8 = **67.3 / 100**.

**Rubric mapping:** 67.3 → 65–79 band → **Minor Revision** by the numeric rubric.

**Editorial decision: MAJOR REVISION.** I round down from the numeric band because the two highest-leverage issues (W1 framing/title, W2 abstract calibration) require changes to the paper's headline claims and title rather than cosmetic edits, and W3 requires structural cuts. No issue is Critical; the paper is fundamentally sound and the requested changes are achievable. A revision that (i) reframes Contribution 1 and the title honestly, (ii) aligns the abstract's verbs and headline numbers with the body, and (iii) makes room by demoting one of {explicit-GNN extension, power case study} would, on the merits, be a clear Accept-class submission.
