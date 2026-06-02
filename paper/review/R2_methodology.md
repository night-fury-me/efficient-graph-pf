# Reviewer 2 (Empirical Methodology) — AEGIS

**Scope.** Experimental methodology only: design rigor, baseline strength/fairness, statistical validity, reproducibility, and claim–evidence support. Theory derivations and literature positioning are out of scope (owned by other reviewers).

**Verdict (empirical rigor).** Broadly honest and unusually well-hedged for an attack/diagnostic paper; the headline claims are *mostly* supported by the cited tables, and the authors pre-empt several reviewer objections in-text. However, the empirical case rests on (a) a non-adversarial 512-query "black-box" baseline, (b) a power-flow surrogate trained on a single narrow load envelope, (c) a missing graph-centrality control on the N-1 task, and (d) heavy reliance on 50-node subgraphs that the paper itself shows are unrepresentative. These are reframing/new-experiment issues, not fatal ones, but two MAJOR items must be closed before the contingency-screening and "black-box" claims are publishable. One genuine internal numerical inconsistency (CRITICAL-for-accuracy, trivial to fix) exists in the case study.

---

## 1. Claim–Evidence Audit

Locations: abstract.tex L2; introduction.tex (Contributions); experiments.tex; case_study.tex.

| # | Headline claim | Cited evidence | Verdict |
|---|---|---|---|
| 1 | **42±8% damage reduction** (top-k edge masking) | experiments.tex L140 (defense para), `tab`-free prose; abstract L2 | **Supported but narrow.** It is *one config*: Cora, IGNN, N=50 subgraph, k=5, 10 seeds, paired Wilcoxon p<0.002. Abstract presents "42±8%" as a general property; it is a single-dataset / single-model / subgraph-only result. **MINOR overclaim of generality.** |
| 2 | **2–4× margin** (ε_crit safety) | experiments.tex (phase-transition para, `fig:phase_transition`); abstract; intro | **Supported.** Defined precisely as empirical margin of ρ(J_z) vs ε_crit over the κ_max∈[0.30,0.99] sweep on Cora. Consistent across abstract/intro/body. OK. |
| 3 | **median τ=+0.99, 39/39 cells positive** | `fig:tau_heatmap`, experiments.tex (cross-dataset transfer); abstract; intro | **Supported.** 390 runs, sign test p<10⁻⁵, per-cell sd 0.0003–0.041 reported. The *edge-weighted* ranking is what hits 39/39; unweighted is 34/39 and is reported honestly. OK. |
| 4 | **+0.996 on Amazon Photo** | experiments.tex (transfer para); abstract | **Supported but caveated correctly** — it is "stratified top-v_ij N-1," and the para concedes full-graph κ≈1.00 (marginally non-contractive). The stratified-sampling qualifier is *not* in the abstract; a reader sees "+0.996" as a plain full-N-1 τ. **MINOR.** |
| 5 | **P@10=0.66–0.81 (power flow)** | `tab:ieee` (case_study.tex); abstract; case study text | **Range correct** (min=case57 .66, max=case118 .81). But see §6: the companion **τ range stated in text (+0.37 to +0.62) is wrong — table shows case57 τ=+0.67.** Also "competitive with industry LODF" is the *AC-angle* target; on LODF's native thermal target AEGIS only reaches P@10=0.60 (footnote). The abstract's framing is defensible; the in-text τ ceiling is an error. |
| 6 | **74–156× per-query damage vs 50-step PGD** | `tab:attack_full` + experiments.tex L43 | **Supported as a *per-query* ratio**, and the text is careful ("one-query diagnostic, not a peak attacker"). But the framing flatters: it is damage *divided by query count*, and PGD's *absolute* equilibrium damage is competitive (Cls-PGD only 15–70% less; Shift-PGD 72–92%). The 74–156× is an efficiency ratio, not a damage-superiority claim — correctly stated in body, but a skim-reader of the abstract ("one query recovers the direction 50-step PGD finds") may over-read it. **MINOR (framing).** |
| 7 | **One query recovers PGD direction at cos=0.99** | experiments.tex L43 | **Slippage.** The cos=0.99 result is recovery by a **separately trained surrogate transfer direction**, *not* by PGD. The abstract says "one query recovers the direction 50-step PGD finds"; the body's cos=0.99 number is surrogate-vs-SVD, and the PGD comparison is the *separate* 74–156× efficiency line. The abstract conflates two distinct experiments. **MAJOR clarity / borderline overclaim** — fix the abstract wording to name the surrogate-transfer experiment. |

**Overclaim list (ranked):**
- (CRITICAL-accuracy) case-study text τ ceiling +0.62 contradicts table +0.67 — §6.
- (MAJOR) Abstract sentence "one query recovers the direction 50-step PGD finds" misattributes the cos=0.99 surrogate-transfer result to PGD — claim #7.
- (MINOR) "42±8%" generality (single config); "+0.996" stratified caveat dropped in abstract; 74–156× is per-query efficiency not absolute damage.

---

## 2. Baseline Strength & Fairness

**2a. The 512-query "black-box search" is a strawman. [MAJOR]**
experiments.tex L43: "44±4% for a 512-query black-box search." Nowhere is the search algorithm specified — it reads as random/coordinate search. The legitimate point being made (SVD direction is model-intrinsic, recoverable by zero-gradient surrogate transfer at cos=0.99) is *fine*; but the 512-query number is used rhetorically to suggest the direction is hard to find by query access, and random search at 512 queries is a weak opponent. A modern query-based attacker (NES, SimBA, Bandits-TD, or Square Attack analogue on the edge simplex) at the same query budget is the fair comparison. **Fix:** report the 44±4% against at least one principled black-box optimizer (NES or SimBA-style) at 512 queries; if the surrogate-transfer cos=0.99 already dominates these, the claim *strengthens*. As written, a reviewer will read it as a chosen-weak baseline.

**2b. Mettack / GR-BCD / PR-BCD budget framing — mostly fair, but flagged. [MINOR→MAJOR risk]**
The paper is explicit and honest: "the budgets shown (k≤10) are AEGIS's early-warning regime ... while GR-BCD/PR-BCD dominate raw damage at the high budgets they target" (experiments.tex, prior-comparison para; `tab:baselines`). The Mettack claim "149/150 paired wins, p<10⁻⁴³" is at k∈{1..5} only. This is a *defensible* scoping (early-warning triage), but the framing selectively compares each external attacker at the budget where AEGIS looks best and concedes the rest in one clause. `tab:baselines` only shows GR-BCD at Pubmed k=10 / Cora k=5 — two cells. **Fix:** show the *full* damage-vs-budget curve (k=1..50) for AEGIS vs GR-BCD/PR-BCD/Mettack in `fig:greedy_topk` (referenced but the crossover budget is never quantified). State the crossover k explicitly so the "early-warning regime" boundary is empirical, not asserted.

**2c. Missing graph-centrality baseline on N-1 (power flow). [MAJOR]**
case_study.tex baselines are LODF, Ejebe–Wollenberg PI, PTDF — all physics/topology flow measures. There is **no degree / betweenness / eigenvector-centrality baseline on the N-1 ranking task.** The whole value proposition is that S_c (a *learned* sensitivity object) adds information beyond raw topology; but the paper's own explanation for why it works ("topology-driven flow concentration," "binary adjacency beats admittance-weighting because a trip removes the line regardless of impedance") *predicts that a pure topology centrality could match it.* Centrality baselines appear only for the classification task at ε=0.01 (degree/betweenness/spectral), not for N-1. **Fix:** add edge-betweenness / line-centrality (and current-flow betweenness) as N-1 rankers on `tab:ieee`. If S_c beats them, the "learned attribution adds value" claim is earned; if not, the case study collapses to "centrality ≈ LODF ≈ S_c," which must be conceded. This is the single most important missing control in the paper.

---

## 3. Statistics

**3a. Test choice — appropriate.** Paired Wilcoxon (defense masking, heuristic baselines, LODF), sign test (39/39 transfer) are correct for paired, non-normal seed/edge samples. OK.

**3b. Absurd p-values are large-n artifacts. [MINOR]**
"p<10⁻⁴³" (Mettack, 149/150 wins) and "p<10⁻⁵" (sign test, 39 cells) — the second is fine (n=39 → 10⁻⁵ is the floor of a sign test, reported honestly as the *floor*). The 10⁻⁴³ is over 150 paired edge-comparisons and is meaningless as a magnitude; it signals nothing beyond "n is large." **Fix:** replace bare p with **effect size + CI** (e.g., median damage ratio with bootstrap 95% CI, or rank-biserial r for Wilcoxon). Report p as "<0.001" and stop there.

**3c. ±SD vs 95% CI inconsistency. [MINOR]**
Setup (experiments.tex L8) declares "reported ± are standard deviations over seeds," and `tab:*` use SD. But `fig:breach` switches to "bands are 95% CIs," and Cora breach is "7.6% (CI [2.2,13.0])." Mixing SD (tables) and CI (breach figure) within one results section is defensible only if flagged. With n=10 seeds, SD and 95% CI differ by ~2.26× (t₉); the reader must not compare a ±SD table number against a CI-band figure. **Fix:** state in each table/figure caption which dispersion is shown (most do; `tab:explicit`/`tab:cross_domain` should reconfirm), and prefer 95% CI everywhere for n=10 since SD over 10 seeds is itself noisy.

**3d. No multiple-comparison correction across the many bolded contrasts. [MINOR]**
The paper makes dozens of significance claims (per-dataset Wilcoxon, 39 sign-test cells, per-case power-flow Wilcoxon p<0.01). No Holm/BH correction is mentioned. For the headline transfer claim (39/39, p<10⁻⁵) correction is irrelevant (single test). For the family of per-dataset/per-architecture contrasts it matters. **Fix:** one sentence stating BH-FDR control across the contrast family, or restrict significance stars to pre-registered primary endpoints.

**3e. n=10 seeds vs claim strength. [acceptable]** n=10 is standard for GNN robustness; effect sizes (3–10× Mettack, 3.2–4.1× random, 42% reduction) are large relative to reported SDs. Adequate. The weak link is *coverage* (one dataset/model for several headline numbers), not n.

---

## 4. Experimental Design

**4a. 50-node BFS subgraph reliance vs its own admission. [MAJOR]**
experiments.tex (full-graph-scale para) concedes: "A 50-node BFS covers only ~1.8% of a citation graph's edges (Cora τ=0.16)." Yet the default for §sec:cross_domain–§sec:defense_ablation — including the headline **42±8% defense** (`tab`-free, N=50), the **four-quadrant attack** (`tab:attack_full`, "50-node subgraph"), and the κ/ε_crit characterisation (`tab:cross_domain`) — is the 50-node subgraph. So several headline numbers live in a regime the paper shows is only 16%-faithful to the full graph. The mitigation (run on full graph at scale, where "edge advantage *amplifies* to 9.82× / 3.25×") is genuine and reassuring for the *ranking* claim, but it does **not** cover the defense (42%) or the attack-quadrant table, which are never re-run at full-graph scale. **Fix:** re-run the top-k defense and at least the SVD-vs-PGD damage comparison on the *full* Cora/Citeseer graph (matrix-free path already exists, validated to N=7,650). If 42% holds full-graph, move that number to the abstract; if it shrinks, rescope. The subgraph τ=0.16 admission is a loaded gun the paper hands the reviewer.

**4b. Power-flow surrogate fidelity vs "contingency screening" claim. [MAJOR]**
case_study.tex Setup: trained on "2,000 load samples per case (70–130% of nominal, *uniform load scaling only*; no seasonal/dispatch/renewable variation)." The paper *does* scope honestly ("rank-*ordering* triage, not post-contingency severity estimation"; abstract says "recovers N-1 rankings ... from a learned surrogate"). But even as triage, a surrogate that has never seen dispatch redispatch, renewable ramps, or topology changes cannot support a *contingency-screening* claim, because N-1 criticality is dominated exactly by the operating-point diversity the training set excludes. The τ=0.37–0.67 numbers are conditional on the test set being drawn from the *same* uniform-scaling envelope (in-distribution), which inflates apparent screening skill. **Fix:** evaluate the surrogate's N-1 ranking on *out-of-envelope* operating points (e.g., an N-1-induced redispatch, or a seasonal-peak case from a different load distribution). If τ/P@10 survive, the triage claim is earned; if they collapse, the case study must be retitled "in-distribution sensitivity attribution," not contingency screening. Also: ground-truth is single-snapshot ℓ₂ angle deviation per AC contingency — fine, but "brute-force N-1" should state convergence handling (non-converged / islanding contingencies are the most critical and the hardest for a GCN surrogate).

**4c. Power-flow finding is partly self-defeating for the method. [MINOR, but tell-tale]**
"*Binary adjacency beats admittance-weighting* (P@10=0.81 vs 0.27)" and "a single-line trip removes the line regardless of impedance" — this says the *signal is topological*, which is precisely why §2c's centrality baseline is mandatory. As stated it inadvertently argues the learned resolvent may be doing little beyond topology.

---

## 5. Reproducibility

**5a. Code release: not stated for the diagnostic path. [MAJOR for reproducibility]**
aegis.tex L64: "Anonymous Author(s)" (correct for blind review). conclusion.tex Disclosure protocol: attack-direction code (Alg. steps 3–4) is *gated*; the **diagnostic-only path (r_v, v_ij; no SVD reconstruction) is "released unconditionally."** Good — but "released" has no artifact: no anonymized repo link, no supplementary-material reference, no DOI. A reviewer cannot reproduce even the gated-free path without code. The hyperparameters *are* well-specified (d=64, lr=0.01, σ-norm via Miyato, rSVD k=p=10 n_iter=2, Neumann tol 10⁻⁶, K∈[20,50]), seeds=10, hardware partially (one GPU, 24 GB implied by "Pubmed exceeds 24 GB"), so re-implementation is *possible* but laborious. **Fix:** provide an anonymized code capsule for the diagnostic path at submission (the disclosure protocol already permits this); state exact GPU model and per-experiment wall-clock.

**5b. Dataset splits / subgraph selection — under-specified. [MINOR→MAJOR]**
- Subgraph selection: "50-node BFS ego-subgraphs centred on the highest-degree node" — reproducible *only* if the BFS tie-breaking and the highest-degree-node tie-breaking are fixed; with 10 seeds, is the *center* reseeded or fixed? Not stated. The full-graph "stratified top-v_ij N-1" (Amazon Photo, +0.996) is *not* defined: stratification variable, strata count, and sample size per stratum are missing — this directly gates the +0.996 headline. **Fix:** specify the stratification and the BFS center policy.
- Train/val/test splits: only "[Planetoid] Cora/Citeseer/Pubmed" implied via `sen2008collective`; standard public splits are *assumed* but not stated. **Fix:** name the split (public planetoid vs random) and ratio.

**5c. Brute-force N-1 ground truth — adequate but incomplete.** case_study.tex defines it as ℓ₂ voltage-angle deviation per AC contingency from PandaPower Newton-Raphson. Reproducible *given* the load samples; the 2,000-sample generator seed and the contingency set (all branches? all non-radial?) are unstated. **Fix:** state contingency enumeration and divergence handling.

---

## 6. Internal Consistency

Cross-checks across abstract / intro / tables / case study:

| Quantity | Where | Status |
|---|---|---|
| κ range 0.14–0.59 | `tab:cross_domain` (Amazon .14, Citeseer .59); scalability para "κ∈[0.14,0.59]" | **Consistent.** |
| ε_crit | `tab:cross_domain` (per-dataset) ↔ formula (1−κ)/‖W‖₂ ↔ intro/abstract | Consistent (presented as closed-form). |
| Spectral gap | framework.tex `fig:sc_heatmap` "43% gap ... cross-benchmark [0.39,0.50]" ↔ experiments.tex defense "43% spectral gap" | **Consistent** (single-instance 43% inside the [0.39,0.50] cross-benchmark range). OK. |
| 390 runs = 39 cells × 10 seeds | experiments.tex, `fig:tau_heatmap`, intro Contributions | **Consistent.** |
| ~6% accuracy cost | experiments.tex Setup ("~6% Cora accuracy") ↔ intro ("~6%") ↔ conclusion ("~6%") | **Consistent.** |
| **Power-flow τ ceiling** | case_study.tex text "**τ=+0.37 to +0.62**" vs `tab:ieee` case57 **τ=+0.67** | **CONTRADICTION.** Text understates max by 0.05; the body even later writes "0.62–0.67 on case57/118" in the Baselines para — so the *same section* gives two different τ ceilings (+0.62 in the intro para, +0.67 in the baselines para). **CRITICAL-accuracy, trivial fix:** make the headline range +0.37 to +0.67 (or clarify the +0.62 refers to a different metric/subset). The abstract avoids τ entirely (states only P@10), so the abstract is safe; the case-study prose is not. |
| P@10 0.66–0.81 | abstract ↔ case study ↔ `tab:ieee` (.66 case57 … .81 case118; case14 .74, case30 .68) | **Consistent** (range endpoints match table min/max). |
| 42±8% | abstract ↔ experiments.tex L140 | **Consistent** numerically; generality caveat is the only issue (§1). |
| "44±4% / 512-query" | experiments.tex only (not in abstract/intro) | Consistent internally; baseline identity unspecified (§2a). |

---

## Severity-Tagged Issue Register

**CRITICAL**
- **C1 — case_study.tex (intro para vs `tab:ieee` & baselines para):** Stated τ ceiling "+0.62" contradicts table case57 "+0.67" and the section's own later "0.62–0.67." *Fix:* correct the range to +0.37 to +0.67 (or disambiguate metric). Trivial edit; flagged CRITICAL only because it is a factual number mismatch a careful reviewer will catch.

**MAJOR**
- **M1 — experiments.tex L43 (`tab:attack_full` context):** 512-query "black-box search" is unspecified/random, a weak opponent used to argue the SVD direction is hard to find. *Fix/experiment:* re-run vs NES or SimBA-style query attacker at 512 queries.
- **M2 — case_study.tex (`tab:ieee`):** No graph-centrality (degree/betweenness/current-flow betweenness) baseline on the N-1 task, despite the paper's own "topology-driven" explanation predicting centrality could match S_c. *Fix/experiment:* add centrality rankers to `tab:ieee`; show S_c beats them or concede.
- **M3 — case_study.tex Setup:** Surrogate trained on uniform 70–130% load scaling only; N-1 evaluated in-distribution, yet claimed as contingency *screening*. *Fix/experiment:* evaluate N-1 ranking on out-of-envelope operating points (redispatch / seasonal peak); rescope if it degrades.
- **M4 — experiments.tex (defense para L140 & `tab:attack_full`):** Headline 42±8% defense and four-quadrant attack live only on 50-node subgraphs, which the paper shows are 16%-faithful (τ=0.16) to the full graph. *Fix/experiment:* re-run defense + SVD-vs-PGD on full Cora/Citeseer (matrix-free path exists).
- **M5 — abstract.tex L2:** "one query recovers the direction 50-step PGD finds" misattributes the cos=0.99 *surrogate-transfer* result to PGD (two separate experiments). *Fix:* rewrite to name the zero-gradient surrogate transfer; keep PGD as the 74–156× efficiency line.
- **M6 — reproducibility (conclusion.tex Disclosure; whole paper):** Diagnostic path declared "released unconditionally" but no anonymized artifact/link/supplement provided; stratified-N-1 protocol for the +0.996 headline undefined; BFS-center/seed policy and dataset-split identity unstated. *Fix:* ship anonymized diagnostic-path code capsule + specify stratification, split, BFS center.

**MINOR**
- m1 — "42±8%" presented in abstract as general; is single dataset/model/subgraph/k. Add qualifier or broaden.
- m2 — "+0.996 (Amazon Photo)" drops the "stratified top-v_ij" qualifier in the abstract.
- m3 — p<10⁻⁴³ (Mettack) is a large-n artifact; report effect size + CI, cap p at "<0.001."
- m4 — SD (tables) vs 95% CI (`fig:breach`) mixed; with n=10 they differ ~2.3×. Standardize on 95% CI; reconfirm dispersion type in `tab:explicit`/`tab:cross_domain` captions.
- m5 — No multiple-comparison (Holm/BH) correction across the contrast family; add a sentence or restrict stars to primary endpoints.
- m6 — 74–156× is a per-query *efficiency* ratio, not absolute damage superiority (PGD absolute damage is competitive); ensure skim-readers cannot over-read it.
- m7 — Power-flow "binary adjacency beats admittance-weighting (0.81 vs 0.27)" inadvertently argues the signal is topological → reinforces need for M2.
- m8 — Brute-force N-1 ground truth: contingency enumeration set and Newton-Raphson divergence/islanding handling unstated (these are the most safety-critical contingencies).

---

## Do the experiments support the headline claims?

**Conditionally yes for the diagnostic/transfer story, conditionally no for the contingency-screening story.** The transfer (39/39, τ=+0.99), ε_crit margin (2–4×), and matrix-free scaling (N=7,650, σ₁ to 0.03%) claims are well-supported and honestly hedged. The power-flow "contingency screening" claim (even scoped as triage) and the "black-box-hard direction" claim are not yet earned: they need (M1) a real black-box attacker, (M2) a centrality control, and (M3) out-of-envelope N-1 evaluation. The 42% defense (M4) should be lifted off the 50-node subgraph before it sits in the abstract. None of these are fatal; all are closable with experiments the authors' existing pipeline can already run.
