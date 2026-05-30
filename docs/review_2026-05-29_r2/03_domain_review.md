# Reviewer 2 — Domain Review (R2)

## Persona

Confirmed. PhD-level researcher with Mettack/Nettack/PR-BCD implementation experience; familiar with the sober-look line (Mujkanović 2022 NeurIPS, Gosch 2023 NeurIPS); recent author on smoothing-based certificates for graphs; frequent ICLR reviewer. My lens is threat-model completeness, head-to-head fairness, the principled vs. rhetorical use of "complementary," adaptive defense ablations, and ethics depth. I did not re-audit Theorem 1's proof (R1) or the power-flow numerics (R3).

## Summary (≤200 words)

R2 has materially improved the empirical scaffolding around AEGIS but the contribution that the paper sells — "closed-form diagnostics across the GNN vulnerability spectrum" — is now load-bearing on framing rather than on dominance. The GR-BCD comparison (Pubmed τ=0.69, Cora τ=0.16) is honest, the AGNNCert comparison (5–10× looser IBP radii, τ=0.08–0.14) is correctly run, and the "complementary" framing of AGNNCert is *principled in mathematical content but rhetorical in placement*: 9/10 Cora seeds have AGNNCert τ < 0.15 with a per-seed p-value distribution that cannot reject the null of zero rank agreement, yet `tab:baselines` reports a single +0.08 number as if it were a property of the pair rather than statistical noise. The full-graph τ on Amazon Photo (G8) is unsupported: `fullgraph_repro.csv` contains only Cora + Citeseer; the manuscript claim "$N{=}7{,}650$ shifts $\tau$ from $-0.14$ to $+0.24$" (experiments.tex L190) has no entry in the R2 CSVs. PR-BCD is name-dropped, never run. Mettack is run at $k\in\{1,\dots,5\}$ — its weakest regime. Sober-look is cited but not engaged. The threat model excludes insertions in §II, which I judge a *defensible scope choice* for this paper's claim but a *material limit on "vulnerability spectrum"*. R2 closes G6/G7 partially and leaves G8 open.

## Threat model audit

**Edge-deletion / weight-perturbation only.** §II (background.tex L25) restricts the perturbation to symmetric, supported-edge-only, continuous $\delta\Ahat$ with $\norm{\delta\Ahat}_F \leq \varepsilon$. Insertion attacks (Nettack-, Mettack-style) are explicitly out of scope; the limitation is also flagged in conclusion.tex (vi). The threat model is therefore *narrower than the GR-BCD/PR-BCD family it benchmarks against* — those attacks operate on the {0,1} candidate set $\bar E$ that includes non-edges, and the AEGIS pipeline cannot represent the Nettack/GIA insertion direction at all.

**Is this excusable for a "vulnerability spectrum" paper?** Partially. The paper's title and abstract sell coverage of "the full adversarial vulnerability spectrum"; the threat model covers approximately *half* of that spectrum (deletion/reweight, not insertion). The honest reading is that AEGIS is the closed-form diagnostic on the *edge-deletion sub-spectrum*. The current title and contribution-list overpromise; conclusion.tex (vi) is the right place but the *introduction* and *abstract* should explicitly say "edge-deletion vulnerability spectrum," not "the full vulnerability spectrum."

**Budget norm.** $\norm{\cdot}_F$ is fine for continuous-relaxation analysis but is non-standard for the structural-attack literature (the GR-BCD / PR-BCD papers use $\ell_0$ budgets on edge flips). The continuous-to-discrete bridge (Prop. \ref{prop:transfer}) is what carries the load here, but R3 should sanity-check the bridge constants.

## Baseline comparisons audit

### vs. GR-BCD (R2_01)

**Fair?** Yes, within the threat-model intersection. The protocol (10 seeds; $k\in\{1,5,10\}$; authors' reference implementation; same trained IGNN checkpoints; Kendall τ on per-edge scores) is the right protocol. CSV verifies: Pubmed at k=10 gives AEGIS 0.356 vs. GR-BCD 0.350, ratio 1.02, τ=0.685 (p≈0); Cora at k=10 gives AEGIS 1.045 vs. GR-BCD 1.440, ratio 0.73, τ=0.159 (p=0.15, not significant). These match Table tab:baselines (Pubmed k=10 column $0.356/0.350$).

**Does AEGIS actually win?** GR-BCD strictly dominates on damage on Cora at every $k$ (0.41–0.73 ratio). The Pubmed convergence is genuine. The paper's claim "the dataset-level $\tau$ acts as a self-diagnostic" is reasonable framing but it is post-hoc: the paper has not pre-registered a τ threshold above which AEGIS is "trustworthy." That makes the diagnostic claim weaker than its presentation.

The win that AEGIS *should* be claiming, but does not claim sharply enough in `tab:baselines`, is the operational profile: (i) no label access at attack/diagnostic time, (ii) closed-form deterministic SVD vs. iterative gradient descent, (iii) the matrix-free pipeline scaling to $N=7{,}650$ where GR-BCD's reference implementation OOMs. The 17× wall-clock penalty at small $N$ (acknowledged in `r2_framing_patches.md` but not in the manuscript) is honest. **Minor:** the wall-clock cost — 32.5 s on Pubmed vs. ~2.6 s on Cora/Citeseer — should be in the manuscript so the operational-profile claim is grounded.

### vs. PR-BCD

**Not run.** The paper says PR-BCD "sits beyond our matrix-free boundary" (experiments.tex L141) and "PR-BCD is the same family's larger-budget variant" (`tab:baselines` caption). There is no PR-BCD entry in `tab:baselines`, no PR-BCD CSV in `results/revision_R2/`, and no head-to-head on Pubmed or Amazon Photo. The introduction (L18) and abstract list PR-BCD as a head-to-head baseline; this is an overclaim. PR-BCD on Amazon Photo at $N=7{,}650$ is exactly where AEGIS's matrix-free claim should be empirically validated against the state-of-the-art iterative attacker at scale — and it isn't.

### vs. AGNNCert (R2_02 / Patch 3)

**Is "complementary" principled or rhetorical?** Mixed.

*Principled part:* AGNNCert produces sound IBP-certified radii; AEGIS $r_v=1/\sigma_1(S_v)$ is a first-order resolvent threshold under Theorem 1(a). These measure different mathematical objects, exactly as the patch text says. The 5–10× tightness gap is the expected order of magnitude for IBP relaxations on multi-layer GNNs.

*Rhetorical part:* The Kendall τ values reported (Cora 0.08, Citeseer 0.09, Pubmed 0.14) are weak rank agreements, but the per-seed picture is worse: Cora τ ranges 0.02–0.14 across 10 seeds, Pubmed has one *negative* τ (seed 137: −0.078). The CSV does not carry per-seed p-values, but a $\tau\approx 0.08$ on $n=50$ paired observations is well above $p=0.05$ for most of these seeds. The honest reading: on Cora and Citeseer, AEGIS and AGNNCert rank nodes *essentially independently*. That is fine for the "they measure different things" claim, but it is not "complementary" in the sense that "combining them gives you something" — there is no experiment in the paper that combines them. The current framing leans on "complementary" to convert an unfavorable disagreement into a feature.

**Was the AGNNCert experiment run at fair hyperparameters?** The protocol (authors' reference parameters, 50-node certification budget matched between methods, 10 seeds, no per-seed tuning) is fair on its face. I would have liked: (a) IBP radii at multiple input perturbation budgets, (b) an evaluation on a deeper IGNN where IBP is known to degrade most, (c) the AGNNCert ranking compared not against AEGIS $r_v$ but against the *attack ranking* AGNNCert was designed to defend against. None of these are run.

**MAJOR:** The Cora median AEGIS radius in the CSV is 0.4075 — `tab:baselines` reports 0.187. Either `tab:baselines` reports a different statistic (e.g. the average-over-seeds-of-median rather than the median-of-medians) or one of the two is wrong. Authors must clarify which number is in the manuscript.

### vs. Mettack

**Budget competitiveness.** Mettack is run at $k\in\{1,\dots,5\}$ (mettack_comparison.py L7, L259–261), and the paper reports "149/150 wins, one-sided sign-test $p{<}10^{-43}$" (experiments.tex L50). This is the Mettack budget regime where Mettack is *known to be weak*: Mettack's strength is exploiting the meta-gradient over many edge flips (typical literature budgets are 5–25% of edges, i.e. hundreds of flips on Cora). At $k=1\ldots 5$ on Cora-scale graphs, Mettack's surrogate-GCN meta-step has barely warmed up, and the Meta-Self objective (classification-margin-based) is not aligned with the equilibrium-shift metric AEGIS optimizes for.

The 149/150 claim is technically true but is essentially a *category error*: AEGIS is being compared to Mettack on AEGIS's home metric ($\norm{\Delta z^\star}$ equilibrium shift) at a budget where Mettack's home metric (margin flip) has not had time to drive damage. The paper would be substantially more credible if it either (i) ran Mettack at $k\in\{50,100,250\}$ on Cora/Citeseer and reported the result, or (ii) downscaled the claim to "AEGIS dominates Mettack on equilibrium-shift damage at small $\ell_0$ budgets, which is the regime relevant to early-warning diagnostics." Currently the abstract-level read of "149/150 wins" implies a much stronger result than the experiment supports.

### vs. LODF

Brief. LODF retargeting in Patch 2 (case57 thermal P@10 = 0.60 best-case; case118 P@10 ≤ 0.20) is honest, and the disclosure of case57-thermal as LODF's strongest corner is the kind of sober reporting Mujkanović 2022 asks for. R3 will go deeper on power-flow physics.

## Related-work coverage audit

**Sober-look engagement.** Mujkanović 2022 and Gosch 2023 are cited in introduction.tex (L8) and related_work.tex (L8). Each citation is one clause: "flag evaluation pitfalls." The defense-ablation section (sec:defense_ablation) invokes paired Wilcoxon "for this reason." But the paper does *not* perform the central evaluation Mujkanović asks for: an **adaptive attacker** against the defense. Mujkanović's main result is that nearly every published GNN defense collapses under adaptive evaluation; the AEGIS defense ablation explicitly states "the attacker is held fixed" (L154) and that "an adaptive variant recomputing $S_c$ would partially erode the gain." This is acknowledgment, not engagement.

**Gosch 2023.** Gosch's main contribution is robust certification with adversarial training and the demonstration that prior structural-defense claims are non-robust. The paper does not run any of Gosch's protocols and does not discuss whether AEGIS's $r_v$ would hold up under Gosch's evaluation. Mention without engagement.

**2024–2025 missing references.** The bib has only **5 papers from 2023–2025**: kim2025physicsinformed (self-cite), li2025agnncert, ieee2024testcases, schuchardt2023localized, gosch2023adversarial, nakiganda2023graph, digiovanni2023oversquashing (and a few 2022s). For a NeurIPS/ICDM-tier paper on graph adversarial robustness in 2026, this is thin. Specifically missing:
- **GIA literature.** Tao et al. *TDGIA*, Chen et al. *GIA-HAO*, the 2023–2024 GraphRobustness Benchmark papers. The paper's introduction motivates fraud-graph insertion attacks but then explicitly excludes insertions from the threat model and cites no GIA literature.
- **2024 PR-BCD updates.** Geisler et al. have follow-ups (*Robustness of GNNs at Scale*, NeurIPS 2024 workshop).
- **Recent certified-smoothing papers.** Scholten et al. 2024 (hierarchical smoothing extensions), Schuchardt et al. 2024 (localized-smoothing follow-ups).
- **Adaptive-attack-on-defense studies.** Gosch et al. 2024 follow-up, Mujkanović has follow-up work.
- **IGNN robustness.** Bai et al. 2023+ stability papers, Winston & Kolter monotone DEQs (already cited only as DEQ background).

This is a real coverage gap, not a stylistic complaint.

## Defense ablation audit

**G6 status.** Open. The non-adaptive defense (top-$k$ $v_{ij}$ masking) is run (Cora IGNN, $N=50$, 10 seeds; 42±8% damage reduction at $k=5$, 61±7% at $k=10$). The paper *explicitly* notes the attacker is held fixed and concedes an adaptive recompute "would partially erode the gain." There is no `tab:defense` showing both adaptive and non-adaptive columns. Given that Mujkanović 2022 is cited as motivation for the experiment, *not running the adaptive column is the central methodological hole of the R2 defense story*.

**Fix:** Add an adaptive column. The adaptive attacker is cheap here — recompute $S_c$ after each top-$k$ mask, restart, report damage. The infrastructure exists (`iem/examples/exp_adaptive_attack.py` is in the tree per my source-tree scan). This needs to be in the manuscript, not a footnote.

## Full-graph τ audit (G8)

**Where in the paper?** experiments.tex L190 (cross-dataset transfer paragraph): "Cold cells ... recover on full-graph matrix-free analysis ($N{=}7{,}650$ shifts $\tau$ from $-0.14$ to $+0.24$). Heterophilic WebKB GCN-4 attains $\tau=+0.21$--$+0.44$."

**Is it convincing?** No. **MAJOR issue.** The R2 CSV `results/revision_R2/fullgraph_repro.csv` contains only Cora and Citeseer rows; there is no Amazon Photo row. The script `scripts/exp_amazon_fullgraph.py` exists and explicitly targets Amazon ($N=7{,}650$, 3 seeds, 200 sampled discrete-truth edges), but the run did not land in the R2 CSVs that are tracked. The "$-0.14 \to +0.24$ on $N=7{,}650$" claim either references an earlier round of results not in `revision_R2/`, or it is a forward-reference to a run that did not complete (run_failed_round2.stdout shows R2_08_fullgraph_repro as PASS but the CSV columns confirm it only saved Cora/Citeseer).

The conclusion.tex (iii) limitation paragraph says the AtkAdv amplifies on the full graph and recommends the matrix-free pipeline at graph scale, but **there is no Amazon Photo full-graph τ in the manuscript's evidence base**. The Cora full-graph CSV reports AEGIS/random ratio 3.51, and the Citeseer ratio 4.60 — these are real results that *do* support the "amplifies on full graph" claim for citation graphs, but they do not address the dense-product-graph (Amazon Photo) cold cell in `fig:tau_heatmap`.

**Fix:** Either run the Amazon Photo full-graph experiment to convergence and ship the CSV, or remove the specific "$-0.14 \to +0.24$ on $N=7{,}650$" sentence and reframe as "the full-graph pipeline recovers cold cells on Cora ($\tau$ rises from $\langle \text{subgraph}\rangle$ to $\langle\text{full}\rangle$, see Tab.~X)." Shipping the specific Amazon number without the data is a citation-of-a-non-existent-experiment.

## Ethics audit

**Tiered code release + 90-day stakeholder notification.** Mentioned three times: experiments.tex L7 ("tiered access (attack-generation behind institutional-affiliation review)"), abstract L1 ("code will be released under tiered access"), conclusion.tex L8 ("90-day stakeholder notification and tiered-access for attack-generation"). 

**Adequate?** Performative without operational detail.

- *Tiered access* is real practice in dual-use ML (Anthropic, OpenAI red-team protocols) but typically requires (i) a named gatekeeper, (ii) a defined review criterion, (iii) a published process. None of these are in the manuscript.
- *90-day stakeholder notification* — to whom? Power-grid operators (NERC, ENTSO-E)? GNN-fraud-detection vendors? Academic GNN benchmarks? The paper does not name the stakeholders. For a paper producing **SVD-optimal closed-form attack directions** with proof of $0.72$--$0.92\times$ PGD parity (experiments.tex L60), the ethics framing needs more than a sentence.
- No mention of (a) responsible disclosure to GR-BCD/Mettack/AGNNCert authors that AEGIS provides a stronger label-free attack template, (b) IRB or equivalent ethics review, (c) data-use statements for the IEEE power-grid benchmarks (which are public but operationally sensitive).

This is closer to best-practice-aware-but-not-best-practice than to insufficient, but a NeurIPS-tier ethics statement would be more substantive. Given the paper produces a closed-form direction that flips up to 27% of Pubmed predictions at $\varepsilon=0.20$, the bar should be higher.

## Strengths (top 3)

1. **The AGNNCert and GR-BCD numbers are run honestly.** Pubmed-GR-BCD τ=0.69 with damage parity is a strong, defensible result. The CSV row-counts (90 GR-BCD, 30 AGNNCert) match the protocols described in `r2_experiments_full_report.md`.
2. **The "first-order threshold vs. sound certificate" disambiguation is correct mathematics.** AGNNCert and $r_v$ measure different objects; the paper says so plainly in Rem. \ref{rem:certificates} and the related_work paragraph. This is the right way to frame a non-dominance result.
3. **The matrix-free scalability claim is internally consistent.** Self-consistency at $N=200$ ($\sigma_1$ agreement 0.03%, per-edge τ=0.999), Neumann truncation bound, rSVD with documented hyperparameters. R1 owns the proof; from a domain standpoint the pipeline is the right pipeline.

## Weaknesses

### MAJOR

**M1. Amazon Photo full-graph τ is claimed but not in evidence.** experiments.tex L190 reports "$N=7{,}650$ shifts $\tau$ from $-0.14$ to $+0.24$"; `fullgraph_repro.csv` has only Cora/Citeseer. Either run the experiment and report the CSV, or remove the claim.

**M2. PR-BCD head-to-head is name-dropped, not run.** Introduction L18, abstract, and `tab:baselines` caption all mention PR-BCD; there is no entry. PR-BCD on Amazon Photo is the natural validation of the matrix-free scaling claim.

**M3. Mettack budget is the regime where Mettack is weak.** $k \in \{1,\dots,5\}$ on Cora is below the published Mettack operating range. The "149/150 wins" headline is misleading at this budget. Either expand the Mettack budget or downscale the claim.

**M4. AGNNCert "complementary" framing converts statistical noise into a feature.** The per-seed τ distribution on Cora cannot reject independence; calling that "complementary" without an experiment that *combines* the two diagnostics is rhetorical.

**M5. Adaptive defense column missing.** Mujkanović 2022 is cited as motivation, but the defense-ablation is non-adaptive only. The paper acknowledges this but does not run it. This is the central sober-look hole.

**M6. AEGIS-radius numerical mismatch.** `tab:baselines` reports Cora AEGIS $r_v=0.187$; `agnncert_comparison.csv` median is 0.4075. Authors must reconcile.

### MINOR

**m1.** Threat-model scope (deletion-only) should be in the *title* or *abstract*, not only the conclusion (vi) limitations.

**m2.** Bib has <8 papers from 2023–2025 for a 2026 graph-adversarial paper. Missing: GIA literature, 2024 PR-BCD follow-ups, 2024 smoothing follow-ups, recent IGNN stability work.

**m3.** "Sober-look engagement" beyond a single clause: the paper should run either (a) Mujkanović's adaptive-attacker protocol on the defense ablation or (b) at least one of Gosch 2023's robust-certification baselines.

**m4.** Ethics statement needs named stakeholders, named gatekeeper, defined review criterion. "Tiered access" without operational detail reads performative.

**m5.** GR-BCD wall-clock comparison should be in the manuscript (currently only in `r2_framing_patches.md` footnote).

**m6.** The phrase "the full adversarial vulnerability spectrum" (abstract, introduction) should become "the edge-deletion vulnerability spectrum" or similar, to match the threat model in §II.

## Recommendation

**Major Revision.**

R2 has done real work to bound the empirical claims (κ_max sweep, defensive theorem rewrite, GR-BCD + AGNNCert numbers), and the framework as a label-free closed-form proxy for edge-deletion structural vulnerability is a legitimate contribution. But the manuscript still over-sells: PR-BCD is absent, Amazon Photo full-graph τ is not in the CSVs, Mettack is run at its weak budget, the defense ablation is non-adaptive, and the AGNNCert "complementary" frame leans on rank-noise. None of these are unfixable. A focused round of revision — run PR-BCD on Pubmed + Amazon Photo, run the Amazon Photo full-graph τ to convergence, expand Mettack to $k=50+$, add the adaptive defense column, replace "full spectrum" with "edge-deletion spectrum" — would produce a paper I would champion. Without that round, the contribution is overstated for the bar at the target venue.

**Confidence: 4.**

## Open questions

1. Where is the Amazon Photo $N=7{,}650$ full-graph τ CSV? Is the manuscript number from an unsaved run?
2. Is `tab:baselines` Cora $r_v=0.187$ a per-seed median averaged over seeds, or a median over the pooled-seed distribution? Why does it differ from `agnncert_comparison.csv`'s 0.4075?
3. Will the authors run PR-BCD on Pubmed and Amazon Photo for R3, or remove PR-BCD from the contribution list?
4. What does an adaptive top-$k$-recompute defense ablation look like on the Cora IGNN — does the 42±8% damage reduction survive, or does Mujkanović's prediction hold?
5. At what Mettack budget does the 149/150 sign-test margin flip — is there a $k$ at which Mettack catches AEGIS on equilibrium-shift damage?
6. Who are the "stakeholders" in the 90-day notification, and what is the gatekeeping criterion for tiered code access?
