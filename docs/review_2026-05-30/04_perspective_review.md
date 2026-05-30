# Peer Review Report — Perspective

## Identity & Focus

**Peer Reviewer 3 — Perspective / Cross-Disciplinary.** Expertise: power-systems contingency analysis (N-1 screening, LODF/PTDF linear sensitivity factors, security-constrained operations) and the deployment/ethics of ML auditing-and-attack tooling in safety-critical settings.

This review is deliberately scoped to the cross-disciplinary and impact surface of the paper. I do **not** audit the core ML proofs (Theorem on the phase transition, the IFT-resolvent derivation, the randomized-SVD machinery); those belong to the methodology referees. I evaluate: (1) validity of the power-flow case study and the "competitive with industry LODF" claim; (2) the gap between motivating safety-critical stakes (drug interactions, fraud) and what is actually evaluated; (3) whether one method genuinely transfers across citation graphs and grids; (4) the dual-use / responsible-disclosure protocol; (5) practitioner actionability. I also score the optional **Significance & Impact** dimension.

Files read: `sections/case_study.tex`, `sections/experiments.tex`, `sections/introduction.tex`, `sections/conclusion.tex`, `sections/abstract.tex`, `sections/background.tex`, `aegis.pdf` (10 pp, letter).

## Recommendation + Confidence

**Recommendation: Minor Revision (weighted average 73/100).**

**Confidence: High** on the power-systems and impact assessment (this is my home turf); **Moderate** on how the cross-domain claims interact with the ML theory I did not re-derive.

The cross-disciplinary reach is genuine and, importantly, the paper is already self-aware: the case study carries explicit envelope caveats, concedes "competitive but not dominant," and the conclusion lists operator-grade deployment as future work. The remaining issues are framing-level (abstract/intro oversell relative to the body) and a small number of claims that should be softened or scoped, not new experiments. None rise to a blocking CRITICAL flag, but two are High-severity overclaims that a power-systems reader will catch immediately.

## Summary Assessment

AEGIS packages three adversarial diagnostics (SVD-optimal direction, per-edge sensitivity ranking, per-node radii) out of a single constrained-sensitivity object $S_c$, and then makes an ambitious cross-domain move: the *same* object that ranks vulnerable edges in citation graphs is applied to a learned AC power-flow surrogate to screen N-1 contingencies on IEEE case14--118 (\cref{sec:power_flow}, \cref{tab:ieee}, \cref{fig:ieee14_case}). That a label-free, model-intrinsic sensitivity score reproduces 7--8 of the brute-force top-10 critical lines, and is statistically ahead of LODF on the AC voltage-angle target for case57/118 (Wilcoxon $p<0.01$), is a legitimately interesting result and the most novel part of the contribution from a cross-disciplinary standpoint.

The problem is the framing ladder. The abstract (p.1) and case-study opener (\cref{sec:power_flow}) state AEGIS is "competitive with industry LODF," and the abstract and introduction motivate the whole paper with "drug-interaction screening" and "fraud detection" as safety-critical stakes. Neither claim survives contact with the body at full strength: (a) LODF is an *exact, closed-form, near-zero-cost* linear sensitivity factor, and the comparison is run on a metric (AC voltage-angle) that is not LODF's native target; on LODF's *own* fair target (thermal overload) AEGIS reaches only P@10=0.60, "competitive but not dominant" (\cref{sec:power_flow}, baselines paragraph); (b) neither drug interactions nor fraud is evaluated anywhere — the empirical datasets are five citation/co-purchase graphs (Cora, Citeseer, Pubmed, Amazon Photo, WikiCS) plus four IEEE cases. The stakes invoked to justify the work are rhetorical, not tested.

The good news: the fixes are almost entirely textual. The body already contains the honest numbers and the right caveats; the abstract and intro simply need to inherit them. With those adjustments the cross-domain story is defensible and even attractive.

## Strengths

1. **Genuine cross-domain reach with a non-trivial positive result.** Reusing one analytical object across citation graphs and an AC power-flow surrogate is a real generality claim, and the grid result is not vacuous: $\tau=+0.37$ to $+0.62$, P@10=0.66--0.81, recovering 7--8 of 10 brute-force-critical lines (\cref{tab:ieee}). The finding that *binary adjacency beats admittance-weighting* (P@10=0.81 vs 0.27 on case118) is a genuine domain insight with a correct physical rationale (a line trip removes the line regardless of impedance), and it is the kind of result a power-systems reader respects.

2. **Unusually honest case-study setup.** The setup paragraph (\cref{sec:power_flow}) volunteers that training data are "uniform load scaling only, a narrow envelope that does not cover seasonal peaks, dispatch shifts, or renewable ramps," and explicitly notes LODF's native metric is DC line-flow change while the benchmark uses AC voltage-angle truth. This is exactly the disclosure a cross-disciplinary referee wants and is rarer than it should be.

3. **Right baselines, fairly reported in the body.** The paper does not only compare to LODF: it adds the Ejebe--Wollenberg performance index and standalone PTDF, and reports that PTDF *without* outage correction is anti-correlated/near-zero ($\tau=-0.14/+0.06$). That last point is a correct and useful demonstration that outage redistribution (not raw flow) is the operative effect — a subtlety many ML-for-grid papers miss. Runtimes are disclosed (LODF <0.13 s; AEGIS 2--23 s).

4. **A serious, structured disclosure protocol exists.** The conclusion proposes a 90-day coordinated-notification window, gates the attack-direction code (Algorithm 1 steps 3--4) behind institutional-ethics review, and releases only the diagnostic-only path unconditionally. For a venue where most attack papers say nothing about dual use, having a tiered protocol at all is above the median.

## Weaknesses

### W1. "Competitive with industry LODF" is an apples-to-oranges headline (abstract p.1; \cref{sec:power_flow})
- **Problem.** LODF is an *exact, analytically derived* linear sensitivity factor computed in closed form from the network admittance/$B$-matrix at essentially zero cost (<0.13 s). AEGIS's $S_c$ is a *learned-surrogate* approximation costing 2--23 s and depending on a trained model with non-trivial error (below). Headlining "competitive with industry LODF" invites the reader to weigh a free exact tool against a costly approximate one as if they were peers. Worse, the headline comparison is scored on AC voltage-angle, which is *not* LODF's native target; on LODF's fair thermal-overload target, AEGIS is only "competitive but not dominant" (P@10=0.60). The body knows this; the abstract and case-study opener do not say it.
- **Why it matters.** A power-systems reader will immediately ask "why would I replace an exact, instantaneous LODF/PTDF screen with a learned surrogate that needs training, has voltage error, and runs 20--150x slower?" If the honest answer is "you wouldn't, for the classical screening task" — and the body implies that — then the value proposition is *not* "competitive replacement" but "a label-free vulnerability layer that happens to recover the same ranking, useful when you already have a GNN surrogate and no contingency labels." That is a fine and defensible claim, but it is a different claim.
- **Suggestion.** Reframe consistently as "recovers LODF-grade rankings as a by-product, without contingency labels or admittance data" rather than "competitive with industry LODF." In the abstract, replace "competitive with industry LODF" with something like "recovering LODF-grade N-1 rankings label-free from a learned surrogate." Explicitly state in the case-study opener that LODF remains the tool of choice when the admittance matrix is available, and position $S_c$ as complementary (model-auditing for an already-deployed GNN-PF surrogate), per \cite{varbella2024contingency}.
- **Severity: High.**

### W2. Surrogate fidelity is marginal on the larger cases, which weakens what "recovering N-1 rankings" certifies (\cref{tab:ieee})
- **Problem.** The reported per-unit RMSE rises to $|V|=0.033$ and $\theta=0.059$ on case57 and $\theta=0.076$ on case118. In power-flow terms a voltage RMSE of 0.033 p.u. (~3.3% of nominal, i.e. several kV on a 138 kV base) and an angle RMSE of 0.076 rad (~4.4 deg) are not operator-grade — they are coarse enough that the surrogate would mis-rank borderline contingencies. The paper benchmarks the *ranking* against brute-force N-1 computed on the *true* AC solver, but the ranking is produced from $S_c$ of an *imperfect* surrogate. So "recovering N-1 rankings" certifies that the surrogate's learned sensitivity structure is topologically correct, not that the surrogate is a faithful power-flow model.
- **Why it matters.** The distinction governs what a practitioner may conclude. The result supports "a coarse GNN-PF surrogate already encodes the right *topological* vulnerability ordering" — genuinely interesting. It does *not* support using this pipeline for security-constrained operations where the magnitude of the post-contingency violation matters, not just the rank. The paper occasionally blurs this (the case-study opener says $S_c$ "recovers brute-force N-1 critical-line severity," but severity = magnitude, whereas the evidence is rank correlation + P@10).
- **Suggestion.** State explicitly that the claim is *rank recovery from a coarse surrogate*, not surrogate fidelity, and that the magnitudes (RMSE in \cref{tab:ieee}) preclude operational use without a higher-fidelity model. Drop "severity" or qualify it as "severity *ordering*." One sentence noting what 0.033 p.u. / 0.076 rad mean physically would pre-empt the obvious referee objection.
- **Severity: Medium.**

### W3. P@10=0.66 is framed as a success without engaging operational acceptability (\cref{tab:ieee}, abstract)
- **Problem.** The lower end of the headline range, P@10=0.66 (case57), means roughly one in three of the top-10 flagged lines is a false positive relative to brute-force N-1 — and, symmetrically, real critical lines are missed. The paper presents 0.66--0.81 as a clean success ("recovers N-1 rankings ... competitive with industry LODF") without ever stating whether a one-in-three error in the top-10 is operationally tolerable.
- **Why it matters.** In grid operations the cost asymmetry is severe: a *missed* critical contingency (false negative) can mean an undetected N-1 violation, while N-1 screening is precisely the safety filter meant to catch these. A P@10 in the 0.6s would be unacceptable as a *replacement* for exact screening; it is, however, perfectly reasonable as a *label-free prioritization heuristic* or a sanity layer on a learned surrogate. The framing must pick the right register.
- **Suggestion.** Add one sentence acknowledging the false-positive/false-negative reading of P@10 and explicitly positioning $S_c$ as a *prioritization/triage* signal rather than a safety-certifying screen. Reporting recall at the operative top-$k$ (how many of the *true* critical lines are caught) would be more meaningful to operators than precision alone; consider adding it if space permits.
- **Severity: Medium.**

### W4. Motivating stakes (drug interactions, fraud) are never evaluated (abstract p.1; \cref{sec:intro})
- **Problem.** The abstract's first sentence and the introduction's opening both invoke "drug-interaction screening" and "fraud detection" as the safety-critical settings that justify the work, each with a citation (\cite{dai2018adversarial}, \cite{zugner2018adversarial}). But the evaluated datasets are Cora, Citeseer, Pubmed, Amazon Photo, WikiCS, and four IEEE power cases (\cref{sec:experiments}, Setup). No molecular-interaction graph and no transaction/fraud graph appears anywhere in the experiments.
- **Why it matters.** This is the classic motivation-vs-evaluation gap. Citation and co-purchase graphs are low-stakes; using high-stakes domains to set up the problem and then never testing them inflates perceived impact. A reader scanning abstract + tables could reasonably believe drug/fraud settings were evaluated. Two of the three "safety-critical" pillars are rhetorical.
- **Suggestion.** Either (a) soften the abstract/intro to "domains such as fraud detection and drug-interaction screening, *where structural errors carry high cost*" and make clear the empirical study covers citation/co-purchase graphs and power grids as *representative* graph types; or (b) add at least one genuinely safety-relevant graph (e.g., a public drug--drug-interaction graph such as a DDI/BioSNAP benchmark, or an OGB fraud-style dataset) to close the gap. Given the page limit, (a) is the realistic fix and is honest.
- **Severity: High** (it touches the central impact narrative and the abstract).

### W5. The "cross-domain transfer" is partly an artifact of a shared proxy, and this is under-acknowledged (\cref{sec:explicit_extension}, \cref{sec:power_flow})
- **Problem.** On the ML side, the discrete ground truth is brute-force N-1 *edge removal* on the GNN; on the grid side, the ground truth is brute-force N-1 *line outage* on the AC solver. So "the same method transfers across domains" is, in part, "the same edge-removal sensitivity proxy correlates with the same edge-removal ground-truth construction in both domains." That is real and useful, but it is a *narrower* statement than "AEGIS generalizes across fundamentally different problem regimes." The regimes also differ in a way the paper should flag: the citation-graph results lean on the matrix-free full-graph pipeline with the edge-weighted ranking $A_{ij}v_{ij}$ reaching $\tau=+0.996$ on Amazon Photo, whereas the *unweighted* binary ranking is what works for the grid (W1's binary-beats-admittance finding). The "one object" is applied with domain-specific edge-weighting choices that are not unified.
- **Why it matters.** A skeptical cross-disciplinary reader will note that the headline $\tau=+0.996$ (Amazon Photo, weighted) and the grid result (binary) use *opposite* edge-weighting conventions, which slightly undercuts the "one method, no retuning" message.
- **Suggestion.** Add a sentence reconciling the two: weighted ranking suits continuous edge-importance domains; binary ranking suits hard line-trip semantics. Frame this as a *feature* (the same $S_c$ supports both readings) rather than leaving the reader to notice the inconsistency. Make explicit that both domains share an edge-removal ground-truth construction, so the transfer claim is about edge-deletion sensitivity specifically.
- **Severity: Medium.**

### W6. The disclosure protocol has credibility gaps for a published attack tool (\cref{sec:conclusion})
- **Problem.** The protocol is structured but leans on unenforceable or under-specified mechanisms: (i) a 90-day coordinated-notification window "addressed to" benchmark maintainers and attack-paper authors — but academic-benchmark maintainers are not vendors with a patch pipeline, so what does the 90 days accomplish operationally? (ii) attack-code gating "behind institutional-affiliation review mediated by an institutional research-ethics office (named in the camera-ready)" — institutional affiliation is a weak filter (it neither prevents misuse by affiliated bad actors nor serves unaffiliated defenders), and the office is unnamed at review time; (iii) the diagnostic-only path is released unconditionally on the grounds it "cannot directly synthesise a perturbation" — but the per-edge ranking $v_{ij}$ *is* the vulnerable-edge list, and the limitations note that the edge-weighted ranking $A_{ij}v_{ij}$ alone reaches $\tau=+0.996$ against true N-1. A ranked list of the most damage-causing edges is most of the attack value even without the SVD reconstruction.
- **Why it matters.** For a tool whose explicit selling point is "which edges, if perturbed, would cause predictions to fail," the line between "diagnostic" and "attack target list" is thin. Releasing the ranking unconditionally while gating the direction may be drawing the gate in the wrong place. As written, the protocol reads as conscientious but somewhat performative — the strongest gate (institutional affiliation) is the weakest filter, and the unconditional release covers the highest-leverage artifact.
- **Suggestion.** (a) Justify *why* the ranking is safe to release given it is itself a target list, or move the full-graph ranking behind the same gate as the direction; (b) replace "institutional-affiliation review" with a defensible criterion (e.g., responsible-disclosure agreement, intended-use attestation), since affiliation is orthogonal to intent; (c) reconsider whether 90-day notification to *academic* maintainers is the right analogue — for the power-grid artifacts, the relevant disclosure analogue is the energy-sector CERT/ISAC process, not OGB maintainers, and the paper should say which model it is invoking. Name the ethics body or at least its type.
- **Severity: Medium.**

### W7. Actionability of the first-order radius for high-stakes decisions is overstated (\cref{rem:certificates}, \cref{sec:conclusion})
- **Problem.** The per-node radius $r_v$ is explicitly "a first-order threshold, not a probabilistic certificate" — a value that "can be violated at larger magnitudes." The paper validates that *every observed breach* satisfies $\varepsilon > r_v$ (so $r_v$ is empirically a reliable *lower* screening bound on the citation graphs). But for the safety-critical decisions the paper invokes (drug, fraud, grid), a first-order threshold that holds only locally and can be exceeded at larger perturbation magnitudes is a *triage* signal, not a decision-grade guarantee. The paper is mostly careful here (the limitation is stated), but the abstract/intro stakes imply more.
- **Why it matters.** A practitioner cannot act on "first-order safe up to $r_v$" as if it were a certificate in a setting where the adversary is not budget-limited. The honest use is "screen and prioritize," which again is the right register but not the one the high-stakes motivation sets up.
- **Suggestion.** In the practitioner-facing framing, state plainly: $r_v$ is a *screening/triage* threshold for prioritizing audit effort, complementary to (not a replacement for) certified-robustness methods, which the paper already contrasts against smoothing. One sentence on the intended workflow (who runs $S_c$, when, on what model, and what action a high $v_{ij}$ / low $r_v$ triggers) would materially improve actionability.
- **Severity: Low–Medium.**

## Detailed Comments

### Case study (power flow)
- The single most valuable domain result is the binary-vs-admittance finding (P@10=0.81 vs 0.27, case118). It is correct, non-obvious to ML readers, and physically well-motivated. Foreground it more; it is currently in a baselines paragraph and a footnote.
- The envelope ratio "$\approx 1.00$ in the PF domain" is reported as a strength, but it is measured on equilibrium shift, not on physical post-contingency quantities; given the surrogate RMSE in \cref{tab:ieee}, a tight first-order envelope on a coarse surrogate does not imply tight prediction of true post-contingency states. Clarify what the envelope certifies.
- LODF $\tau=0.44$--0.58 "on the credible cases" — define "credible cases" (case57/118?) explicitly; the qualifier currently does work without being pinned down.
- The case study uses *full-graph dense* analysis on case14--118 but the headline scalability ($N=7{,}650$) is from Amazon Photo, a citation graph. The grids tested are tiny (max 118 buses). Real screening targets thousands of buses; the paper should not let the Amazon scalability number stand in for grid scalability. State that grid-scale validation (1000+ bus systems) is open.

### Cross-domain
- The four-domain count (\cref{sec:experiments}) is generous: citation graphs (Cora/Citeseer/Pubmed), a co-purchase graph (Amazon Photo), a web/knowledge graph (WikiCS), and power grids. Three of four are homophilous information graphs; only the grid is physically grounded. "4 domains" overstates regime diversity — consider "graph types" or naming them, so the genuine grid transfer is not diluted by counting three citation-family graphs as distinct domains.

### Ethics / dual-use
- Credit where due: the tiered release and the choice to use only public IEEE benchmarks for the grid artifacts are the right instincts. The weakness is internal consistency (W6), not absence of effort.
- Consider stating an explicit *defensive* use case: AEGIS's primary intended user is the model owner hardening their own deployed GNN (the masking experiment, $42\pm8\%$ damage reduction, supports this). Leading with the defensive framing would make the unconditional diagnostic release more coherent.

### Impact / significance
- The defensive payoff (top-edge masking cuts $\sigma_1$ damage 42% vs 11% random, and survives adaptive recomputation) is, from an impact standpoint, the more deployable contribution than the attack direction, yet it is under-sold relative to the attack framing. Re-centering the paper as an *auditing/hardening* tool (with attack capability as the mechanism) would both raise its perceived significance and resolve much of the dual-use tension.
- Net significance: the method is broadly applicable (any continuous-edge-weight GNN), label-free, and matrix-free at scale — real strengths. The impact ceiling is limited by the fact that in the one physically grounded domain, an exact classical tool (LODF) already exists and is cheaper, so the grid contribution is "interesting confirmation" rather than "new capability the field lacked."

## Questions for Authors

1. Given LODF is exact and ~150x faster, what is the concrete practitioner scenario in which $S_c$-based screening is preferred over LODF/PTDF? Is it solely the "no admittance data / already have a GNN surrogate" case? Please state it explicitly.
2. With $|V|$ RMSE up to 0.033 p.u. and $\theta$ up to 0.076 rad (\cref{tab:ieee}), does the surrogate ever *invert* the ordering of two genuinely critical contingencies? What is recall at the operative top-$k$, not just precision?
3. Were any drug-interaction or fraud graphs evaluated and dropped, or were these domains motivational only? If motivational, will you soften the abstract/intro accordingly?
4. The headline $\tau=+0.996$ (Amazon Photo) uses edge-*weighted* ranking; the grid uses *binary*. Is there a single unified ranking choice, or is the edge-weighting domain-selected? If selected, on what basis at deployment time?
5. Why release the per-edge $v_{ij}$ ranking unconditionally when it is itself the vulnerable-edge target list (and $A_{ij}v_{ij}$ alone reaches $\tau=+0.996$)? What stops it from being used directly as an attack list?
6. For the grid artifacts, which disclosure analogue applies — academic-benchmark notification, or energy-sector CERT/ISAC coordinated disclosure? These have very different timelines and recipients.

## Minor Issues

- Abstract (p.1) and \cref{sec:power_flow} use the identical phrase "competitive with industry LODF"; whatever softening is chosen must be applied in both places (and in the figure caption / intro contributions list) to stay consistent.
- "recovers brute-force N-1 critical-line *severity*" (\cref{sec:power_flow}) — evidence is rank correlation + P@10, which is severity *ordering*, not severity magnitude. Adjust wording.
- \cref{fig:ieee14_case} caption reports "P@10 = 0.70 for this run" against the table's $0.74\pm0.12$; the reconciliation sentence is good, but consider plotting the mean run to avoid the apparent mismatch a quick reader sees.
- "credible cases" (baselines paragraph) is undefined — specify which cases.
- The "4 domains" claim recurs (abstract-adjacent intro, \cref{sec:experiments}); see Detailed Comments — naming them would be more precise than the count.
- Runtime footnote: AEGIS 2--23 s vs LODF <0.13 s is a 15--180x gap; this is relevant to the "competitive" framing and deserves a clause in the main text, not only the footnote.

## Dimension Scores

| Dimension | Weight | Score |
|---|---:|---:|
| Originality | 20% | 78 |
| Methodological Rigor | 25% | 72 |
| Evidence | 25% | 70 |
| Coherence | 15% | 74 |
| Writing | 15% | 76 |
| **Weighted average** | | **73.0** |
| *Significance & Impact (separate)* | — | **70** |

**Weighted average computation:** $0.20(78) + 0.25(72) + 0.25(70) + 0.15(74) + 0.15(76) = 15.6 + 18.0 + 17.5 + 11.1 + 11.4 = 73.0$.

**Significance & Impact (70):** The method is broadly applicable and the auditing/hardening payoff is real and deployable. The cross-domain reach to power flow is the standout novelty. The ceiling is set by (a) two of three motivating safety-critical domains being untested, and (b) the one physically grounded domain already being served by an exact, cheaper classical tool, which makes the grid result confirmatory rather than enabling. Re-centering on the defensive use case would raise this.

**Decision: Minor Revision (73.0, band 65--79).** The contribution is sound and the cross-domain result is genuinely interesting; the required changes are predominantly reframing the abstract/intro to inherit the body's honesty (W1, W3, W4, W7), one clarification on surrogate fidelity vs ranking (W2), one reconciliation of the weighted/binary inconsistency (W5), and a tightening of the disclosure protocol's internal consistency (W6). No new core experiments are strictly required, though adding recall-at-$k$ and/or one safety-relevant dataset would lift Evidence and Significance. No CRITICAL blocking issues.
