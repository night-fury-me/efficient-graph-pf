# Reviewer 3 — Perspective Review (Power Systems) (R2)

## Persona

Senior R&D engineer at a transmission utility, secondary academic appointment in operations research. Wood--Wollenberg (Ch.~11) is the desk reference; PTDF, LODF, contingency Performance Index (Ejebe \& Wollenberg, 1979), and the standard FERC/NERC operating-envelope vocabulary are first nature. Has supervised one ML-for-grid postdoc and reads NeurIPS/ICDM grid-GNN papers with skepticism — most "operator-grade" claims do not survive contact with an actual control room. My lens here: does the case study read as honest cross-domain demonstration, or as marketing dressed in case-file numbers?

## Summary (≤200 words)

The authors apply $S_c$ to AC power-flow contingency screening on IEEE case14--case300 and report Kendall $\tau{=}{+}0.37$ to ${+}0.62$ and $\mathrm{P}@10{=}0.66$--$0.81$ against brute-force N-1 ground truth, with LODF and (in passing) Ejebe--Wollenberg PI as physics baselines. Compared to R1, the case study is materially more honest: the abstract calls itself "proof-of-concept, not operator-grade"; case300 is explicitly excluded from headline numbers ($\theta$-RMSE $\approx 22.6^\circ$, "model un-converged"); the operating envelope is disclosed as "70--130\% uniform load scaling only"; binary-vs-admittance is given a defensible methodological argument (N-1 trips are all-or-nothing). However, the LODF retarget experiment (R2\_06) and the PI baseline (R2\_05) — both directly responsive to R1's power-systems requests — are compressed into a single sentence in `experiments.tex` rather than appearing as proper rows in `tab:ieee`; the LODF-thermal P@10=0.60 result is the most consequential newly disclosed number in the paper and it is easy to miss. PTDF as a standalone baseline (G9) is still absent; it is only implicit in the LODF definition. Acceptable as a cross-domain demonstration. Not yet acceptable as a fair head-to-head against power-systems baselines.

## Case study audit

### Operating envelope and ground-truth metric

`case_study.tex` line 8 now discloses the envelope plainly: 2{,}000 load samples at 70--130\% of nominal, *uniform load scaling only*. This is N-0 (intact-system) training with $\pm 30\%$ load perturbation. No N-1 perturbations, no dispatch shifts, no renewable ramps, no generator outages, no seasonal peaks. The N-1 ranking task is therefore an **out-of-distribution transfer claim**, not an in-distribution evaluation. Real contingency screening at a utility is calibrated on the 99th-percentile-load case and stress cases (snowstorm, summer peak, wind drop-off); the regime where contingency severity is operationally interesting is precisely the regime the model has not seen. The disclosure is adequate; the implication is not surfaced strongly enough. The reader should be told once, in plain text, "the $\tau$ values are transfer correlations from a $\pm 30\%$ N-0 training distribution onto N-1 ground truth; in-distribution N-1 calibration is future work."

Ground truth is **$\ell_2$ voltage-angle deviation per AC contingency** (line 8). This is the model's own training-loss surrogate. From a power-systems standpoint this is a defensible choice (it sits between AC angle stability and a coarse damage proxy) but it is **not** how N-1 severity is operationally defined: severity at a control center is a thermal-overload count (MW flow above MVA rating) and a voltage-magnitude-violation count ($|V| \notin [0.95, 1.05]$~p.u.). The authors do not compute those (limitation (iii) in the conclusion). LODF's native quantity is post-contingency *line-flow change*, which is what an operator screens on; AEGIS's native quantity is *learned model sensitivity*. The ground-truth metric chosen is therefore the one most aligned with AEGIS's loss and least aligned with LODF's design intent. This is disclosed ("LODF is benchmarked off-spec ... disadvantages LODF by construction", line 8), and the R2\_06 retarget experiment is the proper response. The disclosure is appropriate.

### AC vs DC choice

The model is trained on AC Newton-Raphson outputs and the ground truth is AC N-1; the LODF baseline is DC. This is the correct experimental choice — AC is operationally relevant — but it tilts the playing field against LODF, which by construction discards reactive flow and voltage. The text now acknowledges this on line 8. Good.

### Admittance vs binary topology

The paper reports binary adjacency beating admittance-weighted ($\mathrm{P}@10{=}0.81$ vs.\ $0.27$ on case118; case\_study.tex line 46). The R2 framing now provides a physical argument: "a single-line trip removes the line regardless of impedance, so binary sensitivity correctly models the all-or-nothing character of N-1." This is a **defensible methodological claim**, not a metric artefact. In contingency ranking, the relevant signal is which lines, if removed, displace the post-contingency state the most; admittance pre-weights the message-passing channel by line susceptance, which down-weights low-impedance lines whose removal is in fact disruptive. The Spearman near-zero between admittance magnitude and N-1 severity ($-0.01$ to $+0.15$) corroborates that high-admittance lines are not preferentially critical. A practitioner reads this as: AEGIS captures topology-level criticality without needing line parameters, which is a genuine feature for screening on incomplete data. The follow-up — "more sophisticated edge encoding with impedance, thermal rating, and voltage level as edge attributes is the right next step" — is the correct next sentence to write. I accept this analysis.

One residual issue: the paper says binary "beats" admittance on $\mathrm{P}@10$, but the runtime-stability and rank-stability analyses are reported only for the binary configuration. If a reviewer asks "is binary universally better, or just on case118," the paper cannot answer.

### case300 caveat handling

Case300 in `tab:ieee` reports $\theta$-RMSE $0.394$~p.u. ($\approx 22.6^\circ$) flagged with $\star$ and called "not a valid operational data point." The table caption now reads "case300 is a scalability stress-test (model un-converged, not an operational point)" and `case_study.tex` line 10 says explicitly: "its apparent $\tau{=}{+}0.72$, $\mathrm{P}@10{=}0.87$ are *conditioned on an unconverged model and excluded from headline numbers*." The abstract numbers ($\tau{=}{+}0.37$ to ${+}0.62$, P@10 $=$ $0.66$--$0.81$) are case14--118 only. This is the right handling.

I would still **move case300 into its own paragraph or a labelled subtable** rather than the same horizontal rule of `tab:ieee`. A casual reader looking at the row "case300 P@10 = 0.87" without parsing the dagger footnote will internalise an inflated headline. A 1pt visual separation, or a separate `tab:case300_stress`, would eliminate the risk. Minor revision item.

### Stability of P@10 / $\tau$ across cases

Across case14, 30, 57, 118: $\tau{=}\{+0.42,+0.37,+0.67,+0.62\}$; P@10 $=\{0.74,0.68,0.66,0.81\}$. The pairwise rank-stability across 10 seeds is $+0.40$--$+0.78$, higher on larger grids. case57 shows the highest $\tau$ but the lowest P@10 — typical of dense graphs where the top-10 is competitive. The values are credible and consistent with what a topology-driven proxy should achieve on small dense networks. Confidence intervals are wide on case14/30 (where the top-10 *is* most of the graph) and tighten on case118. From a practitioner perspective, P@10 in $[0.66, 0.81]$ on case118 means $7$--$8$ of the top-10 critical lines are recovered without solving any contingency — that is a legitimate, screenable signal.

## Baseline audit

### LODF retargets (Patch 2)

`docs/r2_framing_patches.md` Patch 2 proposed a new subsection "Power-grid screening: AEGIS vs.\ LODF across physical metrics" with a `tab:lodf` over three retargets ($\ell_2$ voltage-angle, thermal-overload count, voltage-magnitude violations) on case57 and case118. Per the experimental report, the headline result is: **LODF-thermal P@10 = 0.60 on case57**, within striking distance of AEGIS's $0.66$--$0.81$ band; LODF-voltage on case57 is *anti-correlated* ($\tau = -0.112 \pm 0.001$); on case118 every LODF retarget collapses to P@10 $\leq 0.20$.

What actually appears in the manuscript: a single sentence in `experiments.tex` line 130 — "LODF retargeted to thermal-overload peaks at $\mathrm{P}@10{=}0.60$ on case57 and collapses on case118" — and a single row in `tab:baselines`: "LODF case118 P@10 0.81 vs $\leq 0.20$". There is **no `tab:lodf`**, no separate subsection, no disclosure of the voltage retarget anti-correlation, no per-seed CI, no PTDF-on-its-own. The most operationally meaningful baseline comparison in the paper — LODF-thermal on case57 — is compressed into eleven words of body text in the wrong section.

This is materially honest (the number is reported and AEGIS does not claim to dominate) but **strategically minimal**. The R1 panel's P1.9 / P3.9 concerns asked for a fair-fight on LODF's native metric; the data exist (`docs/r2_experiments_full_report.md` R2\_06) and would fit in a half-column table. A power-systems reader who skimmed the paper would walk away thinking "LODF behind, AEGIS ahead" and would not know that on case57-thermal LODF reaches 0.60 against AEGIS's 0.66. That asymmetry between what the data show and what the reader sees is the textbook definition of motte-and-bailey framing, even though the bailey number is technically printed.

### PTDF (G9)

The R1 panel's G9 explicitly asked for a PTDF baseline. PTDF is the precursor sensitivity to LODF (`LODF_ij = PTDF_ij / (1 - PTDF_jj)`); reporting PTDF would test whether the AEGIS proxy beats the *most basic* DC-PF sensitivity, not just the line-outage closure of it. The paper says (experiments.tex line 6): "LODF as $\mathrm{PTDF}_{ij}/(1{-}\mathrm{PTDF}_{jj})$ from the DC $B$-matrix" — i.e., PTDF is computed as an *internal step* but never reported as its own baseline column. From a power-systems standpoint, PTDF on its own (no outage correction) is a weak baseline and the case-118 ranking it would induce is roughly $\tau \approx 0.20$--$0.35$ in my experience; LODF dominates it precisely because LODF accounts for the post-outage redistribution. So the paper's omission of PTDF probably understates AEGIS's advantage rather than overstates it, but the R1 reviewer asked for it and the authors elected not to add it. The workaround is "LODF subsumes PTDF" — this is technically correct but the R1 ask was specifically for the head-to-head row. Minor revision.

### Performance Index (R2\_05)

I verified `results/revision_R2/pi_baseline.csv` (20 rows, 2 cases $\times$ 10 seeds). Implementation per the report uses Ejebe \& Wollenberg's canonical $\sum_l (P_l / P_l^{\max})^{2n}$ with $n=2$, computed per-contingency from PandaPower load-flow outputs. The PI is then ranked, and Kendall $\tau$ and P@10 are computed against the true N-1 ranking from PandaPower contingency screening. The formulation and ground truth are correct. Numerical results:

- **case57**: $\tau = +0.335 \pm 0.006$, P@10 $= 0.500 \pm 0.000$, $p$-value $\sim 2 \times 10^{-4}$ (significant).
- **case118**: $\tau = +0.101 \pm 0.004$, P@10 $= 0.300 \pm 0.000$, $p$-value $\sim 0.055$ (borderline non-significant).

Two points. First, **the implementation is correct** — exponent $2n$ with $n=2$, $P_l^{\max}$ as line MVA rating, sum over lines $l$, per contingency, ranked. Second, **the result is not in `tab:ieee`**. The integration plan (`r2_experiments_full_report.md` §12) said "R2\_05 PI → Add PI baseline row to the existing case-study table." That row is not in `case_study.tex`; PI appears only as a single clause in `experiments.tex` line 130 ("Ejebe--Wollenberg PI is positively correlated but loses to AEGIS on both grids"). For a power-systems reader the PI is the most canonical screening baseline besides LODF; not putting its numbers in the case-study table is a strategic choice that disadvantages the reader more than it disadvantages AEGIS. The PI result on case57 ($\tau{=}+0.335$, P@10 $=0.50$) is actually a *favorable* comparison for AEGIS ($\tau{=}+0.67$, P@10 $=0.66$); the paper would lose nothing and gain credibility by tabulating it explicitly.

### Comparison framing — honest or motte-and-bailey?

Motte: "On IEEE power-flow benchmarks (proof-of-concept, not operator-grade), $S_c$ correlates with N-1 contingency rankings on case14--118 ($\tau{=}+0.37$ to $+0.62$, P@10 $=0.66$--$0.81$)." (abstract). This is honestly framed and matches `tab:ieee`.

Bailey: the case-study `\textbf{Baselines and stability.}` paragraph in `case_study.tex` line 46 reads "LODF (industry DC screening) attains $\tau{=}0.44$--$0.58$ on credible cases, behind AEGIS's $0.62$--$0.67$ on case57/118 (Wilcoxon $p{<}0.01$)" without noting that LODF retargeted to thermal hits P@10 $= 0.60$ on case57. The experiments.tex paragraph (line 130) does note it. A reader reading only the case-study section gets the more flattering comparison; a reader reading the prior-baselines comparison gets the qualified one. This is the structural definition of a motte-and-bailey: the strong claim is in the prominent section, the qualification in the back-half. **Fix: forward-reference the LODF retarget result into the case-study baselines paragraph.** One sentence: "When LODF is retargeted onto its native thermal-overload metric, case57 P@10 reaches $0.60$ (Table~\ref{tab:baselines}), within striking distance of AEGIS's 0.66."

## Cross-disciplinary impact assessment

This paper is **a useful new tool**, conditional on being read as advertised: a closed-form, label-free ranking proxy that recovers most of the LODF / PI ranking signal without the line-parameter database or PTDF computation, using a GNN trained on AC load flow. For a utility R\&D group with a GNN-PF surrogate, AEGIS gives a "free" first-cut criticality ranking out of the same model. For an academic operations researcher, it is interesting as a sensitivity-spectrum view of a contractive learned operator — a clean Frobenius-norm-bounded analogue of $J^{-1}$-based screening.

It is **not yet operator-grade**, and the authors say so. The N-0-only training, the lack of thermal/voltage-violation severity, the 23-second per-case runtime against LODF's 0.13 seconds, and the 200-node BFS-subgraph requirement for case300 all rule out a control-room deployment as it stands. The "tiered access" code release is appropriate for an adversarial-tool paper, but a control-room can do nothing with code behind institutional review — that is purely an academic-ethics safeguard, not an operator-handoff. The positioning is honest.

It is **not marketing**. Honest cross-domain demonstration with the right caveats. The one thing keeping it from being fully clean is the asymmetric reporting of LODF/PI baselines — strong number in the main table, qualification a section away — which a power-systems reviewer will read as soft-pedaling. Fixable with a forward reference and a PI row.

## Strengths (top 3)

1. **Restraint in framing.** The abstract, introduction, and case-study opening all say "proof-of-concept, not operator-grade." Limitation (v) in the conclusion specifies what operator-grade would require (admittance-weighted edges, broader envelope, thermal/voltage severity). This is exactly the calibration a power-systems reviewer wants to see.
2. **case300 properly demoted.** $\theta$-RMSE $22.6^\circ$ is called what it is — an unconverged model, not an operational point — and excluded from headline numbers. Many ML-for-grid papers would not do this.
3. **Binary-vs-admittance gets a physical argument, not just a number.** "All-or-nothing character of N-1 trips" is the right explanation; it converts a surprising metric finding into a methodological insight.

## Weaknesses (MAJOR / MINOR with location + fix)

- **MAJOR — PI baseline not in `tab:ieee`.** Location: `paper/sections/case_study.tex` table around line 15--40 (`tab:ieee`). The PI numbers (`results/revision_R2/pi_baseline.csv`: case57 $\tau{=}+0.335$, P@10 $=0.50$; case118 $\tau{=}+0.101$, P@10 $=0.30$) exist and are favourable to AEGIS. Fix: add a row "PI (Ejebe--Wollenberg)~\cite{ejebe1979automatic}" to `tab:ieee` for case57 and case118, with "--" for case14/30/300, and a footnote pointing to the formulation. This is the single highest-leverage change for the power-systems reading.

- **MAJOR — LODF-retarget result buried.** Location: `paper/sections/case_study.tex` line 46 (case-study baselines paragraph) and `paper/sections/experiments.tex` line 130. The case-study baselines paragraph reports LODF $\tau{=}0.44$--$0.58$ against AEGIS-native voltage-angle ground truth without disclosing that on its own native metric (thermal-overload on case57) LODF reaches P@10 $= 0.60$. Fix: insert one sentence forward-referencing the LODF-thermal retarget result, or promote Patch 2's `tab:lodf` into the case-study section as originally planned in the integration recommendations (`docs/r2_experiments_full_report.md` §12). Either is acceptable; both honestly disclose the closest comparison.

- **MINOR — PTDF as a standalone baseline omitted (G9).** Location: `paper/sections/experiments.tex` line 6 ("LODF as $\mathrm{PTDF}_{ij}/(1{-}\mathrm{PTDF}_{jj})$"). PTDF is computed internally but not reported as its own baseline. Fix: add a one-row PTDF entry (no outage correction) to `tab:ieee` or `tab:baselines`. PTDF will lose to AEGIS more clearly than LODF does, so this strengthens the case rather than weakens it; the value is in answering G9 directly.

- **MINOR — case300 shares the table with case14--118.** Location: `tab:ieee` in `case_study.tex` lines 15--40. The midrule plus daggered footnote works for a careful reader but invites mis-citation. Fix: split into a separate stress-test subtable or move case300 into the body text only.

- **MINOR — operating-envelope language is buried in the Setup paragraph.** Location: `case_study.tex` line 8. The clause "70--130\% uniform load scaling only, a narrow envelope that does not cover seasonal peaks, dispatch shifts, or renewable ramps" is correct but stated once in a long setup paragraph. Fix: surface this as the first sentence of a "Generalization caveats" or "What this is not" paragraph adjacent to the headline numbers. A reader who only reads the results table should encounter the caveat at the same eye-level.

- **MINOR — N-2 multi-line analysis is a single sentence.** Location: `case_study.tex` line 46. Top-5 overlap 40--64\% edge-wise, 7--18\% pair-wise. The pair-wise number is low and the paper notes it. Fine, but for a power-systems reader N-2 is increasingly normative (NERC TPL-001-5); state once that "N-2 is not within the scope of this proxy" rather than burying the negative result. No new experiment needed.

## Recommendation

**Minor Revision.**

The case study has tightened materially from R1 to R2. Framing is honest (proof-of-concept, not operator-grade), the operating envelope is disclosed, case300 is demoted out of headline numbers, and the binary-vs-admittance finding has a defensible physical argument. The PI baseline implementation is correct and verifies against the canonical Ejebe--Wollenberg formula. The remaining issues are presentational, not substantive: the PI numbers and the LODF-thermal retarget number deserve to appear in `tab:ieee` (or via a tightly cross-referenced sentence in the case-study baselines paragraph) so a power-systems reader does not have to triangulate across two sections to see the closest baseline comparison. None of these fixes invalidate the result; all are within editorial reach in one revision pass.

**Confidence:** 4 / 5.

## Open questions

1. **Generalization across operating envelopes.** Does AEGIS's $\tau$ vs N-1 ranking degrade if the test contingency is computed at a stressed dispatch (e.g., a Grid2Op-style winter-peak load and a generator out) that the GNN was not trained on? The Grid2Op reference is in the limitations; an actual stressed-envelope evaluation would convert "transfer to N-1" into a real generalization claim.
2. **AC-vs-DC sensitivity comparison.** If LODF were redefined on an AC sensitivity matrix (e.g., the full ${\partial S}/{\partial \theta}$ Jacobian of the AC power-flow equations rather than the DC $B$ matrix), would the LODF $\tau$ on case118 still collapse to $\leq 0.20$? The collapse is partly an artefact of DC vs AC. An AC-PTDF baseline would be the genuinely fair comparison.
3. **What does AEGIS rank on a real grid?** The paper closes with "operators have impedance data, so the demonstration's value is conceptual." Are the authors prepared to evaluate $S_c$ on a synthetic-but-realistic case2000 or RTE-1888 with thermal-violation ground truth? That is the bar that converts conceptual into useful.
4. **N-2 specificity.** Pair-wise top-5 overlap with N-2 is 7--18\%. Is the SVD's leading singular direction the wrong object for N-2, or could the top-$k$ singular directions yield N-$k$ rankings? A short experiment with $v_1, v_2, v_3$ jointly would either close or open this question.
5. **Tiered code release operationally.** What does "tiered access with attack-generation behind institutional-affiliation review" mean for a utility analyst without a university affiliation? If the answer is "not accessible," the case study is academic-only; if "accessible to operators," the framing should say so.
