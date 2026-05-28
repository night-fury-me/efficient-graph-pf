# Phase 0 — Reviewer Configuration Card

**Paper.** AEGIS: Mining Graph Structure for Adversarial Vulnerability Analysis of GNNs.
**Format.** IEEEtran conference, single-column-equivalent, **exactly 10 pages** (hard cap confirmed via `pdfinfo`).
**Submitted PDF date.** 2026-05-29.

## Field signature
- **Primary discipline.** Adversarial machine learning on graph-structured data (GNN robustness).
- **Secondary disciplines.** (i) Implicit / deep equilibrium models (DEQ-GNN, IGNN); (ii) numerical linear algebra (Neumann series, randomized SVD, pseudospectra); (iii) cyber-physical systems / power-grid contingency screening (case study only).
- **Paradigm.** Mixed: one formal theorem + three propositions + one observation + one remark for IGNN-class operators, embedded in a heavily empirical study (9 datasets × 7 architectures × 4 domains, 330 runs, plus IEEE 14/30/57/118).
- **Maturity.** Late-stage manuscript; the page is fully saturated (every column is used) and limitations are already enumerated explicitly. Multiple revisions visible in `git status` (`MEMORY` indicates `bulletproof-over-handwaving` discipline).
- **Target venue band.** IEEE conference template + safety-critical framing + tiered-access disclosure language is consistent with IEEE S&P, IEEE TIFS-aligned venues, IEEE Big Data, ICDM (IEEE), or the AdvML / SaTML cluster. (IEEEtran `conference` mode rules out direct ICML/NeurIPS/ICLR submission as the primary target.)

## Page-budget calibration (binding for all reviewers)
The 10-page IEEEtran conference cap is **a hard constraint set by the user, not a defect**. Reviewers MUST NOT request:
- Full proofs in the body for results whose ingredients are classical (Stewart 1990, Trefethen-Embree 2005, Bolte-Pauwels 2021 are correctly cited).
- New sections that would push the paper past 10 pages.
- Restoration of material the authors clearly compressed on purpose (single-paragraph limitations, footnote-style proofs).

Reviewers MAY request, within budget:
- **Reallocation** between sections (e.g., trim a redundant table to make room for a missing analysis).
- **Sharpening** of claim language so abstract / contributions / theorem statements are faithful to the empirical evidence.
- **Movement** of derivations to an appendix / supplementary, as long as the body remains self-contained.
- **Replacing** a weak result with a stronger one when the replacement fits the budget.

This calibration is binding on Phase 1 and Phase 2.

## Dynamically configured reviewer panel

### EIC — Editor-in-Chief (journal/venue fit, originality, significance)
**Persona.** Senior editor at an IEEE-affiliated venue spanning ML security and graph mining (e.g., IEEE TKDE / TIFS area editor with track record at ICDM and IEEE Big Data). Tracks adversarial ML on graphs, has previously edited papers using DEQ / IGNN, comfortable with cyber-physical framings.
**Review preferences.** Wants the paper's contribution to be statable in one sentence; suspicious of scope creep across four domains in 10 pages; weighs whether the unification claim ("no existing method unifies the three outputs") is a real practitioner pain point or an authorial frame.
**Does not duplicate.** Methodology depth (R1's job), domain-specific GNN theory (R2's job), cross-disciplinary power-flow critique (R3's job), or hostile rhetoric (DA's job).

### R1 — Methodology Reviewer (research design, statistical validity, reproducibility)
**Persona.** Adversarial-robustness methodologist with hands-on PRBCD / GR-BCD / Mettack / randomized smoothing experience and a side interest in spectral methods. Reads Halko 2011, knows the difference between operator-norm and Frobenius-norm bounds, has implemented IBP certifiers.
**Particular focus.**
- Definition and semantics of "tightness ratio" (actual / predicted shift) — pushes hard on whether this is a virtue or a bound failure.
- Baseline coverage: is GR-BCD the right state-of-the-art? Where is PRBCD? Where is FGAttack? Where are Gosch/Mujkanovic 2023 sober-look results?
- Subgraph protocol: a 50-node BFS subgraph that correlates τ=0.16 with the full Cora graph is the operating regime of most reported numbers — is this fit-for-purpose?
- Tightness vs breach: are these on different quantities (equilibrium shift vs prediction flip)?
- Variance reporting: are 10 seeds enough; are confidence intervals reported where it matters?

### R2 — Domain Reviewer (literature coverage, theoretical framework, GNN-specific contribution)
**Persona.** GNN-theory researcher with deep familiarity with implicit / equilibrium GNNs (IGNN, MGNNI, DEQ-GNN), graph signal processing, Lipschitz GNNs, and spectral methods. Reads JMLR / TMLR theory tracks. Knows the el-Hamri / Revay / Gama-Bruna stability literature.
**Particular focus.**
- Is the "constrained sensitivity matrix" $S_c$ genuinely a new construction, or a standard symmetrization of the edge-pair Jacobian that the community already uses informally?
- Theorem 1 part (b) "critical regime divergence" — is the asymptotic rate worst-case along the top singular vector of $\hat{A}$, or does it bind on typical directions? Abstract claims "three-regime characterisation" — is the regime structure empirically demonstrated end-to-end?
- Observation 1 graph-independent η bound — depends on $\hat{A}$ symmetric + W spectrally normalized; how general?
- Missing references: Gosch 2024, Mujkanovic 2022, Bojchevski-Günnemann 2019 (collective robustness), Schuchardt 2021, PRBCD (Geisler 2021 cited but as GR-BCD).
- Whether the "any GNN with continuous edge-weight-modulated message passing" umbrella papers over real architectural differences (APPNP vs SAGE vs IGNN have very different sensitivity structures).

### R3 — Perspective Reviewer (cross-disciplinary, practical impact, fundamental assumptions)
**Persona.** Power systems / control engineer with strong CS background; published at IEEE TPWRS and IEEE TPS; familiar with PandaPower, Grid2Op, LODF, PTDF, NERC CIP. Comfortable reading GNN papers but evaluates them by their operational realism.
**Particular focus.**
- LODF baseline: industry workhorse at <0.13s vs AEGIS at 2–23s; the improvement from τ=0.44–0.58 to τ=0.62–0.67 on case57/118 is incremental. Trade-off is unfavorable for operations.
- "Without line-impedance data" advantage: false economy — operators always have impedance data.
- case300 fails: framework does not reach operationally relevant scale (real grids: 1000s–10000s of buses).
- Is the cross-domain case study scientific evidence for the unification claim, or just "we ran it elsewhere"?
- Cyber-physical security framing (NERC CIP-005, CIP-007 mention): is this integrated or decorative?
- Per-unit RMSE on case118: $\theta$ RMSE = 0.076 p.u. ≈ 4.4° is large for an AC PF surrogate; does this contaminate the contingency rankings?

### DA — Devil's Advocate (core argument challenge, logical fallacy, strongest counter-argument)
**Persona.** Hostile-but-fair top-tier reviewer who has rejected many flashy papers and is allergic to unification narratives that don't hold up at the operational scale.
**Particular focus (CRITICAL framing).**
- **Counter-argument #1.** The unification claim ("no existing method unifies the three outputs from a single closed-form computation") is an authorial frame. Practitioners don't shop for unified outputs; they pick the certificate they trust (smoothing) and the attack they fear (PRBCD). The unification is a presentation device, not a contribution.
- **Counter-argument #2.** The framework's most prominent quantitative claim ($S_c$ per-edge rankings) breaks at the operationally relevant scale: 50-node BFS subgraph rankings correlate τ=0.16 with full-graph Cora rankings. The authors flag this as a limitation, but it directly invalidates the "per-edge vulnerability ranking" output for any graph too large to analyze densely.
- **Counter-argument #3.** "Tightness ≥ 1" is reframed as a virtue ("linearisation underestimates damage"), but it is a bound failure. A 1.36 tightness at ε=0.20 means the first-order bound underestimates the true shift by 36%. Reporting this as "safe direction for a diagnostic" papers over a bound looseness.
- **Counter-argument #4.** "Formal track" applies only to a spectrally-norm-constrained IGNN that loses ~6% accuracy on Cora. Practitioners deploying high-accuracy explicit GNNs get only the computational tool, not the guarantees the title promises.
- **Counter-argument #5.** Cherry-picking risk on baselines: Mettack (149/150 wins) is reported on 3 datasets; only GR-BCD is treated as a serious external attack baseline. PRBCD (current SOTA on scale) is absent; Mujkanovic 2022 "Are Defenses Robust?" methodology is absent.

## Confirmation request to user
- Did I correctly identify the venue band (IEEE conference, security-leaning)? If the target is power-systems-specific (e.g., IEEE PES, PSCC), R3 should weight operational realism even more heavily.
- Are there fields you want to swap (e.g., replace R3 power-engineer with R3 GNN-explainability researcher)?

In the absence of explicit user override, the panel above is used for Phase 1.
