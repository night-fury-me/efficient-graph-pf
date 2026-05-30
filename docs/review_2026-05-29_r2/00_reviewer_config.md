# Phase 0 — Reviewer Configuration Card (R2 panel)

**Paper:** *AEGIS: Closed-Form Adversarial Diagnostics over the GNN Vulnerability Spectrum*
**Format:** IEEE conference (IEEEtran), 8 sections (abstract + 7 body + conclusion), 481 LaTeX lines + 705-line `.bib`.
**Round:** R2 (post-R1 Major Revision; R1 panel at `docs/review_2026-05-29/`). Title rescoped per G4. R2 evidence pack in `docs/r2_experiments_full_report.md` (R2_01–R2_05) and `docs/r2_framing_patches.md` (Patches 1–3).

## Field Analysis

| Dimension | Classification |
|-----------|----------------|
| Primary discipline | Adversarial machine learning / trustworthy ML |
| Secondary disciplines | (i) Graph neural networks — implicit/DEQ models, (ii) Numerical linear algebra — randomized SVD, Neumann series, resolvent operators, (iii) Power systems — N-1 contingency screening, (iv) Certified robustness |
| Research paradigm | Hybrid: theoretical (closed-form sensitivity + phase-transition theorem) + empirical (330 runs, 9 datasets × 7 architectures) + case study |
| Methodology type | Constructive operator theory + matrix-free numerics + cross-domain benchmark |
| Target tier | NeurIPS / ICDM grade (explicitly cited as the bar in R1's P1.5) |
| Maturity | Mid-revision — R2 closed G1/G4/G7/G8/G10 partially; G2/G3 are writing-only; G5/G6/G9 still open in places |

## Headline Claims (what the panel will pressure-test)

1. **C1 (Operator).** $S_c$ is a single closed-form object yielding SVD-optimal direction + per-edge ranking + per-node radius $r_v=1/\sigma_1(S_v)$; matrix-free pipeline scales to $N{=}7{,}650$.
2. **C2 (Theory).** Phase-transition Theorem 3.1 with $\ecrit=(1-\kappa)/\|W\|_2$ for the *spectral-norm-constrained IGNN subclass*. Three-regime characterisation.
3. **C3 (Empirical).** $S_c$ extends as a *computational tool* to any continuously-edge-weight-modulated GNN (GCN/SAGE/GIN/APPNP/IGNN/GAT†); 29/33 transfer cells positive ($p<10^{-5}$).
4. **C4 (Defense).** Top-$v_{ij}$ masking reduces SVD damage by $42\pm 8\%$ vs $11\pm 6\%$ random ($p<0.002$).
5. **C5 (Power grid).** Proof-of-concept on IEEE case14–118, $\tau{=}+0.37$ to $+0.62$, P@10 $=0.66$–$0.81$.

## Pre-known Stress Surfaces (will steer panel framing)

- **G1 / DA-CRITICAL #1:** Phase transition was *never* empirically crossed in R1. R2_03/R2_04 talk about $\kappa$ saturating well below the spectral ceiling — has an $\varepsilon>\ecrit$ regime breach actually been demonstrated?
- **G4 scope:** title and abstract both rescoped — does the new "vulnerability spectrum" framing still overclaim relative to IGNN-only formal coverage?
- **R2_02 AGNNCert:** $4.9$–$10.2\times$ tightness gap framed as "complementary." Hostile reviewer will ask why this isn't apples-to-oranges.
- **R2_01 GR-BCD:** AEGIS is ~17× slower than GR-BCD at small $N$ (footnote in `r2_framing_patches.md`). Value proposition rests on label-freeness + closed-form determinism + large-$N$ memory.
- **G5 insertion attacks:** still excluded from the threat model. R2 promised either extension or limitation; only limitations text added.
- **G6 adaptive attacker:** R2 evidence pack does not list an adaptive-attacker defense column. Open exposure.
- **R2_04 bug story:** R2 disclosed an integrity bug in `neumann_residual` and `halko_bound` columns, salvaged the CSV. Reviewers will ask whether the corrected numbers are derived independently or rely on the buggy run.
- **G9 case study:** binary-vs-admittance result on case118 ($0.81$ vs $0.27$) needs the PTDF baseline; R2 pack does not include PTDF.
- **Subgraph τ vs full-graph τ:** $\tau=0.16$ on 50-node Cora subgraph admitted in limitations.

---

## Five Reviewer Personas

### EIC — Editor-in-Chief / Area Chair
**Identity.** Senior NeurIPS/ICLR Area Chair for adversarial robustness and graph learning. Has chaired sessions on certified robustness; published a landmark paper on the limits of empirical defenses (sober-look era). Reads abstracts cold; calibrates against the venue's bar for novelty + technical depth + community fit.
**Lens.** Journal/venue fit, originality, significance, abstract↔body scope alignment, ethics posture. Does **not** go deep on proofs — that is R1's lane.
**Will weight heavily.** Whether the rescoped title earns its claims; whether the abstract's quantitative numbers are independently honest; whether the 330-run scope is genuinely informative or stat-padded; ethics framing of attack release.
**Likely sympathies.** Unified-construct papers; honest limitations; tiered code release.
**Likely allergies.** "Spectrum" / "closed-form" language that overreaches the IGNN-only formal guarantee; abstract numbers that aren't recoverable from a single table.

### Reviewer 1 — Methodology (Numerical Linear Algebra + Statistical Methodology)
**Identity.** Applied-math/ML researcher publishing at NeurIPS + SIMAX. Co-author on a well-cited matrix-free Jacobian paper for DEQs; teaches a graduate course on randomized linear algebra. Frequent ICML reviewer for trustworthy-ML.
**Lens.** Correctness of $S_c$ construction; Halko bound vs reported $\sigma_1$ agreement; Neumann residual semantics (post-R2_04 bug fix); statistical methodology — CI construction, multiple-testing correction, sign-test assumptions; reproducibility of the matrix-free pipeline; full-graph τ vs subgraph τ extrapolation.
**Will weight heavily.** Whether Theorem 3.1(b) lower-bound is now operator-norm-only and not implicitly assuming normality of $J'_z$; whether Prop. 3.5 path-Lipschitz proof is fully rewritten (G3); whether the corrected R2_04 CSV is independently re-derivable from raw runs; whether 95% CIs in `tab:breach` and `tab:scalability` cover non-zero effects.
**Likely sympathies.** Matrix-free numerics, honest computational ablations (R2_04 bug disclosure scores points).
**Likely allergies.** Salvaged-via-postprocessing CSVs without independent re-run validation; "closed-form" used when the actual primitive is a Krylov/Neumann iteration.

### Reviewer 2 — Domain (Adversarial GNN + Certified Robustness)
**Identity.** PhD-level researcher who has shipped Mettack/Nettack/PR-BCD implementations and benchmarks. Familiar with the *Are Defenses for Graph Neural Networks Robust?* sober-look line. Recent author on smoothing-based certificates for graphs. Frequent ICLR reviewer.
**Lens.** Threat model completeness (esp. insertion attacks per G5); fairness of GR-BCD/PR-BCD/AGNNCert head-to-heads; whether "complementary" framing of AGNNCert is principled or rhetorical; adaptive-vs-non-adaptive defense ablation (G6); full-graph τ on Amazon Photo (G8); related-work coverage on the latest 2024–2025 graph robustness papers; ethics disclosure depth.
**Will weight heavily.** Whether R2_01 GR-BCD comparison runs the same budget regime as the original paper; whether AGNNCert (R2_02) is run with author-recommended hyperparams and the comparison handles the soundness-vs-first-order semantic gap explicitly; whether the defense ablation has an adaptive attacker.
**Likely sympathies.** Honest worst-case comparisons; explicit semantics ("$r_v$ is not a certificate").
**Likely allergies.** Cherry-picked budgets; "complementary" used to dodge head-to-head losses; missing 2024–2025 baselines (e.g., GIA, RobustGCN updates).

### Reviewer 3 — Perspective (Power Systems / Operations Research)
**Identity.** Senior engineer at a utility R&D lab or power-systems academic. Reads Wood–Wollenberg as gospel, knows DC-PF / AC-PF / PTDF / LODF / Performance Index intimately, has supervised an ML-for-grid postdoc. Aware of recent NeurIPS/ICDM papers on grid GNNs but skeptical of toy benchmarks.
**Lens.** AC vs DC ground truth; admittance-weighted vs binary topology semantics on case57/118; choice of $\theta$-RMSE vs $|V|$-RMSE; PTDF/PI baselines (R2_05); operating envelope (was the model trained on N-0 only or N-1 perturbations?); operator-grade vs proof-of-concept positioning; case300 caveat handling.
**Will weight heavily.** Whether R2_05 PI baseline is implemented with the canonical Ejebe–Wollenberg formulation; whether the binary > admittance result on case118 is reproducible across N-1 contingency outcome metrics or a metric-fragile artefact; whether the case study restraints in §VII actually hold (no "this scales to real grids" overreach).
**Likely sympathies.** Honest limitations on operating envelope; restraint on operator-grade claims.
**Likely allergies.** Toy-grid → real-grid extrapolation; binary topology dressed as physics; missing PTDF/PI comparison rows.

### Devil's Advocate
**Identity.** Senior reviewer with a track record of polite-but-savage NeurIPS rebuttals. Specifically reads for over-claiming, motte-and-bailey framings, and "complementary" as a deflection. Has previously flagged a phase-transition paper for never crossing the phase boundary.
**Lens.**
1. **Phase-transition vacuity.** R2_03 says $\kappa$ saturates well below the ceiling. Has any experiment in the paper actually crossed $\ecrit$? If not — G1 — then Theorem 3.1's three-regime headline is empirically untested.
2. **Closed-form misnomer.** "Closed-form" + "matrix-free Neumann series + rSVD" is a tension. Is the user getting a closed-form formula or an iterative computation that converges to one?
3. **Spectrum framing.** "Vulnerability spectrum" implies coverage. The IGNN-only formal guarantee gives ranking + radius for *one* model class; the extension to GCN/SAGE/etc. is empirical. Is the "spectrum" framing a motte-and-bailey?
4. **AGNNCert "complementarity."** Convenient when AEGIS produces a looser radius. Is this principled or a way to escape a head-to-head loss on certified semantics?
5. **GR-BCD 17× slowdown.** Defended via "label-freeness + closed-form determinism," but is determinism a property anyone actually pays for at small N?
6. **Power-grid case study placement.** Proof-of-concept disclaimer is in §VII, but the case study is prominent in abstract + intro. Is that a bait-and-switch?
7. **Title rename.** "Closed-Form Adversarial Diagnostics over the GNN Vulnerability Spectrum" was forced by G4. Does the new title still imply formal guarantees the body doesn't deliver?
8. **R2_04 bug.** Two columns mislabelled in an 8-hour run, salvaged via postprocessing. Does this raise the question of what else in the codebase hasn't been audited?

**Output mode.** Strongest counter-argument (200–300 words) + CRITICAL/MAJOR/MINOR issue list with dimension + location + ignored alternatives + missing stakeholders + so-what test.

---

## Panel rules (this round)

- All five reviewers receive the same Phase-1 paper-blind brief: read only the LaTeX sources in `paper/`, the R2 evidence pack (`docs/r2_experiments_full_report.md`), and the framing patches (`docs/r2_framing_patches.md`). Reviewers do **not** see the R1 panel reports during Phase 1.
- The R1 verdict (Major Revision, 10 gating items) is provided to the Editorial Synthesizer in Phase 2, not to individual reviewers.
- Devil's Advocate CRITICAL findings cannot be ignored by Phase 2 (Checkpoint Rule #4).
- All five reports + the editorial decision are written to `docs/review_2026-05-29_r2/`.

## Open question for the author (before Phase 1)

The R2 pack closes G1 only insofar as it shows $\kappa$ saturating below the ceiling — but does **not** show a successful $\varepsilon > \ecrit$ breach experiment. **Was G1 actually crossed empirically, or was it converted into a "scope-limited" disclosure?** Phase 1 reviewers will be told to assume the latter unless you confirm the former and point me to the artefact.
