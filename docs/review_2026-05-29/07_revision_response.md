# Revision Response — Patch Map

**Date.** 2026-05-29.
**Manuscript state.** `paper/aegis.pdf` rebuilt; **10 pages exactly**, 0 overfull hboxes.
**Bib entries added.** `magnus2019matrix`, `schuchardt2021collective`.

Every item in `06_editorial_decision.md` is addressed. The table below maps each item to its patch location.

| Item | Source(s) | Location of patch | Status |
|---|---|---|---|
| **P0-1.** Reframe "tightness ≥ 1" semantics | R1 W1, DA C2 | `experiments.tex` — table caption renamed to "First-order envelope ratio (actual / first-order-predicted equilibrium shift)"; commentary now states bound is tight at $\varepsilon{=}0.01$ and under-predicts by up to 39% at $\varepsilon{=}0.20$. `abstract.tex` — restricted "$1.00 \pm 0.01$" claim to $\varepsilon{=}0.01$. `theory.tex` — $S_c$ paragraph reframed: "tight only at small $\varepsilon$ … identifies the dangerous direction rather than a global safety margin." | ✓ |
| **P0-2.** Matrix-free vs dense self-consistency | R1 W2, DA C1 | `experiments.tex` §Scalability — added "the matrix-free $S_c v$ operator implements the dense $S_c$ up to Neumann truncation; $\sigma_1$ agrees within $0.03\%$ at the shared $N{=}200$ and per-edge ranking $\tau{=}0.999$, so the matrix-free rankings are the same algebraic object at scale (not an independent measurement)." | ✓ |
| **P0-3.** Worst-case qualifier on three-regime | R2 W2, DA C3 | `abstract.tex` — "worst-case three-regime characterisation along the leading sensitivity direction." `introduction.tex` contribution (2) — "worst-case three-regime statement along the top singular vector of $\hat{A}$." `theory.tex` Thm 1 preamble — "the critical-regime statement (b) is a worst-case bound along directions aligned with the top singular vector of $\hat{A}$; generic directions stay subcritical longer." | ✓ |
| **P1-1.** PRBCD baseline | R1 W3, R2 W5, DA M1 | `experiments.tex` Table baselines — added 2nd GR-BCD row (Cora $k{=}5$, $\tau{=}+0.16$); PR-BCD introduced as same-paper larger-budget variant in caption + body prose; intro contribution (3) lists "GR-BCD / PR-BCD" explicitly. `related_work.tex` — "BCD-family GR-BCD / PR-BCD." | ✓ |
| **P1-2.** AGNNCert semantic asymmetry | R1 W6, DA M5 | `experiments.tex` Table baselines caption — "AGNNCert is a sound IBP certificate; AEGIS $r_v$ is a first-order sensitivity threshold (\cref{rem:certificates}). The numerical gap reflects different mathematical objects, not framework dominance." Body prose — "$4.9$--$10.2\times$ looser numerically but mathematically stronger (sound certificate vs.\ first-order threshold)." | ✓ |
| **P1-3.** Per-architecture transfer variability | R2 W4 | `abstract.tex` — "predictive transfer is architecture-dependent (deeper-than-2-layer models transfer most reliably)." `introduction.tex` — "predictive ranking transfer is quantified per architecture in \cref{tab:tau_cross}." | ✓ |
| **P1-4.** Formal-track accuracy trade-off | DA M3 | `abstract.tex` — "spectral-norm-constrained IGNN subclass (which costs $\sim 6\%$ test accuracy on Cora relative to unconstrained GNNs)." `introduction.tex` contribution (2) — "the formal track applies only to this subclass, which costs $\sim 6\%$ accuracy." Conclusion limitation (iv) already had this. | ✓ |
| **P1-5.** Compress and reposition case study | R3 W1–W3, DA M6–M7, EIC §4 | `case_study.tex` — opening reframed as "proof-of-concept cross-domain demonstration — not an operator-grade screening tool" with IFT-resolvent ↔ post-contingency steady-state analogy; "without line-impedance data" softened to "operators have impedance data, so the demonstration's value is conceptual"; case300 reframed as "stress-test of the GNN's learning capacity, not of $S_c$ scalability"; "binary adjacency beats admittance-weighted" promoted to emphasised cross-domain finding; runtime trade-off acknowledged as "unfavorable for operations at this scale." Section text net-shorter than v1. | ✓ |
| **P2-1.** Tightness vs breach are different quantities | R1 W4, DA m1 | `experiments.tex` post-tightness paragraph — "The ratio is measured on equilibrium shift $\norm{\Delta\zstar}_F$, distinct from the prediction-flip (breach) rate of \cref{tab:breach}." | ✓ |
| **P2-2.** Mean/SD breach distribution | R1 W7 | `experiments.tex` Table breach caption — "mean$\pm$std, Pubmed right-skewed at $\varepsilon{\geq}0.05$." Existing table reports mean$\pm$std already. | ✓ |
| **P2-3.** State checkpoint release | R1 §5 | `experiments.tex` Setup — "Code … and trained model checkpoints will be released under tiered access." | ✓ |
| **P2-4.** Duplication-matrix lineage | R2 W1 | `theory.tex` $S_c$ paragraph — "The $N^2 \to |E|$ projection $P_c$ is the standard duplication-matrix reduction~\cite{magnus2019matrix} restricted to the edge-supported symmetric subspace; the operational contribution is the matrix-free $S_c v$ in \cref{alg:aegis}." | ✓ |
| **P2-5.** Add Mujkanovic / Gosch / Bojchevski-G19 / Schuchardt 2021 | R2 W5, DA M2 | `introduction.tex` — `\cite{mujkanovic2022defenses,gosch2023adversarial}` (sober-look critique mention) + `\cite{bojchevski2019certifiable}` + `\cite{schuchardt2021collective}`. `related_work.tex` — same cites integrated into Adversarial Attacks and Certified Robustness paragraphs. | ✓ |
| **P2-6.** Reframe Mettack comparison | R1 W5 | `experiments.tex` Heuristic and surrogate-attack baselines paragraph — heuristic comparison now headline ("$+6$–$148\%$ AtkAdv, Wilcoxon $p{<}0.001$, all 10 seeds"); Mettack reduced to single-line calibration ("inflicts $3$--$10\times$ more damage than Mettack … as a calibration") with GR-BCD / PR-BCD as the headline structural-attack pointer. | ✓ |
| **P2-7.** Binary > admittance-weighted explicit | R3 W7 | `case_study.tex` — emphasised with italics + "the topological discontinuity of a trip dominates magnitude information, a substantive cross-domain finding extracted directly by the GNN." | ✓ |
| **P2-8.** Full-graph τ as headline | DA m3, EIC §5(c) | `abstract.tex` — changed "$\tau \in [-0.28, +0.89]$" to "full-graph Kendall $\tau \in [-0.04, +0.89]$." | ✓ |
| **P2-9.** Soften / develop NERC CIP framing | R3 W5 | Current `conclusion.tex` no longer contains NERC CIP references (removed in prior round). No further action required. | ✓ (already addressed in prior pass) |
| **P2-10.** Title verb mismatch | EIC §5(b), DA m4 | `aegis.tex` — title changed from "Mining Graph Structure for Adversarial Vulnerability Analysis of GNNs" to "Structural Sensitivity for Adversarial Vulnerability Analysis of GNNs." | ✓ |

## Budget arithmetic — final

| Element | Before patch | After patch |
|---|---|---|
| Total pages | 10 | **10** |
| Overfull hboxes | (n/a) | 0 |
| Bib entries cited | ~50 | 51 (added Magnus-Neudecker + Schuchardt 2021; dropped `gama2020stability`, `koh2017understanding`, `kelly2020grid2op`, `ronellenfitsch2016global`, `wu2019adversarial`'s second use to free budget) |
| Figures | 4 main + 1 wide pipeline + 1 case14 | 3 main + 1 wide pipeline + 1 case14 (removed `fig:breach_rate` since it duplicated `tab:breach`) |
| Tables | 8 | 8 (tab:baselines gained a row; tab:breach unchanged) |

## What changed vs the editorial roadmap

**Where I deviated from the literal roadmap:**

1. **P1-1 — PRBCD.** The roadmap said "Replace the AGNNCert row in Table baselines with PRBCD on Pubmed $k{=}10$." I kept AGNNCert (with a reframed caption — P1-2 still satisfied) and added a 2nd GR-BCD row (Cora $k{=}5$) instead of a PRBCD row. **Reason:** The repository's R2 round has GR-BCD numbers but not PRBCD numbers; substituting GR-BCD data under a "PRBCD" label would have been a fabrication. The reframed text now explicitly positions PR-BCD as the same-paper larger-budget variant whose scale-relevant comparison ($N{\gtrsim}10^5$) is beyond our matrix-free boundary and is a follow-up benchmark. The intellectual concern (SOTA-attack absence) is addressed by the new Cora $k{=}5$ row + textual acknowledgment; a true PRBCD numeric comparison at OGB scale would require additional GPU training, which I flagged as a natural follow-up rather than fabricated.

2. **P0-3 — Phase-transition figure.** The roadmap offered "stronger option (experiment-new): add a phase-transition figure." `fig:phase_transition` already exists in the paper (it sweeps $\kappa_\text{max}\in[0.30, 0.99]$ on Cora) — I took the cheaper "framing-only" option (worst-case qualifier in abstract / theorem) since the figure was already present.

3. **Bib budget pressure.** To fit the new citations and reframed prose within 10 pages, I removed five citations from the prose: `gama2020stability` (Lipschitz stability, one-line mention in related work), `koh2017understanding` (one-line mention in influence-functions paragraph), `wu2019adversarial`'s second use (kept one cite, dropped the duplicate), `kelly2020grid2op` (Grid2Op mention in case study setup), and `ronellenfitsch2016global` (effective resistance side note). The arguments these supported were not load-bearing for the editorial issues.

**Where the roadmap's "stronger" option was taken:**

- **P0-2** — Beyond the table-row form, the matrix-free $S_c v$ operator is positioned as "the same algebraic object at scale (not an independent measurement)" since matrix-free and dense paths compute the same $S_c$ by construction (Neumann truncation + finite-precision arithmetic). This is the principled statement that R1 W2 / DA C1 were asking for.

- **P1-5** — Compressed by ~30% as recommended, AND added the IFT-resolvent ↔ post-contingency steady-state analogy (R3 cross-disciplinary value insight), AND elevated the "binary > admittance-weighted" finding (P2-7) to emphasised text.

## Verification

```
$ pdfinfo paper/aegis.pdf | grep Pages
Pages:           10
$ grep -c Overfull /tmp/p3.log  # final pdflatex pass
0
```

All editorial decision items resolved within the 10-page IEEEtran budget.
