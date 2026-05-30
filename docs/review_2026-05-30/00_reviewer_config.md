# Reviewer Configuration Card — Phase 0 Field Analysis

**Manuscript:** *AEGIS: One-Query Adversarial Diagnostics over the GNN Vulnerability Spectrum*
**Review date:** 2026-05-30 · **Round:** 3 (simulated; prior rounds 2026-05-29, 2026-05-29_r2)
**Mode:** `full` (5 reviewers + Editorial Decision)

---

## Field Analysis

| Dimension | Finding |
|-----------|---------|
| **Primary discipline** | Trustworthy / adversarial ML for **Graph Neural Networks** (data mining) |
| **Secondary disciplines** | Implicit/equilibrium deep learning (DEQ/IGNN); randomized numerical linear algebra (Neumann resolvent, randomized SVD); spectral / pseudospectral theory |
| **Tertiary (application)** | Power-systems contingency analysis (N-1, LODF) — case study |
| **Research paradigm** | Constructive method + closed-form theoretical guarantees + large empirical study (quantitative) |
| **Methodology type** | Algorithm design; IFT-based sensitivity operator; matrix-free numerics; adversarial evaluation |
| **Target venue** | IEEE conference format (`IEEEtran`, conference). Calibrated to **ICDM** (IEEE Int'l Conf. on Data Mining), 10-page strict limit, top-tier (~A-rank, competitive acceptance) |
| **Maturity** | **Late-stage / near-submission.** 9 datasets × 4 domains × 10 seeds, p-values, paired Wilcoxon, adaptive-attack discipline, responsible-disclosure protocol. Already through ≥2 prior review rounds |

**Core object.** The *constrained sensitivity matrix* `S_c` — one matrix-free object (Neumann-series resolvent + randomized SVD) claimed to yield three diagnostics at once: (1) SVD-optimal first-order attack direction, (2) per-edge vulnerability rankings, (3) per-node sensitivity radii `r_v`.

**Theory inventory.** 1 theorem (three-regime phase transition for contractive IGNN, `ε_crit=(1−κ)/‖W‖₂`, assumptions A1–A3), 4 propositions (attack direction, per-node radius, continuous→discrete transfer, explicit-GNN extension), 2 remarks. 5 tables, 7 figures, 1 algorithm.

**Headline empirical claims.** one-query SVD direction matches 50-step PGD; continuous→discrete transfer positive in 29/33 cells (p<10⁻⁵), τ=+0.996±0.002 on Amazon Photo (N=7,650); top-edge masking cuts σ₁ damage 42±8% vs 11% random; IEEE power flow P@10=0.66–0.81 "competitive with industry LODF"; scales to N=7,650 at 365 s / 5.5 GB.

---

## Panel Configuration (5 reviewers — independent, non-overlapping)

### EIC — Senior Area Chair, top data-mining venue (ICDM/KDD/TKDE)
Graph-mining + trustworthy-ML generalist. **Focus:** scope fit; framing-level originality ("one object → three diagnostics" — genuine conceptual unification or repackaging of IFT sensitivity?); significance to the ICDM audience; calibration of title/abstract claims vs delivered results; conference-appropriateness. Does **not** verify proofs (R1) or attack-literature minutiae (R2).

### Reviewer 1 — Methodology (THEORY + numerics)  ·  agent: `ml-theory-reviewer`
Expert in implicit/equilibrium deep learning, spectral methods, randomized NLA, adversarial-robustness theory. **Focus:** line-by-line correctness of Thm 1 (conservative-IFT handling of ReLU nonsmoothness under A1; is κ<1 verified or assumed; the η pseudospectral index and Rem. `eta_relu` — proof needs φ'≡1, so does the stated theorem strictly cover trained ReLU models?); Prop. `transfer` rigor (subcritical assumption, first-order vs higher-order bridge); Neumann convergence + randomized-SVD error vs the "one query / matches dense σ₁ to 0.03" claim; statistical reporting; reproducibility.

### Reviewer 2 — Domain (adversarial graph ML)
Expert in graph attacks/defenses & certificates (Nettack, Mettack, PR-BCD/GR-BCD, randomized smoothing, AGNNCert, "sober-look" critiques). **Focus:** literature coverage & positioning (is "each thread supplies at most one diagnostic" fair?); baseline fairness (one-query S_c vs 50-step PGD — apples-to-apples? are GR-BCD/PR-BCD run in their intended budget regime? is "SVD dominates" only in the small-ε regime where first-order is optimal by construction?); honesty of the AGNNCert complementarity framing; net contribution to the graph-adversarial field; missing references.

### Reviewer 3 — Perspective (cross-disciplinary: power systems + safety-critical deployment)
Expert in power-grid contingency analysis (N-1, LODF, PTDF) and deployment/ethics of ML auditing tools. **Focus:** validity of the power-flow case study (is P@10=0.66–0.81 truly "competitive with industry LODF", which is *exact* linear sensitivity? is the GNN a faithful PF model?); whether the safety-critical motivations (drug-interaction, fraud) are *evaluated* or only *motivational*; cross-domain generality; dual-use / disclosure-protocol adequacy; practitioner actionability.

### Devil's Advocate — core-thesis challenger
**Strongest counter-arguments to stress:** (a) "one-query" unification is presentational — three trivial reads (leading singular vector / column norms / margins) of one Jacobian the paper itself says "specialises equilibrium IFT sensitivity"; (b) the headline "matches 50-step PGD" leans on Shift-PGD, which the paper flags as "solver validation, not an independent baseline" — risk of a self-referential comparison; does SVD actually beat the *independent* Cls-PGD outside the small-ε regime? (c) theory↔headline mismatch: Thm 1 covers only contractive spectral-norm IGNN, while the abstract leads with the GCN/SAGE/GIN/APPNP/GAT extension that carries no regime guarantee; (d) cherry-picking (τ=+0.996 on one dataset; 4/33 cells fail — which?; GR-BCD decorrelates on Cora at τ=+0.16); (e) "so what?" — a first-order threshold that "can be violated at larger magnitudes" for the safety-critical decisions the intro invokes.

---

## Output plan
`01_eic_review.md` · `02_methodology_review.md` · `03_domain_review.md` · `04_perspective_review.md` · `05_devils_advocate.md` · `06_editorial_decision.md`

Scoring per `quality_rubrics.md` (Originality 20 / Methodological Rigor 25 / Evidence 25 / Coherence 15 / Writing 15; ≥80 Accept · 65–79 Minor · 50–64 Major · <50 Reject). IRON RULES enforced: reviewers are read-only (never edit the manuscript), evidence-based (cite passages), independent (no cross-referencing); a Devil's Advocate CRITICAL finding blocks an Accept decision.
