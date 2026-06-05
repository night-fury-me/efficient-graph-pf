# Phase 0 — Field Analysis & Reviewer Configuration

**Paper:** *AEGIS: A Matrix-Free Operator — Audit, Certify, and Defend Graph Neural Networks*
**Venue (declared):** AAAI-2026 (anonymous submission), 7pp two-column + unlimited appendix
**Review date:** 2026-06-05 · **Mode:** full (5 reviewers) · **Reviewed build:** `aaai_aegis.tex` (canonical)

## Field classification
- **Primary discipline:** Machine learning — adversarial robustness of graph neural networks.
- **Secondary disciplines:** Implicit/equilibrium models (DEQ/IGNN); conformal prediction / distribution-free uncertainty; numerical linear algebra (IFT sensitivity, Neumann/resolvent, randomized SVD); matrix perturbation theory (pseudospectra, Weyl/Perron).
- **Research paradigm:** Constructive method paper with supporting theory (2 theorems, 4 propositions, 1 lemma, 1 coverage theorem) + a broad empirical study (6 datasets, 7 architectures, 4 domains, 390 runs, 10 seeds).
- **Paper maturity:** Late-stage, heavily revised (the appendix is unusually candid about scope and open gaps — a positive signal). Prior internal review round exists (`review/R1..R4`, `EDITORIAL_DECISION.md`).
- **Target tier:** Top-tier ML conference. Bar = clear novelty *or* a decisive empirical/theoretical advance, fully defensible under hostile review.

## Central claim under test
A single matrix-free operator — the constrained sensitivity matrix `S_c = (I−J_z)^{-1} J_A P_c` — yields, from one randomized SVD pass, (1) an **audit** (optimal first-order attack direction, per-edge ranking, per-node radii), (2) a **certificate** (AEGIS-Conformal robust coverage + closed-form `ε_crit` for contractive IGNNs), and (3) a **defense** (penalize `σ_1(S_c)`). The pitch is *unification + coupling*, not winning any single axis.

## Configured panel (5 reviewers)

| # | Persona | Expertise | Lane (non-overlapping) |
|---|---------|-----------|------------------------|
| EIC | Area Chair, robust/trustworthy ML | Broad ML, GNNs, paper-craft | Fit, originality, significance, headline-vs-delivery gap, presentation/density, tone |
| R1 | Methodology & theory referee | IFT/implicit-diff, conformal, matrix perturbation, empirical rigor | Proof correctness (ε_crit, bracket, conformal), experimental design, statistics, reproducibility |
| R2 | Domain referee | GNN adversarial robustness & certification literature | Novelty of `S_c`, positioning vs DEQ/influence/conformal/spectral-reg, missing refs, contribution increment |
| R3 | Cross-disciplinary / impact referee | Deployed GNNs, applied robustness, dual-use | Real-world applicability of the IGNN/contractive scope, delocalization, power-grid framing, broader impact, fundamental-assumption challenge |
| DA | Devil's Advocate | — | Strongest counter-argument to the unification thesis; cherry-pick / confirmation-bias / "so what" tests |

**Adjustable.** If you'd prefer a referee weighted explicitly toward (a) conformal-prediction theory, (b) power-systems/physics, or (c) numerical linear algebra, say so and I'll re-run that lane. Otherwise the above is the default panel and the reports below stand.
