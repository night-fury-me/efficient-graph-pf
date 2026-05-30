# Phase 0 — Reviewer Configuration

**Paper:** AEGIS: Structural Sensitivity for Adversarial Vulnerability Analysis of GNNs
**Anonymous draft;** IEEEtran conference (`\documentclass[conference]{IEEEtran}`)
**Submission date assumed:** 2026-05-29

## Field analysis

| Field | Value |
|-------|-------|
| Primary discipline | Machine learning — graph neural networks, adversarial robustness |
| Secondary discipline | Numerical linear algebra (IFT, Neumann series, randomised SVD); power systems (case study) |
| Research paradigm | Method contribution + theory + empirical evaluation |
| Methodology type | Algorithmic + formal regime characterisation + 330-run empirical study + cross-domain case study |
| Target venue tier | IEEE conference, plausibly ICDM/IEEE BigData/SaTML; ~10-page budget |
| Paper maturity | Mature — has ablations, paired statistics, ethics, head-to-head baselines, multiple revision marks |
| Calibration | Top-tier conference; hostile-but-fair |

## Reviewer roster (5 personas)

1. **EIC** — Senior PC chair generalist; venue fit, originality, scope clarity, framing.
2. **R1 Methodology** — Adversarial ML + numerical linear algebra (Neumann, randomised SVD, conservative IFT for ReLU, contraction analysis, rank statistics). Owns the theorems.
3. **R2 Domain — GNN adversarial robustness** — Nettack/Mettack/PR-BCD/AGNNCert/smoothing literate; defends ranking-semantics rigor and threat model.
4. **R3 Perspective — Power systems** — AC-PF, N-1/N-k, admittance vs binary edges, PTDF/LODF, operator-grade severity metrics, Grid2Op.
5. **Devil's Advocate** — Hostile reviewer; vacuity, novelty inflation, scope mismatch, dual-use overreach.

All five files dispatched in parallel in Phase 1.
