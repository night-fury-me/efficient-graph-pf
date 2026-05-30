# Revision Round 2 — Experiment Scripts

Generated from the editorial decision letter at
`docs/review_full_2026-05-28/06_editorial_decision.md`.

Each script is a runnable Python file that mirrors the existing
`scripts/exp_*.py` pattern: 10 fixed seeds, output to `results/revision_R2/*.csv`,
imports the `iem.adversarial` / `iem.scalable` API.

| Script | Closes | Effort | Output CSV |
|---|---|---|---|
| `R2_01_grbcd_baseline.py`      | **P1.3** GR-BCD discrete-attack baseline (≥3 datasets) | M–H | `grbcd_baseline.csv` |
| `R2_02_agnncert_comparison.py` | **P1.4** AGNNCert per-node-radii comparison on 50-node subgraphs | M | `agnncert_comparison.csv` |
| `R2_03_stats_reanalysis.py`    | **P1.5 + P2.7** 95% CI + per-cell Wilcoxon + sign-test | L | `stats_reanalysis.csv` |
| `R2_04_matfree_error_bounds.py`| **P1.6** Neumann residual + Halko bound + N=500 dense sanity check | M | `matfree_error_bounds.csv` |
| `R2_05_pi_baseline.py`         | **P1.8** PI (Ejebe–Wollenberg) ranking baseline on case57 / case118 | M | `pi_baseline.csv` |
| `R2_06_lodf_metric_retarget.py`| **P1.9 + P3.9** thermal-overload + voltage-violation metric; AEGIS vs LODF disagreement | M | `lodf_retarget.csv`, `lodf_disagreement.csv` |
| `R2_07_kappa_direct.py`        | **P2.2** report κ = ‖J_z‖₂ directly (replace ρ); (‖Â‖₂, ‖W‖₂, κ) triple | L | `kappa_direct.csv` |
| `R2_08_fullgraph_repro.py`     | **P2.3** full-graph reproduction of `tab:baselines` + `tab:greedy_topk` on Cora + Citeseer | M–H | `fullgraph_repro.csv` |
| `R2_09_iterative_reranking.py` | **P2.4** iterative greedy re-ranking proof-of-concept | M | `iterative_reranking.csv` |
| `R2_10_robust_arch.py`         | **P3.10** apply S_c to RobustGCN + GNNGuard backbones | M | `robust_arch.csv` |

## How to run

All scripts use the same SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
as the rest of the paper.

```bash
cd /home/redwanul/Storage/Work/PR-LAB/GNN_load_flow/GNN_load_flow/GNN/SimpleGNN
.venv/bin/python scripts/revision_R2/R2_01_grbcd_baseline.py
.venv/bin/python scripts/revision_R2/R2_02_agnncert_comparison.py
# ... etc
```

Or all at once (~6–10 hours on a single RTX 4090):

```bash
bash scripts/revision_R2/run_all.sh
```

## Outputs

Each script writes a CSV under `results/revision_R2/`. The CSVs include:
- per-seed values (10 rows per cell)
- mean, std, 95% CI, Wilcoxon p-value where applicable
- a JSON metadata sidecar with hyperparameters

## What still needs manual integration

After running the scripts, paste the resulting numbers into the paper:
- **P1.3 GR-BCD numbers** → `tab:baselines` and `tab:greedy_topk` (add columns)
- **P1.4 AGNNCert numbers** → new `tab:agnncert` after `tab:smoothing`
- **P1.5 CIs + Wilcoxon** → all results tables (replace mean ± std with mean [95% CI] + p-value)
- **P1.6 Halko bound + Neumann residual** → `tab:scalability` (new columns) + a Methodology paragraph
- **P1.8 PI numbers** → `tab:ieee` (new row or column)
- **P1.9 retargeted LODF** → `case_study.tex` LODF comparison paragraph
- **P2.2 κ values** → `tab:cross_domain` (replace ρ with κ; add ‖Â‖₂ + ‖W‖₂ columns)
- **P2.3 full-graph numbers** → Appendix or extended table
- **P2.4 iterative numbers** → `tab:greedy_topk` (new row)
- **P3.10 robust-arch numbers** → new appendix subsection in `experiments.tex`

## Dependencies

These scripts use only the existing `iem.*` API. No new heavy dependencies.

For **P1.4 AGNNCert**, the script implements a deterministic-radius IBP-style
certifier from the equations in Li et al.\ (2025); if the AGNNCert authors
release an official codebase, swap in their implementation.

For **P1.3 GR-BCD**, the script implements Geisler et al.\ (2021) per-edge
block-coordinate descent attack; this is straightforward from the equations
in the paper (no external dependency).

For **P3.10**, RobustGCN / GNNGuard need to be installed (or copied from
their respective repos). The script assumes a stub model class at
`iem/models/robust_gcn.py` — if not present, the script will skip with a
warning.
